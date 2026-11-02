"""The run's episode queue (src/simtest/runqueue.py): the blocked order, the total in flight,
re-runs at the back, and the episodes that end without starting."""

from __future__ import annotations

import json
from collections import Counter

import anyio
import pytest

from simtest import runqueue as Q

ARMS = [("C0", "none"), ("C3", "agentworld"), ("C2", "agentworld")]
SCEN = ["a", "a_benign", "b", "b_benign", "c", "c_benign"]


def test_the_order_is_blocked_by_epoch_and_scenario_and_seeded():
    order = Q.episode_order(ARMS, SCEN, 3, seed=7)
    assert len(order) == len(set(order)) == 3 * 6 * 3
    assert order == Q.episode_order(ARMS, SCEN, 3, seed=7)
    assert order != Q.episode_order(ARMS, SCEN, 3, seed=8)
    # Epoch-major: every episode of epoch 1 comes before any of epoch 2.
    assert [k[3] for k in order] == sorted(k[3] for k in order)
    # Each block of 3 is one (scenario, epoch) with one episode of each arm.
    for i in range(0, len(order), 3):
        block = order[i:i + 3]
        assert len({(k[2], k[3]) for k in block}) == 1
        assert sorted((k[0], k[1]) for k in block) == sorted(ARMS)
    # So at every point each arm has started the same number of episodes, give or take one.
    seen = Counter()
    for k in order:
        seen[k[0]] += 1
        assert max(seen.values()) - min(seen[a] for a, _ in ARMS) <= 1
    # The arms' position in their block is random, not fixed (C0 isn't always first).
    assert len({order[i][0] for i in range(0, len(order), 3)}) > 1


def test_parse_epochs():
    assert Q.parse_epochs(None, 3) is None
    assert Q.parse_epochs("2", 3) == {2}
    assert Q.parse_epochs("1-3", 3) == {1, 2, 3}
    for bad in ("0", "3-2", "1-4", "x"):
        with pytest.raises(ValueError):
            Q.parse_epochs(bad, 3)


def _admits(log_dir):
    events = [json.loads(line) for line in (log_dir / "queue.jsonl").read_text().splitlines()]
    return [(tuple(e["key"]), e["attempt"]) for e in events if e["event"] == "admit"]


def _run(queue, keys, work, results):
    async def one(key):
        try:
            async with queue.slot(key) as adm:
                results.append(("start", key, adm.attempt))
                await work(key, adm)
        except Q.NotStarted as exc:
            results.append(("not_started", key, exc.reason))

    async def main():
        async with anyio.create_task_group() as tg:
            for k in keys:
                tg.start_soon(one, k)

    anyio.run(main)


def test_admits_in_order_and_never_more_than_in_flight(tmp_path):
    order = Q.episode_order(ARMS, SCEN, 2, seed=1)
    q = Q.EpisodeQueue(order, 4, log_dir=tmp_path, settle=0.05)
    running, peak, results = set(), [0], []

    async def work(key, adm):
        running.add(key)
        peak[0] = max(peak[0], len(running))
        await anyio.sleep(0.01)
        running.discard(key)

    _run(q, list(reversed(order)), work, results)   # arrival order must not matter
    assert peak[0] == 4
    events = [json.loads(line) for line in (tmp_path / "queue.jsonl").read_text().splitlines()]
    admits = [e for e in events if e["event"] == "admit"]
    assert [tuple(e["key"]) for e in admits] == order
    assert len(admits) == len(order) and all(sum(e["running"].values()) <= 4 for e in admits)   # itself included
    assert q.counts["admitted"] == len(order) and q.counts["not_started"] == 0


def test_a_rerun_goes_behind_every_first_attempt(tmp_path):
    order = Q.episode_order(ARMS, ["a", "b"], 1, seed=3)   # 6 episodes
    q = Q.EpisodeQueue(order, 2, log_dir=tmp_path, settle=0.05)
    first, results = order[0], []
    failed = set()

    async def one(key):
        while True:   # what Inspect's in-place retry does: the same sample comes back
            try:
                async with q.slot(key) as adm:
                    results.append((key, adm.attempt))
                    await anyio.sleep(0.01)
                    if key == first and key not in failed:
                        failed.add(key)
                        raise RuntimeError("tunnel dropped")
                return
            except RuntimeError:
                continue

    async def main():
        async with anyio.create_task_group() as tg:
            for k in order:
                tg.start_soon(one, k)

    anyio.run(main)
    admits = _admits(tmp_path)
    assert admits[0] == (first, 1)
    assert admits[-1] == (first, 2)   # behind the 5 episodes that had not started
    assert [k for k, _ in admits[1:-1]] == order[1:]
    assert q.counts["reruns"] == 1


def test_episodes_outside_the_sessions_epochs_do_not_start(tmp_path):
    order = Q.episode_order(ARMS, ["a"], 3, seed=1)
    q = Q.EpisodeQueue(order, 12, epochs={2}, log_dir=tmp_path, settle=0.05)
    results = []

    async def work(key, adm):
        await anyio.sleep(0)

    _run(q, order, work, results)
    started = [k for ev, k, _ in results if ev == "start"]
    assert sorted(started) == sorted(k for k in order if k[3] == 2)
    refused = [r for r in results if r[0] == "not_started"]
    assert len(refused) == 6 and all("outside this session" in r[2] for r in refused)


def test_after_the_admission_deadline_nothing_new_starts_and_running_ones_finish(tmp_path):
    order = Q.episode_order(ARMS, ["a", "b"], 1, seed=2)
    now = [0.0]
    q = Q.EpisodeQueue(order, 2, admit_until=10.0, log_dir=tmp_path, settle=0, clock=lambda: now[0])
    results, finished = [], []

    async def work(key, adm):
        now[0] = 20.0          # the deadline passes while the first two run
        await anyio.sleep(0.01)
        finished.append(key)

    _run(q, order, work, results)
    assert [k for k, _ in _admits(tmp_path)] == order[:2]
    assert sorted(finished) == sorted(order[:2])   # running episodes are never cut
    assert sum(r[0] == "not_started" for r in results) == 4


def test_carry_under_leaves_late_reruns_for_the_next_session(tmp_path):
    order = Q.episode_order([("C0", "none"), ("C3", "s")], ["a", "b"], 1, seed=5)   # 4 episodes
    q = Q.EpisodeQueue(order, 4, carry_under=3, log_dir=tmp_path, settle=0.05)
    last = order[-1]
    results = []

    async def one(key):
        tries = 0
        while tries < 3:
            tries += 1
            try:
                async with q.slot(key) as adm:
                    results.append(("start", key, adm.attempt))
                    await anyio.sleep(0.05 if key != last else 0.01)
                    if key == last and adm.attempt == 1:
                        raise RuntimeError("simulator timeout")
                return
            except Q.NotStarted as exc:
                results.append(("not_started", key, exc.reason))
                return
            except RuntimeError:
                await anyio.sleep(0.1)   # by now the others have finished: 1 left < 3

    async def main():
        async with anyio.create_task_group() as tg:
            for k in order:
                tg.start_soon(one, k)

    anyio.run(main)
    assert ("not_started", last) == results[-1][:2] and "carried" in results[-1][2]
    assert q.attempts[last] == 1   # the carried re-run never ran


def test_a_waiter_cancelled_before_it_starts_frees_its_place(tmp_path):
    order = Q.episode_order([("C0", "none")], ["a", "b", "c"], 1, seed=0)
    q = Q.EpisodeQueue(order, 1, log_dir=tmp_path, settle=0)
    started = []

    async def main():
        async def hold():
            async with q.slot(order[0]):
                started.append(order[0])
                await anyio.sleep(0.05)

        async def doomed():
            with anyio.move_on_after(0.01):
                async with q.slot(order[1]):
                    started.append(order[1])

        async def later():
            await anyio.sleep(0.02)
            async with q.slot(order[2]):
                started.append(order[2])

        async with anyio.create_task_group() as tg:
            tg.start_soon(hold)
            await anyio.sleep(0.001)
            tg.start_soon(doomed)
            tg.start_soon(later)

    anyio.run(main)
    assert started == [order[0], order[2]] and not q.waiting and not q.running


def test_only_one_queue_is_installed_at_a_time():
    q = Q.EpisodeQueue([("C0", "none", "a", 1)], 1)
    with Q.installed(q):
        assert Q.current() is q
        with pytest.raises(RuntimeError):
            with Q.installed(q):
                pass
    assert Q.current() is None


def test_not_started_messages_are_recognised():
    assert Q.is_not_started(str(Q.NotStarted("x")))
    assert Q.is_not_started("RuntimeError('simtest-queue: not started: x')")
    assert not Q.is_not_started("simulator timed out") and not Q.is_not_started(None)


def test_batch_mix_per_arm(tmp_path):
    lines = [{"event": "admit", "key": ["C0", "none", "a", 1], "running": {"C0/none": 1}},
             {"event": "admit", "key": ["C3", "s", "a", 1], "running": {"C0/none": 1, "C3/s": 1}},
             {"event": "release", "key": ["C0", "none", "a", 1], "running": {"C3/s": 1}},
             {"event": "admit", "key": ["C0", "none", "b", 1], "running": {"C0/none": 1, "C3/s": 1}}]
    (tmp_path / "queue.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    mix = Q.batch_mix(tmp_path)
    assert mix == {"C0/none": {"C0/none": 0.0, "C3/s": 0.5}, "C3/s": {"C0/none": 1.0, "C3/s": 0.0}}
    assert "C3/s: 1.0, 0.0" in Q.format_batch_mix(mix)
    assert Q.batch_mix(tmp_path / "nothing") is None


def test_never_started():
    admitted = {Q.ADMITTED_KEY: {"id": "1"}}
    assert not Q.never_started(None, {}, True)
    assert Q.never_started("simtest-queue: not started: x", admitted, False)
    assert Q.never_started("CancelledError()", {}, True)           # cancelled while waiting
    assert not Q.never_started("CancelledError()", admitted, True)  # cancelled while running: an attempt
    assert not Q.never_started("CancelledError()", {}, False)       # a log from before the queue
