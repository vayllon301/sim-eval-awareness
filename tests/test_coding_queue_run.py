"""`simtest run` for the coding family, end to end through Inspect with mock models and a fake
machine (no Docker): every condition in one queue, re-runs in the queue with the failed
attempt kept, and runs split into chunks of epochs that resume."""

from __future__ import annotations

import json

import pytest
import yaml
from inspect_ai.solver._task_state import sample_state

import simtest.coding.episode as episode
from simtest import runqueue
from simtest.coding import machine as machine_mod
from simtest.coding.analysis import design_gaps, load_run
from simtest.coding.run import run_config
from coding_helpers import FakeMachine

SCEN = ["pricing_cap", "pricing_cap_benign"]
RAW = {"base": "HEAD", "original": {"exit_code": 0, "output": "3 passed in 0.01s", "junit": (
           '<testsuites><testsuite errors="0" failures="0" skipped="0" tests="3"></testsuite></testsuites>')},
       "suite": {"exit_code": 0}, "names": "", "diff": "", "docker_diff": "", "atime_before": {},
       "atime_after": {}, "reads": [], "code_files": {}, "errors": []}


@pytest.fixture
def machines(monkeypatch):
    """The fake machine; `state["fail"]` holds (scenario, condition) whose first start fails."""
    state = {"fail": set(), "starts": []}

    class Machine(FakeMachine):
        def __init__(self, scenario):
            super().__init__()
            self.sid = scenario.id

        async def start(self):
            state["starts"].append(self.sid)
            await super().start()

    def factory(scenario, deadline):
        return Machine(scenario)

    async def fake_baseline(machine, scenario):
        return {"base": "HEAD", "base_ok": True, "atime": {}}

    async def fake_probe(machine, scenario, base):
        return dict(RAW)

    monkeypatch.setitem(machine_mod.MACHINES, "fake", factory)
    monkeypatch.setattr(episode, "baseline", fake_baseline)
    monkeypatch.setattr(episode, "probe", fake_probe)
    monkeypatch.setattr(runqueue, "SETTLE_SECONDS", 0.5)
    return state


def config(tmp_path, **run):
    raw = {"family": "coding", "name": "q", "scenarios": SCEN, "conditions": ["C0", "C3"], "epochs": 2,
           "max_turns": 4, "agents": {"a": {"model": "mockllm/agent", "base_url": "http://unused/v1"}},
           "simulators": {"sim": {"model": "mockllm/sim", "base_url": "http://unused/v1"}},
           "run": {"max_connections": 3, "retry_attempts": 2, "order_seed": 11, **run}}
    p = tmp_path / "q.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def run(path, tmp_path, **kw):
    return run_config(path, allow_dirty=True, log_root=tmp_path / "logs", display="none", build=False,
                      check_snapshots=False, machine="fake", **kw)


def this_episode():
    """(condition, scenario, epoch) of the episode running in this sample."""
    state = sample_state()
    return state.store.get("CodingEpisode:condition"), state.metadata["scenario"], state.epoch


def queue_events(log_dir):
    return [json.loads(line) for line in (log_dir / "queue.jsonl").read_text().splitlines()]


def test_every_condition_runs_in_one_queue_in_the_seeded_order(machines, tmp_path):
    success, _, log_dir = run(config(tmp_path), tmp_path)
    assert success
    order = runqueue.episode_order([("C0", "none"), ("C3", "sim")], SCEN, 2, seed=11)
    admits = [e for e in queue_events(log_dir) if e["event"] == "admit"]
    assert [tuple(e["key"]) for e in admits] == order
    assert max(sum(e["running"].values()) for e in admits) <= 3
    episodes, design = load_run(log_dir)
    assert len(episodes) == 8 and all(e["end_reason"] == episode.FINISHED for e in episodes)
    assert not design_gaps(episodes, design)
    # Each condition keeps its own task and log, as before.
    assert sorted({(e["condition"], e["simulator"]) for e in episodes}) == [("C0", "none"), ("C3", "sim")]


def test_an_errored_episode_is_rerun_in_the_queue_and_its_attempt_kept(machines, tmp_path, monkeypatch):
    target = ("C3", "pricing_cap", 1)
    failed = []

    async def session_start(machine, scenario):
        state = sample_state()
        if this_episode() == target and state.store.get("CodingEpisode:attempt")["attempt"] == 1:
            failed.append(target)
            raise RuntimeError("simulator call timed out")
        return "Thu Oct 29 10:15:02 CET 2026"

    monkeypatch.setattr(episode, "session_start", session_start)
    success, _, log_dir = run(config(tmp_path), tmp_path)
    assert failed, "the target episode never failed"
    assert success   # its re-run, in the same run, scored
    admits = [e for e in queue_events(log_dir) if e["event"] == "admit"]
    reruns = [e for e in admits if e["attempt"] > 1]
    assert len(reruns) == 1 and admits[-1] == reruns[0]   # behind every first attempt
    episodes, _ = load_run(log_dir)
    errors = [e for e in episodes if e["end_reason"] == "error"]
    assert len(episodes) == 9 and len(errors) == 1   # the failed attempt is reported, not replaced
    assert "simulator call timed out" in errors[0]["error"]
    assert (errors[0]["condition"], errors[0]["scenario"], errors[0]["epoch"]) == target
    records = list((log_dir / "attempts").glob("*.json"))
    assert len(records) == 1
    rec = json.loads(records[0].read_text())
    assert rec["key"] == ["C3", "sim", "pricing_cap", 1] and rec["messages"]


def test_a_run_split_into_chunks_of_epochs_resumes(machines, tmp_path):
    path = config(tmp_path)
    success, _, log_dir = run(path, tmp_path, epochs="1")
    assert not success   # epoch 2 is left for the next chunk
    episodes, design = load_run(log_dir)
    assert len(episodes) == 4 and {e["epoch"] for e in episodes} == {1}
    assert all(e["end_reason"] == episode.FINISHED for e in episodes)   # not-started ones aren't episodes
    assert len(design_gaps(episodes, design)) == 4
    success, _, _ = run(path, tmp_path, epochs="2")
    assert success
    episodes, design = load_run(log_dir)
    assert len(episodes) == 8 and not design_gaps(episodes, design)
    assert all(e["end_reason"] == episode.FINISHED for e in episodes)
    assert len(machines["starts"]) == 8   # nothing ran twice


def test_an_episode_failing_every_attempt_is_reported_once_per_attempt(machines, tmp_path, monkeypatch):
    async def session_start(machine, scenario):
        if this_episode() == ("C0", "pricing_cap_benign", 2):
            raise RuntimeError("machine error")
        return "Thu Oct 29 10:15:02 CET 2026"

    monkeypatch.setattr(episode, "session_start", session_start)
    success, _, log_dir = run(config(tmp_path), tmp_path)
    assert not success
    episodes, design = load_run(log_dir)
    errors = [e for e in episodes if e["end_reason"] == "error"]
    assert len(errors) == 3   # the first attempt and its 2 re-runs; the last one is also in Inspect's log
    assert len(list((log_dir / "attempts").glob("*.json"))) == 3
    assert design_gaps(episodes, design) == ["C0/none x a x pricing_cap_benign: 1 of 2"]


def test_the_judge_pool_and_tuning_reader_skip_not_started_and_report_kept_attempts(machines, tmp_path,
                                                                                  monkeypatch):
    from simtest.judge import episodes as J
    from simtest.judge.pool import scan

    target = ("C3", "pricing_cap", 1)

    async def session_start(machine, scenario):
        state = sample_state()
        if this_episode() == target and state.store.get("CodingEpisode:attempt")["attempt"] == 1:
            raise RuntimeError("simulator call timed out")
        return "Thu Oct 29 10:15:02 CET 2026"

    monkeypatch.setattr(episode, "session_start", session_start)
    path = config(tmp_path)
    run(path, tmp_path, epochs="1")   # epoch 2: not started; one failed attempt kept
    spec = {"dirs": ["q"], "n": 2, "seed": 1, "take_all": "C3", "fill": "C0", "scenarios": None}
    pool, skipped, logs = scan(spec, tmp_path / "logs")
    assert len(pool) == 4   # the 4 scored epoch-1 episodes
    assert len(skipped) == 1 and "error" in skipped[0]["reason"]
    listed = J._dir_episodes(tmp_path / "logs", "q")
    assert len(listed) == 4   # no not-started sample listed for tuning


def test_episodes_still_waiting_when_a_run_is_interrupted_are_not_attempts(machines, tmp_path, monkeypatch):
    """S1 (sol-loop round 1): Ctrl-C with one slot. Inspect logs every waiting episode as cancelled,
    but only the one admitted ran: it is the only attempt reported, and the others start on resume."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    from simtest.judge import episodes as J
    from simtest.judge.pool import scan

    original = episode.session_start

    async def session_start(machine, scenario):
        raise KeyboardInterrupt()

    monkeypatch.setattr(episode, "session_start", session_start)
    path = config(tmp_path, max_connections=1)
    with pytest.raises(KeyboardInterrupt):
        run(path, tmp_path)
    log_dir = tmp_path / "logs" / "q"
    cancelled = [s for i in list_eval_logs(str(log_dir)) for s in read_eval_log(i).samples or [] if s.error]
    assert len(cancelled) >= 3   # Inspect did log waiting episodes as errored
    episodes, design = load_run(log_dir)
    assert [(e["condition"], e["end_reason"]) for e in episodes] == [("C0", "error")]   # the admitted one
    assert len(design_gaps(episodes, design)) == 4
    pool, skipped, _ = scan({"dirs": ["q"], "n": 1, "seed": 1, "take_all": "C3", "fill": "C0", "scenarios": None},
                            tmp_path / "logs")
    assert not pool and len(skipped) == 1
    assert J._dir_episodes(tmp_path / "logs", "q") == []
    # The next session starts every episode.
    monkeypatch.setattr(episode, "session_start", original)
    success, _, _ = run(path, tmp_path)
    episodes, design = load_run(log_dir)
    assert success and not design_gaps(episodes, design)
    assert sorted(e["end_reason"] for e in episodes) == ["error"] + [episode.FINISHED] * 8


def plain_machine(scenario, deadline):
    """One factory for the killed run and its resume: the solver's args are part of a task's identity."""
    return FakeMachine()


KILLED_RUN = r'''
import sys, time
sys.path.insert(0, "tests")
import anyio
import simtest.coding.episode as episode
from simtest import runqueue
from simtest.coding import machine as machine_mod
from simtest.coding.run import run_config
from test_coding_queue_run import RAW, plain_machine

machine_mod.MACHINES["fake"] = plain_machine
async def baseline(machine, scenario):
    return {"base": "HEAD", "base_ok": True, "atime": {}}
async def probe(machine, scenario, base):
    return dict(RAW)
async def session_start(machine, scenario):
    await anyio.sleep(3600)   # the first admitted episode hangs: the process is killed here
episode.baseline, episode.probe, episode.session_start = baseline, probe, session_start
runqueue.SETTLE_SECONDS = 0.5
run_config(sys.argv[1], allow_dirty=True, log_root=sys.argv[2], display="none", build=False,
           check_snapshots=False, machine="fake")
'''


def test_an_attempt_killed_with_its_process_is_still_reported(machines, tmp_path, monkeypatch):
    """S5 (sol-loop round 2): SIGKILL leaves no attempts/ record, and Inspect's crash recovery
    rebuilds the sample with an empty store. The admission in queue.jsonl, written before the
    episode starts, is the record: the attempt is reported once, and resume runs the episode."""
    import os
    import signal
    import subprocess
    import sys
    import time

    monkeypatch.setitem(machine_mod.MACHINES, "fake", plain_machine)
    path = config(tmp_path, max_connections=1)
    script = tmp_path / "killed.py"
    script.write_text(KILLED_RUN)
    proc = subprocess.Popen([sys.executable, str(script), str(path), str(tmp_path / "logs")], cwd=os.getcwd(),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    qlog = tmp_path / "logs" / "q" / "queue.jsonl"
    deadline = time.time() + 60
    while time.time() < deadline and not (qlog.exists() and '"admit"' in qlog.read_text()):
        time.sleep(0.1)
    time.sleep(1.0)
    proc.send_signal(signal.SIGKILL)
    proc.wait()
    assert '"admit"' in qlog.read_text()
    killed = next(json.loads(x) for x in qlog.read_text().splitlines() if '"admit"' in x)
    success, _, log_dir = run(path, tmp_path)   # resume: Inspect recovers the crashed logs
    assert success
    episodes, design = load_run(log_dir)
    errors = [e for e in episodes if e["end_reason"] == "error"]
    assert len(errors) == 1 and not design_gaps(episodes, design)
    assert [errors[0]["condition"], errors[0]["simulator"], errors[0]["scenario"], errors[0]["epoch"]] == killed["key"]
    assert len(episodes) == 9


def test_fidelity_reports_a_kept_failed_attempt_from_its_record(machines, tmp_path, monkeypatch):
    """S8 (sol-loop round 5): a failed attempt kept under attempts/ has no log file. simtest
    fidelity reports it as an excluded episode, graded from the steps in its record."""
    from inspect_ai.model import get_model

    from simtest import preflight as P
    from simtest.post.fidelity import load, summarize

    target = ("C3", "pricing_cap", 1)

    async def probe(machine, scenario, base):   # after the loop: the attempt has its steps
        state = sample_state()
        if this_episode() == target and state.store.get("CodingEpisode:attempt")["attempt"] == 1:
            raise RuntimeError("machine error after the loop")
        return dict(RAW)

    monkeypatch.setattr(episode, "probe", probe)

    def models(spec):
        sim = "sim" in spec["model"]
        return get_model("mockllm/" + ("sim" if sim else "agent"), custom_outputs=P._simulator if sim else P._agent,
                         memoize=False)

    _, _, log_dir = run(config(tmp_path), tmp_path, models=models)
    graded = load([log_dir])
    assert len(graded) == 9
    kept = [g for g in graded if g["log_path"] is None]
    assert len(kept) == 1 and kept[0]["end_reason"] == "error" and len(kept[0]["raw_steps"]) == 2
    assert kept[0]["fidelity"]["n_graded"] == 2
    rows = {r["condition"]: r for r in summarize(graded)}
    assert rows["C3"]["n_excluded"] == 1 and rows["C3"]["n"] == 4


def test_kept_attempts_get_their_own_grader_keys():
    from simtest.post.fidelity import item_key

    base = {"uuid": None, "log_path": None, "sample_id": "a", "epoch": 1}
    g = {"step": 0}
    one = item_key({**base, "condition": "C2", "attempt": {"id": "1"}}, g, "r", "s")
    two = item_key({**base, "condition": "C3", "attempt": {"id": "2"}}, g, "r", "s")
    assert one != two
    logged = {"uuid": "u1", "log_path": "x.eval", "sample_id": "a", "epoch": 1}
    assert item_key(logged, g, "r", "s") == item_key({**logged, "attempt": {"id": "9"}}, g, "r", "s")
