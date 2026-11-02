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
