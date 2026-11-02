"""`simtest preflight`: a config end to end offline, with scripted models (src/simtest/preflight.py)."""

from __future__ import annotations

import pytest

import simtest.coding.episode as episode
from simtest import preflight as P
from test_coding_queue_run import config, machines  # noqa: F401 (the fixture)


def test_preflight_runs_every_episode_and_records_the_commit(machines, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(P, "git_dirty", lambda: False)
    monkeypatch.setattr(P, "_commit", lambda: "abc123")
    ok, problems, log_dir, recorded = P.preflight(config(tmp_path), root=tmp_path / "pf", machine="fake",
                                                  build=False, check_snapshots=False)
    assert ok and not problems and recorded, problems
    assert (tmp_path / "pf" / "q.ok").read_text() == "abc123\n"
    # The scripted agent ran its commands on the machine, and C3 called its simulator.
    assert machines["starts"] and len(machines["starts"]) == 8


def test_a_failing_preflight_lists_the_problem_and_records_nothing(machines, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(P, "git_dirty", lambda: False)
    monkeypatch.setattr(P, "_commit", lambda: "abc123")
    (tmp_path / "pf").mkdir()
    (tmp_path / "pf" / "q.ok").write_text("old\n")

    async def broken(machine, scenario):
        raise RuntimeError("the image has no date")

    monkeypatch.setattr(episode, "session_start", broken)
    ok, problems, _, recorded = P.preflight(config(tmp_path, retry_attempts=0), root=tmp_path / "pf",
                                            machine="fake", build=False, check_snapshots=False)
    assert not ok and not recorded and not (tmp_path / "pf" / "q.ok").exists()
    assert any("the image has no date" in p for p in problems)
    assert any(p.startswith("missing:") for p in problems)


def test_a_dirty_tree_passes_but_is_not_recorded(machines, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(P, "git_dirty", lambda: True)
    ok, _, _, recorded = P.preflight(config(tmp_path), epochs="1", root=tmp_path / "pf", machine="fake",
                                     build=False, check_snapshots=False)
    assert ok and not recorded


@pytest.mark.docker
def test_preflight_on_the_real_flows_images(tmp_path, monkeypatch):
    """The real Docker images, one pair in C0 and C3, 1 epoch (skipped without Docker)."""
    import yaml

    from coding_helpers import docker_available

    if not docker_available():
        pytest.skip("Docker is not available")
    monkeypatch.setattr(P, "git_dirty", lambda: True)
    raw = yaml.safe_load(open("configs/coding/pilot.yaml"))
    raw.update(name="pf-flows", scenarios=["flows_audit", "flows_audit_benign"], conditions=["C0", "C3"],
               epochs=1, simulators_for={"C3": ["agentworld"]})
    path = tmp_path / "pf.yaml"
    path.write_text(yaml.safe_dump(raw))
    ok, problems, _, _ = P.preflight(path, root=tmp_path / "pf")
    assert ok, problems
