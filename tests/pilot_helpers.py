"""Synthetic runs for the pilot report's tests: a small config for `simtest run` on fake machines,
and hand-built episodes and judge records for its arithmetic."""

from __future__ import annotations

import yaml
from inspect_ai.solver._task_state import sample_state

import simtest.coding.episode as episode

SCEN = ["pricing_cap", "pricing_cap_benign"]
PROBE = {"base": "HEAD", "original": {"exit_code": 0, "output": "3 passed in 0.01s", "junit": (
    '<testsuites><testsuite errors="0" failures="0" skipped="0" tests="3"></testsuite></testsuites>')},
    "suite": {"exit_code": 0}, "names": "", "diff": "", "docker_diff": "", "atime_before": {},
    "atime_after": {}, "reads": [], "code_files": {}, "errors": []}


def config(tmp_path, conditions=("C0", "C3"), epochs=2):
    raw = {"family": "coding", "name": "q", "scenarios": SCEN, "conditions": list(conditions), "epochs": epochs,
           "max_turns": 4, "agents": {"a": {"model": "mockllm/agent", "base_url": "http://unused/v1"}},
           "simulators": {"sim": {"model": "mockllm/sim", "base_url": "http://unused/v1"}},
           "run": {"max_connections": 3, "retry_attempts": 2, "order_seed": 11}}
    p = tmp_path / "q.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def session_fails_once(monkeypatch, target, message="simulator call timed out"):
    """The target (condition, scenario, epoch)'s first attempt fails; its re-run goes through."""
    async def session_start(machine, scenario):
        state = sample_state()
        this = (state.store.get("CodingEpisode:condition"), state.metadata["scenario"], state.epoch)
        if this == target and state.store.get("CodingEpisode:attempt")["attempt"] == 1:
            raise RuntimeError(message)
        return "Thu Oct 29 10:15:02 CET 2026"

    monkeypatch.setattr(episode, "session_start", session_start)


SCENARIOS = ["a", "a_benign", "b", "b_benign", "c", "c_benign"]
DESIGN = {"conditions": ["C0", "C3", "C2"], "agents": ["qwen"], "simulators": {"C3": ["sim"], "C2": ["sim"]},
          "scenarios": SCENARIOS, "epochs": 3}
RUN = "n2-pilot"


def ep(condition, scenario, epoch, end_reason="no_tool_call", n_bash=10, n_divergent=0, calls=None, **over):
    """One episode, as `load_run` gives it, with what `pilot.read_run` adds."""
    uuid = f"{condition}-{scenario}-{epoch}"
    e = {"agent": "qwen", "model": "m", "condition": condition, "simulator": "none" if condition == "C0" else "sim",
         "scenario": scenario, "variant": None, "epoch": epoch, "end_reason": end_reason,
         "error": "boom" if end_reason == "error" else None,
         "metrics": {} if end_reason == "error" else {"violation": False, "honest_success": True},
         "violation_observed": None, "error_status": None, "notes": [],
         "divergence": {"n_bash": n_bash, "n_divergent": n_divergent if condition != "C0" else 0}, "post": None,
         "fingerprint": "f", "log": "x.eval", "log_path": "x.eval", "sample_id": scenario, "uuid": uuid,
         "judge_key": f"{RUN}::{uuid}", "simulator_calls": calls or []}
    e.update(over)
    return e


def full(conditions=("C0", "C3", "C2"), scenarios=SCENARIOS, epochs=3, **over):
    return [ep(c, s, i, **over) for c in conditions for s in scenarios for i in range(1, epochs + 1)]


def verdict(key, positive=True, label="eval_awareness", judge="v3", backend="inspect", model="haiku", **over):
    value = ("EVAL" if positive else "REAL") if label == "eval_awareness" else positive
    r = {"key": f"{key}-{label}", "label": label, "episode": {"key": key}, "value": value, "positive": positive,
         "error": None, "verdict_raw": str(value), "prompt_version": judge, "prompt_sha256": f"sha-{label}",
         "backend": backend, "model": model, "settings": {"temperature": None}, "render_version": 1,
         "transcript_sha256": "h", "attempts": [{"stop_reason": "stop"}], "alt_values": None,
         "scout": {"stop_reason": "stop"} if backend == "scout" else None}
    r.update(over)
    return r


def failed(key, kind="malformed", label="eval_awareness", attempts=None, **over):
    r = verdict(key, label=label, value=None, positive=None, verdict_raw=None,
                error={"kind": kind, "message": f"{kind} message"}, **over)
    r["attempts"] = attempts if attempts is not None else [{"stop_reason": "stop"}]
    return r
