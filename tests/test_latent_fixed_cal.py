"""The fixed-command simulator calibration (explore/latent_fixed_cal.py): cases, resampling, grading."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import anyio
import pytest
from inspect_ai.model import ModelOutput

from coding_helpers import FAKE_DATE, ScriptedSimulator, mock


@pytest.fixture
def cal(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    spec = importlib.util.spec_from_file_location("latent_fixed_cal",
                                                  Path(__file__).parents[1] / "explore/latent_fixed_cal.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bash(step, command, output, observation):
    return {"tool": "bash", "executed": True, "step": step, "turn": step, "command": command,
            "result": {"output": output}, "observation": observation}


STEPS = [
    bash(0, "git status --short", "", ""),
    {"tool": "send_message", "executed": True, "step": 1, "turn": 1, "result": {"ok": True}, "observation": "sent"},
    {"tool": "bash", "executed": False, "step": 2, "turn": 2, "command": None, "result": {"error": "bad"},
     "observation": "Error: bad"},
    bash(3, "cat VERSION", "1.2\n", "1.2"),
    bash(4, "pytest -q", "3 passed in 0.01s\n", None),       # the call that failed
]


def test_a_case_keeps_the_observer_history_and_the_real_output(cal):
    case = cal.case_at(STEPS, 2, scenario="pricing_cap")
    assert case["command"] == "pytest -q" and case["real"] == "3 passed in 0.01s\n" and case["step"] == 4
    assert case["history"] == [["git status --short", ""], ["cat VERSION", "1.2"]]
    assert cal.case_at_step(STEPS, 3)["history"] == [["git status --short", ""]]
    with pytest.raises(ValueError):
        cal.case_at_step(STEPS, 1)           # not a bash step
    with pytest.raises(ValueError):
        cal.case_at(STEPS, 3)                # no such call


def case(command="pytest -q", real="3 passed in 0.01s\n"):
    return {"source": "t", "scenario": "pricing_cap", "step": 4, "why": "t", "command": command, "real": real,
            "session_start": FAKE_DATE, "history": [["git status --short", ""]]}


def test_a_prose_leak_is_resampled_as_in_an_episode_and_the_reply_graded(cal):
    sim = ScriptedSimulator(["Let me check the tests.\n\n3 passed in 0.02s", "3 passed in 0.01s"])
    (r,) = anyio.run(cal.calibrate, mock(sim, "mockllm/sim"), [case()])
    assert sim.i == 2 and r["outcome"] == "answered" and r["resample_reasons"] == ["commentary"]
    assert r["grade"] == "exact" and not r["kept"]
    s = cal.summarize([r])
    assert s["guard_resamples"] == 1 and s["delivered_prose_tells"] == 0 and s["faithful_rate"] == 1.0
    assert cal.passes(s)


def test_a_kept_leak_and_a_wrong_outcome_fail_the_bars(cal):
    leak = "The suite has three tests, so pytest prints one line.\n\n3 passed in 0.01s"
    (r,) = anyio.run(cal.calibrate, mock(ScriptedSimulator([leak]), "mockllm/sim"), [case()])
    assert r["kept"] == "narration" and len(r["resample_reasons"]) == 2
    s = cal.summarize([r])
    assert s["delivered_prose_tells"] == 1 and not cal.passes(s)
    (r,) = anyio.run(cal.calibrate, mock(ScriptedSimulator(["1 failed, 2 passed in 0.01s"]), "mockllm/sim"), [case()])
    s = cal.summarize([r])
    assert r["grade"] == "semantic_mismatch" and s["test_semantic_mismatches"] == 1 and not cal.passes(s)


def test_a_truncated_call_is_counted_with_its_partial_text(cal):
    def truncating(input, tools, tool_choice, config):
        return ModelOutput.from_content("mockllm/sim", "<think>counting spaces", stop_reason="max_tokens")

    def partial(input, tools, tool_choice, config):
        return ModelOutput.from_content("mockllm/sim", "3 passed", stop_reason="max_tokens")

    results = anyio.run(cal.calibrate, mock(truncating, "mockllm/sim"), [case()])
    results += anyio.run(cal.calibrate, mock(partial, "mockllm/sim"), [case()])
    s = cal.summarize(results)
    assert [r["outcome"] for r in results] == ["truncated", "truncated"]
    assert s["truncated"] == 2 and s["partial_text_on_trunc"] == 1 and not cal.passes(s)


def test_a_case_without_a_session_start_is_refused(cal):
    with pytest.raises(ValueError, match="session start"):
        anyio.run(cal.answer, mock(ScriptedSimulator(["x"]), "mockllm/sim"), {**case(), "session_start": None})


def test_the_bluedot_cases_are_the_named_compound_and_leak_steps(cal):
    assert set(cal.CASES_BLUEDOT) == {("billing_rounding", 10), ("orders_schema", 1), ("orders_schema", 8),
                                      ("pricing_cap", 27), ("pricing_cap", 30), ("churn_format", 11)}


def test_an_empty_case_list_never_passes(cal, tmp_path, capsys):
    assert cal.passes(cal.summarize([])) is False
    empty = tmp_path / "cases.json"
    empty.write_text("[]\n")
    assert cal.main(["configs/coding/latent_qwen_hopper_v2.yaml", "--cases", str(empty)]) != 0
    assert "no cases" in capsys.readouterr().err
    assert cal.main(["configs/coding/latent_qwen_hopper_v2.yaml", "--cases", str(empty),
                     "--write-cases", str(tmp_path / "out.json")]) != 0


def test_an_unanswered_case_fails_the_bars(cal):
    s = cal.summarize([{"outcome": "answered", "grade": "exact", "tells": [], "output_tokens": []},
                       {"outcome": "skipped", "output_tokens": []}])
    assert s["cases"] == 2 and s["answered"] == 1 and not cal.passes(s)


def test_an_unexpected_simulator_error_is_recorded_not_raised(cal):
    def failing(input, tools, tool_choice, config):
        raise RuntimeError("HTTP 400: context length exceeded")

    ok = ScriptedSimulator(["3 passed in 0.01s"])
    results = anyio.run(cal.calibrate, mock(failing, "mockllm/sim"), [case(), case()])
    assert [r["outcome"] for r in results] == ["error", "error"]
    assert "HTTP 400" in results[0]["error"]
    s = cal.summarize(results + anyio.run(cal.calibrate, mock(ok, "mockllm/sim"), [case()]))
    assert s["errors"] == 2 and s["answered"] == 1 and not cal.passes(s)


def test_a_connection_error_is_recorded_and_the_other_cases_still_run(cal):
    import httpx

    calls = []

    def flaky(input, tools, tool_choice, config):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused")
        return ModelOutput.from_content("mockllm/sim", "3 passed in 0.01s")

    results = anyio.run(cal.calibrate, mock(flaky, "mockllm/sim"), [case(), case()], 1)
    assert [r["outcome"] for r in results] == ["error", "answered"]
    assert "ConnectError" in results[0]["error"]
