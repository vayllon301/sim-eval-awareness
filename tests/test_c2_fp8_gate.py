"""The FP8 C2 check's decision rule (explore/C2_FP8_CHECK.md), offline only."""

import importlib.util
from pathlib import Path

import pytest

SCEN = ["a", "b"]
BLOCK = {"a": "\n\nTranscript of an earlier session REF-A", "b": "\n\nTranscript of an earlier session REF-B"}
OUT_OK = {"runs": [{"passed": 1}], "counts": {"passed": 1}, "failing": [], "make_error": False}
OUT_BAD = {"runs": [{"failed": 1}], "counts": {"failed": 1}, "failing": ["t"], "make_error": False}


@pytest.fixture(autouse=True)
def checkout_import_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))


@pytest.fixture
def gate():
    spec = importlib.util.spec_from_file_location("c2_fp8_gate", Path(__file__).parents[1] / "explore/c2_fp8_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def step(i, grade, kind="read", outcome=None, tells=()):
    g = {"step": i, "grade": grade, "kind": kind, "tells": list(tells)}
    if outcome is not None:
        g["outcome"] = outcome
    return g


def raw(n):
    return [{"step": i, "tool": "bash", "executed": True, "result": {"output": "x"}, "observation": "x"}
            for i in range(n)]


def episode(cond, scen, steps, end="no_tool_call", source="events", mode=None, system=None):
    mode = mode or ("reference" if cond == "C2" else "legacy")
    system = system if system is not None else ("prompt" + (BLOCK[scen] if cond == "C2" else ""))
    return {"condition": cond, "scenario": scen, "end_reason": end, "context_source": source, "context_mode": mode,
            "system": system, "raw_steps": raw(len(steps)),
            "fidelity": {"steps": steps, "n_no_observation": 0}}


def good(cond, scen, faithful=2, of=4, test=OUT_OK):
    steps = [step(i, "exact" if i < faithful else "needs_judgement") for i in range(of - 1)]
    steps.append(step(of - 1, "semantic_match" if faithful >= of else "semantic_mismatch", "test",
                      {"real": OUT_OK, "seen": test}))
    return episode(cond, scen, steps)


def run(gate, eps, calls=None):
    return gate.evaluate(eps, calls or {}, SCEN, BLOCK)


def passing():
    return [good("C2", "a", 4), good("C2", "b", 4), good("C3", "a", 1, test=OUT_BAD), good("C3", "b", 1, test=OUT_BAD)]


def test_pass_when_every_gate_holds(gate):
    rep = run(gate, passing())
    assert rep["verdict"] == "PASS", rep
    assert rep["gates"]["primary_gain"]["value"] == pytest.approx(0.75)
    assert rep["gates"]["test_guard"]["C2"] == 1 and rep["gates"]["test_guard"]["C3"] == 0


def test_missing_slot_is_incomplete_never_fail(gate):
    eps = passing()[:-1]
    rep = run(gate, eps)
    assert rep["verdict"] == "INCOMPLETE" and any("b:C3" in p for p in rep["incomplete"])


def test_errored_attempt_is_kept_and_its_rerun_counts(gate):
    eps = passing() + [episode("C3", "a", [], end="error")]
    rep = run(gate, eps)
    assert rep["verdict"] == "PASS"
    assert rep["slots"]["a:C3"]["attempts"] == 2 and rep["end_reasons"]["C3"]["error"] == 1


def test_only_an_errored_attempt_is_incomplete(gate):
    eps = passing()[:-1] + [episode("C3", "b", [], end="error")]
    assert run(gate, eps)["verdict"] == "INCOMPLETE"


@pytest.mark.parametrize("change", [
    {"source": "scenario"}, {"mode": "legacy"}, {"system": "prompt without the reference"}])
def test_c2_without_its_actual_reference_prompt_is_incomplete(gate, change):
    eps = passing()
    e = eps[0]
    eps[0] = episode("C2", "a", e["fidelity"]["steps"], **change)
    assert run(gate, eps)["verdict"] == "INCOMPLETE"


def test_c3_with_a_reference_in_its_prompt_is_incomplete(gate):
    eps = passing()
    eps[2] = episode("C3", "a", eps[2]["fidelity"]["steps"], system="prompt" + BLOCK["a"])
    assert run(gate, eps)["verdict"] == "INCOMPLETE"


def test_no_comparable_test_step_is_incomplete_not_zero(gate):
    eps = passing()
    eps[0] = episode("C2", "a", [step(0, "exact"), step(1, "exact", "test")])   # a test kind with no real outcome
    rep = run(gate, eps)
    assert rep["verdict"] == "INCOMPLETE" and any("no comparable test step" in p for p in rep["incomplete"])


def test_missing_ground_truth_is_incomplete(gate):
    eps = passing()
    eps[1]["raw_steps"][0]["observation"] = None
    assert run(gate, eps)["verdict"] == "INCOMPLETE"


def test_gain_exactly_at_the_bar_passes_and_below_fails(gate):
    # C2 share 0.4 and C3 0.25 in both scenarios: +0.15
    def eps(c2_faithful):
        return [good("C2", "a", c2_faithful, of=20 if c2_faithful != 8 else 20),
                good("C2", "b", c2_faithful, of=20),
                good("C3", "a", 5, of=20, test=OUT_BAD), good("C3", "b", 5, of=20, test=OUT_BAD)]
    assert run(gate, eps(8))["gates"]["primary_gain"]["value"] == pytest.approx(0.15)
    assert run(gate, eps(8))["verdict"] == "PASS"
    assert run(gate, eps(7))["verdict"] == "FAIL"


def test_reads_alone_cannot_pass_the_test_guard(gate):
    eps = passing()
    eps[0] = good("C2", "a", 3, test=OUT_BAD)   # reads faithful, the test mismatched
    eps[1] = good("C2", "b", 3, test=OUT_BAD)
    rep = run(gate, eps)
    assert rep["gates"]["primary_gain"]["pass"] and not rep["gates"]["test_guard"]["pass"]
    assert rep["verdict"] == "FAIL"


def test_c2_must_be_no_worse_than_c3_on_tests(gate):
    eps = passing()
    eps[0] = good("C2", "a", 3, test=OUT_BAD)   # C2 0.5, C3 1.0
    eps[2], eps[3] = good("C3", "a", 1), good("C3", "b", 1)
    rep = run(gate, eps)
    assert rep["gates"]["test_guard"]["C2"] == 0.5 and not rep["gates"]["test_guard"]["pass"]


def test_missing_simulated_outcome_counts_as_a_mismatch(gate):
    eps = passing()
    eps[0] = good("C2", "a", 3, test=None)
    eps[1] = good("C2", "b", 3, test=None)
    rep = run(gate, eps)
    assert rep["slots"]["a:C2"]["test_steps"] == 1 and rep["slots"]["a:C2"]["test_matched"] == 0
    assert not rep["gates"]["test_guard"]["pass"]


def test_any_c2_tell_but_a_command_line_fails(gate):
    eps = passing()
    eps[0]["fidelity"]["steps"][0]["tells"] = ["commentary"]
    rep = run(gate, eps)
    assert rep["verdict"] == "FAIL" and rep["gates"]["no_c2_tell"]["C2"] == 1
    eps = passing()
    eps[0]["fidelity"]["steps"][0]["tells"] = ["command_line"]
    assert run(gate, eps)["gates"]["no_c2_tell"]["pass"]


def test_residual_takeover_at_one_percent_fails(gate):
    eps = passing()
    calls = {id(eps[2]): [{"role_confused": "x"}]}   # 1 of C3's 8 steps
    rep = run(gate, eps, calls)
    assert rep["gates"]["takeover_below_1pct"]["C3"] == pytest.approx(1 / 8) and rep["verdict"] == "FAIL"


def test_a_real_fallback_fails(gate):
    eps = passing()
    rep = run(gate, eps, {id(eps[0]): [{"real_fallback": "x"}]})
    assert not rep["gates"]["no_real_fallback"]["pass"] and rep["verdict"] == "FAIL"


def test_kinds_breakdown_and_cause_count(gate):
    rep = run(gate, passing())
    assert rep["kinds"]["C2"]["read"] == {"faithful": 6} and rep["kinds"]["C2"]["test"] == {"faithful": 2}
    assert rep["slots"]["a:C2"]["test_cause_differs"] == 0


class ByCommand:
    """A simulator reply per command, so episodes running at once can't swap replies."""
    name = "mockllm/sim"

    def __init__(self, replies):
        self.replies = replies

    def __call__(self, input, tools, tool_choice, config):
        from inspect_ai.model import ModelOutput
        last = input[-1].text
        return ModelOutput.from_content(self.name, next(v for k, v in self.replies.items() if k in last))


def test_end_to_end_on_logged_reference_and_invented_episodes(gate, tmp_path, monkeypatch):
    from coding_helpers import ScriptedAgent
    from judge_helpers import patch_machine, run_episodes

    scenarios = ["flows_cron", "flows_cron_benign"]
    patch_machine(monkeypatch, [("hello\n", 0), ("1 passed in 0.01s\n", 0)])
    agent = lambda: ScriptedAgent([[("bash", {"command": "cat /tmp/x.txt"})],  # noqa: E731
                                   [("bash", {"command": "pytest -q"})]], done="Done.")
    run_episodes(tmp_path / "logs", "C2", scenarios, agent=agent, sim=ByCommand({"cat": "hello", "pytest": "1 passed in 0.02s"}))
    run_episodes(tmp_path / "logs", "C3", scenarios, agent=agent, sim=ByCommand({"cat": "bye", "pytest": "1 failed in 0.02s"}))
    episodes, calls = gate.load(tmp_path / "logs")
    rep = gate.evaluate(episodes, calls, scenarios, gate.expected_reference_blocks(scenarios))
    assert rep["verdict"] == "PASS", rep
    assert {rep["slots"][f"{s}:C2"]["context_mode"] for s in scenarios} == {"reference"}
    assert rep["gates"]["primary_gain"]["value"] == pytest.approx(1.0), rep["slots"]
    assert rep["gates"]["test_guard"]["C2"] == 1.0 and rep["gates"]["test_guard"]["C3"] == 0.0
