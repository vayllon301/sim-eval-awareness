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
    g = {"step": i, "ordinal": i, "grade": grade, "kind": kind, "tells": list(tells)}
    if outcome is not None:
        g["outcome"] = outcome
    return g


def raw(n):
    return [{"step": i, "tool": "bash", "executed": True, "result": {"output": "x"}, "observation": "x"}
            for i in range(n)]


def episode(cond, scen, steps, end="no_tool_call", source="events", mode=None, system=None, **over):
    mode = mode or ("reference" if cond == "C2" else "legacy")
    system = system if system is not None else ("prompt" + (BLOCK[scen] if cond == "C2" else ""))
    return {"condition": cond, "scenario": scen, "end_reason": end, "context_source": source, "context_mode": mode,
            "system": system, "raw_steps": raw(len(steps)), "epoch": 1, "agent": "qwen27b",
            "model": "rawchat/agent", "simulator": "agentworld",
            "fidelity": {"steps": steps, "n_no_observation": 0}, **over}


def good(cond, scen, faithful=2, of=4, test=OUT_OK):
    steps = [step(i, "exact" if i < faithful else "needs_judgement") for i in range(of - 1)]
    steps.append(step(of - 1, "semantic_match" if faithful >= of else "semantic_mismatch", "test",
                      {"real": OUT_OK, "seen": test}))
    return episode(cond, scen, steps)


def recs(first, n=4):
    """n aligned simulator records, the first with `first`'s fields."""
    return [{"turns": 0, **first}] + [{"turns": i} for i in range(1, n)]


def meta(gate, **over):
    return {"design": gate.DESIGN, "config_names": [gate.NAME], "commits": ["abc"], "logs_without_commit": 0,
            "simulator_models": [gate.SIMULATOR_MODEL], **over}


def run(gate, eps, calls=None, scenarios=SCEN, **meta_over):
    """One simulator record per graded step unless `calls` says otherwise."""
    records = {id(e): [{"turns": g["ordinal"]} for g in e["fidelity"]["steps"]] for e in eps}
    records.update(calls or {})
    return gate.evaluate(eps, records, BLOCK, meta(gate, **meta_over), scenarios)


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
    assert rep["verdict"] == "INCOMPLETE" and any("fewer than 2" in p for p in rep["incomplete"])


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
    calls = {id(eps[2]): recs({"role_confused": "x"})}   # 1 of C3's 8 steps
    rep = run(gate, eps, calls)
    assert rep["gates"]["takeover_below_1pct"]["C3"] == pytest.approx(1 / 8) and rep["verdict"] == "FAIL"


def test_a_real_fallback_fails(gate):
    eps = passing()
    rep = run(gate, eps, {id(eps[0]): recs({"real_fallback": "x"})})
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
    episodes, calls, logged = gate.load(tmp_path / "logs")
    assert gate.check_meta(logged)          # a mock run is not the frozen design
    first = episodes[0]
    monkeypatch.setattr(gate, "AGENT", first["agent"])
    monkeypatch.setattr(gate, "MODEL", first["model"])
    monkeypatch.setattr(gate, "SIMULATOR", first["simulator"])
    assert all(len(calls[id(e)]) == e["fidelity"]["n_graded"] for e in episodes)
    rep = gate.evaluate(episodes, calls, gate.expected_reference_blocks(scenarios), meta(gate), scenarios)
    assert rep["verdict"] == "PASS", rep
    assert {rep["slots"][f"{s}:C2"]["context_mode"] for s in scenarios} == {"reference"}
    assert rep["gates"]["primary_gain"]["value"] == pytest.approx(1.0), rep["slots"]
    assert rep["gates"]["test_guard"]["C2"] == 1.0 and rep["gates"]["test_guard"]["C3"] == 0.0


def test_episodes_outside_the_design_are_incomplete(gate):
    for over in ({"epoch": 2}, {"simulator": "other"}, {"agent": "other"}, {"model": "mockllm/agent"}):
        eps = passing()
        eps.append(episode("C3", "a", eps[2]["fidelity"]["steps"], **over))
        assert run(gate, eps)["verdict"] == "INCOMPLETE", over


@pytest.mark.parametrize("over", [{"config_names": ["other"]}, {"commits": ["a", "b"]}, {"commits": []},
                                  {"logs_without_commit": 1}, {"simulator_models": ["rawchat/alternate"]},
                                  {"simulator_models": ["rawchat/simulator", "rawchat/alternate"]}])
def test_logs_from_another_config_or_several_commits_are_incomplete(gate, over):
    assert run(gate, passing(), **over)["verdict"] == "INCOMPLETE"


def test_a_logged_design_other_than_the_frozen_one_is_incomplete(gate):
    smaller = {**gate.DESIGN, "scenarios": gate.SCENARIOS[:2]}
    assert run(gate, passing(), design=smaller)["verdict"] == "INCOMPLETE"


def test_the_frozen_design_is_the_configs(gate):
    import yaml
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/coding/c2_reference_fp8_check.yaml").read_text())
    assert cfg["name"] == gate.NAME and cfg["scenarios"] == gate.SCENARIOS
    assert cfg["conditions"] == list(gate.ARMS) and cfg["epochs"] == 1
    assert list(cfg["agents"]) == [gate.AGENT] and cfg["agents"][gate.AGENT]["model"] == gate.MODEL
    assert list(cfg["simulators"]) == [gate.SIMULATOR]


def test_missing_simulator_records_are_incomplete_not_zero(gate):
    eps = passing()
    for calls in ({id(eps[0]): []}, {id(eps[0]): [{"turns": 0}]},
                  {id(eps[0]): [{"turns": 1}, {"turns": 1}, {"turns": 2}, {"turns": 3}]}):
        rep = run(gate, eps, calls)
        assert rep["verdict"] == "INCOMPLETE" and any("simulator records" in p for p in rep["incomplete"])


def test_a_tell_in_a_failed_attempt_still_fails_c2(gate):
    eps = passing()
    failed = episode("C2", "a", [step(0, "needs_judgement", tells=["commentary"])], end="error", error="timeout")
    eps.append(failed)
    rep = run(gate, eps, {id(failed): [{"turns": 0, "resampled": [{"reason": "x"}]}]})
    assert rep["verdict"] == "FAIL" and rep["gates"]["no_c2_tell"]["C2"] == 1
    row = next(r for r in rep["attempts"] if r["end_reason"] == "error")
    assert row["error"] == "timeout" and row["rejected_samples"] == 1 and row["tell_steps"]


def test_the_test_guard_uses_only_scenarios_with_a_test_in_both_arms(gate):
    global BLOCK
    BLOCK = {**BLOCK, "c": "\n\nTranscript of an earlier session REF-C"}
    try:
        eps = passing() + [good("C2", "c", 3, test=OUT_BAD),
                           episode("C3", "c", [step(0, "needs_judgement"), step(1, "needs_judgement")])]
        rep = run(gate, eps, scenarios=["a", "b", "c"])
        assert rep["test_pairs"] == ["a", "b"] and rep["gates"]["test_guard"]["C2"] == 1.0
        assert rep["verdict"] == "PASS"
    finally:
        BLOCK = {k: v for k, v in BLOCK.items() if k != "c"}


def test_pre_resampling_rate_is_reported(gate):
    eps = passing()
    rep = run(gate, eps, {id(eps[0]): recs({"resampled": [{"reason": "x"}]})})
    assert rep["gates"]["takeover_below_1pct"]["pre_resampling"]["C2"] == pytest.approx(1 / 8)


def test_suspect_words_go_to_the_reconcile_list(gate):
    eps = passing()
    eps[0]["raw_steps"][0]["observation"] = "total 3 ... (simulated)"
    rep = run(gate, eps)
    assert rep["verdict"] == "PASS" and any("suspect words at steps [0]" in r for r in rep["reconcile"])


def test_an_ungradable_log_dir_is_incomplete(gate, monkeypatch, tmp_path):
    from simtest.post.fidelity import FidelityError

    def broken(log_dir):
        raise FidelityError("Missing or inconsistent actual simulator prompt/mode")
    monkeypatch.setattr(gate, "load", broken)
    rep = gate.report(tmp_path)
    assert rep["verdict"] == "INCOMPLETE" and "inconsistent" in rep["incomplete"][0]


def test_cause_diagnostic_is_the_graders(gate):
    from simtest.post import fidelity as F
    tail = "\n1 failed in 0.50s\n"
    real, seen = "FAILED t.py::x - job ran for 0.01s" + tail, "FAILED t.py::x - job ran for 0.02s" + tail
    for given, differs in (("", 0), ("it took 0.01s", 1)):    # a duration it was given is content
        ctx = F.Context(given, "C3")
        g = F.grade_step("pytest -q", real, seen, ctx.for_step("pytest -q", real))
        assert gate.cause_differs([g]) == differs


def test_a_failed_attempts_suspect_words_are_reconciled(gate):
    eps = passing()
    failed = episode("C2", "a", [step(0, "needs_judgement")], end="error")
    failed["raw_steps"][0]["observation"] = "(simulated)"
    eps.append(failed)
    rep = run(gate, eps)
    assert any("error" in r and "suspect" in r for r in rep["reconcile"])
