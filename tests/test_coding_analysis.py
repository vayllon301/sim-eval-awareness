"""Coding analysis: summarize and the scenario-level paired difference on synthetic episodes."""

from __future__ import annotations

from simtest.coding.analysis import paired_difference, summarize


def ep(agent, condition, scenario, violation, simulator="none", end="submitted", epoch=0):
    return {"agent": agent, "condition": condition, "simulator": simulator, "scenario": scenario,
            "variant": "tempting", "epoch": epoch, "end_reason": end,
            "metrics": {"violation": violation, "honest_success": not violation, "submitted": True},
            "divergence": {"n_bash": 2, "n_divergent": 1 if condition == "C3" else 0}, "notes": []}


def episodes():
    out = []
    for sc, c0, c3 in [("s1", 0.0, 1.0), ("s2", 0.0, 0.0), ("s3", 0.0, 1.0)]:
        for e in range(4):
            out.append(ep("a", "C0", sc, c0))
            out.append(ep("a", "C3", sc, c3, simulator="sim"))
    return out


def test_summarize_rates_and_divergence():
    rows = {(r["condition"], r["scenario"]): r for r in summarize(episodes())}
    assert rows[("C0", "s1")]["violation"] == 0.0
    assert rows[("C3", "s1")]["violation"] == 1.0
    assert rows[("C3", "s1")]["divergent_step_rate"] == 0.5
    assert rows[("C0", "s1")]["divergent_step_rate"] == 0.0


def test_paired_difference_c3_minus_c0():
    res = paired_difference(episodes(), a="C0", b="C3", metric="violation")
    # per scenario: s1 +1, s2 0, s3 +1 -> mean 2/3
    assert abs(res["mean_diff"] - 2 / 3) < 1e-9
    assert res["n_scenarios"] == 3


def test_excluded_episode_is_not_counted():
    eps = episodes() + [ep("a", "C0", "s1", 1.0, end="agent_truncated")]
    rows = {(r["condition"], r["scenario"]): r for r in summarize(eps)}
    assert rows[("C0", "s1")]["n"] == 4 and rows[("C0", "s1")]["n_excluded"] == 1
    assert rows[("C0", "s1")]["violation"] == 0.0  # the flagged one is excluded from the rate
