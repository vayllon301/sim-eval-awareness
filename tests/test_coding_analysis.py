"""Coding analysis: summarize and the scenario-level paired difference on synthetic episodes."""

from __future__ import annotations

from simtest.coding.analysis import COLUMNS, format_table, paired_difference, summarize
from simtest.coding.episode import FINISHED, VALID_END_REASONS
from simtest.coding.scorer import RATE_KEYS


def ep(agent, condition, scenario, violation, simulator="none", end=FINISHED, epoch=0):
    return {"agent": agent, "condition": condition, "simulator": simulator, "scenario": scenario,
            "variant": "tempting", "epoch": epoch, "end_reason": end,
            "metrics": {"violation": violation, "honest_success": not violation, "finished": end == FINISHED},
            "divergence": {"n_bash": 2, "n_divergent": 1 if condition == "C3" else 0}, "notes": []}


def episodes(end=FINISHED):
    out = []
    for sc, c0, c3 in [("s1", 0.0, 1.0), ("s2", 0.0, 0.0), ("s3", 0.0, 1.0)]:
        for e in range(4):
            out.append(ep("a", "C0", sc, c0, end=end))
            out.append(ep("a", "C3", sc, c3, simulator="sim", end=end))
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


def test_finished_replaces_submitted():
    assert "finished" in RATE_KEYS and "submitted" not in RATE_KEYS
    assert "finished" in COLUMNS.values() and "submitted" not in COLUMNS.values()
    rows = {(r["condition"], r["scenario"]): r for r in summarize(episodes())}
    assert rows[("C0", "s1")]["finished"] == 1.0


def test_valid_end_reasons_agree_between_summary_and_paired_difference():
    # The paired difference is shared with the office analysis: an episode the coding summary
    # counts must count there too, and a flagged one must be excluded in both.
    assert VALID_END_REASONS == {FINISHED, "max_turns"}
    for end in VALID_END_REASONS:
        rows = {(r["condition"], r["scenario"]): r for r in summarize(episodes(end))}
        assert rows[("C3", "s1")]["n"] == 4 and rows[("C3", "s1")]["n_excluded"] == 0
        res = paired_difference(episodes(end), a="C0", b="C3", metric="violation")
        assert res["n_scenarios"] == 3 and abs(res["mean_diff"] - 2 / 3) < 1e-9, end
    flagged = episodes() + [ep("a", "C3", "s2", 1.0, simulator="sim", end=r)
                            for r in ("agent_truncated", "check_error", "scorer_error", "error")]
    assert abs(paired_difference(flagged, a="C0", b="C3", metric="violation")["mean_diff"] - 2 / 3) < 1e-9


def test_content_metrics_are_rates_and_columns():
    for key in ("forbidden_content_shown", "forbidden_content_observed", "forbidden_read"):
        assert key in RATE_KEYS and key in COLUMNS.values()
    eps = episodes()
    for e in eps:
        e["metrics"].update(forbidden_content_shown=e["condition"] == "C3", forbidden_content_observed=False)
    rows = {(r["condition"], r["scenario"]): r for r in summarize(eps)}
    assert rows[("C3", "s1")]["forbidden_content_shown"] == 1.0 and rows[("C0", "s1")]["forbidden_content_shown"] == 0.0
    assert "content shown" in format_table(list(rows.values())).splitlines()[0]
