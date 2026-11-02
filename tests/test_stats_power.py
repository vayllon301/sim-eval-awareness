"""Power of the exact sign-flip test by simulation (stats/power.py)."""

from __future__ import annotations

import json
import math

import pytest

from simtest.stats.power import (cheapest_designs, format_grid, format_ranking, planned_repeats, power_grid,
                                 simulate_power)


def se(p, n):
    return math.sqrt(p * (1 - p) / n)


def test_under_the_null_power_is_at_most_alpha():
    # c3 == c0 with spread: the test's size, which a discrete exact test keeps at or below alpha
    for S, n in ((8, 10), (12, 20), (16, 30)):
        r = simulate_power(S, n, 0.3, 0.3, spread=1.0, effect_spread=0.0, reps=4000, seed=1)
        assert r["power"] <= 0.05 + 3 * se(0.05, 4000), (S, n, r["power"])
    # with 16 scenarios the attainable sizes are fine-grained: the size comes close to alpha
    r = simulate_power(16, 30, 0.5, 0.5, reps=4000, seed=2)
    assert 0.02 <= r["power"] <= 0.05 + 3 * se(0.05, 4000)


def test_power_grows_with_the_effect():
    rs = [simulate_power(10, 10, 0.1, c3, spread=0.5, reps=3000, seed=3) for c3 in (0.15, 0.3, 0.5, 0.8)]
    powers = [r["power"] for r in rs]
    assert powers == sorted(powers)
    assert rs[0]["ci"][1] < rs[2]["ci"][0]          # clearly apart, beyond Monte Carlo error
    assert powers[-1] > 0.95


def test_power_grows_with_repeats_and_scenarios():
    by_n = [simulate_power(8, n, 0.1, 0.3, reps=3000, seed=4)["power"] for n in (5, 15, 30)]
    assert by_n == sorted(by_n)
    by_s = [simulate_power(S, 10, 0.1, 0.3, reps=3000, seed=5)["power"] for S in (8, 12, 16)]
    assert by_s == sorted(by_s)


def test_too_few_scenarios_can_never_reject():
    # 5 scenarios: the smallest two-sided p is 2/32 = 0.0625 > 0.05, whatever the effect
    r = simulate_power(5, 30, 0.0, 1.0, reps=500, seed=6)
    assert r["power"] == 0.0 and r["cannot_reject"] == 1.0
    # one-sided, 5 scenarios can reach 1/32 < 0.05
    assert simulate_power(5, 30, 0.0, 1.0, reps=500, seed=6, alternative="greater")["power"] == 1.0


def test_c0_at_zero_and_ties():
    r = simulate_power(8, 5, 0.0, 0.05, reps=2000, seed=7)
    assert r["mean_rate_c0"] == 0.0
    assert r["mean_ties"] > 4          # most scenarios show 0 of 5 in both conditions
    assert 0.0 <= r["power"] <= 1.0


def test_reproducible_per_cell():
    a = simulate_power(10, 10, 0.1, 0.4, spread=1.0, reps=1000, seed=8)
    b = power_grid(0.1, 0.4, scenarios=(8, 10), repeats=(10,), spread=1.0, effect_spread=0.0, reps=1000, seed=8)[1]
    assert a["power"] == b["power"]      # a cell's result doesn't depend on the rest of the grid
    assert simulate_power(10, 10, 0.1, 0.4, spread=1.0, reps=1000, seed=9)["seed"] == 9


def test_bad_inputs():
    with pytest.raises(ValueError):
        simulate_power(8, 10, 1.2, 0.5)
    with pytest.raises(ValueError):
        simulate_power(8, 10, 0.1, 0.5, spread=-1)
    with pytest.raises(ValueError):
        simulate_power(0, 10, 0.1, 0.5)


def test_grid_table_and_cli(capsys):
    from simtest.cli import main

    rows = power_grid(0.1, 0.6, scenarios=(8, 12), repeats=(5, 10), reps=200, seed=1)
    text = format_grid(rows, label="development, not the pilot")
    assert text.startswith("[development, not the pilot] Power of the exact sign-flip test")
    assert "S=8 | S=12" in text and "n=5 |" in text and "n=10 |" in text
    main(["power", "--c0", "0.1", "--c3", "0.6", "--scenarios", "8", "--repeats", "5", "--reps", "100",
          "--label", "development, not the pilot", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["label"] == "development, not the pilot" and out["rows"][0]["scenarios"] == 8
    assert "ranking" not in out                     # no --cost, no ranking


def test_planned_repeats_cover_the_exclusion_rate():
    """The expected scored n is never below the n the power calculation asks for: the smallest
    n_planned with n_planned x (1 - exclusion) >= n, so ceil(n / (1 - exclusion)). Rounding
    the expected count instead let it fall up to half an episode short (n=4 at 10% had no
    inflation), and fewer episodes than the power calculation asks for lower the experiment."""
    assert [planned_repeats(n) for n in (1, 5, 30)] == [1, 5, 30]           # no exclusions: as designed
    assert planned_repeats(4, 0.1) == 5            # 4 x 0.9 = 3.6 < 4; 5 x 0.9 = 4.5
    assert planned_repeats(3, 0.1) == 4
    assert planned_repeats(10, 0.1) == 12          # 11 x 0.9 = 9.9 < 10
    assert planned_repeats(20, 0.1) == 23          # 22 x 0.9 = 19.8 < 20
    assert planned_repeats(5, 0.1) == 6
    assert planned_repeats(9, 0.1) == 10           # exactly 9.0: enough
    from fractions import Fraction

    for e in (0.0, 0.05, 0.1, 0.125, 0.15, 0.3):
        keep = 1 - Fraction(str(e))
        for n in range(1, 41):
            m = planned_repeats(n, e)
            assert m * keep >= n and (m - 1) * keep < n, (n, e, m)
    with pytest.raises(ValueError):
        planned_repeats(10, 1.0)
    with pytest.raises(ValueError):
        planned_repeats(10, -0.1)


COST = {"C0": 0.4, "C1": 4, "C2": 4, "C3": 4}          # 12.4 units per scenario and repeat


def _row(S, n, power, lo, hi, c0=0.1, c3=0.35):
    return {"scenarios": S, "repeats": n, "c0": c0, "c3": c3, "spread": 0.5, "effect_spread": 0.5, "alpha": 0.05,
            "alternative": "two-sided", "reps": 4000, "seed": 1, "power": power, "ci": [lo, hi]}


def _cells(res):
    return [(d["scenarios"], d["repeats"]) for d in res["designs"]]


def test_cheapest_reaches_the_target_on_the_lower_end_of_the_interval():
    rows = [_row(8, 10, 0.82, 0.79, 0.85),           # the estimate passes, the interval doesn't
            _row(8, 20, 0.86, 0.80, 0.91),           # lower end exactly at the target: reaches it
            _row(12, 20, 0.95, 0.94, 0.96)]
    res = cheapest_designs(rows, COST, target=0.8)
    assert _cells(res) == [(8, 20), (12, 20)]
    c = res["cheapest"]
    assert (c["scenarios"], c["repeats"], c["planned_repeats"]) == (8, 20, 20)
    assert c["episodes"] == 8 * 20 * 4 and c["units"] == pytest.approx(8 * 20 * 12.4)
    assert c["power"] == 0.86 and c["ci"] == [0.80, 0.91]
    assert res["units_per_set"] == pytest.approx(12.4) and res["conditions"] == ["C0", "C1", "C2", "C3"]


def test_cheapest_ties_in_cost_go_to_more_scenarios():
    ok = (0.9, 0.85, 0.95)
    rows = [_row(8, 30, *ok), _row(16, 15, *ok), _row(20, 30, *ok), _row(12, 20, *ok)]   # 240, 240, 600, 240
    assert _cells(cheapest_designs(rows, COST)) == [(16, 15), (12, 20), (8, 30), (20, 30)]


def test_cheapest_costs_the_planned_episodes():
    ok = (0.9, 0.85, 0.95)
    rows = [_row(12, 10, *ok), _row(10, 12, *ok)]
    assert _cells(cheapest_designs(rows, COST)) == [(12, 10), (10, 12)]     # 120 = 120: more scenarios first
    res = cheapest_designs(rows, COST, exclusion=0.1)
    assert _cells(res) == [(10, 12), (12, 10)]         # planned 10 x 14 = 140 < 12 x 12 = 144
    c = res["cheapest"]
    assert c["planned_repeats"] == 14 and c["episodes"] == 10 * 14 * 4
    assert c["units"] == pytest.approx(10 * 14 * 12.4) and c["power"] == 0.9     # power at the scored n
    # only the run's conditions are paid for
    res = cheapest_designs(rows, COST, conditions=["C3", "C0"])
    assert res["conditions"] == ["C0", "C3"] and res["cheapest"]["units"] == pytest.approx(120 * 4.4)
    assert res["cheapest"]["episodes"] == 120 * 2


def test_cheapest_never_reports_fewer_scenarios_than_the_floor():
    rows = [_row(6, 10, 0.99, 0.98, 1.0), _row(8, 10, 0.5, 0.48, 0.52), _row(8, 30, 0.9, 0.85, 0.95)]
    res = cheapest_designs(rows, COST, min_scenarios=8)
    assert _cells(res) == [(8, 30)]
    res = cheapest_designs(rows[:2], COST, min_scenarios=8)
    assert res["designs"] == [] and res["cheapest"] is None
    assert (res["best"]["scenarios"], res["best"]["power"]) == (8, 0.5)       # not the 6-scenario cell
    assert _cells(cheapest_designs(rows, COST, min_scenarios=6))[0] == (6, 10)


def test_cheapest_when_none_reaches_the_target():
    rows = [_row(8, 10, 0.5, 0.48, 0.52), _row(16, 30, 0.71, 0.70, 0.72), _row(12, 30, 0.71, 0.70, 0.72)]
    res = cheapest_designs(rows, COST, target=0.8)
    assert res["designs"] == [] and res["cheapest"] is None
    assert (res["best"]["scenarios"], res["best"]["repeats"]) == (12, 30)   # equal power: the cheaper one
    text = format_ranking(res)
    assert "No design in the grid reaches power 0.8 with at least 8 scenarios" in text
    assert "Best power found: S=12, n=30 (30 planned): 0.710 [0.700, 0.720]" in text
    # no cell with enough scenarios at all
    text = format_ranking(cheapest_designs([_row(6, 10, 0.99, 0.98, 1.0)], COST))
    assert "No design in the grid has at least 8 scenarios" in text


def test_cheapest_within_a_budget():
    rows = [_row(8, 10, 0.6, 0.58, 0.62), _row(8, 20, 0.9, 0.85, 0.95), _row(12, 20, 0.95, 0.94, 0.96)]
    assert _cells(cheapest_designs(rows, COST, budget=2000)) == [(8, 20)]          # 1,984 and 2,976 units
    res = cheapest_designs(rows, COST, budget=1000)
    assert res["designs"] == [] and res["best"]["repeats"] == 10        # the best power within the budget
    assert (res["cheapest_over_budget"]["scenarios"], res["cheapest_over_budget"]["repeats"]) == (8, 20)
    text = format_ranking(res)
    assert "No design within 1,000 units reaches power 0.8 with at least 8 scenarios" in text
    assert "The cheapest that does costs 1,984 units: S=8, n=20" in text


def test_cheapest_bad_inputs():
    rows = [_row(8, 10, 0.9, 0.85, 0.95)]
    for kw in ({"conditions": ["C1", "C3"]},                     # the test needs C0 and C3
               {"conditions": ["C0", "C3", "C4"]},
               {"target": 1.0}, {"target": 0.0}, {"min_scenarios": 0}, {"exclusion": 1.0}, {"budget": 0}):
        with pytest.raises(ValueError):
            cheapest_designs(rows, COST, **kw)
    with pytest.raises(ValueError):
        cheapest_designs(rows, {"C0": 0.4, "C3": -1})
    with pytest.raises(ValueError):
        cheapest_designs(rows, {"C0": 0.4, "C2": 4})                      # no cost for C3
    with pytest.raises(ValueError):
        cheapest_designs(rows, {"C0": 0.0, "C3": 0.0})                    # nothing costs anything
    with pytest.raises(ValueError):
        cheapest_designs(rows + [_row(8, 10, 0.9, 0.85, 0.95, c3=0.5)], COST)    # one (c0, c3) pair at a time


RANKED = ["power", "--c0", "0", "--c3", "0.9", "--scenarios", "6", "8", "12", "--repeats", "5", "10", "--reps", "200",
          "--cost", "C0=0.4", "C1=4", "C2=4", "C3=4", "--exclusion", "0.1", "--label", "development"]
HEADER = "S | n | planned n | episodes | units | power [95% MC interval]"


def test_cli_ranks_designs_by_cost(capsys):
    from simtest.cli import main

    main(RANKED)
    grid, ranking = capsys.readouterr().out.split("\n\n[development] Cheapest designs that reach power 0.8: ")
    assert "S=6 | S=8 | S=12" in grid               # the grid keeps every cell
    assert ranking.startswith("C0 0, C3 0.9, logit spread 0 shared + 0 in the effect")
    assert "the lower end of its 95% Monte Carlo interval is at least 0.8, not the point estimate" in ranking
    assert "C0, C1, C2, C3 at 0.4 + 4 + 4 + 4 = 12.4 units per scenario and repeat; at least 8 scenarios" in ranking
    assert "planned n x 0.9 >= n" in ranking
    lines = ranking.splitlines()
    table = lines[lines.index(HEADER) + 1:-1]
    # every cell reaches the target; S=6 is cheaper but below the floor; 5 and 10 scored need 6 and 12 planned
    assert [row.split(" | ")[:5] for row in table] == [["8", "5", "6", "192", "595.2"],
                                                       ["12", "5", "6", "288", "892.8"],
                                                       ["8", "10", "12", "384", "1,190.4"],
                                                       ["12", "10", "12", "576", "1,785.6"]]
    assert lines[-1].startswith("Cheapest: S=8, n=5 (6 planned) per scenario and condition, 192 episodes, 595.2 units")

    main(RANKED[:4] + ["0.9", "0.95"] + RANKED[5:] + ["--json"])       # one ranking per (c0, c3) pair
    out = json.loads(capsys.readouterr().out)
    assert [r["c3"] for r in out["ranking"]] == [0.9, 0.95]
    first = out["ranking"][0]
    assert first["cheapest"]["scenarios"] == 8 and first["cheapest"]["planned_repeats"] == 6
    assert first["exclusion"] == 0.1 and first["target"] == 0.8 and first["rule"] == "ci[0] >= target"
    assert all(d["ci"][0] >= 0.8 and d["scenarios"] >= 8 for d in first["designs"])


def test_cli_ranking_when_none_reaches_and_within_a_budget(capsys):
    from simtest.cli import main

    main(["power", "--c0", "0.1", "--c3", "0.12", "--scenarios", "8", "--repeats", "5", "--reps", "100",
          "--cost", "C0=1", "C3=1"])
    text = capsys.readouterr().out
    assert "No design in the grid reaches power 0.8 with at least 8 scenarios" in text
    assert "Best power found: S=8, n=5 (5 planned): " in text
    main(RANKED + ["--budget", "500"])
    text = capsys.readouterr().out
    assert "No design within 500 units reaches power 0.8 with at least 8 scenarios" in text
    assert "The cheapest that does costs 595.2 units: S=8, n=5 (6 planned)" in text
    main(RANKED + ["--budget", "900", "--target", "0.9", "--min-scenarios", "6"])
    lines = capsys.readouterr().out.splitlines()
    table = lines[lines.index(HEADER) + 1:-1]
    # 446.4, 595.2, then 892.8 twice (n=10 plans 12 at 10%): the tie goes to more scenarios
    assert [row.split(" | ")[:2] for row in table] == [["6", "5"], ["8", "5"], ["12", "5"], ["6", "10"]]


@pytest.mark.parametrize("extra, why", [
    (["--cost", "C0=x", "C3=4"], "COND=UNITS"), (["--cost", "C0", "C3=4"], "COND=UNITS"),
    (["--cost", "C0=0.4", "C9=4"], "unknown condition C9"), (["--cost", "C0=-1", "C3=4"], "at least 0"),
    (["--cost", "C0=0.4", "C0=1", "C3=4"], "C0 is given twice"), (["--cost", "C1=4", "C3=4"], "C0 and C3"),
    (["--target", "0.9"], "need --cost"), (["--exclusion", "0.1"], "need --cost"),
    (["--cost", "C0=0.4", "C3=4", "--exclusion", "1"], "exclusion"),
    (["--cost", "C0=0.4", "C3=4", "--target", "1.2"], "target"),
    (["--cost", "C0=0.4", "C3=4", "--budget", "0"], "budget"),
    (["--cost", "C0=0.4", "C3=4", "--min-scenarios", "0"], "min_scenarios")])
def test_cli_rejects_bad_ranking_options(extra, why, capsys):
    from simtest.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["power", "--c0", "0.1", "--c3", "0.3", "--scenarios", "8", "--repeats", "5", "--reps", "50"] + extra)
    message = f"{exc.value.code} {capsys.readouterr().err}"
    assert why in message and "unrecognized" not in message, message
