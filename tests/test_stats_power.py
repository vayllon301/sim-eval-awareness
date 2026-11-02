"""Power of the exact sign-flip test by simulation (stats/power.py)."""

from __future__ import annotations

import json
import math

import pytest

from simtest.stats.power import format_grid, power_grid, simulate_power


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
