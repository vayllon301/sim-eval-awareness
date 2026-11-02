"""The primary test (stats/signflip.py): the exact sign-flip test on per-scenario differences,
its null distribution, the tie rule, and its use in the coding summary's paired difference."""

from __future__ import annotations

import itertools
from fractions import Fraction

import pytest

from simtest.stats.signflip import integer_p, min_p, null_counts, sign_flip_test, to_integers


def brute_p(diffs, alternative="two-sided"):
    """Every sign pattern enumerated, in exact fractions."""
    d = [Fraction(x).limit_denominator(10**6) for x in diffs]
    t = sum(d)
    hits = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        s = sum(e * x for e, x in zip(signs, d))
        hits += (abs(s) >= abs(t)) if alternative == "two-sided" else (s >= t if alternative == "greater" else s <= t)
    return hits / 2 ** len(d)


def test_null_distribution_of_a_small_case():
    # differences 1, 2: patterns +1+2=3, +1-2=-1, -1+2=1, -1-2=-3
    assert null_counts([1, 2]) == {3: 1, -1: 1, 1: 1, -3: 1}
    # a tie doubles every count: flipping a zero changes nothing
    assert null_counts([1, 2, 0]) == {3: 2, -1: 2, 1: 2, -3: 2}
    # equal magnitudes add up: 1, 1 -> 2 once, 0 twice, -2 once
    assert null_counts([1, 1]) == {2: 1, 0: 2, -2: 1}
    assert sum(null_counts([3, 1, 4, 1, 5]).values()) == 2**5


def test_six_scenarios_with_one_tie_cannot_go_below_0_0625():
    # PLAN.md: "with 6 scenarios and one tie, the smallest two-sided p is 0.0625"
    res = sign_flip_test([0.5, 0.3, 0.2, 0.1, 0.4, 0.0])
    assert res["p"] == 0.0625 == 2 / 32
    assert res["min_p"] == 0.0625 and res["n_ties"] == 1 and res["n_scenarios"] == 6
    assert sign_flip_test([0.5, 0.3, 0.2, 0.1, 0.4, 0.6])["p"] == 2 / 64
    assert min_p(6, 1) == 0.0625 and min_p(8, 0) == 2 / 256 and min_p(5, 0) == 0.0625


def test_p_values_by_hand():
    # 1, 2, 3: T = 6; |T*| >= 6 only for all + or all -: 2/8
    assert sign_flip_test([1, 2, 3])["p"] == 0.25
    # 3, -1, 1: T = 3; the 8 patterns give +-5, +-3, +-3, +-1, so 6 of 8 have |T*| >= 3
    assert sign_flip_test([3, -1, 1])["p"] == 0.75
    # one-sided: T = 6 is the largest of 8
    assert sign_flip_test([1, 2, 3], alternative="greater")["p"] == 1 / 8
    assert sign_flip_test([1, 2, 3], alternative="less")["p"] == 1.0
    # all zeros: every pattern ties the observed 0
    assert sign_flip_test([0, 0, 0])["p"] == 1.0


@pytest.mark.parametrize("diffs", [
    [0.3, -0.1, 0.5, 0.2, 0.0, 0.7, -0.2],
    [1 / 3, 2 / 3, -1 / 6, 0.5, 1 / 3],          # rates with denominators 3 and 6: exact ties between patterns
    [0.1, 0.1, 0.1, -0.1, 0.2, 0.0, 0.0, 0.3],
])
def test_matches_brute_force(diffs):
    for alt in ("two-sided", "greater", "less"):
        assert sign_flip_test(diffs, alternative=alt)["p"] == pytest.approx(brute_p(diffs, alt), abs=1e-15)


def test_ties_never_change_p():
    base = [0.4, 0.1, -0.2, 0.3, 0.5]
    with_ties = base + [0.0, 0.0]
    assert sign_flip_test(base)["p"] == sign_flip_test(with_ties)["p"]
    assert sign_flip_test(with_ties)["n_ties"] == 2
    assert sign_flip_test(with_ties)["min_p"] == min_p(7, 2) == min_p(5, 0)


def test_the_statistic_is_the_sum_and_order_does_not_matter():
    d = [0.25, -0.5, 0.75, 1.0]
    res = sign_flip_test(d)
    assert res["statistic"] == 1.5 and res["mean"] == 0.375
    assert sign_flip_test(list(reversed(d)))["p"] == res["p"]
    # rescaling every difference doesn't change p (the sum is scale-free in its ranking)
    assert sign_flip_test([x * 3 for x in d])["p"] == res["p"]


def test_integer_fast_path_agrees_with_the_test():
    for vals in ([5, 3, 2, 1, 4, 0], [3, -1, 1], [2, 2, -2, 0, 7, -1, 4, 4], [0, 0]):
        for alt in ("two-sided", "greater", "less"):
            assert integer_p(vals, alt) == sign_flip_test(vals, alt)["p"]


def test_rational_differences_are_exact():
    vals, scale = to_integers([1 / 3, 1 / 6, -0.5])
    assert scale == 6 and vals == [2, 1, -3]
    # 1/3 + 1/3 against 2/3 is an exact tie between patterns, not a rounding accident
    assert sign_flip_test([1 / 3, 1 / 3, 2 / 3])["p"] == brute_p([1 / 3, 1 / 3, 2 / 3])


def test_bad_inputs():
    with pytest.raises(ValueError):
        sign_flip_test([0.1, float("nan")])
    with pytest.raises(ValueError):
        sign_flip_test([0.1], alternative="sideways")
    assert sign_flip_test([])["n_scenarios"] == 0


def test_coding_paired_difference_carries_the_sign_flip_test():
    from simtest.coding.analysis import format_paired, paired_difference

    eps = []
    for i, (k0, k3) in enumerate([(0, 3), (1, 4), (0, 2), (1, 1), (0, 4), (2, 4)]):   # one tie (s3)
        for e in range(4):
            for cond, k, sim in (("C0", k0, "none"), ("C3", k3, "sim")):
                eps.append({"agent": "a", "condition": cond, "simulator": sim, "scenario": f"s{i}", "epoch": e,
                            "end_reason": "no_tool_call", "metrics": {"violation": e < k}})
    res = paired_difference(eps, a="C0", b="C3", metric="violation")
    assert res["sign_flip"]["n_scenarios"] == 6 and res["sign_flip"]["n_ties"] == 1
    assert res["sign_flip"]["p"] == 0.0625
    assert "primary test: exact sign-flip two-sided p=0.0625" in format_paired(res)
