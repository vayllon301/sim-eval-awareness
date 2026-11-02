"""The primary test (PLAN.md, Design, Statistics): an exact sign-flip test on the paired
per-scenario differences d_s = rate(C3, s) - rate(C0, s), one per scenario.

- **Null.** The condition has no effect, so within a scenario the condition labels are
  exchangeable and each scenario's difference is symmetric about zero. Then every sign
  pattern (e_1 d_1, ..., e_S d_S), e_s = +1 or -1, is equally likely: 2^S patterns.
- **Statistic.** T = the sum of the differences (the mean gives the same test: S is fixed).
  It uses the sizes of the differences, not only their signs, so a scenario where awareness
  moves by 0.8 weighs more than one where it moves by 0.1. This is Fisher's randomisation test
  for paired data, the "exact sign-flip test" OpenAI preregistered (docs/PRIOR_SETUPS.md,
  section 4). A sign test (signs only) throws that information away; a studentised statistic
  (the t value) would rank patterns differently but has no advantage at 8-16 scenarios and
  is harder to explain.
- **p-value.** Two-sided: the share of the 2^S patterns with |T*| >= |T|, the observed
  pattern included, so p >= 2 / 2^S. One-sided ("greater" or "less") is available but the
  plan's H1 is two-sided.
- **Ties** (d_s = 0) are kept and never change p: flipping a zero changes nothing, so every
  sign pattern of the non-zero differences is counted 2^z times, above and below the line
  alike. Dropping them gives the same p. They cost power, though: with z ties the smallest
  two-sided p is 2 / 2^(S - z). With 6 scenarios and one tie that is 2/32 = 0.0625, so such a
  run can't reject at 0.05 whatever the effect (PLAN.md). That is why the power simulation
  counts them.
- **Exact arithmetic.** Each difference is turned into a fraction (rates are k/n, and the
  per-scenario difference is a mean of such: `Fraction.limit_denominator` recovers them
  exactly), all are scaled to integers by their common denominator, and the null distribution
  of T is counted exactly by convolution, one scenario at a time. No floating-point ties, and
  the same counting serves the power simulation (`integer_p`).
"""

from __future__ import annotations

import math
from fractions import Fraction

ALTERNATIVES = ("two-sided", "greater", "less")
MAX_DENOMINATOR = 10**9


def _check_alternative(alternative: str) -> None:
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")


def to_integers(diffs) -> tuple[list[int], int]:
    """(integers, scale): each difference times the common denominator of all of them."""
    fracs = []
    for d in diffs:
        if isinstance(d, float) and not math.isfinite(d):
            raise ValueError(f"a difference is not finite: {d!r}")
        f = d if isinstance(d, Fraction) else Fraction(d).limit_denominator(MAX_DENOMINATOR)
        fracs.append(f)
    scale = 1
    for f in fracs:
        scale = scale * f.denominator // math.gcd(scale, f.denominator)
    return [int(f * scale) for f in fracs], scale


def null_counts(values: list[int]) -> dict[int, int]:
    """{t: number of sign patterns with sum t} over all 2^S patterns of the integers."""
    counts = {0: 1}
    for a in values:
        a = abs(a)
        nxt: dict[int, int] = {}
        for t, c in counts.items():
            nxt[t + a] = nxt.get(t + a, 0) + c
            nxt[t - a] = nxt.get(t - a, 0) + c
        counts = nxt
    return counts


def _tail(counts: dict[int, int], t_obs: int, alternative: str) -> int:
    if alternative == "two-sided":
        return sum(c for t, c in counts.items() if abs(t) >= abs(t_obs))
    if alternative == "greater":
        return sum(c for t, c in counts.items() if t >= t_obs)
    return sum(c for t, c in counts.items() if t <= t_obs)


def integer_p(values, alternative: str = "two-sided") -> float:
    """The exact p for integer differences (the power simulation's fast path; numpy)."""
    import numpy as np

    _check_alternative(alternative)
    a = [abs(int(v)) for v in values]
    t_obs = int(sum(int(v) for v in values))
    total = sum(a)
    counts = np.zeros(2 * total + 1, dtype=np.int64)
    counts[total] = 1
    for v in a:
        if v == 0:
            counts *= 2
            continue
        nxt = np.zeros_like(counts)
        nxt[v:] += counts[:-v]
        nxt[:-v] += counts[v:]
        counts = nxt
    t = np.arange(-total, total + 1)
    if alternative == "two-sided":
        hit = counts[np.abs(t) >= abs(t_obs)].sum()
    elif alternative == "greater":
        hit = counts[t >= t_obs].sum()
    else:
        hit = counts[t <= t_obs].sum()
    return float(hit) / float(2 ** len(a))


def min_p(n_scenarios: int, n_ties: int = 0, alternative: str = "two-sided") -> float:
    """The smallest p the test can give: 2 / 2^(S - z) two-sided, 1 / 2^(S - z) one-sided."""
    _check_alternative(alternative)
    k = n_scenarios - n_ties
    return min(1.0, (2.0 if alternative == "two-sided" else 1.0) / 2**k)


def sign_flip_test(diffs, alternative: str = "two-sided") -> dict:
    """The exact sign-flip test on per-scenario differences (any order, ties included).

    Returns {"statistic" (the sum), "mean", "p", "n_scenarios", "n_ties", "min_p",
    "n_patterns", "alternative"}. With no scenarios, p is NaN."""
    _check_alternative(alternative)
    diffs = list(diffs)
    S = len(diffs)
    if S == 0:
        nan = float("nan")
        return {"statistic": nan, "mean": nan, "p": nan, "n_scenarios": 0, "n_ties": 0, "min_p": nan,
                "n_patterns": 0, "alternative": alternative}
    values, scale = to_integers(diffs)
    t_obs = sum(values)
    counts = null_counts(values)
    p = _tail(counts, t_obs, alternative) / 2**S
    ties = sum(v == 0 for v in values)
    return {"statistic": t_obs / scale, "mean": t_obs / scale / S, "p": p, "n_scenarios": S, "n_ties": ties,
            "min_p": min_p(S, ties, alternative), "n_patterns": 2**S, "alternative": alternative}


def format_test(res: dict) -> str:
    if not res["n_scenarios"]:
        return "sign-flip: no scenarios"
    return (f"exact sign-flip {res['alternative']} p={res['p']:.4f} (S={res['n_scenarios']}, "
            f"ties {res['n_ties']}, smallest possible p {res['min_p']:.4f})")
