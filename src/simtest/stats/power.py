"""Power of the exact sign-flip test, by simulation (PLAN.md, Design: "Power is computed by
simulation for this test over the number of scenarios and repeats, ties included").

One simulated run:
- S scenarios. Each gets true awareness rates for C0 and C3, spread around a centre on the
  logit scale:
      logit p0_s = logit(c0) + b_s
      logit p3_s = logit(c3) + b_s + e_s
  with b_s ~ Normal(0, spread) (a scenario that raises awareness raises it in both
  conditions) and e_s ~ Normal(0, effect_spread) (the C3 effect differs by scenario). `c0`
  and `c3` are the rates of a typical scenario (the median on the logit scale), not the mean
  rate. c0 = 0 keeps every scenario's C0 rate at zero ("C0 may sit at zero").
- n episodes per scenario and condition: k0_s ~ Binomial(n, p0_s), k3_s ~ Binomial(n, p3_s).
- The difference d_s = (k3_s - k0_s) / n, ties (d_s = 0) included, and the exact sign-flip
  test of `signflip` on the S differences (the same counting, on the integers k3_s - k0_s).
- Rejected when p <= alpha.

Power is the share of rejected runs, with a Wilson 95% interval for the Monte Carlo error.
Each grid cell has its own seed, derived from the base seed and the cell, so a cell's result
doesn't depend on the rest of the grid. Excluded episodes (flagged outputs) would lower n
below the design's; the simulation assumes none.
"""

from __future__ import annotations

import json

from ..analysis import wilson
from .signflip import ALTERNATIVES, integer_p, min_p

DEFAULT_SCENARIOS = (8, 10, 12, 14, 16)
DEFAULT_REPEATS = (5, 10, 15, 20, 25, 30)


def _expit(x):
    import numpy as np

    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    import math

    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf
    return math.log(p / (1 - p))


def _seed(seed: int, *cell) -> "object":
    import numpy as np

    return np.random.SeedSequence([seed, *[int(round(float(c) * 1_000_000)) for c in cell]])


def _check(name: str, p: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"{name} must be a rate between 0 and 1, got {p}")


def simulate_power(n_scenarios: int, repeats: int, c0: float, c3: float, spread: float = 0.0,
                   effect_spread: float = 0.0, alpha: float = 0.05, reps: int = 10_000, seed: int = 0,
                   alternative: str = "two-sided") -> dict:
    """Power of the exact sign-flip test for one design cell (see the module docstring)."""
    import numpy as np

    _check("c0", c0)
    _check("c3", c3)
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}")
    if n_scenarios < 1 or repeats < 1 or reps < 1:
        raise ValueError("scenarios, repeats and reps must be at least 1")
    if spread < 0 or effect_spread < 0:
        raise ValueError("spreads are standard deviations: they can't be negative")
    rng = np.random.default_rng(_seed(seed, n_scenarios, repeats, c0, c3, spread, effect_spread, alpha,
                                      ALTERNATIVES.index(alternative)))
    S, n = n_scenarios, repeats
    b = rng.normal(0.0, spread, (reps, S)) if spread else np.zeros((reps, S))
    e = rng.normal(0.0, effect_spread, (reps, S)) if effect_spread else np.zeros((reps, S))
    p0 = _expit(_logit(c0) + b)
    p3 = _expit(_logit(c3) + b + e)
    k0 = rng.binomial(n, p0)
    k3 = rng.binomial(n, p3)
    d = k3 - k0
    cache: dict[tuple, float] = {}
    rejected = cannot = 0
    for row in d:
        mags = tuple(sorted(abs(int(v)) for v in row))
        t = int(row.sum())
        key = (mags, abs(t)) if alternative == "two-sided" else (mags, t)
        if key not in cache:
            cache[key] = integer_p([int(v) for v in row], alternative)
        rejected += cache[key] <= alpha
        cannot += min_p(S, mags.count(0), alternative) > alpha
    ties = (d == 0).sum(axis=1)
    lo, hi = wilson(int(rejected), reps)
    return {"scenarios": S, "repeats": n, "c0": c0, "c3": c3, "spread": spread, "effect_spread": effect_spread,
            "alpha": alpha, "alternative": alternative, "reps": reps, "seed": seed,
            "power": rejected / reps, "ci": [lo, hi], "mean_ties": float(ties.mean()),
            "cannot_reject": cannot / reps, "mean_diff": float((d / n).mean()),
            "mean_rate_c0": float(p0.mean()), "mean_rate_c3": float(p3.mean())}


def planned_repeats(n: int, exclusion: float = 0.0) -> int:
    """Episodes to plan per scenario and condition so that `n` are scored after exclusions.

    The smallest n_planned with round(n_planned x (1 - exclusion)) >= n, where round is to the
    nearest whole number and a half rounds down (a half episode is never counted as scored). The
    arithmetic is exact on the decimal given (0.1 is one tenth). exclusion = 0 gives n.
    """
    import math
    from fractions import Fraction

    if n < 1:
        raise ValueError("repeats must be at least 1")
    if not 0.0 <= exclusion < 1.0:
        raise ValueError(f"exclusion must be a rate in [0, 1), got {exclusion}")
    keep = 1 - Fraction(str(exclusion))
    m = n
    while math.ceil(m * keep - Fraction(1, 2)) < n:
        m += 1
    return m


def power_grid(c0: float, c3: float, scenarios=DEFAULT_SCENARIOS, repeats=DEFAULT_REPEATS, **kw) -> list[dict]:
    """simulate_power over every S in `scenarios` and n in `repeats`."""
    return [simulate_power(S, n, c0, c3, **kw) for S in scenarios for n in repeats]


def format_grid(rows: list[dict], label: str | None = None) -> str:
    """One table per (c0, c3, spread, effect_spread): repeats down, scenarios across, each cell
    `power [95% MC interval]`."""
    out = []
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["c0"], r["c3"], r["spread"], r["effect_spread"], r["alpha"], r["alternative"]),
                          []).append(r)
    for (c0, c3, sp, esp, alpha, alt), rs in groups.items():
        head = (f"Power of the exact sign-flip test ({alt}, alpha {alpha}): C0 {c0:g}, C3 {c3:g}, logit spread "
                f"{sp:g} shared + {esp:g} in the effect; {rs[0]['reps']:,} runs per cell, seed {rs[0]['seed']}")
        if label:
            head = f"[{label}] " + head
        out.append(head)
        Ss = sorted({r["scenarios"] for r in rs})
        ns = sorted({r["repeats"] for r in rs})
        cell = {(r["scenarios"], r["repeats"]): r for r in rs}
        out.append("repeats \\ scenarios | " + " | ".join(f"S={S}" for S in Ss))
        for n in ns:
            vals = []
            for S in Ss:
                r = cell.get((S, n))
                vals.append("-" if r is None else f"{r['power']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
            out.append(f"n={n} | " + " | ".join(vals))
        ties = ", ".join(f"S={S}: {sum(cell[(S, n)]['mean_ties'] for n in ns if (S, n) in cell) / len(ns):.2f}"
                         for S in Ss)
        out.append(f"mean ties per run (averaged over n): {ties}")
        rr = rs[0]
        out.append(f"mean true rates across scenarios: C0 {rr['mean_rate_c0']:.3f}, C3 {rr['mean_rate_c3']:.3f}")
        out.append("")
    return "\n".join(out).rstrip()


def add_parser(sub) -> None:
    p = sub.add_parser("power", help="power of the exact sign-flip test by simulation (no model calls)")
    p.add_argument("--c0", type=float, nargs="+", required=True, help="C0 awareness rate of a typical scenario")
    p.add_argument("--c3", type=float, nargs="+", required=True, help="C3 awareness rate of a typical scenario")
    p.add_argument("--spread", type=float, default=0.0,
                   help="SD across scenarios on the logit scale, shared by both conditions (default 0)")
    p.add_argument("--effect-spread", type=float, default=None,
                   help="SD of the C3 effect across scenarios on the logit scale (default: --spread)")
    p.add_argument("--scenarios", type=int, nargs="+", default=list(DEFAULT_SCENARIOS))
    p.add_argument("--repeats", type=int, nargs="+", default=list(DEFAULT_REPEATS),
                   help="episodes per scenario and condition")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--alternative", choices=ALTERNATIVES, default="two-sided")
    p.add_argument("--reps", type=int, default=10_000, help="simulated runs per cell")
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--label", default=None, help='printed on every table, e.g. "development, not the pilot"')
    p.add_argument("--json", action="store_true")


def main(args) -> None:
    esp = args.spread if args.effect_spread is None else args.effect_spread
    rows = []
    for c0 in args.c0:
        for c3 in args.c3:
            rows += power_grid(c0, c3, scenarios=args.scenarios, repeats=args.repeats, spread=args.spread,
                               effect_spread=esp, alpha=args.alpha, reps=args.reps, seed=args.seed,
                               alternative=args.alternative)
    if args.json:
        print(json.dumps({"label": args.label, "rows": rows}, indent=1))
    else:
        print(format_grid(rows, label=args.label))
