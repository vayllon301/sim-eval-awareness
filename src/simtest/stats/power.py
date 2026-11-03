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
below the design's; the simulation assumes none, so its n is the scored n.

The cheapest design for a power target (`cheapest_designs`, `simtest power --cost`). The run
gives every scenario all its conditions with the same n, so a design of S scenarios costs
S x n_planned x (the sum of its conditions' units per episode). Over the grid:
- A design reaches the target when the lower end of its 95% Monte Carlo interval does, not
  its estimate, so the cheapest cell isn't picked for its Monte Carlo noise.
- No design below a floor of scenarios (8 by default); ties in cost go to more scenarios.
- n_planned covers an expected exclusion rate: the smallest n_planned with
  n_planned x (1 - exclusion) >= n, so ceil(n / (1 - exclusion)) (`planned_repeats`).
The test, alpha and the simulation are the same with or without costs.
"""

from __future__ import annotations

import json
import math

from ..analysis import wilson
from ..observers import CONDITIONS
from .signflip import ALTERNATIVES, integer_p, min_p

DEFAULT_SCENARIOS = (8, 10, 12, 14, 16)
DEFAULT_REPEATS = (5, 10, 15, 20, 25, 30)
DEFAULT_TARGET = 0.8
DEFAULT_MIN_SCENARIOS = 8


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

    The smallest n_planned whose expected scored count, n_planned x (1 - exclusion), is at least n:
    ceil(n / (1 - exclusion)). The expected count is never rounded up, so it never falls below
    the n the power calculation asks for. The arithmetic is exact on the decimal given (0.1 is one
    tenth). exclusion = 0 gives n.
    """
    import math
    from fractions import Fraction

    if n < 1:
        raise ValueError("repeats must be at least 1")
    if not 0.0 <= exclusion < 1.0:
        raise ValueError(f"exclusion must be a rate in [0, 1), got {exclusion}")
    keep = 1 - Fraction(str(exclusion))
    m = n
    while m * keep < n:
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


def _ranking_inputs(cost: dict, conditions, target: float, min_scenarios: int, exclusion: float,
                    budget: float | None):
    """Check the inputs of `cheapest_designs`; return the run's conditions in order and the exact
    units one scenario and repeat costs across them."""
    from fractions import Fraction

    if not 0.0 < target < 1.0:
        raise ValueError(f"target must be a power between 0 and 1, got {target}")
    if min_scenarios < 1:
        raise ValueError(f"min_scenarios must be at least 1, got {min_scenarios}")
    if not 0.0 <= exclusion < 1.0:
        raise ValueError(f"exclusion must be a rate in [0, 1), got {exclusion}")
    if budget is not None and not budget > 0:
        raise ValueError(f"budget must be positive, got {budget}")
    conds = list(cost) if conditions is None else list(conditions)
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise ValueError(f"unknown conditions {bad} (known: {list(CONDITIONS)})")
    if "C0" not in conds or "C3" not in conds:
        raise ValueError("the run must include C0 and C3: the test compares them")
    missing = [c for c in conds if c not in cost]
    if missing:
        raise ValueError(f"no cost given for {missing}")
    for c in conds:
        if not math.isfinite(float(cost[c])) or cost[c] < 0:
            raise ValueError(f"the cost of {c} must be a finite number of units, at least 0; got {cost[c]}")
    conds = [c for c in CONDITIONS if c in conds]
    per_set = sum(Fraction(str(cost[c])) for c in conds)
    if per_set == 0:
        raise ValueError("every condition costs 0 units: there is nothing to minimise")
    return conds, per_set


def cheapest_designs(rows: list[dict], cost: dict, conditions=None, target: float = DEFAULT_TARGET,
                     min_scenarios: int = DEFAULT_MIN_SCENARIOS, exclusion: float = 0.0,
                     budget: float | None = None) -> dict:
    """The grid's designs that reach the target power, cheapest first.

    `rows` are one (c0, c3) pair's cells from `power_grid`. `cost` gives the units one episode
    costs in each condition, and `conditions` the run's conditions (default: the keys of `cost`).
    They must include C0 and C3, the two the test compares. Every design runs every condition
    with the same n, so a design of S scenarios and n scored repeats costs
        S x planned_repeats(n, exclusion) x (the sum of the run's conditions' costs)
    units, excluded episodes included.

    - A design reaches the target when the lower end of its 95% Monte Carlo interval is at
      least `target` (ci[0] >= target), not its point estimate: the cheapest cell by estimate
      would often be the one Monte Carlo noise pushed up.
    - Designs with fewer than `min_scenarios` scenarios are never reported, nor are designs
      above `budget` units when a budget is given.
    - Cheapest first; ties in cost go to more scenarios (the scenario is the unit of the test).

    Returns the inputs, `cells` (the grid cells with enough scenarios), `designs` (the ranking),
    `cheapest` (its first design or None), `best` (the highest power among the designs
    considered, reached or not) and, when a budget leaves nothing, `cheapest_over_budget`. The
    power is the simulated one at the scored n; nothing here changes the test.
    """
    from fractions import Fraction

    if not rows:
        raise ValueError("no design cells to rank")
    pairs = {(r["c0"], r["c3"], r["spread"], r["effect_spread"], r["alpha"], r["alternative"]) for r in rows}
    if len(pairs) > 1:
        raise ValueError("rank one (c0, c3) pair at a time")
    conds, per_set = _ranking_inputs(cost, conditions, target, min_scenarios, exclusion, budget)
    planned = {n: planned_repeats(n, exclusion) for n in {r["repeats"] for r in rows}}
    seen, considered, over = set(), [], []
    for r in rows:
        S, n = r["scenarios"], r["repeats"]
        if S < min_scenarios or (S, n) in seen:
            continue
        seen.add((S, n))
        units = S * planned[n] * per_set
        d = {"scenarios": S, "repeats": n, "planned_repeats": planned[n],
             "episodes": S * planned[n] * len(conds), "units": units, "power": r["power"], "ci": list(r["ci"])}
        (considered if budget is None or units <= Fraction(str(budget)) else over).append(d)

    def by_cost(d):
        return (d["units"], -d["scenarios"], d["repeats"])

    def reached(ds):
        return sorted((d for d in ds if d["ci"][0] >= target), key=by_cost)

    designs = reached(considered)
    best = min(considered, key=lambda d: (-d["power"], *by_cost(d)), default=None)
    over_budget = (reached(over) or [None])[0] if budget is not None and not designs else None

    def out(d):
        return None if d is None else {**d, "units": float(d["units"])}

    r0 = rows[0]
    return {"c0": r0["c0"], "c3": r0["c3"], "spread": r0["spread"], "effect_spread": r0["effect_spread"],
            "alpha": r0["alpha"], "alternative": r0["alternative"], "reps": r0["reps"], "seed": r0["seed"],
            "target": target, "rule": "ci[0] >= target", "conditions": conds,
            "cost": {c: cost[c] for c in conds}, "units_per_set": float(per_set),
            "min_scenarios": min_scenarios, "exclusion": exclusion, "budget": budget, "cells": len(seen),
            "designs": [out(d) for d in designs], "cheapest": out(designs[0]) if designs else None,
            "best": out(best), "cheapest_over_budget": out(over_budget)}


def _units(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _design(d: dict) -> str:
    return f"S={d['scenarios']}, n={d['repeats']} ({d['planned_repeats']} planned)"


def _power(d: dict) -> str:
    return f"{d['power']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]"


def format_ranking(res: dict, label: str | None = None) -> str:
    """The ranking of `cheapest_designs` as text: its rules, the designs that reach the target
    cheapest first, then one line naming the cheapest (or saying that none reaches it). `label`
    is printed first, as on the grid's tables."""
    t, ex, conds = res["target"], res["exclusion"], res["conditions"]
    costs = " + ".join(_units(float(res["cost"][c])) for c in conds)
    budget = f"; within {_units(res['budget'])} units" if res["budget"] is not None else ""
    head = (f"Cheapest designs that reach power {t:g}: C0 {res['c0']:g}, C3 {res['c3']:g}, logit spread "
            f"{res['spread']:g} shared + {res['effect_spread']:g} in the effect; {res['reps']:,} runs per cell, "
            f"seed {res['seed']}")
    out = [f"[{label}] {head}" if label else head,
           f"- a design reaches it when the lower end of its 95% Monte Carlo interval is at least {t:g}, not the "
           f"point estimate (the cheapest cell by estimate would often be Monte Carlo noise)",
           f"- conditions {', '.join(conds)} at {costs} = {_units(res['units_per_set'])} units per scenario and "
           f"repeat; at least {res['min_scenarios']} scenarios{budget}; ties in cost go to more scenarios"]
    if ex:
        out.append(f"- n is scored; planned n covers {ex * 100:g}% exclusions: the smallest planned n with "
                   f"planned n x {1 - ex:g} >= n")
    else:
        out.append("- no exclusion rate given: planned n = n")
    if res["designs"]:
        out.append("S | n | planned n | episodes | units | power [95% MC interval]")
        for d in res["designs"]:
            out.append(f"{d['scenarios']} | {d['repeats']} | {d['planned_repeats']} | {d['episodes']:,} | "
                       f"{_units(d['units'])} | {_power(d)}")
        c = res["cheapest"]
        out.append(f"Cheapest: {_design(c)} per scenario and condition, {c['episodes']:,} episodes, "
                   f"{_units(c['units'])} units, power {_power(c)}. Its power comes from the runs that picked "
                   f"it; to confirm it, rerun that cell alone with another --seed.")
        return "\n".join(out)
    if not res["cells"]:
        out.append(f"No design in the grid has at least {res['min_scenarios']} scenarios (--min-scenarios).")
        return "\n".join(out)
    where = f"within {_units(res['budget'])} units" if res["budget"] is not None else "in the grid"
    out.append(f"No design {where} reaches power {t:g} with at least {res['min_scenarios']} scenarios, on the "
               f"lower end of its 95% Monte Carlo interval.")
    if res["best"] is not None:
        b = res["best"]
        out.append(f"Best power found: {_design(b)}: {_power(b)}, {b['episodes']:,} episodes, "
                   f"{_units(b['units'])} units.")
    o = res["cheapest_over_budget"]
    if o is not None:
        out.append(f"The cheapest that does costs {_units(o['units'])} units: {_design(o)}, "
                   f"{o['episodes']:,} episodes, power {_power(o)}.")
    else:
        out.append("Widen --scenarios or --repeats.")
    return "\n".join(out)


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
    g = p.add_argument_group("cheapest design", "with --cost, rank the grid's designs that reach --target by cost")
    g.add_argument("--cost", type=_cost_item, nargs="+", default=None, metavar="COND=UNITS",
                   help="units one episode costs in each of the run's conditions, e.g. C0=0.4 C1=4 C2=4 C3=4; "
                        "the conditions given are the run's and must include C0 and C3")
    g.add_argument("--target", type=float, default=None,
                   help="power to reach, on the lower end of the 95%% Monte Carlo interval "
                        f"(default {DEFAULT_TARGET})")
    g.add_argument("--min-scenarios", type=int, default=None,
                   help=f"never report a design with fewer scenarios (default {DEFAULT_MIN_SCENARIOS})")
    g.add_argument("--exclusion", type=float, default=None,
                   help="expected share of excluded episodes (default 0). The planned repeats are the smallest "
                        "n_planned with n_planned x (1 - exclusion) >= n, the grid's scored n; costs count the "
                        "planned episodes")
    g.add_argument("--budget", type=float, default=None, help="list only designs within this many units")


def _cost_item(text: str) -> tuple[str, float]:
    import argparse

    cond, sep, units = text.partition("=")
    try:
        value = float(units)
    except ValueError:
        value = None
    if not sep or value is None:
        raise argparse.ArgumentTypeError(f"{text!r}: give COND=UNITS with a number of units, e.g. C0=0.4")
    if cond not in CONDITIONS:
        raise argparse.ArgumentTypeError(f"unknown condition {cond} (known: {', '.join(CONDITIONS)})")
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r}: the units must be a finite number, at least 0")
    return cond, value


def main(args) -> None:
    esp = args.spread if args.effect_spread is None else args.effect_spread
    ranking_options = (args.target, args.min_scenarios, args.exclusion, args.budget)
    if args.cost is None and any(v is not None for v in ranking_options):
        raise SystemExit("--target, --min-scenarios, --exclusion and --budget rank designs by cost: they need --cost")
    if args.cost is not None:
        cost = {}
        for cond, units in args.cost:
            if cond in cost:
                raise SystemExit(f"--cost: {cond} is given twice")
            cost[cond] = units
        rank = {"cost": cost, "target": DEFAULT_TARGET if args.target is None else args.target,
                "min_scenarios": DEFAULT_MIN_SCENARIOS if args.min_scenarios is None else args.min_scenarios,
                "exclusion": args.exclusion or 0.0, "budget": args.budget}
        try:
            _ranking_inputs(conditions=None, **rank)          # before the simulation, not after it
        except ValueError as exc:
            raise SystemExit(f"--cost and its options: {exc}") from exc
    rows, ranking = [], []
    for c0 in args.c0:
        for c3 in args.c3:
            pair = power_grid(c0, c3, scenarios=args.scenarios, repeats=args.repeats, spread=args.spread,
                              effect_spread=esp, alpha=args.alpha, reps=args.reps, seed=args.seed,
                              alternative=args.alternative)
            rows += pair
            if args.cost is not None:
                ranking.append(cheapest_designs(pair, **rank))
    if args.json:
        out = {"label": args.label, "rows": rows}
        if args.cost is not None:
            out["ranking"] = ranking
        print(json.dumps(out, indent=1))
    else:
        print("\n\n".join([format_grid(rows, label=args.label)]
                           + [format_ranking(r, label=args.label) for r in ranking]))
