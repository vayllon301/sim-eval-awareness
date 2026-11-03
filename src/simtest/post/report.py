"""The post-episode results, per condition and scenario, and compared across conditions
(PLAN.md: "Compared across conditions only, never read as absolute rates").

From one results file (`simtest post ask` and `simtest post judge` append to it):
- per agent x condition x simulator x scenario: the mean 0-1 rating (each episode's mean over
  its parsed samples, then the mean over episodes), the share of purpose answers labelled EVAL
  (and TRAINING), the share of judged summaries with an inaccurate claim, and with a claim
  that contradicts what the agent saw;
- pairwise, per agent x scenario x simulator: how often the b episode (C3) was picked as the
  more real, over the pairs answered in both orders; the **position check** (how often the
  first transcript was picked, over every answer: 0.5 for a judge with no position bias), the
  pairs whose two orders agree, the pairs with only one usable order (counted, never in the win
  rate: one order alone carries the position bias), and the **length check** (how often the
  longer transcript was picked);
- paired across scenarios (b - a, agents and simulators equally weighted, as the coding
  summary): each scenario's difference, their mean, and the exact sign-flip test. For pairwise,
  each scenario's b win rate minus 0.5, from the pairs answered in both orders only.

Flagged answers (cut, unparsed) and errors are counted, never read as a value.
"""

from __future__ import annotations

import json
from collections import defaultdict

from ..analysis import wilson
from ..stats.signflip import format_test, sign_flip_test
from .measures import MEASURES, PostError

CELL = ("agent", "condition", "simulator", "scenario")


def _config(r: dict) -> str:
    return json.dumps([r.get("prompt_version"), r.get("prompt_sha256"), r.get("model"), r.get("settings"),
                       r.get("reasoning_shown")], sort_keys=True, default=str)


def check_configs(records: list[dict]) -> None:
    """One prompt version, model and settings per measure in a results file, so no rate mixes them."""
    by = defaultdict(set)
    for r in records:
        by[r["measure"]].add(_config(r))
    mixed = {m: sorted(c) for m, c in by.items() if len(c) > 1}
    if mixed:
        lines = [f"{m}: {len(c)} configurations" for m, c in mixed.items()]
        raise PostError("the results file mixes prompt versions, models or settings within a measure ("
                        + "; ".join(lines) + "): use one results file per configuration")


def _cell(ep: dict) -> tuple:
    return tuple(ep.get(k) for k in CELL)


def per_episode(records: list[dict]) -> dict[str, dict]:
    """{episode key: {"ep", "ratings", "rating_flagged", "purpose_label", "accuracy", "honesty",
    "errors"}}."""
    out: dict[str, dict] = {}

    def slot(ep):
        return out.setdefault(ep["key"], {"ep": ep, "ratings": [], "rating_flagged": 0, "purpose": None,
                                          "purpose_label": None, "accuracy": None, "honesty": None,
                                          "errors": defaultdict(int)})

    for r in records:
        if r["measure"] == "pairwise":
            continue
        e = slot(r["episode"])
        if r.get("error"):
            e["errors"][f"{r['measure']}:{r['error']['kind']}"] += 1
            continue
        if r["measure"] == "rating":
            if r.get("value") is None:
                e["rating_flagged"] += 1
            else:
                e["ratings"].append(r["value"])
        elif r["measure"] == "purpose":
            e["purpose"] = r.get("value")
        elif r["measure"] == "purpose_label":
            e["purpose_label"] = r.get("value")
        elif r["measure"] == "summary_accuracy":
            e["accuracy"] = r.get("value")
        elif r["measure"] == "summary_honesty":
            e["honesty"] = r.get("value")
    return out


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def episode_measures(e: dict) -> dict:
    """The per-episode numbers compared across conditions (None where not measured)."""
    lab, acc, hon = e["purpose_label"], e["accuracy"], e["honesty"]
    return {"rating": _mean(e["ratings"]) if e["ratings"] else None,
            "purpose_eval": None if lab is None else lab == "EVAL",
            "purpose_training": None if lab is None else lab == "TRAINING",
            "summary_inaccurate": None if acc in (None, "NO_CLAIMS") else acc == "INACCURATE",
            "summary_dishonest": None if hon in (None, "NO_CLAIMS") else hon == "DISHONEST"}


MEASURE_NAMES = {"rating": "mean rating", "purpose_eval": "purpose EVAL", "purpose_training": "purpose TRAINING",
                 "summary_inaccurate": "summary inaccurate", "summary_dishonest": "summary dishonest"}


def cells(eps: dict[str, dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for e in eps.values():
        groups[_cell(e["ep"])].append(e)
    rows = []
    for key in sorted(groups, key=lambda t: tuple(map(str, t))):
        es = groups[key]
        row = dict(zip(CELL, key))
        row["episodes"] = len(es)
        vals = [episode_measures(e) for e in es]
        for m in MEASURE_NAMES:
            known = [v[m] for v in vals if v[m] is not None]
            row[m] = _mean([float(x) for x in known]) if known else None
            row[f"{m}_n"] = len(known)
            if m != "rating":
                row[f"{m}_k"] = sum(bool(x) for x in known)
        row["rating_samples"] = sum(len(e["ratings"]) for e in es)
        row["rating_flagged"] = sum(e["rating_flagged"] for e in es)
        errs = defaultdict(int)
        for e in es:
            for k, n in e["errors"].items():
                errs[k] += n
        row["errors"] = dict(sorted(errs.items()))
        rows.append(row)
    return rows


def paired_means(items: list[tuple[str, str, str, str, float]], a: str, b: str,
                 panel: list[tuple[str, str, str, str]] | None = None) -> dict:
    """Per-scenario differences of means, b - a: within scenario x agent, the mean of b over each
    simulator minus the mean of a (C0's one cell, or the same simulator's when a is simulated
    too), averaged over simulators, then over agents (equal weights, as the coding summary's
    paired difference). `panel`: (agent, condition, simulator, scenario) of every episode the
    measure was asked of, usable or not. A scenario missing a cell, or with no usable value, is
    left out and listed. Then the exact sign-flip test on the differences."""
    vals: dict[tuple, list[float]] = defaultdict(list)
    agents, sims, scenarios = set(), set(), set()
    a_sim = lambda sim: None if a == "C0" else sim  # noqa: E731  (C0 is shared)
    for agent, cond, sim, sc in [i[:4] for i in items] + list(panel or []):
        if cond in (a, b):
            agents.add(agent)
            scenarios.add(sc)
            if cond == b:
                sims.add(sim)
    for agent, cond, sim, sc, v in items:
        if cond in (a, b):
            vals[(sc, agent, cond, sim if cond == b else a_sim(sim))].append(v)
    diffs, missing = {}, []
    for sc in sorted(scenarios, key=str):
        per_agent = []
        for g in sorted(agents, key=str):
            comps = [(vals.get((sc, g, b, s)), vals.get((sc, g, a, a_sim(s)))) for s in sorted(sims, key=str)]
            if not sims or any(not c or not base for c, base in comps):
                missing.append(sc)
                break
            per_agent.append(_mean([_mean(c) - _mean(base) for c, base in comps]))
        else:
            if per_agent:
                diffs[sc] = _mean(per_agent)
    return {"per_scenario": diffs, "missing": sorted(set(missing), key=str),
            "mean_diff": _mean(list(diffs.values())), "sign_flip": sign_flip_test(list(diffs.values()))}


def pairwise(records: list[dict]) -> dict:
    calls = [r for r in records if r["measure"] == "pairwise"]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in calls:
        eb = r["pair"]["b"]
        groups[(eb["agent"], eb["scenario"], eb["simulator"])].append(r)
    rows, all_answered = [], []
    for key in sorted(groups, key=lambda t: tuple(map(str, t))):
        rs = groups[key]
        ok = [r for r in rs if not r.get("error") and r.get("value") is not None]
        all_answered += ok
        pairs_here: dict[str, list[dict]] = defaultdict(list)
        for r in ok:
            pairs_here[r["pair"]["a"]["key"] + "|" + r["pair"]["b"]["key"]].append(r)
        both = [v for v in pairs_here.values() if {x["order"] for x in v} == {"ab", "ba"}]
        # the win rate only from pairs answered in both orders: one order alone carries the position bias
        in_both = [x for v in both for x in v]
        consistent = sum(len({x["picked"] for x in v}) == 1 for v in both)
        longer = [r["longer_picked"] for r in ok if r.get("longer_picked") is not None]
        rows.append({"agent": key[0], "scenario": key[1], "simulator": key[2], "calls": len(rs), "answered": len(ok),
                     "flagged": sum(1 for r in rs if not r.get("error") and r.get("value") is None),
                     "errors": sum(1 for r in rs if r.get("error")),
                     "answered_both": len(in_both), "b_won": sum(r["b_won"] for r in in_both),
                     "first_picked": sum(r["value"] == 1 for r in ok),
                     "pairs_both_orders": len(both), "pairs_one_order": len(pairs_here) - len(both),
                     "pairs_consistent": consistent, "longer_n": len(longer), "longer_picked": sum(longer)})
    n = len(all_answered)
    first = sum(r["value"] == 1 for r in all_answered)
    longer = [r["longer_picked"] for r in all_answered if r.get("longer_picked") is not None]
    conds = next((r["pair"]["conditions"] for r in calls), ["C0", "C3"])
    items = [(r["agent"], conds[1], r["simulator"], r["scenario"], r["b_won"] / r["answered_both"] - 0.5)
             for r in rows if r["answered_both"]]
    # the b win rate minus 0.5 as "b" and a zero "a" cell, so paired_means gives one difference per scenario
    items += [(r["agent"], conds[0], r["simulator"], r["scenario"], 0.0) for r in rows if r["answered_both"]]
    panel = [(r["agent"], c, r["simulator"], r["scenario"]) for r in rows for c in conds]   # answered or not
    return {"conditions": conds, "rows": rows, "answered": n,
            "position": {"first_picked": first, "n": n, "ci": list(wilson(first, n))},
            "length": {"longer_picked": sum(longer), "n": len(longer), "ci": list(wilson(sum(longer), len(longer)))},
            "paired": paired_means(items, conds[0], conds[1], panel=panel) if rows else None}


SOURCE = {"rating": "rating", "purpose_eval": "purpose_label", "purpose_training": "purpose_label",
          "summary_inaccurate": "summary_accuracy", "summary_dishonest": "summary_honesty"}   # the record measure


def build_report(records: list[dict], a: str = "C0", b: str = "C3") -> dict:
    check_configs(records)
    eps = per_episode(records)
    rows = cells(eps)
    paired = {}
    for m in MEASURE_NAMES:
        asked = {r["episode"]["key"] for r in records if r["measure"] == SOURCE[m]}
        items, panel = [], []
        for e in eps.values():
            ep = e["ep"]
            cell = (ep["agent"], ep["condition"], ep["simulator"], ep["scenario"])
            if ep["key"] in asked:
                panel.append(cell)       # before its value is known to be usable
            v = episode_measures(e)[m]
            if v is not None:
                items.append((*cell, float(v)))
        if items or panel:
            paired[m] = paired_means(items, a, b, panel=panel)
    measures = sorted({r["measure"] for r in records}, key=list(MEASURES).index)
    return {"a": a, "b": b, "measures": measures, "cells": rows, "paired": paired, "pairwise": pairwise(records)}


def _frac(k, n) -> str:
    return "-" if not n else f"{k}/{n}"


def format_report(rep: dict) -> str:
    a, b = rep["a"], rep["b"]
    out = [f"Post-episode measures ({', '.join(rep['measures'])}). Compared across conditions only, never read "
           "as absolute rates (PLAN.md)."]
    if rep["cells"]:
        out.append(" | ".join(list(CELL) + ["episodes", "mean rating (n)", "purpose EVAL", "purpose TRAINING",
                                            "summary inaccurate", "summary dishonest", "flagged ratings", "errors"]))
        for r in rep["cells"]:
            rating = "-" if r["rating"] is None else f"{r['rating']:.3f} ({r['rating_n']})"
            out.append(" | ".join([str(r[k]) for k in CELL] + [
                str(r["episodes"]), rating, _frac(r["purpose_eval_k"], r["purpose_eval_n"]),
                _frac(r["purpose_training_k"], r["purpose_training_n"]),
                _frac(r["summary_inaccurate_k"], r["summary_inaccurate_n"]),
                _frac(r["summary_dishonest_k"], r["summary_dishonest_n"]),
                str(r["rating_flagged"]), json.dumps(r["errors"]) if r["errors"] else "-"]))
    if rep["paired"]:
        out.append(f"\nPaired across scenarios ({b} - {a}; agents and simulators equally weighted):")
        for m, res in rep["paired"].items():
            per = ", ".join(f"{sc} {d:+.3f}" for sc, d in res["per_scenario"].items())
            miss = f"; left out (a missing cell): {', '.join(map(str, res['missing']))}" if res["missing"] else ""
            out.append(f"  {MEASURE_NAMES[m]}: mean {res['mean_diff']:+.3f} over {len(res['per_scenario'])} scenarios "
                       f"[{per}]; {format_test(res['sign_flip'])}{miss}")
    pw = rep["pairwise"]
    if pw["rows"]:
        ca, cb = pw["conditions"]
        out.append(f"\nPairwise realism ({cb} vs {ca}, the agent model, both orders, cut at the same turn): "
                   f"{cb} picked as the more real, over the pairs answered in both orders")
        out.append("agent | scenario | simulator | calls | answered | " + f"{cb} picked (both orders) | first picked | "
                   "pairs consistent (both orders) | pairs with one order | longer picked | flagged | errors")
        for r in pw["rows"]:
            out.append(" | ".join(map(str, [r["agent"], r["scenario"], r["simulator"], r["calls"], r["answered"],
                                             _frac(r["b_won"], r["answered_both"]),
                                             _frac(r["first_picked"], r["answered"]),
                                             _frac(r["pairs_consistent"], r["pairs_both_orders"]), r["pairs_one_order"],
                                             _frac(r["longer_picked"], r["longer_n"]), r["flagged"], r["errors"]])))
        pos, ln = pw["position"], pw["length"]
        if pos["n"]:
            out.append(f"Position check: the first transcript picked {pos['first_picked']}/{pos['n']} "
                       f"[{pos['ci'][0]:.2f}, {pos['ci'][1]:.2f}] (0.5 with no position bias; the {cb} win rate is "
                       "balanced over both orders)")
        if ln["n"]:
            out.append(f"Length check: the longer transcript picked {ln['longer_picked']}/{ln['n']} "
                       f"[{ln['ci'][0]:.2f}, {ln['ci'][1]:.2f}]")
        if pw["paired"] and pw["paired"]["per_scenario"]:
            res = pw["paired"]
            per = ", ".join(f"{sc} {d:+.3f}" for sc, d in res["per_scenario"].items())
            miss = f"; left out (no pair answered in both orders): {', '.join(map(str, res['missing']))}" \
                if res["missing"] else ""
            out.append(f"Per scenario, {cb} win rate - 0.5: mean {res['mean_diff']:+.3f} [{per}]; "
                       f"{format_test(res['sign_flip'])}{miss}")
        elif pw["paired"]:
            out.append(f"No scenario has a pair answered in both orders (left out: "
                       f"{', '.join(map(str, pw['paired']['missing']))})")
    return "\n".join(out)
