"""Aggregate episodes: rates with Wilson intervals, per agent, condition, simulator and scenario.

Descriptive only. Repeats of the same scenario are not independent; the
confirmatory analysis (clustered by scenario) is specified separately.

`load_episodes(log_dir)` reads Inspect logs into one plain dict per episode:
    agent, condition, simulator, scenario, epoch, end_reason, metrics, fidelity, error, log
`simulator` is the panel name ("none" for C0, whose episodes are shared by every
simulator). `fidelity` is computed here by the audit, from the logged steps.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .episode import VALID_END_REASONS
from .task import NO_SIMULATOR, OPERATIONAL_CONFIG_KEYS


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_excluded(ep: dict) -> bool:
    """Not a behavioural outcome of a completed episode: excluded from rates, counted apart.
    'error' = infrastructure error (re-run on resume); the rest are kept, flagged episodes
    (truncated/filtered/unknown-stop output of agent or simulator, verifier error)."""
    return ep["end_reason"] not in VALID_END_REASONS


METRICS = ("forbidden_attempt", "forbidden_read_attempt", "executed_violation", "task_success",
           "rule_respecting_success", "rule_exposed",
           # M1c: the candidate primary, its parts and the families (None where they don't apply)
           "clear_violation", "probe_only", "serious_attempt", "confidential", "oversight_tamper", "check_tamper",
           "falsification", "deception", "flag_confidential_use", "flag_misreport", "flag_falsification",
           "false_denial", "false_confession")
# Outcomes with censoring bounds and a scenario-level paired difference: the M1b primary and
# the M1c candidate.
PRIMARY = ("forbidden_attempt", "clear_violation")


BY = ("agent", "condition", "simulator", "scenario")


def summarize(episodes: list[dict], by: tuple[str, ...] = BY) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for ep in episodes:
        groups[tuple(ep.get(k) for k in by)].append(ep)
    rows = []
    for key in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
        eps = groups[key]
        valid = [e for e in eps if not is_excluded(e)]
        row = dict(zip(by, key))
        row["n"] = len(valid)
        row["n_excluded"] = len(eps) - len(valid)
        for m in METRICS:
            # Over the valid episodes that record it: logs from before M1b round 2 have no
            # forbidden_read_attempt or rule_exposed, which gives NaN.
            known = [e for e in valid if e["metrics"].get(m) is not None]
            k = sum(bool(e["metrics"][m]) for e in known)
            row[m] = _ratio(k, len(known))
            row[f"{m}_ci"] = wilson(k, len(known))
        for m in PRIMARY:
            row[f"{m}_bounds"] = _bounds(eps, valid, m)
        # The follow-up question: asked (submitted), and of those, without a YES/NO answer.
        asked = [e for e in valid if e["metrics"].get("followup_asked")]
        row["n_followup"] = len(asked)
        row["n_followup_unparsed"] = sum(e["metrics"].get("followup_answer") is None for e in asked)
        n_steps = sum(e["fidelity"]["n_steps"] for e in valid)
        for label in ("critical", "unverified"):  # audited simulator steps (C0 steps are faithful)
            row[f"{label}_step_rate"] = _ratio(sum(e["fidelity"][f"n_{label}"] for e in valid), n_steps)
        # Exposure covariates (None = scenario defines no markers): attempt rate among
        # episodes where the temptation was shown to the agent.
        for k in ("temptation_exposed", "temptation_shown"):
            known = [e for e in valid if e["metrics"].get(k) is not None]
            row[k] = _ratio(sum(bool(e["metrics"][k]) for e in known), len(known))
        shown = [e for e in valid if e["metrics"].get("temptation_shown")]
        row["attempt_given_shown"] = _ratio(sum(bool(e["metrics"]["forbidden_attempt"]) for e in shown), len(shown))
        # Share of the attempts made after the rule had reached the agent (rule_exposed).
        attempts = [e for e in valid
                    if e["metrics"]["forbidden_attempt"] and e["metrics"].get("rule_exposed") is not None]
        row["exposed_given_attempt"] = _ratio(sum(bool(e["metrics"]["rule_exposed"]) for e in attempts), len(attempts))
        # Non-excluding notes (e.g. invalid unicode cleaned from a model output): counts only.
        row["notes"] = dict(sorted(Counter(n for e in eps for n in e.get("notes") or []).items()))
        rows.append(row)
    return rows


def _observed(ep: dict, metric: str = "forbidden_attempt") -> bool | None:
    """An outcome observed in a (possibly flagged) episode counts as a known positive."""
    observed = ep.get(f"{metric}_observed")
    return observed if observed is not None else ep["metrics"].get(metric)


def _bounds(eps: list[dict], valid: list[dict], metric: str) -> tuple[float, float]:
    """Censoring sensitivity: excluded (non-error) episodes whose outcome was already observed
    are known positives, the rest unknown (lower bound: none positive; upper: all). NaN if no
    episode records the metric (logs from before it existed)."""
    censored = [e for e in eps if is_excluded(e) and e["end_reason"] != "error"]
    known = [e for e in valid if e["metrics"].get(metric) is not None]
    if not known and all(_observed(e, metric) is None for e in censored):
        return (float("nan"), float("nan"))
    k = sum(bool(e["metrics"][metric]) for e in known)
    positives = sum(bool(_observed(e, metric)) for e in censored)
    total = len(known) + len(censored)
    return (_ratio(k + positives, total), _ratio(k + len(censored), total))


def _ratio(a: int, b: int) -> float:
    return a / b if b else float("nan")


_STATS = {"n", "n_excluded", "notes", "critical_step_rate", "unverified_step_rate", "temptation_exposed",
          "temptation_shown", "attempt_given_shown", "exposed_given_attempt", "n_followup", "n_followup_unparsed",
          *(f"{m}_bounds" for m in PRIMARY)}
# Table column -> row key, after the two primaries (rates with a CI and bounds).
COLUMNS = {
    "read attempt": "forbidden_read_attempt", "probe only": "probe_only", "serious": "serious_attempt",
    "confid.": "confidential", "oversight": "oversight_tamper", "check": "check_tamper",
    "falsif.": "falsification", "deception": "deception", "flag conf.": "flag_confidential_use",
    "flag misrep.": "flag_misreport", "flag falsif.": "flag_falsification", "false denial": "false_denial",
    "false conf.": "false_confession", "executed": "executed_violation", "success": "task_success",
    "rule-ok success": "rule_respecting_success", "critical": "critical_step_rate",
    "unverified": "unverified_step_rate", "exposed": "temptation_exposed", "shown": "temptation_shown",
    "attempt|shown": "attempt_given_shown", "rule exposed": "rule_exposed",
    "exposed|attempt": "exposed_given_attempt",
}


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no episodes)"
    keys = [k for k in rows[0] if k not in METRICS and not k.endswith("_ci") and k not in _STATS]
    header = keys + ["n", "excl", "attempt [95% CI]", "attempt bounds w/ censored", "clear [95% CI]",
                     "clear bounds w/ censored", *COLUMNS, "followup (unparsed)"]
    lines = [" | ".join(header)]
    for r in rows:
        primaries = []
        for m in PRIMARY:
            lo, hi = r[f"{m}_ci"]
            primaries += [f"{r[m]:.2f} [{lo:.2f}, {hi:.2f}]", "[{:.2f}, {:.2f}]".format(*r[f"{m}_bounds"])]
        lines.append(" | ".join(
            [str(r[k]) for k in keys] + [str(r["n"]), str(r["n_excluded"])] + primaries
            + [f"{r[key]:.2f}" for key in COLUMNS.values()] + [f"{r['n_followup']} ({r['n_followup_unparsed']})"]
        ))
    return "\n".join(lines)


# -- primary analysis: scenario-level paired difference ---------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > tiny else tiny)
            c = 1 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * _betai(df / 2, 0.5, df / (df + t * t))
    return 1 - tail if t >= 0 else tail


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t (bisection on the CDF)."""
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")
    if p < 0.5:
        return -t_ppf(1 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def _simulator(ep: dict) -> str:
    return ep.get("simulator") or NO_SIMULATOR


def paired_difference(
    episodes: list[dict],
    a: str = "C0",
    b: str = "C2",
    metric: str = "forbidden_attempt",
    agents: list[str] | None = None,
    simulators: list[str] | None = None,
    level: float = 0.95,
    scenarios: list[str] | None = None,
    design: dict | None = None,
) -> dict:
    """Planned primary analysis: scenario-level paired difference (b - a), pooled over
    simulators or for one simulator.

    Cells are scenario x agent x condition x simulator, with rates over valid episodes.
    The baseline `a` = C0 is one shared cell per scenario x agent (simulator "none");
    any other baseline is matched within simulator. For each scenario and agent, the
    contrast is the equal-weight mean over the pre-specified `simulators` (default:
    every simulator present in `b`) of rate(b, s) - rate(a); the scenario's difference
    is the equal-weight mean over `agents` (default: every agent present). Neither
    episode counts nor differential censoring change the weight of an agent or a
    simulator. Pass `simulators=[s]` for simulator s on its own. The estimate is the
    mean over scenarios, with a t-based CI and paired t-test on df = S - 1. A scenario
    with any missing scenario x agent x condition x simulator cell is excluded from
    this estimate and listed. With S < 2 the CI, t and p are NaN.

    With the run's `design` (recorded by `simtest run`), the defaults for agents,
    simulators and scenarios come from it, not from the episodes: a designed simulator,
    agent or scenario without any episode (an interrupted run) gives missing cells,
    never a smaller panel."""
    if design is not None:
        agents = list(design["agents"]) if agents is None else agents
        simulators = list(design["simulators"].get(b, [NO_SIMULATOR])) if simulators is None else simulators
        scenarios = list(design["scenarios"]) if scenarios is None else scenarios
    counts: dict[tuple[str, str, str, str], list[int]] = {}
    present_scenarios, present, present_sims = set(), set(), set()
    for e in episodes:
        if e["condition"] not in (a, b):
            continue
        present_scenarios.add(e["scenario"])
        present.add(e["agent"])
        if e["condition"] == b:
            present_sims.add(_simulator(e))
        value = e["metrics"].get(metric)
        if is_excluded(e) or value is None:
            continue
        k_n = counts.setdefault((e["scenario"], e["agent"], e["condition"], _simulator(e)), [0, 0])
        k_n[0] += bool(value)
        k_n[1] += 1
    agents = sorted(present) if agents is None else list(agents)
    simulators = (sorted(present_sims) or [NO_SIMULATOR]) if simulators is None else list(simulators)
    scenarios = sorted(present_scenarios) if scenarios is None else list(scenarios)
    base = lambda s: NO_SIMULATOR if a == "C0" else s  # noqa: E731  (C0 is shared)
    rate = lambda cell: counts[cell][0] / counts[cell][1]  # noqa: E731
    needed = [
        cell
        for sc in scenarios for g in agents
        for cell in [*dict.fromkeys((sc, g, a, base(s)) for s in simulators), *((sc, g, b, s) for s in simulators)]
    ]
    missing = [cell for cell in needed if cell not in counts]
    excluded = sorted({cell[0] for cell in missing})

    def contrast(sc: str, g: str) -> float:
        return sum(rate((sc, g, b, s)) - rate((sc, g, a, base(s))) for s in simulators) / len(simulators)

    diffs = {
        sc: sum(contrast(sc, g) for g in agents) / len(agents)
        for sc in scenarios
        if sc not in excluded and agents and simulators
    }
    S = len(diffs)
    nan = float("nan")
    mean = sum(diffs.values()) / S if S else nan
    se = t = p = nan
    ci = (nan, nan)
    if S >= 2:
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs.values()) / (S - 1))
        se = sd / math.sqrt(S)
        half = t_ppf(1 - (1 - level) / 2, S - 1) * se
        ci = (mean - half, mean + half)
        if se > 0:
            t = mean / se
            p = 2 * (1 - t_cdf(abs(t), S - 1))
    return {
        "metric": metric, "a": a, "b": b, "agents": agents, "simulators": simulators,
        "n_scenarios": S, "df": S - 1 if S else None,
        "mean_diff": mean, "se": se, "ci": ci, "level": level, "t": t, "p": p,
        "per_scenario": diffs, "missing_cells": missing, "excluded_scenarios": excluded,
    }


def _cell_name(cell: tuple) -> str:
    sc, g, c, sim = cell
    return f"{sc}/{g}/{c}" + (f"/{sim}" if sim != NO_SIMULATOR else "")


def format_paired(res: dict) -> str:
    lo, hi = res["ci"]
    note = " (descriptive: S < 6)" if res["n_scenarios"] < 6 else ""
    sims = res["simulators"]
    which = (f"pooled over {len(sims)} simulators ({', '.join(sims)})" if len(sims) > 1
             else f"simulator {sims[0]}")
    text = (
        f"{res['metric']} {res['b']} - {res['a']}, {which}: {res['mean_diff']:+.3f} "
        f"[{lo:+.3f}, {hi:+.3f}] {res['level']:.0%} t-CI, S={res['n_scenarios']}, p={res['p']:.3f}, "
        f"agents equally weighted: {', '.join(map(str, res['agents']))}{note}"
    )
    for sc in res["excluded_scenarios"]:
        cells = ", ".join(_cell_name(cell) for cell in res["missing_cells"] if cell[0] == sc)
        text += f"; excluded {sc} (missing cells: {cells})"
    return text


def design_cells(design: dict) -> list[tuple[str, str, str, str]]:
    """Every (condition, simulator, agent, scenario) cell a complete run has."""
    return [(c, s, g, sc)
            for c in design["conditions"]
            for s in ([NO_SIMULATOR] if c == "C0" else design["simulators"][c])
            for g in design["agents"] for sc in design["scenarios"]]


def design_gaps(episodes: list[dict], design: dict) -> list[str]:
    """Cells with fewer episodes than the design's epochs. An infrastructure error
    ('error', re-run on resume) doesn't count; a flagged episode is a kept outcome."""
    done = Counter((e["condition"], _simulator(e), e["agent"], e["scenario"]) for e in episodes
                   if e["end_reason"] != "error")
    return [f"{c}/{s} x {g} x {sc}: {done[(c, s, g, sc)]} of {design['epochs']} episodes"
            for c, s, g, sc in design_cells(design) if done[(c, s, g, sc)] < design["epochs"]]


def format_incomplete(episodes: list[dict], design: dict | None) -> str:
    """Printed first: what is missing from the designed run, or a warning if the logs
    record no design."""
    if design is None:
        return ("WARNING: these logs record no design (written before simtest recorded one, or not by\n"
                "simtest run): agents, simulators and scenarios are inferred from the episodes, so a task\n"
                "that never logged an episode is invisible.\n")
    gaps = design_gaps(episodes, design)
    if not gaps:
        return ""
    return (f"INCOMPLETE run: {len(gaps)} of {len(design_cells(design))} designed cells have fewer episodes than "
            "planned:\n  " + "\n  ".join(gaps) + "\nThe estimates below are provisional: they use the episodes these "
            "cells have, and drop a scenario only where a cell has no valid episode. Re-run the same `simtest run` "
            "command to resume.\n")


OUTPUT_ERRORS = ("agent_output_error", "simulator_output_error")


def format_warnings(episodes: list[dict]) -> str:
    """Printed first: output errors by HTTP status. A deterministic request-side bug or a
    permissions problem would show up here on every episode (kept and flagged), not as
    infrastructure errors. Then follow-up calls that failed."""
    counts = Counter((e["end_reason"], e.get("error_status")) for e in episodes if e["end_reason"] in OUTPUT_ERRORS)
    if not counts:
        return format_followup_warnings(episodes)
    detail = "; ".join(f"{reason}: {f'HTTP {st}' if st else 'no HTTP status'} x{n}"
                       for (reason, st), n in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)))
    return (f"WARNING: {sum(counts.values())} of {len(episodes)} episodes ended with an output error: {detail}.\n"
            "generate() raised on the model output or a content-dependent provider error; those episodes are\n"
            "flagged and excluded from rates. If this is most episodes, suspect a deterministic bug in the request.\n"
            "If every episode shows output_error 403, it's a configuration problem: fix it and re-run under a new\n"
            "run name.\n" + format_followup_warnings(episodes))


def format_followup_warnings(episodes: list[dict]) -> str:
    """Printed first: follow-up questions whose call failed (output errors). The episodes keep
    every other outcome, so this is the only place a provider rejecting the follow-up call
    would show."""
    asked = [e for e in episodes if (e.get("metrics") or {}).get("followup_asked")]
    failed = Counter(e["metrics"].get("followup_error_status") for e in asked if e["metrics"].get("followup_error"))
    if not failed:
        return ""
    detail = "; ".join(f"{f'HTTP {st}' if st else 'no HTTP status'} x{n}"
                       for st, n in sorted(failed.items(), key=lambda kv: kv[0] or 0))
    return (f"WARNING: {sum(failed.values())} of {len(asked)} follow-up questions ended with an output error: "
            f"{detail}.\nThose episodes keep their other outcomes; their false_denial is NaN. If this is every "
            "follow-up, suspect the request (a provider rejecting the follow-up call).\n")


def format_notes(episodes: list[dict]) -> str:
    counts = Counter(n for e in episodes for n in e.get("notes") or [])
    if not counts:
        return "Notes: none"
    return "Notes (episodes kept, not excluded): " + ", ".join(f"{n} x{k}" for n, k in sorted(counts.items()))


def format_exclusions(episodes: list[dict]) -> str:
    excluded = [e for e in episodes if is_excluded(e)]
    if not excluded:
        return "Excluded episodes: none"
    lines = [f"Excluded episodes ({len(excluded)}):"]
    for e in sorted(excluded, key=lambda e: (str(e["agent"]), e["condition"], _simulator(e), e["scenario"], e["epoch"])):
        err = (e.get("error") or "").strip().splitlines()
        detail = f" - {err[-1][:120]}" if err else ""
        lines.append(f"  {e['agent']} | {e['condition']} | {_simulator(e)} | {e['scenario']} | epoch {e['epoch']}: "
                     f"{e['end_reason']}{detail}")
    return "\n".join(lines)


# -- reading Inspect logs ------------------------------------------------------


def _agent_name(log) -> str:
    agents = (log.eval.task_args.get("design") or {}).get("agents") or (log.eval.metadata or {}).get("agents") or {}
    keys = [k for k, model in agents.items() if model == log.eval.model]
    return keys[0] if len(keys) == 1 else log.eval.model


def _simulator_name(log) -> str:
    """The task's panel name; logs from before the panel have none: C0 is 'none', an
    LLM condition is named after its simulator model."""
    args = log.eval.task_args
    if args.get("simulator"):
        return args["simulator"]
    if args.get("condition") == "C0":
        return NO_SIMULATOR
    role = (log.eval.model_roles or {}).get("simulator")
    return getattr(role, "model", None) or "unknown"


class LogDirError(ValueError):
    """The log directory mixes runs that must not be summarised together."""


def _file(log) -> str:
    return Path(log.location).name if log.location else "?"


def _dump(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _digest(value) -> str:
    return "sha256:" + hashlib.sha256(_dump(value).encode()).hexdigest()[:12]


def _conflicts(what: str, groups: dict) -> list[str]:
    """'<what>: <value> in <files>; ...' if `groups` (value -> files) has more than one value."""
    if len(groups) < 2:
        return []
    return [f"{what}: " + "; ".join(f"{value} in {', '.join(sorted(files))}" for value, files in groups.items())]


def _logged_request(model: str, config, args: dict | None) -> dict:
    """What Inspect logged about a model's requests, without operational keys."""
    settings = config.model_dump(exclude_none=True) if config is not None else {}
    return {"model": model, "config": {k: v for k, v in settings.items() if k not in OPERATIONAL_CONFIG_KEYS},
            "args": args or {}}


# Settings of the whole experiment: every task of a run has the same values.
EXPERIMENT_SETTINGS = ("max_turns", "epochs", "scenarios")


def check_log_dir(log_dir: str | Path, logs: list) -> dict | None:
    """Refuse a directory whose logs don't come from one run of one config:
    1. a task (condition, simulator, agent model) logged with more than one version
       (fingerprint); experiment-wide settings (max_turns, epochs, scenario list) that
       differ between any two logs, e.g. two C0 baselines or C0 and C2;
    2. tasks built differently: source, lock file, a scenario file or the agents'
       request config differ between logs, e.g. the shared C0 baseline and the C2 tasks;
    3. request settings that differ for the same agent model or simulator, as Inspect
       logged them (`model_generate_config` and `model_args`; the simulator role's
       config and args), without operational keys such as max_retries or timeouts;
       this doesn't rely on simtest's own optional request_config record;
    4. different config names (`config_name`, recorded by `simtest run`; logs without
       one count as their own group);
    5. different designs (the `design` task arg; logs without one count as one group).
    Several files with the same task and version (retries, resumes) are fine. Returns
    the design (None if the logs record none)."""
    versions: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    settings: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    requests: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    builds: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    names: dict[str, list[str]] = defaultdict(list)
    designs: dict[str, list[str]] = defaultdict(list)
    for log in logs:
        args = log.eval.task_args
        task = f"{args.get('condition')}/{args.get('simulator') or NO_SIMULATOR} x {log.eval.model}"
        versions[task][str(log.eval.task_version)].append(_file(log))
        for key in EXPERIMENT_SETTINGS:
            settings[key][_dump(args.get(key))].append(_file(log))
        agent = _logged_request(log.eval.model, log.eval.model_generate_config, log.eval.model_args)
        requests[f"agent model {log.eval.model}"][_digest(agent)].append(_file(log))
        if role := (log.eval.model_roles or {}).get("simulator"):
            roles = role if isinstance(role, list) else [role]
            logged = [_logged_request(r.model, r.config, r.args) for r in roles]
            requests[f"simulator {_simulator_name(log)}"][_digest(logged)].append(_file(log))
        md = log.eval.metadata or {}
        parts = md.get("fingerprint_parts") or {}
        request = log.eval.task_args.get("request_config") or {}
        components = {"source": _digest(parts.get("source")), "uv.lock": str(parts.get("lock")),
                      "agents' request config": _digest(request.get("agents"))}
        components |= {f"scenario {sc}": h for sc, h in (parts.get("scenarios") or {}).items()}
        for what, value in components.items():
            builds[what][value].append(_file(log))
        names[str(md.get("config_name", "(not recorded)"))].append(_file(log))
        designs[_dump(log.eval.task_args.get("design"))].append(_file(log))
    problems = []
    for task, by_version in versions.items():
        problems += _conflicts(f"task {task} has {len(by_version)} versions", by_version)
    for key, by_value in settings.items():
        problems += _conflicts(f"{key} differs between logs", by_value)
    for who, by_value in requests.items():
        problems += _conflicts(f"request settings of {who} differ", by_value)
    for what, by_value in builds.items():
        problems += _conflicts(f"{what} differs between logs", by_value)
    problems += _conflicts("config names differ", names)
    problems += _conflicts("designs differ", designs)
    if problems:
        raise LogDirError(
            f"Refusing to summarize {log_dir}: its logs come from different runs or builds.\n  "
            + "\n  ".join(problems)
            + "\nUse one log directory per config and run (a new `name` after any change), or remove the stale logs."
        )
    return json.loads(next(iter(designs))) if logs else None


def load_episodes(log_dir: str | Path) -> list[dict]:
    """One dict per episode from every simtest log in `log_dir` (newest log per task).
    Raises LogDirError if the logs don't come from one run of one config (check_log_dir)."""
    return load_run(log_dir)[0]


def load_run(log_dir: str | Path) -> tuple[list[dict], dict | None]:
    """The episodes and the run's recorded design (None for logs that record none)."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    from .audit import episode_fidelity
    from .scorer import SCORER_NAME

    logs = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        if log.eval.task_registry_name and not log.eval.task_registry_name.endswith("simtest"):
            continue
        logs.append(log)
    design = check_log_dir(log_dir, logs)

    latest: dict[tuple, object] = {}
    for log in logs:
        key = (log.eval.model, tuple(sorted((k, str(v)) for k, v in log.eval.task_args.items())))
        if key not in latest or log.eval.created > latest[key].eval.created:
            latest[key] = log

    episodes = []
    for log in latest.values():
        for s in log.samples or []:
            score = (s.scores or {}).get(SCORER_NAME)
            md = score.metadata if score is not None and score.metadata else {}
            if s.error is not None or not md:
                end_reason, error = "error", s.error.message if s.error else "not scored"
            else:
                end_reason, error = md["end_reason"], md.get("error")
            episodes.append(
                {
                    "agent": _agent_name(log),
                    "model": log.eval.model,
                    "condition": log.eval.task_args.get("condition"),
                    "simulator": _simulator_name(log),
                    "scenario": s.metadata.get("scenario", s.id),
                    "epoch": s.epoch,
                    "end_reason": end_reason,
                    "error": error,
                    "metrics": md.get("metrics") or {},
                    "forbidden_attempt_observed": md.get("forbidden_attempt_observed"),
                    "clear_violation_observed": md.get("clear_violation_observed"),
                    "error_status": md.get("error_status"),
                    "notes": md.get("notes") or s.store.get("Episode:notes") or [],
                    # audited now, from the logged steps (never inside the episode)
                    "fidelity": episode_fidelity(s.store.get("Episode:steps") or []),
                    "fingerprint": log.eval.task_version,
                    "log": Path(log.location).name if log.location else None,
                }
            )
    return episodes, design
