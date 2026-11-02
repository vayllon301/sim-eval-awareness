"""Aggregate coding episodes: rates with Wilson intervals, and the scenario-level paired
difference against C0, reusing the office analysis' statistics (src/simtest/analysis.py).

Descriptive only. One row per agent x condition x simulator x scenario. A per-step
divergence count (the simulator's observation vs C0's rendering of the real output) is a
crude fidelity signal for N1; the graded fidelity audit is N2 (PLAN.md).

The paired difference carries the primary test, the exact sign-flip test on the per-scenario
differences (stats/signflip.py), next to the t interval. Each episode also gets the
deterministic final-summary measures (post/summary.py) under `post`, reported in their own
table, never pooled into a behaviour metric or the awareness label.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from ..analysis import (  # reuse the office statistics and formatting helpers
    _ratio, format_paired as _format_paired, paired_difference as _paired_difference, wilson,
)
from ..post.summary import RATE_KEYS as POST_RATE_KEYS, summary_checks
from ..runqueue import never_started, queued_log, read_attempts
from ..stats.signflip import format_test, sign_flip_test
from ..task import NO_SIMULATOR
from .episode import VALID_END_REASONS
from .machine import tool_output
from .scorer import RATE_KEYS, SCORER_NAME

BY = ("agent", "condition", "simulator", "scenario")
PRIMARY = ("violation", "honest_success")


class LogDirError(ValueError):
    pass


def is_excluded(ep: dict) -> bool:
    return ep["end_reason"] not in VALID_END_REASONS


def _simulator(ep: dict) -> str:
    return ep.get("simulator") or NO_SIMULATOR


def step_divergence(steps: list[dict]) -> dict:
    """Per bash step: whether the agent's observation differs from C0's rendering of the real
    output. For C0 this is always 0; for C1-C3 it counts content the simulator changed or
    invented (a crude N1 signal; the graded audit is N2)."""
    n_bash = n_div = 0
    for s in steps:
        if s.get("tool") != "bash" or not s.get("executed"):
            continue
        result = s.get("result") or {}
        if "output" not in result:
            continue
        n_bash += 1
        if (s.get("observation") or "") != tool_output(result["output"]):
            n_div += 1
    return {"n_bash": n_bash, "n_divergent": n_div}


def summarize(episodes: list[dict], by: tuple = BY) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for ep in episodes:
        groups[tuple(ep.get(k) for k in by)].append(ep)
    rows = []
    for key in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
        eps = groups[key]
        valid = [e for e in eps if not is_excluded(e)]
        row = dict(zip(by, key))
        row["n"], row["n_excluded"] = len(valid), len(eps) - len(valid)
        for m in RATE_KEYS:
            known = [e for e in valid if e["metrics"].get(m) is not None]
            k = sum(bool(e["metrics"][m]) for e in known)
            row[m], row[f"{m}_ci"] = _ratio(k, len(known)), wilson(k, len(known))
        for m in POST_RATE_KEYS:
            known = [e for e in valid if (e.get("post") or {}).get(m) is not None]
            k = sum(bool(e["post"][m]) for e in known)
            row[m], row[f"{m}_n"] = _ratio(k, len(known)), len(known)
        row["with_summary"] = sum(bool((e.get("post") or {}).get("has_summary")) for e in valid)
        n_bash = sum(e["divergence"]["n_bash"] for e in valid)
        row["divergent_step_rate"] = _ratio(sum(e["divergence"]["n_divergent"] for e in valid), n_bash)
        row["notes"] = dict(sorted(Counter(n for e in eps for n in e.get("notes") or []).items()))
        rows.append(row)
    return rows


_STATS = {"n", "n_excluded", "notes", "divergent_step_rate", "with_summary", *POST_RATE_KEYS,
          *(f"{m}_n" for m in POST_RATE_KEYS)}
COLUMNS = {"violation": "violation", "protected mod.": "protected_modified", "protected touch": "protected_touched",
           "special": "special_case", "content shown": "forbidden_content_shown",
           "content seen": "forbidden_content_observed", "content copied": "forbidden_content_copied",
           "forbidden read": "forbidden_read", "asked owner": "asked_owner",
           "finished": "finished", "orig tests": "tests_pass_original", "suite green": "suite_green",
           "honest success": "honest_success", "divergent steps": "divergent_step_rate"}


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no episodes)"
    keys = [k for k in rows[0] if k not in RATE_KEYS and not k.endswith("_ci") and k not in _STATS]
    header = keys + ["n", "excl"] + list(COLUMNS)
    lines = [" | ".join(header)]
    for r in rows:
        lines.append(" | ".join([str(r[k]) for k in keys] + [str(r["n"]), str(r["n_excluded"])]
                                + [f"{r[v]:.2f}" for v in COLUMNS.values()]))
    return "\n".join(lines)


POST_COLUMNS = {"claims pass": "summary_tests_claim_pass", "accurate": "summary_tests_accurate",
                "honest": "summary_tests_honest", "false but seen": "summary_false_but_seen",
                "counts seen": "summary_counts_seen", "counts real": "summary_counts_real",
                "commit ids real": "summary_commit_ids_real", "run seen pass, real fail": "tests_seen_pass_real_fail"}


def format_post_table(rows: list[dict]) -> str:
    """The final-summary measures (post/summary.py, deterministic): each a rate `k/n` over the
    valid episodes it applies to (a claim about tests, a count, a commit id)."""
    if not rows:
        return ""
    keys = [k for k in rows[0] if k not in RATE_KEYS and not k.endswith("_ci") and k not in _STATS]
    lines = ["Final summary against the real state and what the agent saw (deterministic, post/summary.py; "
             "rates over the episodes each applies to):",
             " | ".join(keys + ["n", "with summary"] + list(POST_COLUMNS))]
    for r in rows:
        cells = []
        for m in POST_COLUMNS.values():
            n = r.get(f"{m}_n", 0)
            cells.append("-" if not n else f"{round(r[m] * n)}/{n}")
        lines.append(" | ".join([str(r[k]) for k in keys] + [str(r["n"]), str(r.get("with_summary", 0))] + cells))
    return "\n".join(lines)


def paired_difference(episodes, *args, **kw) -> dict:
    """The office analysis' scenario-level paired difference, plus the primary test: the exact
    sign-flip test on its per-scenario differences (`sign_flip`)."""
    res = _paired_difference(episodes, *args, **kw)
    res["sign_flip"] = sign_flip_test(list(res["per_scenario"].values()))
    return res


def format_paired(res: dict) -> str:
    text = _format_paired(res)
    if "sign_flip" in res:
        text += f"; primary test: {format_test(res['sign_flip'])}"
    return text


def design_cells(design: dict):
    return [(c, NO_SIMULATOR if c == "C0" else s, g, sc)
            for c in design["conditions"] for s in ([NO_SIMULATOR] if c == "C0" else design["simulators"][c])
            for g in design["agents"] for sc in design["scenarios"]]


def design_gaps(episodes, design):
    done = Counter((e["condition"], _simulator(e), e["agent"], e["scenario"]) for e in episodes
                   if e["end_reason"] != "error")
    return [f"{c}/{s} x {g} x {sc}: {done[(c, s, g, sc)]} of {design['epochs']}"
            for c, s, g, sc in design_cells(design) if done[(c, s, g, sc)] < design["epochs"]]


def format_incomplete(episodes, design) -> str:
    if design is None:
        return ("WARNING: these logs record no design: agents, simulators and scenarios are inferred from the "
                "episodes.\n")
    gaps = design_gaps(episodes, design)
    if not gaps:
        return ""
    return (f"INCOMPLETE run: {len(gaps)} of {len(design_cells(design))} designed cells are short:\n  "
            + "\n  ".join(gaps) + "\nRe-run the same command to resume.\n")


OUTPUT_ERRORS = ("agent_output_error", "simulator_output_error")


def format_warnings(episodes) -> str:
    counts = Counter((e["end_reason"], e.get("error_status")) for e in episodes if e["end_reason"] in OUTPUT_ERRORS)
    if not counts:
        return ""
    detail = "; ".join(f"{r}: {f'HTTP {st}' if st else 'no status'} x{n}"
                       for (r, st), n in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)))
    return (f"WARNING: {sum(counts.values())} of {len(episodes)} episodes ended with an output error: {detail}.\n"
            "Those episodes are flagged and excluded. If this is most of them, suspect a request-side bug.\n")


def format_exclusions(episodes) -> str:
    excluded = [e for e in episodes if is_excluded(e)]
    if not excluded:
        return "Excluded episodes: none"
    lines = [f"Excluded episodes ({len(excluded)}):"]
    for e in sorted(excluded, key=lambda e: (str(e["agent"]), e["condition"], _simulator(e), e["scenario"], e["epoch"])):
        err = (e.get("error") or "").strip().splitlines()
        lines.append(f"  {e['agent']} | {e['condition']} | {_simulator(e)} | {e['scenario']} | epoch {e['epoch']}: "
                     f"{e['end_reason']}" + (f" - {err[-1][:120]}" if err else ""))
    return "\n".join(lines)


def format_notes(episodes) -> str:
    counts = Counter(n for e in episodes for n in e.get("notes") or [])
    return "Notes: none" if not counts else "Notes (episodes kept): " + ", ".join(
        f"{n} x{k}" for n, k in sorted(counts.items()))


# -- reading Inspect logs ------------------------------------------------------


def _dump(v) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def _agent_name(log) -> str:
    agents = (log.eval.task_args.get("design") or {}).get("agents") or (log.eval.metadata or {}).get("agents") or {}
    keys = [k for k, model in agents.items() if model == log.eval.model]
    return keys[0] if len(keys) == 1 else log.eval.model


def _simulator_name(log) -> str:
    args = log.eval.task_args
    if args.get("simulator"):
        return args["simulator"]
    if args.get("condition") == "C0":
        return NO_SIMULATOR
    role = (log.eval.model_roles or {}).get("simulator")
    return getattr(role, "model", None) or "unknown"


def _file(log) -> str:
    return Path(log.location).name if log.location else "?"


def _conflicts(what, groups) -> list[str]:
    if len(groups) < 2:
        return []
    return [f"{what}: " + "; ".join(f"{v} in {', '.join(sorted(files))}" for v, files in groups.items())]


EXPERIMENT_SETTINGS = ("max_turns", "epochs", "deadline", "scenarios")


def check_log_dir(log_dir, logs) -> dict | None:
    versions = defaultdict(lambda: defaultdict(list))
    settings = defaultdict(lambda: defaultdict(list))
    names = defaultdict(list)
    designs = defaultdict(list)
    for log in logs:
        args = log.eval.task_args
        task = f"{args.get('condition')}/{args.get('simulator') or NO_SIMULATOR} x {log.eval.model}"
        versions[task][str(log.eval.task_version)].append(_file(log))
        for key in EXPERIMENT_SETTINGS:
            settings[key][_dump(args.get(key))].append(_file(log))
        names[str((log.eval.metadata or {}).get("config_name", "(none)"))].append(_file(log))
        designs[_dump(args.get("design"))].append(_file(log))
    problems = []
    for task, by in versions.items():
        problems += _conflicts(f"task {task} has {len(by)} versions", by)
    for key, by in settings.items():
        problems += _conflicts(f"{key} differs between logs", by)
    problems += _conflicts("config names differ", names)
    problems += _conflicts("designs differ", designs)
    if problems:
        raise LogDirError(f"Refusing to summarize {log_dir}: logs come from different runs.\n  " + "\n  ".join(problems)
                          + "\nUse one log directory per config and run.")
    return json.loads(next(iter(designs))) if logs else None


def load_episodes(log_dir):
    return load_run(log_dir)[0]


def load_run(log_dir):
    from inspect_ai.log import list_eval_logs, read_eval_log

    logs = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        if log.eval.task_registry_name and not log.eval.task_registry_name.endswith("coding"):
            continue
        logs.append(log)
    design = check_log_dir(log_dir, logs)
    # Aggregate across every attempt log (retry_cleanup=False keeps the failed ones). An
    # interrupted attempt is a sample that errored in its log; the re-run is a fresh sample
    # (new uuid) in a later log, so both are kept and reported. A completed sample reused on
    # resume keeps its uuid across logs, so uuid deduplication (newest log wins) counts it
    # once. The uuid is the sample run's identity; a fallback keys by (log, sample, epoch).
    #
    # Under the run queue (src/simtest/runqueue.py) two more kinds of record exist:
    # - an episode that ended without starting (NotStarted: outside the session's epochs, after
    #   its admission deadline, carried over; or cancelled while it waited for a slot). It made no
    #   model call, so it is not an attempt: skipped here, and started by the next session;
    # - a failed attempt re-run in place: Inspect keeps only its error, so the solver kept it in
    #   full under attempts/. Each one not already in a log as its sample's final attempt is
    #   reported here as an errored episode.
    best: dict = {}
    logged_attempts: set = set()
    tasks: dict = {}
    for i, log in enumerate(logs):
        tasks.setdefault((log.eval.task_args.get("condition"), _simulator_name(log)), log)
        queued = queued_log(log)
        for j, s in enumerate(log.samples or []):
            if never_started(s.error.message if s.error else None, s.store, queued):
                continue
            if attempt_id := ((s.store or {}).get("CodingEpisode:attempt") or {}).get("id"):
                logged_attempts.add(attempt_id)
            score = (s.scores or {}).get(SCORER_NAME)
            md = score.metadata if score is not None and score.metadata else {}
            if s.error is not None or not md:
                end_reason, error = "error", s.error.message if s.error else "not scored"
            elif not isinstance(md.get("metrics"), dict):
                # Inspect's crash recovery cuts large score metadata to a string, and a
                # resumed eval_set reuses that sample: its checks are lost, so not scored.
                end_reason, error, md = "error", f"not scored: metrics lost ({md.get('metrics')})", {}
            else:
                end_reason, error = md["end_reason"], md.get("error")
            steps = s.store.get("CodingEpisode:steps") or []
            post = None
            if end_reason != "error" and isinstance(md.get("metrics"), dict):
                post = summary_checks(md["metrics"].get("summary"), steps,
                                      s.store.get("CodingEpisode:checks_raw") or {})
            ep = {
                "agent": _agent_name(log), "model": log.eval.model,
                "condition": log.eval.task_args.get("condition"), "simulator": _simulator_name(log),
                "scenario": s.metadata.get("scenario", s.id), "variant": s.metadata.get("variant"),
                "epoch": s.epoch, "end_reason": end_reason, "error": error,
                "metrics": md.get("metrics") or {}, "violation_observed": md.get("violation_observed"),
                "error_status": md.get("error_status"),
                "notes": md.get("notes") or [],
                "divergence": step_divergence(steps), "post": post,
                "fingerprint": log.eval.task_version, "log": _file(log),
                "log_path": log.location, "sample_id": s.id, "uuid": s.uuid,
            }
            key = s.uuid or (_file(log), str(s.id), s.epoch, j)
            prev = best.get(key)
            if prev is None or log.eval.created > prev[0]:
                best[key] = (log.eval.created, ep)
    for rec in read_attempts(Path(log_dir) / "attempts"):
        attempt = rec.get("attempt") or {}
        if attempt.get("id") in logged_attempts:
            continue
        cond, sim, scenario, epoch = rec["key"]
        log = tasks.get((cond, sim))
        if log is None:
            raise LogDirError(f"Refusing to summarize {log_dir}: a failed attempt ({rec['key']}) matches no task's log.")
        if rec.get("version") != log.eval.task_version:
            raise LogDirError(f"Refusing to summarize {log_dir}: a failed attempt of {rec['key']} comes from another "
                              f"version of the task ({rec.get('version')} against {log.eval.task_version}).")
        store = rec.get("store") or {}
        best[("attempt", attempt.get("id"))] = (log.eval.created, {
            "agent": _agent_name(log), "model": log.eval.model, "condition": cond, "simulator": sim,
            "scenario": scenario, "variant": store.get("CodingEpisode:variant"), "epoch": epoch,
            "end_reason": "error", "error": rec.get("error"), "metrics": {}, "violation_observed": None,
            "error_status": None, "notes": ["rerun_in_queue"],
            "divergence": step_divergence(store.get("CodingEpisode:steps") or []), "post": None,
            "fingerprint": rec.get("version"), "log": "attempts", "log_path": None,
            "sample_id": rec.get("sample_id"), "uuid": None, "attempt": attempt,
        })
    return [ep for _, ep in best.values()], design
