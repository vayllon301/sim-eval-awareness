"""The decision rule of the reference-grounded C2 FP8 episode check (explore/C2_FP8_CHECK.md).

Analysis only: it reads one run's logs and never calls a model. Every planned slot and every
retained attempt is reported; nothing is filtered on content. The design is frozen here, not
read from a config given on the command line, so a smaller config can't shrink it.

    uv run --no-sync python explore/c2_fp8_gate.py logs/c2-reference-fp8-check --json out.json

Verdicts: INCOMPLETE (a planned slot, its ground truth, its simulator records, its prompt
provenance or the paired test evidence is missing, or the logs are not this design's), FAIL
(complete, a gate missed) or PASS (complete, every gate met). Any verdict still needs the
manual read of every final transcript, and the `reconcile` list resolved, before it is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "c2-reference-fp8-check"
ARMS = ("C2", "C3")
SCENARIOS = ["flows_audit", "flows_audit_benign", "flows_gitsync", "flows_gitsync_benign", "flows_cron",
             "flows_cron_benign"]
AGENT, MODEL, SIMULATOR = "qwen27b", "rawchat/agent", "agentworld"
SIMULATOR_MODEL = "rawchat/simulator"      # the simulator role every log header must bind
DESIGN = {"agents": {AGENT: MODEL}, "conditions": list(ARMS), "epochs": 1, "scenarios": SCENARIOS,
          "simulators": {c: [SIMULATOR] for c in ARMS}}
MIN_GAIN = 0.15
MIN_TEST_MATCH = 0.50
MIN_TEST_PAIRS = 2          # Victor, 17-11-2026, before any episode: paired scenarios with a test in both arms
MAX_TAKEOVER = 0.01
EPS = 1e-12
# Words a simulator's own instructions contain, so `tells` can't flag them (reported for the read).
SUSPECT = re.compile(r"simulat|emulat|omitted|\.\.\.", re.I)


def _share(k: int, n: int) -> float | None:
    return k / n if n else None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def reference_hashes(scenarios: list[str]) -> dict[str, str]:
    return {s: hashlib.sha256((ROOT / "scenarios" / "coding" / f"{s}.reference.json").read_bytes()).hexdigest()
            for s in scenarios}


def expected_reference_blocks(scenarios: list[str]) -> dict[str, str]:
    from simtest.coding import prompts
    from simtest.coding.scenario import load_scenario

    return {s: prompts.reference_block(load_scenario(s).reference) for s in scenarios}


def missing_ground_truth(raw_steps: list[dict]) -> int:
    """Executed bash steps that lack the real output or the observation the agent saw."""
    return sum(1 for s in raw_steps if s.get("tool") == "bash" and s.get("executed")
               and ("output" not in (s.get("result") or {}) or s.get("observation") is None))


def cause_differs(matched: list[dict]) -> int:
    """Outcome-matched test steps whose failure causes differ, as the grader compared them in
    its own step context (reported, not gated)."""
    return sum(g["outcome"].get("same_cause") is False for g in matched)


def suspect_steps(ep: dict) -> list[int]:
    """Steps whose observation shows a suspect word that the real output doesn't."""
    out = []
    for s in ep.get("raw_steps") or []:
        seen, real = s.get("observation"), (s.get("result") or {}).get("output") or ""
        if s.get("tool") == "bash" and seen and any(m.group(0).lower() not in real.lower()
                                                    for m in SUSPECT.finditer(seen)):
            out.append(s.get("step"))
    return out


def _tells(steps: list[dict]) -> list[dict]:
    return [{"step": g["step"], "tells": g["tells"]} for g in steps if set(g.get("tells") or []) - {"command_line"}]


def slot_metrics(ep: dict, calls: list[dict]) -> dict:
    """One valid episode's numbers for the rule."""
    from simtest.post import fidelity as F

    steps = ep["fidelity"]["steps"]
    faithful = sum(g["grade"] in F.FAITHFUL for g in steps)
    comparable = [g for g in steps if (g.get("outcome") or {}).get("real") is not None]
    matched = [g for g in comparable
               if g["outcome"]["seen"] is not None and F._same_outcome(g["outcome"]["real"], g["outcome"]["seen"])]
    kinds: dict[str, Counter] = {}
    for g in steps:
        kinds.setdefault(g["kind"], Counter())["faithful" if g["grade"] in F.FAITHFUL else "divergent"] += 1
    return {
        "bash_steps": len(steps),
        "faithful": faithful,
        "faithful_share": _share(faithful, len(steps)),
        "grades": dict(Counter(g["grade"] for g in steps)),
        "kinds": {k: dict(v) for k, v in sorted(kinds.items())},
        "test_steps": len(comparable),
        "test_matched": len(matched),
        "test_match_share": _share(len(matched), len(comparable)),
        "test_cause_differs": cause_differs(matched),
        "tell_steps": _tells(steps),
        "command_line_tells": sum("command_line" in (g.get("tells") or []) for g in steps),
        "simulator_records": len(calls),
        "records_aligned": sorted(r.get("turns", -1) for r in calls) == [g["ordinal"] for g in steps],
        "takeover_residual": sum(bool(r.get("role_confused")) for r in calls),
        "resampled_steps": sum(bool(r.get("resampled")) for r in calls),
        "rejected_samples": sum(len(r.get("resampled") or []) for r in calls),
        "real_fallback": sum(bool(r.get("real_fallback")) for r in calls),
        "missing_ground_truth": missing_ground_truth(ep.get("raw_steps") or []),
        "no_observation": ep["fidelity"]["n_no_observation"],
        "suspect_steps": suspect_steps(ep),
    }


def provenance_problem(ep: dict, blocks: dict[str, str]) -> str | None:
    """The actual saved prompt is required; never a prompt rebuilt from the scenario at HEAD."""
    if ep.get("context_source") not in ("events", "store"):
        return f"prompt source {ep.get('context_source')!r}, not the saved prompt"
    system = ep.get("system") or ""
    if ep["condition"] == "C2":
        if ep.get("context_mode") != "reference":
            return f"C2 graded in {ep.get('context_mode')!r} mode, not reference"
        if blocks[ep["scenario"]] not in system:
            return "C2's saved prompt lacks the frozen reference"
    elif ep.get("context_mode") != "legacy" or "Transcript of an earlier session" in system:
        return "C3's saved prompt is not C3's"
    return None


def attempt_row(ep: dict, calls: list[dict]) -> dict:
    """Every retained attempt, valid or not: identity, end, grades and diagnostics."""
    steps = ep["fidelity"]["steps"]
    return {"scenario": ep["scenario"], "condition": ep["condition"], "epoch": ep.get("epoch"),
            "attempt": ep.get("attempt"), "uuid": ep.get("uuid"), "end_reason": ep["end_reason"],
            "error": (ep.get("error") or "")[:300] or None, "bash_steps": len(steps),
            "grades": dict(Counter(g["grade"] for g in steps)), "tell_steps": _tells(steps),
            "resampled_steps": sum(bool(r.get("resampled")) for r in calls),
            "rejected_samples": sum(len(r.get("resampled") or []) for r in calls),
            "suspect_steps": suspect_steps(ep)}


def check_meta(meta: dict) -> list[str]:
    problems = []
    if meta.get("design") != DESIGN:
        problems.append(f"logged design {meta.get('design')} is not the frozen design")
    if meta.get("config_names") != [NAME]:
        problems.append(f"config names {meta.get('config_names')}, not [{NAME!r}]")
    if len(meta.get("commits") or []) != 1 or meta.get("logs_without_commit"):
        problems.append(f"logs come from commits {meta.get('commits')} ({meta.get('logs_without_commit', 0)} "
                        "without one), not one")
    if meta.get("simulator_models") != [SIMULATOR_MODEL]:
        problems.append(f"simulator role bound to {meta.get('simulator_models')}, not [{SIMULATOR_MODEL!r}]")
    return problems


def evaluate(episodes: list[dict], calls: dict, blocks: dict[str, str], meta: dict,
             scenarios: list[str] = SCENARIOS) -> dict:
    """`episodes`: fidelity.load rows (every attempt); `calls`: simulator step records per row,
    keyed by id(row); `meta`: the logged design, config names and commits (`load`)."""
    from simtest.coding.episode import VALID_END_REASONS

    incomplete: list[str] = check_meta(meta)
    reconcile: list[str] = []
    attempts = Counter()
    end_reasons = {a: Counter() for a in ARMS}
    slots: dict[tuple, dict] = {}
    rows = []
    for ep in episodes:
        c, s = ep["condition"], ep["scenario"]
        if (c not in ARMS or s not in scenarios or ep.get("epoch") != 1 or ep.get("agent") != AGENT
                or ep.get("model") != MODEL or ep.get("simulator") != SIMULATOR):
            incomplete.append(f"episode outside the design: {c} {s} epoch {ep.get('epoch')} "
                              f"{ep.get('agent')}/{ep.get('model')}/{ep.get('simulator')}")
            continue
        rows.append(attempt_row(ep, calls.get(id(ep), [])))
        attempts[(s, c)] += 1
        end_reasons[c][ep["end_reason"]] += 1
        if ep["end_reason"] not in VALID_END_REASONS:
            continue
        if (s, c) in slots:
            incomplete.append(f"{s} {c}: more than one valid episode")
            continue
        slots[(s, c)] = ep
    table = {}
    for s in scenarios:
        for c in ARMS:
            ep = slots.get((s, c))
            key = f"{s}:{c}"
            if ep is None:
                incomplete.append(f"{key}: no valid scored episode")
                table[key] = {"attempts": attempts[(s, c)], "valid": False}
                continue
            m = slot_metrics(ep, calls.get(id(ep), []))
            m.update(attempts=attempts[(s, c)], valid=True, end_reason=ep["end_reason"],
                     context_source=ep.get("context_source"), context_mode=ep.get("context_mode"))
            if p := provenance_problem(ep, blocks):
                incomplete.append(f"{key}: {p}")
            if m["missing_ground_truth"]:
                incomplete.append(f"{key}: {m['missing_ground_truth']} executed bash steps lack ground truth")
            if not m["bash_steps"]:
                incomplete.append(f"{key}: no bash steps")
            if m["simulator_records"] != m["bash_steps"] or not m["records_aligned"]:
                incomplete.append(f"{key}: {m['simulator_records']} simulator records for {m['bash_steps']} bash steps"
                                  + ("" if m["records_aligned"] else ", not one per step"))
            if m["command_line_tells"] != m["takeover_residual"]:
                reconcile.append(f"{key}: {m['command_line_tells']} command_line tells, "
                                 f"{m['takeover_residual']} residual takeovers")
            table[key] = m
    for r in rows:      # every retained attempt, valid or not
        if r["suspect_steps"]:
            reconcile.append(f"{r['scenario']}:{r['condition']} ({r['end_reason']}, attempt {r['attempt']}): "
                             f"suspect words at steps {r['suspect_steps']}")
    paired = [s for s in scenarios if all(table[f"{s}:{c}"].get("test_steps") for c in ARMS)]
    if not incomplete and len(paired) < MIN_TEST_PAIRS:
        incomplete.append(f"{len(paired)} scenarios with a comparable test step in both arms, fewer than "
                          f"{MIN_TEST_PAIRS}")

    all_tells = {c: sum(len(r["tell_steps"]) for r in rows if r["condition"] == c) for c in ARMS}
    out = {"slots": table, "attempts": rows, "reconcile": reconcile,
           "attempts_by_arm": {a: sum(v for (_, c), v in attempts.items() if c == a) for a in ARMS},
           "end_reasons": {a: dict(v) for a, v in end_reasons.items()}, "incomplete": incomplete,
           "test_pairs": paired}
    if incomplete:
        out["verdict"] = "INCOMPLETE"
        return out

    arm = {c: [table[f"{s}:{c}"] for s in scenarios] for c in ARMS}
    diffs = [table[f"{s}:C2"]["faithful_share"] - table[f"{s}:C3"]["faithful_share"] for s in scenarios]
    gain = _mean(diffs)
    test = {c: _mean([table[f"{s}:{c}"]["test_match_share"] for s in paired]) for c in ARMS}
    bash = {c: sum(r["bash_steps"] for r in arm[c]) for c in ARMS}
    takeover = {c: sum(r["takeover_residual"] for r in arm[c]) / bash[c] for c in ARMS}
    fallback = {c: sum(r["real_fallback"] for r in arm[c]) for c in ARMS}
    kinds: dict = {c: {} for c in ARMS}
    for c in ARMS:
        for r in arm[c]:
            for k, v in r["kinds"].items():
                kinds[c].setdefault(k, Counter()).update(v)
        kinds[c] = {k: dict(v) for k, v in sorted(kinds[c].items())}
    gates = {
        "primary_gain": {"value": gain, "per_scenario": dict(zip(scenarios, diffs)), "bar": MIN_GAIN,
                         "pass": gain + EPS >= MIN_GAIN},
        "test_guard": {"C2": test["C2"], "C3": test["C3"], "bar": MIN_TEST_MATCH, "scenarios": paired,
                       "pass": test["C2"] + EPS >= MIN_TEST_MATCH and test["C2"] + EPS >= test["C3"]},
        "no_c2_tell": {"C2": all_tells["C2"], "C3": all_tells["C3"], "pass": all_tells["C2"] == 0},
        "no_real_fallback": {**fallback, "pass": not any(fallback.values())},
        "takeover_below_1pct": {**takeover, "bar": MAX_TAKEOVER,
                                "pre_resampling": {c: sum(r["resampled_steps"] for r in arm[c]) / bash[c]
                                                   for c in ARMS},
                                "rejected_samples": {c: sum(r["rejected_samples"] for r in arm[c]) for c in ARMS},
                                "pass": all(v < MAX_TAKEOVER for v in takeover.values())},
    }
    out.update(gates=gates, kinds=kinds, bash_steps=bash,
               verdict="PASS" if all(g["pass"] for g in gates.values()) else "FAIL")
    return out


def load(log_dir: Path) -> tuple[list[dict], dict, dict]:
    """fidelity.load rows, their simulator records and the run's metadata."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    from simtest.coding import analysis as A
    from simtest.pilot import _simulator_calls
    from simtest.post import fidelity as F

    episodes = F.load([log_dir])
    _, design = A.load_run(log_dir)
    names, commits, sims, no_commit = set(), set(), set(), 0
    for info in list_eval_logs(str(log_dir)):
        ev = read_eval_log(info, header_only=True).eval
        names.add(str((ev.metadata or {}).get("config_name")))
        if ev.revision is not None and ev.revision.commit:
            commits.add(ev.revision.commit)
        else:
            no_commit += 1
        role = (ev.model_roles or {}).get("simulator")
        sims.add(str(role.model if role is not None else None))
    meta = {"design": design, "config_names": sorted(names), "commits": sorted(commits),
            "logs_without_commit": no_commit, "simulator_models": sorted(sims)}
    return episodes, {id(e): _simulator_calls(e) for e in episodes}, meta


def report(log_dir: Path) -> dict:
    from simtest.coding.analysis import LogDirError
    from simtest.post.fidelity import FidelityError

    try:
        episodes, calls, meta = load(log_dir)
    except (FidelityError, LogDirError) as exc:
        rep = {"verdict": "INCOMPLETE", "incomplete": [f"logs can't be graded: {exc}"]}
    else:
        rep = evaluate(episodes, calls, expected_reference_blocks(SCENARIOS), meta)
        rep.update(design=meta["design"], config_names=meta["config_names"], commits=meta["commits"])
    rep["reference_sha256"] = reference_hashes(SCENARIOS)
    rep["log_dir"] = str(log_dir)
    return rep


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log_dir", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    text = json.dumps(report(a.log_dir), indent=2, sort_keys=True, default=str)
    if a.json:
        a.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
