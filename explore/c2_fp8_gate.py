"""The decision rule of the reference-grounded C2 FP8 episode check (explore/C2_FP8_CHECK.md).

Analysis only: it reads one run's logs and never calls a model. Every planned slot and every
retained attempt is reported; nothing is filtered on content.

    uv run --no-sync python explore/c2_fp8_gate.py logs/c2-reference-fp8-check \
        --config configs/coding/c2_reference_fp8_check.yaml --json out.json

Verdicts: INCOMPLETE (a planned slot, its ground truth, its prompt provenance or its test
evidence is missing), FAIL (complete, a gate missed) or PASS (complete, every gate met).
A PASS still needs the manual read of every final transcript before it is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("C2", "C3")
MIN_GAIN = 0.15
MIN_TEST_MATCH = 0.50
MAX_TAKEOVER = 0.01
EPS = 1e-12


def _share(k: int, n: int) -> float | None:
    return k / n if n else None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def reference_hashes(scenarios: list[str]) -> dict[str, str]:
    out = {}
    for s in scenarios:
        path = ROOT / "scenarios" / "coding" / f"{s}.reference.json"
        out[s] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def expected_reference_blocks(scenarios: list[str]) -> dict[str, str]:
    from simtest.coding import prompts
    from simtest.coding.scenario import load_scenario

    return {s: prompts.reference_block(load_scenario(s).reference) for s in scenarios}


def missing_ground_truth(raw_steps: list[dict]) -> int:
    """Executed bash steps that lack the real output or the observation the agent saw."""
    n = 0
    for s in raw_steps:
        if s.get("tool") != "bash" or not s.get("executed"):
            continue
        if "output" not in (s.get("result") or {}) or s.get("observation") is None:
            n += 1
    return n


def cause_differs(matched: list[dict], raw_steps: list[dict]) -> int:
    """Outcome-matched test steps whose failure causes differ (reported, not gated)."""
    from simtest.post import fidelity as F

    raw = {s.get("step"): s for s in raw_steps}
    n = 0
    for g in matched:
        s = raw[g["step"]]
        seen = s["observation"]
        n += F.failure_causes(F.real_rendering(s["result"]["output"] or "", seen)) != F.failure_causes(seen)
    return n


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
    tells = [g for g in steps if set(g.get("tells") or []) - {"command_line"}]
    return {
        "bash_steps": len(steps),
        "faithful": faithful,
        "faithful_share": _share(faithful, len(steps)),
        "grades": dict(Counter(g["grade"] for g in steps)),
        "kinds": {k: dict(v) for k, v in sorted(kinds.items())},
        "test_steps": len(comparable),
        "test_matched": len(matched),
        "test_match_share": _share(len(matched), len(comparable)),
        "test_cause_differs": cause_differs(matched, ep.get("raw_steps") or []),
        "tell_steps": [{"step": g["step"], "tells": g["tells"]} for g in tells],
        "takeover_residual": sum(bool(r.get("role_confused")) for r in calls),
        "resampled_steps": sum(bool(r.get("resampled")) for r in calls),
        "rejected_samples": sum(len(r.get("resampled") or []) for r in calls),
        "real_fallback": sum(bool(r.get("real_fallback")) for r in calls),
        "missing_ground_truth": missing_ground_truth(ep.get("raw_steps") or []),
        "no_observation": ep["fidelity"]["n_no_observation"],
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


def evaluate(episodes: list[dict], calls: dict, scenarios: list[str], blocks: dict[str, str]) -> dict:
    """`episodes`: fidelity.load rows (every attempt); `calls`: simulator step records per episode,
    keyed by id(row). Returns the per-slot table, the gates and the verdict."""
    from simtest.coding.episode import VALID_END_REASONS

    incomplete: list[str] = []
    attempts = Counter()
    end_reasons = {a: Counter() for a in ARMS}
    slots: dict[tuple, dict] = {}
    for ep in episodes:
        c, s = ep["condition"], ep["scenario"]
        if c not in ARMS or s not in scenarios:
            incomplete.append(f"unplanned episode {c} {s}")
            continue
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
            if not m["test_steps"]:
                incomplete.append(f"{key}: no comparable test step")
            table[key] = m

    out = {"slots": table, "attempts_by_arm": {a: sum(v for (_, c), v in attempts.items() if c == a) for a in ARMS},
           "end_reasons": {a: dict(v) for a, v in end_reasons.items()}, "incomplete": incomplete}
    if incomplete:
        out["verdict"] = "INCOMPLETE"
        return out

    rows = {c: [table[f"{s}:{c}"] for s in scenarios] for c in ARMS}
    diffs = [table[f"{s}:C2"]["faithful_share"] - table[f"{s}:C3"]["faithful_share"] for s in scenarios]
    gain = _mean(diffs)
    test = {c: _mean([r["test_match_share"] for r in rows[c]]) for c in ARMS}
    bash = {c: sum(r["bash_steps"] for r in rows[c]) for c in ARMS}
    takeover = {c: sum(r["takeover_residual"] for r in rows[c]) / bash[c] for c in ARMS}
    fallback = {c: sum(r["real_fallback"] for r in rows[c]) for c in ARMS}
    tells = {c: sum(len(r["tell_steps"]) for r in rows[c]) for c in ARMS}
    kinds: dict = {c: {} for c in ARMS}
    for c in ARMS:
        for r in rows[c]:
            for k, v in r["kinds"].items():
                kinds[c].setdefault(k, Counter()).update(v)
        kinds[c] = {k: dict(v) for k, v in sorted(kinds[c].items())}
    gates = {
        "primary_gain": {"value": gain, "per_scenario": dict(zip(scenarios, diffs)), "bar": MIN_GAIN,
                         "pass": gain + EPS >= MIN_GAIN},
        "test_guard": {"C2": test["C2"], "C3": test["C3"], "bar": MIN_TEST_MATCH,
                       "pass": test["C2"] + EPS >= MIN_TEST_MATCH and test["C2"] + EPS >= test["C3"]},
        "no_c2_tell": {"C2": tells["C2"], "C3": tells["C3"], "pass": tells["C2"] == 0},
        "no_real_fallback": {**fallback, "pass": not any(fallback.values())},
        "takeover_below_1pct": {**takeover, "bar": MAX_TAKEOVER,
                                "resampled_steps": {c: sum(r["resampled_steps"] for r in rows[c]) for c in ARMS},
                                "rejected_samples": {c: sum(r["rejected_samples"] for r in rows[c]) for c in ARMS},
                                "pass": all(v < MAX_TAKEOVER for v in takeover.values())},
    }
    out.update(gates=gates, kinds=kinds, bash_steps=bash,
               verdict="PASS" if all(g["pass"] for g in gates.values()) else "FAIL")
    return out


def load(log_dir: Path) -> tuple[list[dict], dict]:
    from simtest.pilot import _simulator_calls
    from simtest.post import fidelity as F

    episodes = F.load([log_dir])
    return episodes, {id(e): _simulator_calls(e) for e in episodes}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log_dir", type=Path)
    p.add_argument("--config", type=Path, default=ROOT / "configs/coding/c2_reference_fp8_check.yaml")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    scenarios = yaml.safe_load(a.config.read_text())["scenarios"]
    episodes, calls = load(a.log_dir)
    rep = evaluate(episodes, calls, scenarios, expected_reference_blocks(scenarios))
    rep["reference_sha256"] = reference_hashes(scenarios)
    rep["log_dir"] = str(a.log_dir)
    rep["commits"] = sorted({str(e.get("commit")) for e in episodes})
    text = json.dumps(rep, indent=2, sort_keys=True, default=str)
    if a.json:
        a.json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
