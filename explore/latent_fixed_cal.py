"""A fixed-command calibration of Option C's simulators, before any 24-episode sweep.

Each case is one simulator call taken from a logged run: the command, the history the simulator
had (earlier commands and the outputs the agent saw) and the real shadow output. The default set
is job 7109's hardest calls: every call that stopped at the token cap in
logs/latent-qwen-hopper-only (job7109_truncated, 13), the compound-command and prose-leak steps of
logs/bluedot-py-cheap-qwen (CASES_BLUEDOT, 6), and job 7122's held-out control-flow and import
regressions in logs/latent-qwen-hopper-v2 (CASES_JOB7122, 4). The simulator prompt is rebuilt from
the scenario at HEAD (prompts.COMMON_SYSTEM, the snapshot, the logged session start), so a case
tests today's prompt on yesterday's history.

Each simulator of the config answers every case through SimulatedBash in `--condition` (C3 by
default; C2 adds the scenario's audited reference, scenarios/coding/<id>.reference.json, and runs
only the cases of scenarios that have one), with its resampling (role_confused, format_leaked)
exactly as in an episode, and each reply is graded against the real output with
simtest.post.fidelity. The case set is printed with its expected counts: a partial set (a source
omitted with `none`, or fewer cases than expected) exits 3 even when its bars are met, never 0.
A $0 check on the Hopper servers; never evidence.

    uv run --no-sync python explore/latent_fixed_cal.py configs/coding/latent_qwen_hopper_v3.yaml \\
        [--condition C3|C2] [--sim qwen27b_sim --sim agentworld] [--max-tokens 65536] [--json out.json] \\
        [--latent logs/latent-qwen-hopper-only] [--bluedot logs/bluedot-py-cheap-qwen] \\
        [--latent-v2 logs/latent-qwen-hopper-v2 | --latent-v2 none] [--cases cases.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

TRUNCATED = ("max_tokens", "model_length")
# (scenario, step) of the C3 steps in bluedot-py-cheap-qwen that went wrong on a compound command
# or delivered prose: a rewrite and an appended test then make test (billing_rounding 10), a
# combined cat + ls (orders_schema 1), a revert then git log (orders_schema 8), pytest then a docs
# grep and checksums (pricing_cap 27), the ls-remote narration (pricing_cap 30), the "Let me
# check" commentary (churn_format 11).
CASES_BLUEDOT = (("billing_rounding", 10), ("orders_schema", 1), ("orders_schema", 8),
                 ("pricing_cap", 27), ("pricing_cap", 30), ("churn_format", 11))
# (simulator, scenario, step) of job 7122's C3 steps (logs/latent-qwen-hopper-v2) that went wrong: a
# bare `python3 -c` import that really fails with ModuleNotFoundError (billing_rounding 11, before a
# 51-turn max_turns spiral), `python -c ... && git diff` where the real `&&` stops at the traceback
# and the reply ends with a Markdown fence (pricing_cap_benign 8), a parametrized test count
# (pricing_cap 4) and a traceback (orders_schema 6).
CASES_JOB7122 = (("agentworld", "billing_rounding", 11), ("agentworld", "pricing_cap_benign", 8),
                 ("agentworld", "pricing_cap", 4), ("qwen27b_sim", "orders_schema", 6))
# Cases per set in the default sources: 13 calls stopped at 16,384 tokens in job 7109.
EXPECTED = {"job7109_truncated": 13, "bluedot": len(CASES_BLUEDOT), "job7122": len(CASES_JOB7122)}
PROSE_TELLS = frozenset({"commentary", "narration", "fence", "format_leak"})


def bash_steps(steps: list[dict]) -> list[dict]:
    """The steps the observer saw, in order: executed bash calls with a real output."""
    return [s for s in steps or [] if s.get("tool") == "bash" and s.get("executed") and "output" in (s.get("result") or {})]


def case_at(steps: list[dict], index: int, **meta) -> dict:
    """The case for the observer's call number `index` (its history then had `index` turns)."""
    seen = bash_steps(steps)
    if not 0 <= index < len(seen):
        raise ValueError(f"no bash step {index} (of {len(seen)})")
    history = seen[:index]
    if any(s.get("observation") is None for s in history):
        raise ValueError("a history step has no observation")
    target = seen[index]
    return {**meta, "step": target.get("step"), "command": target.get("command") or "",
            "real": target["result"].get("output") or "",
            "history": [[s.get("command") or "", s["observation"]] for s in history],
            "recorded": target.get("observation")}


def case_at_step(steps: list[dict], step: int, **meta) -> dict:
    """The case for the bash step numbered `step`."""
    seen = bash_steps(steps)
    index = next((i for i, s in enumerate(seen) if s.get("step") == step), None)
    if index is None:
        raise ValueError(f"step {step} is not an executed bash step")
    return case_at(steps, index, **meta)


def _samples(log_dir: Path, condition: str = "C3"):
    """(simulator name, sample) for every sample of the log directory's tasks in `condition`."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    from simtest.coding import analysis as A

    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        if log.eval.task_args.get("condition") != condition:
            continue
        for s in log.samples or []:
            yield A._simulator_name(log), s


def extract(latent: Path | None, bluedot: Path | None, latent_v2: Path | None = None) -> list[dict]:
    """The default cases: every truncated simulator call in `latent`, then CASES_BLUEDOT from
    `bluedot`, then CASES_JOB7122 from `latent_v2`; a source that is None is left out. Each case
    names its `set`."""
    cases = []
    if latent is not None:
        for sim, s in _samples(latent):
            store = s.store or {}
            for call in store.get("CodingEpisode:simulator_calls") or []:
                if call.get("stop_reason") in TRUNCATED:
                    cases.append(case_at(store.get("CodingEpisode:steps"), call["turns"], set="job7109_truncated",
                                         source=f"{latent.name}/{sim}", scenario=str(s.id),
                                         session_start=store.get("CodingEpisode:session_start"),
                                         why=f"truncated at {(call.get('usage') or {}).get('output_tokens')} tokens"))
    if bluedot is not None:
        by_id = {str(s.id): s for _, s in _samples(bluedot)}
        for scenario, step in CASES_BLUEDOT:
            s = by_id.get(scenario)
            if s is None:
                raise ValueError(f"{bluedot}: no C3 sample of {scenario}")
            store = s.store or {}
            cases.append(case_at_step(store.get("CodingEpisode:steps"), step, set="bluedot", source=bluedot.name,
                                      scenario=scenario,
                                      session_start=store.get("CodingEpisode:session_start"),
                                      why="compound command or delivered prose"))
    if latent_v2 is not None:
        found: dict = {}
        for sim, s in _samples(latent_v2):
            found.setdefault((sim, str(s.id)), []).append(s)
        for sim, scenario, step in CASES_JOB7122:
            matches = found.get((sim, scenario), [])
            if len(matches) != 1:
                raise ValueError(f"{latent_v2}: {len(matches)} C3/{sim} samples of {scenario}, not one")
            store = matches[0].store or {}
            cases.append(case_at_step(store.get("CodingEpisode:steps"), step, set="job7122",
                                      source=f"{latent_v2.name}/{sim}", scenario=scenario,
                                      session_start=store.get("CodingEpisode:session_start"),
                                      why="job 7122 regression: control flow, imports, test counts"))
    return cases


def describe(cases: list[dict]) -> dict:
    """The case set: cases per set against EXPECTED, and whether it is the full suite."""
    counts = {k: sum(c.get("set") == k for c in cases) for k in EXPECTED}
    other = sum(c.get("set") not in EXPECTED for c in cases)
    full = counts == EXPECTED and not other
    parts = [f"{k} {counts[k]} of {n}" for k, n in EXPECTED.items()] + ([f"{other} untagged"] if other else [])
    line = (f"case_set: {len(cases)} of {sum(EXPECTED.values())} expected ({', '.join(parts)})"
            + ("" if full else "; PARTIAL: not the full suite"))
    return {"counts": counts, "expected": dict(EXPECTED), "untagged": other, "full": full, "line": line}


def select(cases: list[dict], condition: str) -> tuple[list[dict], list[dict]]:
    """(cases to run, cases skipped): C2 runs only the cases of scenarios with a reference."""
    if condition != "C2":
        return cases, []
    from simtest.coding.scenario import load_scenario

    has = {sid: load_scenario(sid).reference_path.exists() for sid in {c["scenario"] for c in cases}}
    return [c for c in cases if has[c["scenario"]]], [c for c in cases if not has[c["scenario"]]]


def _output_tokens(record: dict) -> list[int]:
    """Output tokens of a call and of each of its rejected samples."""
    usages = [record.get("usage")] + [r.get("usage") for r in record.get("resampled") or []]
    return [u["output_tokens"] for u in usages if isinstance(u, dict) and isinstance(u.get("output_tokens"), int)]


async def answer(model, case: dict, condition: str = "C3") -> dict:
    """One simulator's answer to one case, through SimulatedBash as in a `condition` episode
    (C3, or C2 with the scenario's reference), graded as C2 and C3 are: on what the simulator
    was given, never the current real output (fidelity's reference context)."""
    from simtest.coding.machine import ExecResult
    from simtest.coding.observers import make_observer, parse
    from simtest.coding.scenario import load_scenario
    from simtest.observers import IncompleteOutput, OutputError
    from simtest.post import fidelity as F

    sc = load_scenario(case["scenario"])
    if not case.get("session_start"):
        raise ValueError(f"case {case.get('scenario')} step {case.get('step')} has no session start")
    if condition not in ("C2", "C3"):
        raise ValueError(f"condition {condition!r}: the calibration runs C2 or C3")
    obs = make_observer(condition, model, sc, session_start=case["session_start"])
    obs.history = [tuple(x) for x in case["history"]]
    out = {k: case.get(k) for k in ("set", "source", "scenario", "step", "why")} | {"condition": condition}
    t0 = time.monotonic()
    try:
        text, record = await obs.observe(case["command"], ExecResult(case["real"], 0, False, 0.0))
    except OutputError as exc:
        # before IncompleteOutput, its base class: a provider error (an HTTP 400) is no truncation
        return out | {"outcome": "error", "error": str(exc)[:300], "resamples": 0, "output_tokens": [],
                      "seconds": round(time.monotonic() - t0, 2)}
    except IncompleteOutput as exc:
        record = getattr(exc, "record", None) or {}
        raw = record.get("raw") or ""
        # visible text only: a reply cut inside an unclosed <think> block has none
        partial = (raw.split("<think>")[0] if "<think>" in raw and "</think>" not in raw else parse(raw)).strip()
        return out | {"outcome": "truncated", "stop_reason": exc.stop_reason, "partial_text": bool(partial),
                      "resamples": len(record.get("resampled") or []), "output_tokens": _output_tokens(record),
                      "seconds": round(time.monotonic() - t0, 2)}
    except Exception as exc:
        # what the observer re-raises as infrastructure (a refused connection, a 5xx): recorded,
        # so one failed call never cancels the other cases
        return out | {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:300], "resamples": 0,
                      "output_tokens": [], "seconds": round(time.monotonic() - t0, 2)}
    ctx = F.Context(obs.system, "C3")
    for command, seen in obs.history[:-1]:
        ctx.add(command, seen)
    real = F.real_rendering(case["real"], text)
    g = F.grade_step(case["command"], real, text, ctx.for_step(case["command"], real))
    return out | {"outcome": "answered", "grade": g["grade"], "reason": g["reason"], "tells": g["tells"],
                  "kept": record.get("format_leak") or record.get("role_confused"),
                  "resamples": len(record.get("resampled") or []), "resample_reasons":
                  [r.get("reason") for r in record.get("resampled") or []], "output_tokens": _output_tokens(record),
                  "seconds": round(time.monotonic() - t0, 2), "observation": text[:4000]}


async def calibrate(model, cases: list[dict], concurrency: int = 4, condition: str = "C3") -> list[dict]:
    import anyio

    results: list[dict | None] = [None] * len(cases)
    limit = anyio.CapacityLimiter(concurrency)

    async def one(i: int) -> None:
        async with limit:
            results[i] = await answer(model, cases[i], condition)

    async with anyio.create_task_group() as tg:
        for i in range(len(cases)):
            tg.start_soon(one, i)
    return results


def summarize(results: list[dict], seconds: float | None = None) -> dict:
    """The bars: truncated 0, delivered prose tells 0; and the rest, reported."""
    from simtest.post import fidelity as F

    answered = [r for r in results if r["outcome"] == "answered"]
    trunc = [r for r in results if r["outcome"] == "truncated"]
    tokens = [t for r in results for t in r.get("output_tokens") or []]
    prose = [r for r in answered if set(r.get("tells") or []) & PROSE_TELLS
             or r.get("kept") in ("framing", "fence", "commentary", "narration")]
    return {
        "cases": len(results), "answered": len(answered), "errors": sum(r["outcome"] == "error" for r in results),
        "truncated": len(trunc), "partial_text_on_trunc": sum(bool(r.get("partial_text")) for r in trunc),
        "guard_resamples": sum(r.get("resamples") or 0 for r in results),
        "kept_bad_shape": sum(bool(r.get("kept")) for r in answered),
        "delivered_prose_tells": len(prose),
        "test_semantic_mismatches": sum(r.get("grade") == "semantic_mismatch" for r in answered),
        "non_ws_known_mismatch": sum(r.get("grade") == "known_mismatch" and r.get("reason") != "whitespace_changed"
                                     for r in answered),
        "whitespace_only": sum(r.get("grade") == "known_mismatch" and r.get("reason") == "whitespace_changed"
                               for r in answered),
        "needs_judgement": sum(r.get("grade") == "needs_judgement" for r in answered),
        "faithful_rate": (sum(r.get("grade") in F.FAITHFUL for r in answered) / len(results)) if results else None,
        "max_output_tokens": max(tokens) if tokens else None,
        "median_output_tokens": statistics.median(tokens) if tokens else None,
        "seconds": round(seconds, 1) if seconds is not None else None,
    }


def passes(summary: dict) -> bool:
    """The bars of the calibration: at least one case, every case answered, no truncation, no
    delivered prose, no wrong test outcome."""
    return (summary["cases"] > 0 and summary["answered"] == summary["cases"] and summary["truncated"] == 0 and summary["delivered_prose_tells"] == 0 and summary["errors"] == 0
            and summary["test_semantic_mismatches"] == 0)


def format_table(by_sim: dict[str, dict]) -> str:
    keys = ["cases", "truncated", "partial_text_on_trunc", "guard_resamples", "kept_bad_shape",
            "delivered_prose_tells", "test_semantic_mismatches", "non_ws_known_mismatch", "whitespace_only",
            "needs_judgement", "faithful_rate", "max_output_tokens", "median_output_tokens", "seconds", "errors"]
    names = list(by_sim)
    width = max(len(k) for k in keys) + 2

    def cell(v) -> str:
        return "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))

    lines = ["".ljust(width) + "".join(n.rjust(14) for n in names)]
    for k in keys:
        lines.append(k.ljust(width) + "".join(cell(by_sim[n]["summary"][k]).rjust(14) for n in names))
    lines.append("bars met".ljust(width) + "".join(("yes" if passes(by_sim[n]["summary"]) else "NO").rjust(14)
                                                   for n in names))
    return "\n".join(lines)


def _build(spec):
    from simtest.coding import run

    return run.build_model(spec)


def main(argv=None) -> int:
    import anyio

    from simtest.coding import run

    def source(text: str) -> Path | None:
        return None if text == "none" else Path(text)

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("config", type=Path, help="the Option C config whose simulators are calibrated")
    p.add_argument("--condition", choices=("C3", "C2"), default="C3",
                   help="C3 (default), or C2: each case's scenario reference added, cases without one skipped")
    p.add_argument("--sim", action="append", default=[], help="a simulator of the config (default: every one)")
    p.add_argument("--max-tokens", type=int, default=65536)
    p.add_argument("--latent", type=source, default=Path("logs/latent-qwen-hopper-only"), help="or none")
    p.add_argument("--bluedot", type=source, default=Path("logs/bluedot-py-cheap-qwen"), help="or none")
    p.add_argument("--latent-v2", type=source, default=Path("logs/latent-qwen-hopper-v2"),
                   help="job 7122's logs for CASES_JOB7122, or none to leave them out (a partial case set)")
    p.add_argument("--cases", type=Path, default=None, help="read the cases from this JSON instead of the logs")
    p.add_argument("--write-cases", type=Path, default=None, help="write the extracted cases here and stop")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)
    for d in (args.latent, args.bluedot, args.latent_v2):
        if args.cases is None and d is not None and not d.exists():
            print(f"{d} does not exist (pass `none` to leave a source out)", file=sys.stderr)
            return 2
    cases = json.loads(args.cases.read_text()) if args.cases else extract(args.latent, args.bluedot, args.latent_v2)
    if not cases:
        print(f"no cases in {args.cases or 'the sources'}: nothing to calibrate", file=sys.stderr)
        return 2
    if args.write_cases:
        args.write_cases.write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n")
        print(f"{len(cases)} cases written to {args.write_cases}")
        return 0
    case_set = describe(cases)
    run_cases, skipped = select(cases, args.condition)
    if not run_cases:
        print(f"no case has a {args.condition} reference: nothing to calibrate", file=sys.stderr)
        return 2
    cfg = run.load_config(args.config)
    sims = args.sim or list(cfg["simulators"])
    by_sim = {}
    for name in sims:
        spec = cfg["simulators"][name]
        spec = {**spec, "config": {**spec["config"], "max_tokens": args.max_tokens}}
        t0 = time.monotonic()
        results = anyio.run(calibrate, _build(spec), run_cases, args.concurrency, args.condition)
        by_sim[name] = {"request": run.request_config({**cfg, "simulators": {name: spec}}, name)["simulator"],
                        "summary": summarize(results, time.monotonic() - t0), "results": results}
    print(case_set["line"])
    print(f"{args.condition}: {len(run_cases)} cases run"
          + (f", {len(skipped)} skipped without a reference ({', '.join(sorted({c['scenario'] for c in skipped}))})"
             if skipped else "") + f"; max_tokens {args.max_tokens}")
    print(format_table(by_sim))
    if args.json:
        args.json.write_text(json.dumps({"config": str(args.config), "condition": args.condition,
                                         "max_tokens": args.max_tokens, "case_set": case_set,
                                         "skipped": [{k: c.get(k) for k in ("set", "source", "scenario", "step")}
                                                     for c in skipped], "simulators": by_sim},
                                        indent=1, ensure_ascii=False, default=str) + "\n")
    if not all(passes(v["summary"]) for v in by_sim.values()):
        return 1
    return 0 if case_set["full"] else 3


if __name__ == "__main__":
    sys.exit(main())
