"""Command line: simtest {scenarios,demo,run,summarize}."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import (
    BY, PRIMARY, LogDirError, format_exclusions, format_incomplete, format_notes, format_paired, format_table, format_warnings,
    load_run, paired_difference, summarize,
)
from .scenarios import list_scenarios, load_scenario


def print_summary(log_dir: Path, by: tuple[str, ...] = BY) -> list[dict]:
    try:
        episodes, design = load_run(log_dir)
    except LogDirError as exc:
        raise SystemExit(str(exc)) from exc
    rows = summarize(episodes, by=by)
    if warnings := format_warnings(episodes):  # first: a request-side bug shows up here
        print(warnings)
    if incomplete := format_incomplete(episodes, design):
        print(incomplete)
    print(format_table(rows))
    # The design (if recorded) says which conditions and simulators the run has, so an
    # interrupted run shows missing cells instead of a smaller panel.
    conditions = sorted(design["conditions"] if design else {e["condition"] for e in episodes})
    if len(conditions) > 1:  # scenario-level paired difference vs the baseline condition
        base = "C0" if "C0" in conditions else conditions[0]
        # each primary the logs record (clear_violation from M1c on)
        metrics = [m for m in PRIMARY if any(e["metrics"].get(m) is not None for e in episodes)]
        print("\nScenario-level paired difference (within scenario x agent; agents and simulators equally "
              "weighted). Pooled first, then each simulator:")
        for b in conditions:
            if b == base:
                continue
            sims = (design["simulators"][b] if design
                    else sorted({e.get("simulator") or "none" for e in episodes if e["condition"] == b}))
            for m in metrics:
                print("  " + format_paired(paired_difference(episodes, a=base, b=b, metric=m, design=design)))
                if len(sims) > 1:  # with one simulator the pooled line is that simulator's
                    for s in sims:
                        print("    " + format_paired(paired_difference(episodes, a=base, b=b, metric=m,
                                                                       simulators=[s], design=design)))
    print()
    print(format_exclusions(episodes))
    print(format_notes(episodes))
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="simtest")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scenarios", help="list scenarios")
    d = sub.add_parser("demo", help="offline demo: scripted agents + two mock simulators (mockllm)")
    d.add_argument("--log-dir", default="logs/demo", help="each run writes to a new timestamped subdirectory")
    r = sub.add_parser("run", help="run an experiment config (resumable)")
    r.add_argument("config", type=Path)
    r.add_argument("--allow-dirty", action="store_true", help="allow a dirty git working tree")
    s = sub.add_parser("summarize", help="summarise a log directory")
    s.add_argument("log_dir", type=Path)
    s.add_argument("--by", default=",".join(BY))
    s.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    if args.cmd == "scenarios":
        for sid in list_scenarios():
            sc = load_scenario(sid)
            print(f"{sid} [{sc.split}] — {sc.title}")
    elif args.cmd == "demo":
        from .demo import run_demo

        run_dir = run_demo(args.log_dir)
        print_summary(run_dir)
        print(f"\nOffline demo: scripted agents + mock simulators echo and lossy (no API calls). Logs: {run_dir}")
        print(f"View transcripts: uv run inspect view --log-dir {run_dir}")
    elif args.cmd == "run":
        from .run import run_config

        success, _, log_dir = run_config(args.config, allow_dirty=args.allow_dirty)
        print(f"\nLogs: {log_dir} ({'complete' if success else 'INCOMPLETE: re-run the same command to resume'})")
        print_summary(log_dir)
    elif args.cmd == "summarize":
        rows = print_summary(args.log_dir, by=tuple(args.by.split(",")))
        if args.json:
            print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
