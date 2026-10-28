"""Command line: simtest {scenarios,demo,run,summarize,images,snapshot}.

Two families share the CLI: the office testbed (src/simtest, the first design) and the
coding family (src/simtest/coding). `run` picks the family from the config's `family:` key
(default office); `summarize` picks it from the logs (a coding log's task name ends in
`coding`). `scenarios` lists both. `images` and `snapshot` are coding-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import (
    BY, PRIMARY, LogDirError, format_exclusions, format_incomplete, format_notes, format_paired, format_table,
    format_warnings, load_run, paired_difference, summarize,
)
from .scenarios import list_scenarios, load_scenario


def _is_coding_config(path: Path) -> bool:
    import yaml

    try:
        raw = yaml.safe_load(path.read_text())
    except Exception:
        return False
    return isinstance(raw, dict) and raw.get("family") == "coding"


def _is_coding_logdir(log_dir: Path) -> bool:
    from inspect_ai.log import list_eval_logs, read_eval_log

    try:
        infos = list_eval_logs(str(log_dir))
    except Exception:
        return False
    for info in infos:
        name = read_eval_log(info, header_only=True).eval.task_registry_name or ""
        if name.endswith("coding"):
            return True
    return False


def _print(episodes, design, rows, mod) -> None:
    if warnings := mod.format_warnings(episodes):
        print(warnings)
    if incomplete := mod.format_incomplete(episodes, design):
        print(incomplete)
    print(mod.format_table(rows))
    conditions = sorted(design["conditions"] if design else {e["condition"] for e in episodes})
    if len(conditions) > 1:
        base = "C0" if "C0" in conditions else conditions[0]
        metrics = [m for m in mod.PRIMARY if any(e["metrics"].get(m) is not None for e in episodes)]
        print("\nScenario-level paired difference (within scenario x agent; agents and simulators equally "
              "weighted). Pooled first, then each simulator:")
        for b in conditions:
            if b == base:
                continue
            sims = (design["simulators"][b] if design
                    else sorted({e.get("simulator") or "none" for e in episodes if e["condition"] == b}))
            for m in metrics:
                print("  " + mod.format_paired(mod.paired_difference(episodes, a=base, b=b, metric=m, design=design)))
                if len(sims) > 1:
                    for s in sims:
                        print("    " + mod.format_paired(mod.paired_difference(episodes, a=base, b=b, metric=m,
                                                                               simulators=[s], design=design)))
    print()
    print(mod.format_exclusions(episodes))
    print(mod.format_notes(episodes))


def print_summary(log_dir: Path, by: tuple[str, ...] = BY) -> list[dict]:
    """Office-family summary. `load_run`/`summarize`/`paired_difference` are module-level so
    tests can patch them; coding log dirs are summarised by print_summary_coding."""
    if _is_coding_logdir(log_dir):
        return print_summary_coding(log_dir, by=None)
    try:
        episodes, design = load_run(log_dir)
    except LogDirError as exc:
        raise SystemExit(str(exc)) from exc
    import sys as _sys

    rows = summarize(episodes, by=by)
    mod = _sys.modules[__name__]  # so monkeypatched cli.* are used
    _print(episodes, design, rows, mod)
    return rows


def print_summary_coding(log_dir: Path, by=None) -> list[dict]:
    from .coding import analysis as A

    by = by or A.BY
    try:
        episodes, design = A.load_run(log_dir)
    except A.LogDirError as exc:
        raise SystemExit(str(exc)) from exc
    rows = A.summarize(episodes, by=by)
    _print(episodes, design, rows, A)
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="simtest")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scenarios", help="list scenarios (office and coding)")
    d = sub.add_parser("demo", help="offline demo: scripted agents + two mock simulators (mockllm)")
    d.add_argument("--log-dir", default="logs/demo")
    r = sub.add_parser("run", help="run an experiment config (resumable)")
    r.add_argument("config", type=Path)
    r.add_argument("--allow-dirty", action="store_true")
    r.add_argument("--no-build", action="store_true", help="coding: don't build images before the run")
    s = sub.add_parser("summarize", help="summarise a log directory")
    s.add_argument("log_dir", type=Path)
    s.add_argument("--by", default=None)
    s.add_argument("--json", action="store_true")
    im = sub.add_parser("images", help="coding: build the dev-box and scenario Docker images")
    im.add_argument("--rebuild", action="store_true")
    im.add_argument("--clean", action="store_true", help="also remove leftover simtest containers")
    sn = sub.add_parser("snapshot", help="coding: (re)generate or check a scenario's initial-state snapshot")
    sn.add_argument("scenario", nargs="?", help="scenario id (default: all)")
    sn.add_argument("--check", action="store_true", help="compare the frozen snapshot with a fresh container")
    args = p.parse_args(argv)

    if args.cmd == "scenarios":
        from .coding.scenario import list_scenarios as coding_list, load_scenario as coding_load

        for sid in list_scenarios():
            sc = load_scenario(sid)
            print(f"{sid} [office, {sc.split}] — {sc.title}")
        for sid in coding_list():
            sc = coding_load(sid)
            print(f"{sid} [coding, {sc.split}, {sc.variant}] — {sc.title}")
    elif args.cmd == "demo":
        from .demo import run_demo

        run_dir = run_demo(args.log_dir)
        print_summary(run_dir)
        print(f"\nOffline demo: scripted agents + mock simulators echo and lossy (no API calls). Logs: {run_dir}")
        print(f"View transcripts: uv run inspect view --log-dir {run_dir}")
    elif args.cmd == "run":
        if _is_coding_config(args.config):
            from .coding.run import run_config

            success, _, log_dir = run_config(args.config, allow_dirty=args.allow_dirty, build=not args.no_build)
        else:
            from .run import run_config

            success, _, log_dir = run_config(args.config, allow_dirty=args.allow_dirty)
        print(f"\nLogs: {log_dir} ({'complete' if success else 'INCOMPLETE: re-run to resume'})")
        print_summary(log_dir)
    elif args.cmd == "summarize":
        by = tuple(args.by.split(",")) if args.by else None
        rows = print_summary(args.log_dir, by=by or BY) if not _is_coding_logdir(args.log_dir) \
            else print_summary_coding(args.log_dir, by=by)
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
    elif args.cmd == "images":
        from .coding.images import build_all, clean

        if args.clean:
            print(f"removed {clean()} leftover container(s)")
        tags = build_all(rebuild=args.rebuild)
        print("images ready:\n  " + "\n  ".join(tags))
    elif args.cmd == "snapshot":
        from .coding.scenario import list_scenarios as coding_list, load_scenario as coding_load
        from .coding.snapshot import check, write

        for sid in [args.scenario] if args.scenario else coding_list():
            sc = coding_load(sid)
            if args.check:
                diff = check(sc)
                print(f"{sid}: {'up to date' if not diff else 'DIFFERS'}")
                if diff:
                    print("\n".join(diff[:60]))
            else:
                write(sc)
                print(f"{sid}: wrote {sc.snapshot_path}")


if __name__ == "__main__":
    main()
