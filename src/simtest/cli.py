"""Command line: simtest {scenarios,demo,run,preflight,plan,summarize,images,snapshot,label,judge,power,post,fidelity,equiv}.

Two families share the CLI: the office testbed (src/simtest, the first design) and the
coding family (src/simtest/coding). `run` picks the family from the config's `family:` key
(default office); `summarize` picks it from the logs (a coding log's task name ends in
`coding`). `scenarios` lists both. `images` and `snapshot` are coding-only. `label` builds
the hand-label sample and serves the labelling page; `judge` runs and checks the awareness
judge (src/simtest/judge). `power` simulates the power of the primary test (src/simtest/stats);
`post` runs and reports the post-episode measures (src/simtest/post); `fidelity` grades each
simulated bash step against shadow execution (src/simtest/post/fidelity.py). `equiv` checks that
two vLLM backends compute the same models (src/simtest/equiv.py, docs/BACKENDS.md). `plan`
estimates a run's GPU hours and units, and splits it into jobs (src/simtest/stats/replay.py);
`preflight` runs a config end to end offline before a GPU session (src/simtest/preflight.py).
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
    if post := A.format_post_table(rows):
        print()
        print(post)
    from .runqueue import batch_mix, format_batch_mix

    if mix := format_batch_mix(batch_mix(log_dir)):
        print()
        print(mix)
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
    r.add_argument("--epochs", default=None, help="coding: only these epochs this session (N or A-B)")
    r.add_argument("--admit-for", type=float, default=None,
                   help="coding: start no episode after this many hours (running ones finish)")
    r.add_argument("--carry-under", type=int, default=0,
                   help="coding: when fewer than K episodes are left, leave the rest for the next session")
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
    lb = sub.add_parser("label", help="hand labels: build the sample manifest, serve the labelling page")
    lsub = lb.add_subparsers(dest="label_cmd", required=True)
    lm = lsub.add_parser("manifest", help="draw the hand-label sample from a pool spec (YAML) and write its manifest")
    lm.add_argument("pool", type=Path)
    lm.add_argument("--out", type=Path, default=None, help="default: manifest.json next to the pool spec")
    lm.add_argument("--force", action="store_true", help="replace an existing manifest")
    ls = lsub.add_parser("serve", help="serve the labelling page for a manifest (local only)")
    ls.add_argument("manifest", type=Path)
    ls.add_argument("--labels", type=Path, default=None, help="default: labels.jsonl next to the manifest")
    ls.add_argument("--host", default="127.0.0.1")
    ls.add_argument("--port", type=int, default=8766)
    from .post.fidelity import add_parser as add_fidelity_parser

    add_fidelity_parser(sub)
    from .judge.command import add_parser as add_judge_parser
    from .post.command import add_parser as add_post_parser
    from .stats.power import add_parser as add_power_parser

    add_judge_parser(sub)
    add_power_parser(sub)
    add_post_parser(sub)
    from .equiv import add_parser as add_equiv_parser
    from .preflight import add_parser as add_preflight_parser
    from .stats.replay import add_parser as add_plan_parser

    add_equiv_parser(sub)
    add_plan_parser(sub)
    add_preflight_parser(sub)
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

            success, _, log_dir = run_config(args.config, allow_dirty=args.allow_dirty, build=not args.no_build,
                                             epochs=args.epochs, admit_for=args.admit_for,
                                             carry_under=args.carry_under)
        else:
            if args.epochs or args.admit_for is not None or args.carry_under:
                raise SystemExit("--epochs, --admit-for and --carry-under are for coding configs")
            from .run import run_config

            success, _, log_dir = run_config(args.config, allow_dirty=args.allow_dirty)
        print(f"\nLogs: {log_dir} ({'complete' if success else 'INCOMPLETE: re-run to resume'})")
        print_summary(log_dir)
        if not success:
            raise SystemExit(3)   # resumable: session scripts tell this from an error
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
    elif args.cmd == "label" and args.label_cmd == "manifest":
        from .judge.pool import PoolError, build_manifest, format_counts, write_manifest

        out = args.out or args.pool.with_name("manifest.json")
        if out.exists() and not args.force:
            raise SystemExit(f"{out} exists: labels refer to its label ids. Use --force to replace it.")
        try:
            manifest = build_manifest(args.pool)
            write_manifest(manifest, out, force=args.force)
        except PoolError as exc:
            raise SystemExit(str(exc)) from exc
        print(format_counts(manifest))
        print(f"wrote {out}")
    elif args.cmd == "label" and args.label_cmd == "serve":
        from .judge.server import serve

        serve(args.manifest, args.labels, host=args.host, port=args.port)
    elif args.cmd == "fidelity":
        from .post.fidelity import main as fidelity_main

        fidelity_main(args)
    elif args.cmd == "judge":
        from .judge.command import main as judge_main

        judge_main(args)
    elif args.cmd == "power":
        from .stats.power import main as power_main

        power_main(args)
    elif args.cmd == "post":
        from .post.command import main as post_main

        post_main(args)
    elif args.cmd == "preflight":
        from .preflight import main as preflight_main

        raise SystemExit(preflight_main(args))
    elif args.cmd == "plan":
        from .stats.replay import main as plan_main

        plan_main(args)
    elif args.cmd == "equiv":
        from .equiv import main as equiv_main

        equiv_main(args)


if __name__ == "__main__":
    main()
