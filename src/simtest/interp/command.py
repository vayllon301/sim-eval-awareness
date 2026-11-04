"""`simtest interp`: the laptop's side of reading a run (docs/INTERP.md, "Reading a run"). The
readings themselves run in the GPU venv (explore/interp/read_run.py), which has no `simtest` command.

    simtest interp export logs/n2-pilot                     # episodes for the reader, as plain JSON
    simtest interp server-ids logs/read/n2-pilot/episodes   # the IDs of the server's own renderer
    simtest interp phrases logs/n2-pilot --judge J --kappa K --episodes E --out FILE    # DRAFT rule
    simtest interp report logs/n2-pilot --readings R [--judge J --kappa K] [--phrases F] [--json]
"""

from __future__ import annotations

import json
from pathlib import Path


def default_dir(log_dir: Path, name: str) -> Path:
    """Reader files live beside the run's logs, not in them: Inspect lists every JSON under a log
    directory as a log."""
    return log_dir.parent / "read" / log_dir.name / name


def add_parser(sub) -> None:
    p = sub.add_parser("interp", help="read a run from the inside: export, server IDs, sentences, report (no model calls)")
    s = p.add_subparsers(dest="interp_cmd", required=True)
    e = s.add_parser("export", help="a run's episodes for the reader, as plain JSON")
    e.add_argument("log_dir", type=Path)
    e.add_argument("--out", type=Path, default=None, help="default: logs/read/<name>/episodes")
    i = s.add_parser("server-ids", help="the token IDs of the agent's server for each episode's last request")
    i.add_argument("episodes", type=Path)
    i.add_argument("--url", default="http://localhost:18000")
    q = s.add_parser("phrases", help="the logit monitor's sentences from the pilot's verbalisations (DRAFT rule)")
    q.add_argument("log_dir", type=Path)
    q.add_argument("--judge", type=Path, required=True), q.add_argument("--kappa", type=Path, required=True)
    q.add_argument("--episodes", type=Path, default=None), q.add_argument("--out", type=Path, required=True)
    q.add_argument("--draft", type=Path, default=Path("explore/interp/data/logit_monitor_phrases.json"))
    r = s.add_parser("report", help="each monitor against the judge's labels and behaviour")
    r.add_argument("log_dir", type=Path)
    r.add_argument("--readings", type=Path, default=None, help="default: logs/read/<name>/readings")
    r.add_argument("--judge", type=Path, default=None), r.add_argument("--kappa", type=Path, default=None)
    r.add_argument("--phrases", type=Path, default=None, help="the phrase file, when its sentences were picked on this run")
    r.add_argument("--layer", type=int, default=None), r.add_argument("--json", action="store_true")


def _run(log_dir: Path) -> tuple[list[dict], dict]:
    from ..coding import analysis as A
    from ..judge import episodes as E

    episodes, design = A.load_run(log_dir)
    if design is None:
        raise SystemExit(f"{log_dir}: these logs record no design; a run's come from `simtest run`")
    for e in episodes:
        e["judge_key"] = None if A.is_excluded(e) else E.run_episode(log_dir, e)["key"]
    return episodes, design


def _judge(judge: Path, kappa: Path | None) -> tuple[list[dict], dict]:
    from ..judge import judge as J
    from ..judge import kappa as K

    results = J.read_results(judge)
    if not results:
        raise SystemExit(f"no results in {judge}")
    return results, K.pass_state(kappa, K.judge_identity(results))


def main(args) -> None:
    from .readrun import ReadError, server_ids

    cmd = args.interp_cmd
    if cmd == "export":
        from .episodes import AdapterError, export_run

        out = args.out or default_dir(args.log_dir, "episodes")
        try:
            m = export_run(args.log_dir, out)
        except AdapterError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"{out}: {m['counts']}")
        for r in m["episodes"]:
            if r["status"] != "exported":
                print(f"  {r['status']}: {r['condition']} {r['scenario']} epoch {r['epoch']}: {r['reason']}")
    elif cmd == "server-ids":
        try:
            print(server_ids(args.episodes, args.url))
        except (ReadError, OSError) as exc:
            raise SystemExit(str(exc)) from exc
    elif cmd == "phrases":
        from .phrases import pick, reasoning_turns
        from .reader import load_episode

        episodes, _ = _run(args.log_dir)
        results, kappa = _judge(args.judge, args.kappa)
        if not kappa["valid"]:
            raise SystemExit(f"NOT VALID: {kappa['reason']}. Sentences come only from a judge that passed.")
        where = args.episodes or default_dir(args.log_dir, "episodes")
        texts = lambda uuid: reasoning_turns(load_episode(where / f"{uuid}.json")) if (where / f"{uuid}.json").exists() else []
        doc = pick(results, episodes, texts, json.loads(args.draft.read_text()), run=args.log_dir.name)
        args.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {args.out}: " + ", ".join(f"{g} {len(v)}" for g, v in doc["groups"].items()))
    elif cmd == "report":
        from . import validate as V

        episodes, design = _run(args.log_dir)
        readings = V.load_readings(args.readings or default_dir(args.log_dir, "readings"))
        results, kappa = _judge(args.judge, args.kappa) if args.judge else (None, None)
        provenance = json.loads(args.phrases.read_text()).get("provenance") if args.phrases else None
        rep = V.build(episodes, design, results, readings, layer=args.layer or V.reading_layer(readings), kappa=kappa,
                      run=args.log_dir.name, provenance=provenance)
        print(V.dumps(rep) if args.json else V.format_report(rep))
