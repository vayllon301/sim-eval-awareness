"""`simtest post ...`: the post-episode measures. See docs/TESTBED.md, "Post-episode measures".

    simtest post ask <log_dir>... --config configs/coding/<run>.yaml --prompts v1 --out post.jsonl --yes
    simtest post judge <log_dir>... --model openrouter/<org>/<model> --prompts v1 --out post.jsonl --yes
    simtest post report post.jsonl

`ask` puts the purpose question, the 0-1 rating (3 samples) and the pairwise realism judgement
to the agent model: the run config's agent with the episodes' settings (`--config`), or any
Inspect model (`--model`, e.g. mockllm/model offline). `judge` labels the purpose answers and
judges the final summaries' accuracy and honesty, on any provider through Inspect's model API.
`report` prints the results per condition and scenario and compared across conditions.

Any call to a model other than mockllm needs `--yes` (an endpoint on Colab spends compute
units, OpenRouter spends credit); without it the command prints the plan and stops. Such a run
also refuses uncommitted changes under post/prompts, and warns when the prompt version isn't
frozen (a FROZEN file).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from . import measures as M


def _common(p) -> None:
    p.add_argument("log_dirs", nargs="+", type=Path, help="coding log directories (as for simtest summarize)")
    p.add_argument("--prompts", required=True, help="prompt version (a directory in post/prompts)")
    p.add_argument("--out", type=Path, required=True, help="results JSONL (appended; cached calls are skipped)")
    p.add_argument("--only", default=None, help="comma-separated substrings of episode keys")
    p.add_argument("--with-reasoning", action="store_true",
                   help="show the agent's reasoning in the transcript (default: left out; recorded)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--model-args", default=None, help="provider options as JSON (OpenRouter's provider order, say)")
    p.add_argument("--yes", action="store_true", help="make the calls; without it, print the plan")


def add_parser(sub) -> None:
    pp = sub.add_parser("post", help="post-episode measures: purpose, rating, pairwise realism, summary judges")
    psub = pp.add_subparsers(dest="post_cmd", required=True)
    ask = psub.add_parser("ask", help="ask the agent model, in a fresh context: purpose, rating, pairwise")
    _common(ask)
    who = ask.add_mutually_exclusive_group(required=True)
    who.add_argument("--config", type=Path, help="a coding run config: its agent, with the episodes' settings")
    who.add_argument("--model", default=None, help="any Inspect model (mockllm/model offline)")
    ask.add_argument("--agent", default=None, help="the agent's name in --config (default: its only agent)")
    ask.add_argument("--generate-config", default=None,
                     help='with --model: generation settings as JSON, e.g. \'{"temperature": 1.0, "top_p": 0.95, '
                          '"top_k": 20, "max_tokens": 32768}\'')
    ask.add_argument("--measures", default=",".join(M.AGENT_MEASURES))
    ask.add_argument("--a", default="C0", help="pairwise: the baseline condition (default C0)")
    ask.add_argument("--b", default="C3", help="pairwise: the condition compared with it (default C3)")
    ask.add_argument("--cut-max", type=int, default=None,
                     help="pairwise: show at most this many agent turns (fix it before the run; default no cap)")
    ask.add_argument("--budget", type=int, default=131_072, help="the model's context: prompt + output tokens")
    jd = psub.add_parser("judge", help="LLM judges: the purpose answers' labels, the summaries' accuracy and honesty")
    _common(jd)
    jd.add_argument("--model", required=True, help="an Inspect model (openrouter/..., mockllm/model offline)")
    jd.add_argument("--measures", default=",".join(M.JUDGE_MEASURES))
    jd.add_argument("--temperature", default="0", help="a number, or 'none' for the provider default")
    jd.add_argument("--max-tokens", type=int, default=16000, help="output tokens, reasoning included")
    jd.add_argument("--reasoning-effort", default=None)
    jd.add_argument("--seed", type=int, default=None)
    jd.add_argument("--budget", type=int, default=200_000, help="token budget for one call (prompt + output)")
    rp = psub.add_parser("report", help="the results per condition and scenario, compared across conditions")
    rp.add_argument("results", type=Path)
    rp.add_argument("--a", default="C0")
    rp.add_argument("--b", default="C3")
    rp.add_argument("--json", action="store_true")


def _json_arg(value, name):
    if value is None:
        return None
    obj = json.loads(value)
    if not isinstance(obj, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return obj


def _measures(args, allowed) -> list[str]:
    ms = [m.strip() for m in args.measures.split(",") if m.strip()]
    if bad := [m for m in ms if m not in allowed]:
        raise SystemExit(f"unknown measures {bad} for this command (allowed: {list(allowed)})")
    return ms


def _episodes(args):
    eps, skipped = M.load_episodes(args.log_dirs)
    if args.only:
        want = [w.strip() for w in args.only.split(",")]
        eps = [e for e in eps if any(w in e["key"] for w in want)]
    for s in skipped:
        print(f"skipped {s['key']}: {s['reason']}")
    return eps


def _paid(model: str) -> bool:
    return not str(model).startswith("mockllm/")


def _guard(args, backend) -> bool:
    """True if the calls may be made."""
    if not _paid(backend.model):
        return True
    if not args.yes:
        print("Not called: this model is not mockllm. Add --yes to make the calls.")
        return False
    M.code_state.cache_clear()          # read the prompts' state now, not as it was at planning
    state = M.code_state(args.prompts)
    if state["prompts_dirty"]:
        raise SystemExit("post/prompts has uncommitted changes: commit them first")
    if not state["prompts_frozen"]:
        print(f"note: prompt version {args.prompts} is not frozen (no FROZEN file); each result records it")
    return True


def _run(args, calls, skipped, backend) -> None:
    for s in skipped:
        print(f"skipped {s['measure']} {s.get('key')}: {s['reason']}")
    store = M.ResultStore(args.out)
    print(M.plan_summary(calls, store, backend))
    if not _guard(args, backend):
        return
    if any(store.done(c["key"]) is None for c in calls):
        try:
            backend.ensure()
        except Exception as exc:
            raise SystemExit(f"could not set up {backend.model}: {exc}") from exc
    asyncio.run(M.run_calls(calls, backend, store, concurrency=args.concurrency))
    print(f"results: {args.out}")


def main(args: argparse.Namespace) -> None:
    from .report import build_report, format_report

    if args.post_cmd == "report":
        recs = list(M.ResultStore(args.results).latest.values())
        try:
            rep = build_report(recs, a=args.a, b=args.b)
        except M.PostError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(rep, indent=1, default=str) if args.json else format_report(rep))
        return
    try:
        if args.post_cmd == "ask":
            ms = _measures(args, M.AGENT_MEASURES)
            if args.config:
                if args.generate_config or args.model_args:
                    raise SystemExit("--generate-config and --model-args go with --model; --config uses the run's "
                                     "own settings")
                backend = M.agent_backend(args.config, args.agent)
            else:
                backend = M.ModelBackend(args.model, config=_json_arg(args.generate_config, "--generate-config"),
                                         model_args=_json_arg(args.model_args, "--model-args"))
            prompts = M.load_prompts(args.prompts, ms)
            calls, skipped = M.plan_agent(_episodes(args), M.read_sample, prompts, args.prompts, backend,
                                          reasoning=args.with_reasoning, a=args.a, b=args.b, cut_max=args.cut_max,
                                          budget=args.budget)
        else:
            ms = _measures(args, M.JUDGE_MEASURES)
            temp = None if str(args.temperature).lower() == "none" else float(args.temperature)
            cfg = {k: v for k, v in {"temperature": temp, "max_tokens": args.max_tokens,
                                     "reasoning_effort": args.reasoning_effort, "seed": args.seed}.items()
                   if v is not None}
            backend = M.ModelBackend(args.model, config=cfg, model_args=_json_arg(args.model_args, "--model-args"))
            prompts = M.load_prompts(args.prompts, ms)
            store = M.ResultStore(args.out)
            calls, skipped = M.plan_judge(_episodes(args), M.read_sample, prompts, args.prompts, backend, store,
                                          budget=args.budget, reasoning=args.with_reasoning)
    except M.PostError as exc:
        raise SystemExit(str(exc)) from exc
    _run(args, calls, skipped, backend)
