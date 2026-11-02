"""`simtest judge ...`: run the awareness judge, tune it, size the prompts, and compare it with
the hand labels. See docs/TESTBED.md, "The awareness judge".

    simtest judge tune <tuning.yaml> --backend codex --prompts v1 --out tune.jsonl
    simtest judge run <manifest.json> --backend inspect --model openrouter/... --prompts v3 --out kappa.jsonl --yes
    simtest judge kappa <kappa.jsonl> <labels.jsonl>
    simtest judge sizes <manifest.json | tuning.yaml> --prompts v3
    simtest judge show <results.jsonl>

`tune` reads only tuning episodes and refuses the test set. `run` reads only the test set,
with a frozen prompt version (a FROZEN file, committed: the prompt directory must be clean),
on Inspect's model API. Any paid call (Inspect with a model other than mockllm) needs
`--yes`; without it the command prints the plan and stops.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import episodes as E
from . import judge as J


def add_parser(sub) -> None:
    jp = sub.add_parser("judge", help="awareness judge: tune, run on the test set, kappa against the hand labels")
    jsub = jp.add_subparsers(dest="judge_cmd", required=True)

    def common(p, backend_default):
        p.add_argument("--backend", choices=("inspect", "codex"), default=backend_default)
        p.add_argument("--model", default=None, help="inspect: a model id (openrouter/...); codex: default gpt-6.1-sol")
        p.add_argument("--prompts", required=True, help="prompt version (a directory in judge/prompts)")
        p.add_argument("--labels", default=None, help="comma-separated labels (default: all five)")
        p.add_argument("--out", type=Path, required=True, help="results JSONL (appended; cached calls are skipped)")
        p.add_argument("--concurrency", type=int, default=4)
        p.add_argument("--budget", type=int, default=200_000, help="token budget for one call (prompt + output)")
        p.add_argument("--temperature", default="0", help="inspect: a number, or 'none' for the provider default")
        p.add_argument("--max-tokens", type=int, default=16000, help="inspect: output tokens, reasoning included")
        p.add_argument("--reasoning-effort", default=None, help="inspect: none by default; codex: high by default")
        p.add_argument("--seed", type=int, default=None, help="inspect: a sampling seed, where the provider takes one")
        p.add_argument("--model-args", default=None,
                       help='inspect: provider options as JSON, e.g. \'{"provider": {"order": ["openai"], '
                            '"allow_fallbacks": false}}\' for OpenRouter')
        p.add_argument("--yes", action="store_true", help="make the calls (paid backends); without it, print the plan")

    t = jsub.add_parser("tune", help="run the judge on tuning episodes (never the test set)")
    t.add_argument("spec", type=Path, help="tuning spec (YAML)")
    t.add_argument("--only", default=None, help="comma-separated episode keys (or substrings of them)")
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--min-turns", type=int, default=1, help="leave out episodes with fewer agent turns (default 1)")
    common(t, "codex")
    r = jsub.add_parser("run", help="run the frozen judge on a manifest's drawn episodes (the kappa pass)")
    r.add_argument("manifest", type=Path)
    common(r, "inspect")
    k = jsub.add_parser("kappa", help="Cohen's kappa per label against the hand labels, with a bootstrap 95%% CI")
    k.add_argument("results", type=Path)
    k.add_argument("labels", type=Path)
    k.add_argument("--boot", type=int, default=10000)
    k.add_argument("--json", action="store_true")
    s = jsub.add_parser("sizes", help="prompt sizes in tokens against the budget (numbers only, no text)")
    s.add_argument("source", type=Path, help="a manifest (its drawn episodes) or a tuning spec")
    s.add_argument("--prompts", required=True)
    s.add_argument("--budget", type=int, default=200_000)
    s.add_argument("--max-tokens", type=int, default=16000)
    sh = jsub.add_parser("show", help="verdict counts, errors and quote failures in a results file")
    sh.add_argument("results", type=Path)


def _backend(args):
    from .backends import CodexBackend, InspectBackend

    if args.backend == "codex":
        return CodexBackend(model=args.model or "gpt-6.1-sol", reasoning_effort=args.reasoning_effort or "high")
    if not args.model:
        raise SystemExit("--model is required with --backend inspect")
    temp = None if str(args.temperature).lower() == "none" else float(args.temperature)
    margs = json.loads(args.model_args) if args.model_args else None
    if margs is not None and not isinstance(margs, dict):
        raise SystemExit("--model-args must be a JSON object")
    return InspectBackend(args.model, temperature=temp, max_tokens=args.max_tokens,
                          reasoning_effort=args.reasoning_effort, seed=args.seed, model_args=margs)


def _paid(args) -> bool:
    return args.backend == "inspect" and not (args.model or "").startswith("mockllm/")


def _labels(args) -> list[str] | None:
    return [x.strip() for x in args.labels.split(",")] if args.labels else None


def _plan_summary(calls, store, backend) -> str:
    todo = [c for c in calls if store.done(c["key"]) is None]
    tokens = sum(c["base"]["prompt_tokens"] for c in todo)
    too_long = sum(not c["fits"] for c in calls)
    eps = len({c["base"]["episode"]["key"] for c in calls})
    return (f"{eps} episodes x {len({c['label'] for c in calls})} labels = {len(calls)} calls; "
            f"{len(calls) - len(todo)} cached, {len(todo)} to make, ~{tokens:,} prompt tokens "
            f"(o200k_base), {too_long} too long for the budget; backend {backend.name} {backend.model} "
            f"{json.dumps(backend.settings)}")


def _execute(args, episodes, read, guard=None, min_turns: int = 0) -> None:
    prompts = J.load_prompts(args.prompts, _labels(args))
    backend = _backend(args)
    calls, refused = J.plan_calls(episodes, read, prompts, args.prompts, backend, args.budget, guard=guard)
    for r in refused:
        print(f"refused {r['key']}: {r['reason']}")
    if short := {c["base"]["episode"]["key"] for c in calls if c["base"]["n_turns"] < min_turns}:
        print(f"left out {len(short)} episode(s) with fewer than {min_turns} agent turn(s)")
        calls = [c for c in calls if c["base"]["n_turns"] >= min_turns]
    store = J.ResultStore(args.out)
    print(_plan_summary(calls, store, backend))
    if _paid(args) and not args.yes:
        print("Not called: this backend spends credit. Add --yes to make the calls.")
        return
    asyncio.run(J.run_calls(calls, backend, store, concurrency=args.concurrency))
    print(show(list(store.latest.values()), keys={c["key"] for c in calls}))
    print(f"results: {args.out}")


def show(results: list[dict], keys: set[str] | None = None) -> str:
    rs = [r for r in results if keys is None or r["key"] in keys]
    by = defaultdict(Counter)
    qn, qf, errs = Counter(), Counter(), Counter()
    for r in rs:
        lab = r["label"]
        if r.get("error"):
            errs[(lab, r["error"]["kind"])] += 1
            by[lab]["error"] += 1
            continue
        v = r["verdict_raw"] if r["verdict_raw"] else ("flagged" if r["positive"] else "not flagged")
        by[lab][v] += 1
        qn[lab] += len(r["quotes"])
        qf[lab] += r["quote_failures"]
    lines = []
    for lab in J.LABELS:
        if lab in by:
            lines.append(f"{lab}: " + ", ".join(f"{k} {n}" for k, n in sorted(by[lab].items())) +
                         f"; quotes {qn[lab]}, unverified {qf[lab]}")
    for (lab, kind), n in sorted(errs.items()):
        lines.append(f"  errors {lab} {kind}: {n}")
    return "\n".join(lines)


def _sizes(args) -> None:
    import statistics

    from . import transcript as T

    prompts = J.load_prompts(args.prompts)
    longest = max(prompts.values(), key=lambda p: len(p["text"]))
    if args.source.suffix == ".json":
        guard = E.TestSetGuard(args.source)
        eps, read = guard.drawn, (lambda ep: E.read_sample(guard.root, ep))
    else:
        spec = E.load_tuning_spec(args.source)
        guard = E.TestSetGuard((args.source.parent / spec["root"]).resolve() / spec["test_set"])
        eps, _, root = E.tuning_set(args.source, guard=guard)
        read = lambda ep: E.read_sample(root, ep)  # noqa: E731
    sizes = []
    for ep in eps:
        steps = T.episode_steps(read(ep))
        if args.source.suffix != ".json" and guard.reason(ep, steps):
            continue                       # a tuning episode the test set refuses
        sizes.append(J.count_tokens(J.build_prompt(longest["text"], T.render_text(steps)))[0])
    sizes.sort(reverse=True)
    over = sum(not J.fits(n, args.max_tokens, args.budget) for n in sizes)
    print(f"{len(sizes)} episodes; prompt tokens (the longest prompt, o200k_base): max {sizes[0]:,}, "
          f"top 5 {sizes[:5]}, median {statistics.median(sizes):,.0f}, sum {sum(sizes):,}")
    print(f"labels {len(prompts)}: all calls ~{sum(sizes) * len(prompts):,} prompt tokens; "
          f"{over} episode(s) might not fit {args.budget:,} with {args.max_tokens:,} output tokens and a 25% margin")


def main(args: argparse.Namespace) -> None:
    from .kappa import dumps, format_report, kappa_files

    if args.judge_cmd == "tune":
        spec = E.load_tuning_spec(args.spec)
        guard = E.TestSetGuard((args.spec.parent / spec["root"]).resolve() / spec["test_set"])
        eps, refused, root = E.tuning_set(args.spec, guard=guard)
        for r in refused:
            print(f"refused {r['key']}: {r['reason']}")
        if args.only:
            want = [w.strip() for w in args.only.split(",")]
            eps = [e for e in eps if any(w in e["key"] for w in want)]
        if args.limit:
            eps = eps[:args.limit]
        _execute(args, eps, lambda ep: E.read_sample(root, ep), guard=guard, min_turns=args.min_turns)
    elif args.judge_cmd == "run":
        if args.backend != "inspect":
            raise SystemExit("the test set is judged on Inspect's model API only (a fixed model and settings "
                             "anyone can rerun); codex is for tuning")
        if not J.is_frozen(args.prompts):
            raise SystemExit(f"prompt version {args.prompts} is not frozen (no FROZEN file): the test set is "
                             "judged only with a frozen, committed prompt version")
        if J.code_state()["prompts_dirty"]:
            raise SystemExit("judge/prompts has uncommitted changes: commit them first")
        guard = E.TestSetGuard(args.manifest)
        _execute(args, guard.drawn, lambda ep: E.read_sample(guard.root, ep))
    elif args.judge_cmd == "kappa":
        rep = kappa_files(args.results, args.labels, n_boot=args.boot)
        print(dumps(rep) if args.json else format_report(rep))
    elif args.judge_cmd == "sizes":
        _sizes(args)
    elif args.judge_cmd == "show":
        print(show(J.read_results(args.results)))
