"""Backend equivalence check (docs/BACKENDS.md): do two vLLM backends compute the same model?

Saved episodes are teacher-forced on OpenAI-compatible vLLM servers: each context's tokens go
in as the prompt, with prompt log-probs and no sampling, so every position's log-probability
of the recorded token and the server's top-1 token are read without generating anything.
Two score files are then compared position by position.

    simtest equiv contexts <logs...> --out contexts.jsonl      # offline: from .eval logs
    simtest equiv score contexts.jsonl --agent-url URL --simulator-url URL --concurrency 1 12 --out g4
    simtest equiv compare g4-c1.jsonl hopper-c1.jsonl --noise g4-c12.jsonl

- **Contexts.** Per episode, the agent's last call (its prompt is the whole episode) and a few
  of the simulator's calls, spread over the episode with the last kept. A context is the exact
  request Inspect logged plus the reply, as the next turn would send it back (`rawchat`).
- **Tokens.** Each server renders and tokenizes a context itself (`/tokenize`, its own chat
  template), and the IDs' hash is stored. Two files can only be compared on identical IDs: a
  backend that renders differently is not the same backend.
- **Noise floor.** The simulator's FP8 MoE scales activations per tensor over the whole batch
  (docs/INTERP.md), so even one server gives different numbers when requests share a batch.
  Scoring the same contexts one at a time and N at a time measures that noise.
- **Pass rule (DRAFT):** for each model, the gap between the two backends (both one request at
  a time) is no bigger than the reference backend's own batching noise: mean |difference in
  log-prob| no larger, top-1 agreement no lower. Both models must pass. Different token IDs
  fail; a context missing from any file makes the check INCOMPLETE, never PASS.

Never score while episodes run: the requests would join the episodes' batches.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

FORMAT = 1
MODELS = ("agent", "simulator")
GOOD_STOPS = ("stop", "tool_calls")


class EquivError(ValueError):
    pass


@dataclass
class Server:
    url: str          # the OpenAI-compatible base, e.g. http://localhost:18000/v1
    model: str        # the served model name

    @property
    def root(self) -> str:
        u = self.url.rstrip("/")
        return u[:-3] if u.endswith("/v1") else u


def sha256_tokens(tokens: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


# --- contexts from saved episodes -------------------------------------------------------------


def event_model(event) -> str:
    """'simulator' for the simulator's calls (Inspect's model role), else 'agent'."""
    return "simulator" if getattr(event, "role", None) == "simulator" else "agent"


def final_message(response: dict) -> dict | None:
    """The reply as the next request would send it back (provider.py: content, reasoning, tool
    calls), or None when there is no complete reply."""
    choices = (response or {}).get("choices") or []
    if not choices or choices[0].get("finish_reason") not in GOOD_STOPS:
        return None
    m = choices[0].get("message") or {}
    msg = {"role": "assistant", "content": m.get("content") or "",
           "reasoning": m.get("reasoning") or m.get("reasoning_content") or ""}
    if m.get("tool_calls"):
        msg["tool_calls"] = [{"id": c["id"], "type": c.get("type", "function"),
                              "function": {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}}
                             for c in m["tool_calls"]]
    return msg


def context_from_event(event, model: str) -> dict | None:
    """One model call as a context (request messages + reply), or None if it can't be one."""
    call = getattr(event, "call", None)
    if getattr(event, "error", None) or call is None:
        return None
    request = call.request or {}
    if not isinstance(request.get("messages"), list):
        return None
    reply = final_message(call.response or {})
    if reply is None:
        return None
    usage = (call.response or {}).get("usage") or {}
    return {"model": model, "messages": [*request["messages"], reply], "tools": request.get("tools") or None,
            "prompt_tokens": usage.get("prompt_tokens")}


def spaced(items: list, k: int) -> list:
    """k items spread evenly over the list, the first and the last included (all if fewer)."""
    n = len(items)
    if k >= n:
        return list(items)
    if k == 1:
        return [items[-1]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def _log_paths(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(p.glob("*.eval"))
        elif p.suffix == ".eval" and p.exists():
            out.append(p)
        else:
            raise EquivError(f"{p}: not a log directory or an .eval file")
    return out


def extract_contexts(paths, episodes: int = 12, sim_calls: int = 2) -> tuple[list[dict], dict]:
    """Contexts from `episodes` episodes spread over the logs' completed samples (a sample run
    in several attempt logs counts once, from the newest log; log files are named by their
    start time). Returns (contexts, notes)."""
    from inspect_ai.log import read_eval_log, read_eval_log_sample, read_eval_log_sample_summaries

    picked: dict[str, tuple] = {}
    errored = 0
    conditions = {}
    for path in sorted(_log_paths(paths), key=lambda p: p.name):
        head = read_eval_log(str(path), header_only=True)
        conditions[str(path)] = (head.eval.task_args or {}).get("condition")
        for s in read_eval_log_sample_summaries(str(path)):
            key = s.uuid or f"{path}::{s.id}::{s.epoch}"
            if s.error:
                errored += 1
                continue
            picked[key] = (path, s.id, s.epoch)
    samples = sorted(picked.values(), key=lambda t: (t[0].name, str(t[1]), t[2]))
    contexts, unusable = [], 0
    for path, sid, epoch in spaced(samples, episodes):
        sample = read_eval_log_sample(str(path), id=sid, epoch=epoch, resolve_attachments=True)
        events = {m: [] for m in MODELS}
        for ev in sample.events:
            if ev.event == "model":
                m = event_model(ev)
                events[m].append(ev)
        for model in MODELS:
            usable = []
            for i, ev in enumerate(events[model]):
                c = context_from_event(ev, model)
                if c is None:
                    unusable += 1
                else:
                    usable.append((i, c))
            chosen = usable[-1:] if model == "agent" else spaced(usable, sim_calls)
            for i, c in chosen:
                contexts.append({"id": f"{path.name}::{sid}::{epoch}::{model}::{i}", "log": str(path),
                                 "sample_id": sid, "epoch": epoch, "condition": conditions[str(path)],
                                 "event": i, **c})
    return contexts, {"samples": len(samples), "episodes": min(episodes, len(samples)), "errored_samples": errored,
                      "unusable_calls": unusable}


# --- scoring on a server ----------------------------------------------------------------------


def parse_prompt_logprobs(entries: list, tokens: list[int]) -> tuple[list, list]:
    """vLLM's prompt_logprobs (one dict per position, None first) as the log-prob of each
    recorded token and the server's top-1 token at each position."""
    if not isinstance(entries, list) or len(entries) != len(tokens):
        raise EquivError(f"prompt_logprobs has {len(entries) if isinstance(entries, list) else 'no'} positions "
                         f"for {len(tokens)} tokens")
    lp, top1 = [None], [None]
    for i in range(1, len(tokens)):
        d = entries[i] or {}
        forced = d.get(str(tokens[i]))
        if forced is None:
            raise EquivError(f"position {i}: no log-prob for its token {tokens[i]}")
        lp.append(float(forced["logprob"]))
        ranked = [int(k) for k, v in d.items() if v.get("rank") == 1]
        top1.append(min(ranked) if ranked else int(max(d, key=lambda k: d[k]["logprob"])))
    return lp, top1


def _transport():
    """The HTTP transport (None: the network). Tests replace it with a mocked server."""
    return None


async def _post(client, url: str, body: dict) -> dict:
    r = await client.post(url, json=body)
    if r.status_code != 200:
        raise EquivError(f"{url}: HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


async def _tokenize(client, server: Server, messages, tools, add_generation_prompt: bool) -> list[int]:
    body = {"model": server.model, "messages": messages, "add_generation_prompt": add_generation_prompt}
    if tools:
        body["tools"] = tools
    return (await _post(client, f"{server.root}/tokenize", body))["tokens"]


async def server_info(server: Server, transport=None) -> dict:
    import httpx

    async with httpx.AsyncClient(transport=transport, timeout=60) as client:
        models = (await client.get(f"{server.url.rstrip('/')}/models")).json().get("data") or []
        r = await client.get(f"{server.root}/version")
        version = r.json().get("version") if r.status_code == 200 else None
    return {"url": server.url, "served_model": server.model, "models": [m.get("id") for m in models],
            "max_model_len": models[0].get("max_model_len") if models else None, "version": version}


async def score(contexts: list[dict], servers: dict[str, Server], concurrency: int = 1, top_k: int = 1,
                max_tokens: int | None = None, transport=None, progress=None) -> list[dict]:
    """Score each context whose model has a server. Every context is tokenized first; then up
    to `concurrency` scoring requests per server are in flight at once. A failed context gives
    an error row, never an exception."""
    import httpx

    todo = [c for c in contexts if c["model"] in servers]
    rows: dict[str, dict] = {}
    prepared: list[tuple[dict, list[int], int | None]] = []
    async with httpx.AsyncClient(transport=transport, timeout=httpx.Timeout(3600, connect=30)) as client:
        for c in todo:   # tokenizing is cheap and off the GPU: one at a time
            server = servers[c["model"]]
            try:
                full = await _tokenize(client, server, c["messages"], c.get("tools"), False)
                prompt = await _tokenize(client, server, c["messages"][:-1], c.get("tools"), True)
            except Exception as exc:
                rows[c["id"]] = {"kind": "error", "id": c["id"], "model": c["model"], "stage": "tokenize",
                                 "error": f"{type(exc).__name__}: {exc}"[:500]}
                continue
            start = len(prompt) if full[:len(prompt)] == prompt else None
            prepared.append((c, full, start))

        sems = {m: asyncio.Semaphore(concurrency) for m in servers}

        async def one(c, full, start):
            server = servers[c["model"]]
            tokens = full[:max_tokens] if max_tokens else full
            body = {"model": server.model, "prompt": tokens, "max_tokens": 1, "temperature": 0,
                    "prompt_logprobs": top_k}
            async with sems[c["model"]]:
                t0 = time.monotonic()
                try:
                    out = await _post(client, f"{server.url.rstrip('/')}/completions", body)
                    lp, top1 = parse_prompt_logprobs(out["choices"][0].get("prompt_logprobs"), tokens)
                except Exception as exc:
                    rows[c["id"]] = {"kind": "error", "id": c["id"], "model": c["model"], "stage": "score",
                                     "error": f"{type(exc).__name__}: {exc}"[:500]}
                    return
            logged = c.get("prompt_tokens")
            rows[c["id"]] = {
                "kind": "score", "id": c["id"], "model": c["model"], "n_tokens": len(tokens),
                "tokens_sha256": sha256_tokens(tokens), "truncated": len(tokens) < len(full),
                "response_start": start if start is not None and start < len(tokens) else None,
                "prompt_tokens_logged": logged,
                "prompt_matches_log": None if logged is None or start is None else start == logged,
                "seconds": round(time.monotonic() - t0, 2), "logprob": lp, "top1": top1}
            if progress:
                progress(rows[c["id"]])

        await asyncio.gather(*(one(*p) for p in prepared))
    return [rows[c["id"]] for c in todo if c["id"] in rows]


# --- files --------------------------------------------------------------------------------------


def write_scores(path: Path, header: dict, rows: list[dict], force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise EquivError(f"{path} exists (use --force to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"kind": "header", "format": FORMAT, **header}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_scores(path: Path) -> tuple[dict, list[dict]]:
    lines = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if not lines or lines[0].get("kind") != "header":
        raise EquivError(f"{path}: not a score file (no header)")
    header = {k: v for k, v in lines[0].items() if k not in ("kind", "format")}
    return header, lines[1:]


def write_contexts(path: Path, contexts: list[dict], force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise EquivError(f"{path} exists (use --force to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(c) + "\n" for c in contexts))


def read_contexts(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


# --- comparing ----------------------------------------------------------------------------------


def _quantile(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def _stats(pairs: list[tuple[float, float, bool]]) -> dict:
    """pairs: (log-prob a, log-prob b, same top-1) per position."""
    diffs = [abs(a - b) for a, b, _ in pairs if math.isfinite(a) and math.isfinite(b)]
    return {"positions": len(pairs), "non_finite": len(pairs) - len(diffs),
            "mean_abs_dlogprob": statistics.fmean(diffs) if diffs else None,
            "median_abs_dlogprob": _quantile(diffs, 0.5), "p99_abs_dlogprob": _quantile(diffs, 0.99),
            "max_abs_dlogprob": max(diffs) if diffs else None,
            "top1_agreement": sum(s for _, _, s in pairs) / len(pairs) if pairs else None}


def gap(rows_a: list[dict], rows_b: list[dict]) -> dict[str, dict]:
    """Per model: how far b's numbers are from a's, over the contexts both scored on identical
    tokens. Every position after the first counts; `response` is the reply's own tokens."""
    a = {r["id"]: r for r in rows_a if r.get("kind") == "score"}
    b = {r["id"]: r for r in rows_b if r.get("kind") == "score"}
    ids = {r["id"]: r["model"] for r in [*rows_a, *rows_b] if "id" in r}
    out = {}
    for model in sorted(set(ids.values())):
        mine = sorted(i for i, m in ids.items() if m == model)
        allpos, reply, per = [], [], []
        missing, mismatch = [], []
        for i in mine:
            ra, rb = a.get(i), b.get(i)
            if ra is None or rb is None:
                missing.append(i)
                continue
            if ra["tokens_sha256"] != rb["tokens_sha256"]:
                mismatch.append(i)
                continue
            pairs = [(ra["logprob"][k], rb["logprob"][k], ra["top1"][k] == rb["top1"][k])
                     for k in range(1, ra["n_tokens"])]
            allpos += pairs
            start = ra.get("response_start")
            if start is not None and start == rb.get("response_start"):
                reply += pairs[start - 1:]
            s = _stats(pairs)
            per.append({"id": i, "positions": s["positions"], "mean_abs_dlogprob": s["mean_abs_dlogprob"],
                        "top1_agreement": s["top1_agreement"],
                        "_sum": sum(abs(x - y) for x, y, _ in pairs if math.isfinite(x) and math.isfinite(y)),
                        "_finite": s["positions"] - s["non_finite"], "_agree": sum(t for _, _, t in pairs)})
        out[model] = {"contexts": len(per), **_stats(allpos), "response": _stats(reply),
                      "missing": missing, "token_mismatch": mismatch, "per_context": per}
    return out


def _bootstrap(cross: list[dict], noise: list[dict], reps: int = 2000, seed: int = 20261102) -> dict | None:
    """95% interval of (cross - noise) for both metrics, resampling contexts (paired: the same
    contexts in both). Informational; the rule uses the point estimates."""
    n = {c["id"]: c for c in noise}
    paired = [(c, n[c["id"]]) for c in cross if c["id"] in n]
    if len(paired) < 2:
        return None
    rng = random.Random(seed)
    d_mean, d_top = [], []
    for _ in range(reps):
        pick = [paired[rng.randrange(len(paired))] for _ in paired]

        def pooled(side, key):
            den = sum(p[side]["_finite" if key == "_sum" else "positions"] for p in pick)
            return sum(p[side][key] for p in pick) / den if den else 0.0

        d_mean.append(pooled(0, "_sum") - pooled(1, "_sum"))
        d_top.append(pooled(0, "_agree") - pooled(1, "_agree"))
    d_mean.sort()
    d_top.sort()
    lo, hi = int(0.025 * reps), int(0.975 * reps) - 1
    return {"mean_abs_dlogprob": [d_mean[lo], d_mean[hi]], "top1_agreement": [d_top[lo], d_top[hi]]}


def verdict(cross: dict[str, dict], noise: dict[str, dict], models=None) -> dict:
    """The DRAFT pass rule (module docstring): per model, cross mean |d log-prob| <= noise's
    and cross top-1 agreement >= noise's. PASS only if every model passes on complete data."""
    models = sorted(models or set(cross) | set(noise))
    result, statuses = {}, []
    for m in models:
        c, n = cross.get(m), noise.get(m)
        problems = []
        if c is None or n is None or not c["contexts"] or not n["contexts"]:
            problems.append("no contexts compared")
        else:
            if c["token_mismatch"] or n["token_mismatch"]:
                problems.append(f"different token IDs in {len(c['token_mismatch']) + len(n['token_mismatch'])} context(s)")
            if c["missing"] or n["missing"]:
                problems.append(f"{len(c['missing']) + len(n['missing'])} context(s) missing or errored")
            if {x["id"] for x in c["per_context"]} != {x["id"] for x in n["per_context"]}:
                problems.append("the two comparisons cover different contexts")
        if problems and any("different token" in p for p in problems):
            status = "FAIL"
        elif problems:
            status = "INCOMPLETE"
        else:
            checks = {"mean_abs_dlogprob": c["mean_abs_dlogprob"] <= n["mean_abs_dlogprob"],
                      "top1_agreement": c["top1_agreement"] >= n["top1_agreement"]}
            status = "PASS" if all(checks.values()) else "FAIL"
        entry = {"status": status, "pass": status == "PASS", "problems": problems}
        if not problems:
            entry["checks"] = checks
            entry["cross"] = {k: c[k] for k in ("mean_abs_dlogprob", "top1_agreement", "positions")}
            entry["noise"] = {k: n[k] for k in ("mean_abs_dlogprob", "top1_agreement", "positions")}
            entry["ratio_mean_abs_dlogprob"] = (c["mean_abs_dlogprob"] / n["mean_abs_dlogprob"]
                                                if n["mean_abs_dlogprob"] else None)
            entry["bootstrap_cross_minus_noise"] = _bootstrap(c["per_context"], n["per_context"])
        result[m] = entry
        statuses.append(status)
    if not statuses or "INCOMPLETE" in statuses and "FAIL" not in statuses:
        status = "INCOMPLETE"
    elif "FAIL" in statuses:
        status = "FAIL"
    else:
        status = "PASS"
    return {"status": status, "models": result}


# --- command line -------------------------------------------------------------------------------


def _fmt(x, nd=4):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def format_gap(name: str, g: dict) -> str:
    lines = [f"{name}:"]
    for m, s in g.items():
        r = s["response"]
        lines.append(f"  {m}: {s['contexts']} contexts, {s['positions']} positions; |d logprob| mean "
                     f"{_fmt(s['mean_abs_dlogprob'])}, median {_fmt(s['median_abs_dlogprob'])}, p99 "
                     f"{_fmt(s['p99_abs_dlogprob'])}, max {_fmt(s['max_abs_dlogprob'])}; top-1 agreement "
                     f"{_fmt(s['top1_agreement'])}; reply only ({r['positions']} positions): mean "
                     f"{_fmt(r['mean_abs_dlogprob'])}, top-1 {_fmt(r['top1_agreement'])}")
        if s["missing"]:
            lines.append(f"    missing or errored: {len(s['missing'])}")
        if s["token_mismatch"]:
            lines.append(f"    DIFFERENT TOKEN IDS: {len(s['token_mismatch'])} (the backends render differently)")
        if s["non_finite"]:
            lines.append(f"    non-finite log-probs left out: {s['non_finite']}")
    return "\n".join(lines)


def _check_headers(*pairs) -> list[str]:
    """Refuse files scored on different contexts or settings; return warnings."""
    (pa, ha), *rest = pairs
    warnings = []
    for p, h in rest:
        for key in ("contexts_sha256", "top_k", "max_tokens"):
            if h.get(key) != ha.get(key):
                what = "different contexts" if key == "contexts_sha256" else f"different {key}"
                raise SystemExit(f"{pa} and {p} were scored on {what} ({ha.get(key)} vs {h.get(key)})")
    versions = {p: {m: (s or {}).get("version") for m, s in (h.get("servers") or {}).items()} for p, h in pairs}
    if len({json.dumps(v, sort_keys=True) for v in versions.values()}) > 1:
        warnings.append(f"vLLM versions differ: {versions}")
    return warnings


def main_compare(reference: Path, candidate: Path, noise: Path | None, as_json: bool = False) -> dict:
    ha, ra = read_scores(reference)
    hb, rb = read_scores(candidate)
    files = [(reference, ha), (candidate, hb)]
    if noise is not None:
        hn, rn = read_scores(noise)
        files.append((noise, hn))
    warnings = _check_headers(*files)
    cross = gap(ra, rb)
    result = {"reference": str(reference), "candidate": str(candidate), "gap": cross, "warnings": warnings}
    if noise is not None:
        floor = gap(ra, rn)
        result["noise"] = str(noise)
        result["noise_gap"] = floor
        result["verdict"] = verdict(cross, floor)
    if as_json:
        print(json.dumps(result, indent=1, default=str))
        return result
    for w in warnings:
        print(f"WARNING: {w}")
    print(format_gap(f"gap {reference} -> {candidate} (concurrency {ha.get('concurrency')} vs "
                     f"{hb.get('concurrency')})", cross))
    if noise is not None:
        print(format_gap(f"noise floor {reference} -> {noise} (concurrency {ha.get('concurrency')} vs "
                         f"{hn.get('concurrency')})", floor))
        v = result["verdict"]
        print(f"\nPass rule (DRAFT): the gap is no bigger than the noise floor, for every model -> {v['status']}")
        for m, e in v["models"].items():
            if e["problems"]:
                print(f"  {m}: {e['status']}: " + "; ".join(e["problems"]))
            else:
                b = e["bootstrap_cross_minus_noise"] or {}
                print(f"  {m}: {e['status']}: mean |d logprob| {_fmt(e['cross']['mean_abs_dlogprob'])} vs noise "
                      f"{_fmt(e['noise']['mean_abs_dlogprob'])} (ratio {_fmt(e['ratio_mean_abs_dlogprob'], 2)}); "
                      f"top-1 {_fmt(e['cross']['top1_agreement'])} vs {_fmt(e['noise']['top1_agreement'])}; "
                      f"95% bootstrap of the differences {b or '-'}")
    return result


def add_parser(sub) -> None:
    p = sub.add_parser("equiv", help="backend equivalence: teacher-force saved episodes on two vLLM backends")
    s = p.add_subparsers(dest="equiv_cmd", required=True)
    c = s.add_parser("contexts", help="contexts (both models' calls) from .eval logs (offline)")
    c.add_argument("logs", nargs="+", type=Path, help="log directories or .eval files")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--episodes", type=int, default=12, help="episodes, spread over the completed samples")
    c.add_argument("--sim-calls", type=int, default=2, help="simulator calls per episode (the last one kept)")
    c.add_argument("--force", action="store_true")
    sc = s.add_parser("score", help="teacher-force the contexts on running servers (no sampling)")
    sc.add_argument("contexts", type=Path)
    sc.add_argument("--agent-url", default=None, help="e.g. http://localhost:18000/v1")
    sc.add_argument("--simulator-url", default=None, help="e.g. http://localhost:18001/v1")
    sc.add_argument("--agent-model", default="agent", help="served model name (serve.sh: agent)")
    sc.add_argument("--simulator-model", default="simulator", help="served model name (serve.sh: simulator)")
    sc.add_argument("--concurrency", type=int, nargs="+", default=[1],
                    help="requests in flight per server; one pass and one file per value (e.g. 1 12)")
    sc.add_argument("--top-k", type=int, default=1, help="prompt_logprobs per position (1: the top-1 is enough)")
    sc.add_argument("--max-tokens", type=int, default=None, help="cut every context to its first N tokens")
    sc.add_argument("--out", required=True, help="prefix: writes <prefix>-c<concurrency>.jsonl")
    sc.add_argument("--label", default=None, help='e.g. "G4, session 6"')
    sc.add_argument("--force", action="store_true")
    cp = s.add_parser("compare", help="gap between two score files; with --noise, the DRAFT pass rule")
    cp.add_argument("reference", type=Path)
    cp.add_argument("candidate", type=Path)
    cp.add_argument("--noise", type=Path, default=None, help="the reference server's own file at higher concurrency")
    cp.add_argument("--json", action="store_true")


def main(args) -> None:
    if args.equiv_cmd == "contexts":
        try:
            contexts, notes = extract_contexts(args.logs, episodes=args.episodes, sim_calls=args.sim_calls)
            write_contexts(args.out, contexts, force=args.force)
        except EquivError as exc:
            raise SystemExit(str(exc)) from exc
        by = {m: sum(c["model"] == m for c in contexts) for m in MODELS}
        print(f"{len(contexts)} contexts ({by['agent']} agent, {by['simulator']} simulator) from "
              f"{notes['episodes']} of {notes['samples']} completed episodes; {notes['errored_samples']} errored "
              f"samples left out, {notes['unusable_calls']} unusable calls; wrote {args.out}")
    elif args.equiv_cmd == "score":
        servers = {}
        if args.agent_url:
            servers["agent"] = Server(args.agent_url, args.agent_model)
        if args.simulator_url:
            servers["simulator"] = Server(args.simulator_url, args.simulator_model)
        if not servers:
            raise SystemExit("give --agent-url, --simulator-url or both")
        contexts = read_contexts(args.contexts)
        digest = hashlib.sha256(Path(args.contexts).read_bytes()).hexdigest()
        info = {m: asyncio.run(server_info(s, transport=_transport())) for m, s in servers.items()}
        for conc in args.concurrency:
            out = Path(f"{args.out}-c{conc}.jsonl")
            if out.exists() and not args.force:
                raise SystemExit(f"{out} exists (use --force to replace it)")
            started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            rows = asyncio.run(score(contexts, servers, concurrency=conc, top_k=args.top_k,
                                     max_tokens=args.max_tokens, transport=_transport()))
            header = {"label": args.label, "contexts": str(args.contexts), "contexts_sha256": digest,
                      "concurrency": conc, "top_k": args.top_k, "max_tokens": args.max_tokens, "servers": info,
                      "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            write_scores(out, header, rows, force=args.force)
            errors = [r for r in rows if r["kind"] == "error"]
            flagged = sum(r.get("prompt_matches_log") is False for r in rows)
            print(f"concurrency {conc}: {len(rows) - len(errors)} scored, {len(errors)} errors"
                  + (f", {flagged} whose prompt length differs from the episode's" if flagged else "")
                  + f"; wrote {out}")
            for e in errors[:5]:
                print(f"  {e['id']} ({e['stage']}): {e['error']}")
    elif args.equiv_cmd == "compare":
        main_compare(args.reference, args.candidate, args.noise, as_json=args.json)
