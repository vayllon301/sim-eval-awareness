"""Reading a run: every exported episode (episodes.py) through one backend, resumable. Reference:
docs/INTERP.md, "Reading a run". Runs in the GPU venv, so nothing here needs Inspect.

Two parts, each its own file per episode, so each can run on its own engine and session:
- `proj`: one forward pass. `<uuid>.npz` and `<uuid>.summary.json` (reader.Reading): projections onto
  every vector at every chosen layer, per token; each turn's prompt end; SAE top-k latents when asked.
  Never raw activations.
- `logit`: the logit monitor, `<uuid>.logit.json`. It needs a backend with `continuation_scan`: on
  vLLM's prompt log-probs the pilot would take about 430M tokens against 2.3M (docs/INTERP.md), so
  that path is refused unless asked for.

Every output names its episode (uuid and the sha256 of the exported file), the vector file's sha256,
the phrases' sha256 and the backend. `read.json` holds the run's settings: a second call with other
settings is refused, since the directory would mix two readings; give it another directory.

Resuming: a finished file is never redone, and a file only gets its name once complete. The longest
episode goes first, so a memory failure shows in the first minutes. A failed episode is listed and
tried again on the next call; 3 failures in a row stop the run (the engine is probably gone). The
budget stops before an episode starts, never during one.

The token IDs. The pass uses the rendering of `reader.render`, checked against what the server saw:
- `<uuid>.ids.json`, from `server_ids`: the IDs of the server's own renderer for the episode's last
  request (vLLM's /tokenize, the same renderer as chat completions, so no GPU work and no change to
  any episode). Every earlier prompt is a prefix of that one. Different IDs refuse the episode.
- Without it, the check is lengths only: each turn's rendered prompt against the server's count.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .meta import file_sha256, git_commit, now
from .reader import Reading, completion_checks, load_episode, read, render

PARTS = ("proj", "logit")
MAX_FAILURES = 3
# The served context (serve.sh): vllm_backend.ENGINE's 16K is the smoke test's, and session 5's agent
# prompts reach 80K tokens.
READ_ENGINE = {"max_model_len": 131072}


class ReadError(ValueError):
    pass


def make_backend(engine: str, model: str | Path, *, quantization: str | None = None):
    """vllm: the reference engine (vllm-lens), as gpu_smoke.py's lens step. hf: the same FP8 files on
    transformers. mlx: the laptop's weights, development only."""
    if engine == "vllm":
        os.environ.pop("VLLM_LENS_DISABLE", None)
        os.environ.setdefault("VLLM_LENS_STRICT_COMPATIBILITY", "1")
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        from .vllm_backend import VLLMBackend

        return VLLMBackend(model, quantization=quantization, engine=READ_ENGINE)
    if engine == "hf":
        from .hf_backend import HFBackend

        return HFBackend(model)
    if engine == "mlx":
        from .mlx_backend import MLXBackend

        return MLXBackend(model)
    raise ReadError(f"unknown engine {engine!r}")


def transfer_sample(rows: Sequence[dict], per_condition: int) -> list[str]:
    """The episodes of the SAE transfer check, fixed before any is read: per condition, the first
    `per_condition` scored episodes by epoch and scenario, so each condition gives a tempting and a
    benign episode of one world."""
    out = []
    for c in sorted({r["condition"] for r in rows}):
        eps = [r for r in rows if r["condition"] == c and r["scored"] and r["status"] == "exported"]
        out += [r["uuid"] for r in sorted(eps, key=lambda r: (r["epoch"], r["scenario"]))[:per_condition]]
    return out


def gpu_used_mib() -> int | None:
    """Memory in use on the busiest GPU, or None on a machine without nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    vals = [int(x) for x in out.stdout.split() if x.isdigit()]
    return max(vals) if out.returncode == 0 and vals else None


def _rows(episodes_dir: Path) -> list[dict]:
    path = episodes_dir / "manifest.json"
    if not path.exists():
        raise ReadError(f"no manifest in {episodes_dir}: run `simtest interp export` first")
    rows = [r for r in json.loads(path.read_text())["episodes"] if r["status"] == "exported"]
    return sorted(rows, key=lambda r: (-(r.get("tokens") or 0), r["uuid"]))


def _server_ids(episodes_dir: Path, uuid: str) -> list[int] | None:
    p = episodes_dir / f"{uuid}.ids.json"
    return json.loads(p.read_text())["ids"] if p.exists() else None


def _check_ids(ids, server: list[int] | None) -> dict | None:
    """The check against the server's IDs, or None when there are none."""
    if server is None:
        return None
    mine = [int(i) for i in ids[:len(server)]]
    at = next((i for i, (a, b) in enumerate(zip(mine, server)) if a != b), None)
    if at is None and len(mine) != len(server):
        at = len(mine)
    if at is not None:
        raise ReadError(f"the rendered IDs and the server's first differ at token {at}: this is not what the agent saw")
    return {"source": "server", "n": len(server), "equal": True, "first_difference": None}


def _mismatches(checks: Sequence[dict]) -> list[int]:
    return [c["turn"] for c in checks if c["recorded"] != c["rendered"]]


def _write_json(path: Path, obj: Any, **kw) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, **kw))
    os.replace(tmp, path)


def _read_proj(backend, ep, row: dict, episodes_dir: Path, out_dir: Path, vectors, layers, names, saes, vectors_path,
               allow_other_model) -> int:
    server = _check_ids(render(ep, backend.tokenizer).ids, _server_ids(episodes_dir, row["uuid"]))
    r = read(backend, ep, vectors, layers, names=names, saes=saes, vectors_path=str(vectors_path) if vectors_path else None,
             allow_other_model=allow_other_model)
    bad = _mismatches(r.meta["prompt_checks"])
    r.meta["episode_sha256"] = row["sha256"]
    r.meta["ids_check"] = server or {"source": "lengths", "n": len(r.meta["prompt_checks"]), "equal": not bad,
                                     "first_difference": bad[0] if bad else None}
    r.meta["prompt_mismatches"] = bad
    r.meta["completion_checks"] = completion_checks(r.token_ids, r.prompt_end, ep.meta.get("completion_tokens"),
                                                    backend.eos_token_ids)
    tmp = out_dir / f"{row['uuid']}.tmp.npz"
    r.save(tmp)
    os.replace(out_dir / f"{row['uuid']}.tmp.summary.json", out_dir / f"{row['uuid']}.summary.json")
    os.replace(tmp, out_dir / f"{row['uuid']}.npz")
    return len(r.token_ids)


def _read_logit(backend, ep, row: dict, episodes_dir: Path, out_dir: Path, phrases, phrases_path, phrase_turns) -> int:
    from .monitors import logit_monitor

    rd = render(ep, backend.tokenizer)
    _check_ids(rd.ids, _server_ids(episodes_dir, row["uuid"]))
    mon = logit_monitor(backend, rd.ids, rd.role, rd.turn, phrases["groups"], turns=phrase_turns)
    _write_json(out_dir / f"{row['uuid']}.logit.json",
                {"episode": ep.meta, "episode_sha256": row["sha256"], "backend": backend.info, "created": now(),
                 "git_commit": git_commit(), "logit_monitor": {
                     "phrases_path": str(phrases_path) if phrases_path else None, "status": phrases.get("status"),
                     "phrases_sha256": _sha(phrases_path), **mon}}, sort_keys=True)
    return len(rd.ids)


def _sha(path) -> str | None:
    return file_sha256(path) if path and Path(path).exists() else None


def _settings(backend, parts, vectors, vectors_path, layers, names, saes, phrases, phrases_path, phrase_turns) -> dict:
    out: dict[str, Any] = {"parts": sorted(parts), "fingerprint": backend.info.get("fingerprint"),
                           "engine": backend.info.get("engine")}
    if "proj" in parts:
        out.update(vectors_sha256=_sha(vectors_path), layers=sorted(int(L) for L in layers),
                   names=sorted(names or vectors.names),
                   sae={str(L): getattr(s, "info", {}) for L, s in sorted((saes or {}).items())})
    if "logit" in parts:
        out.update(phrases=phrases["groups"], phrase_turns=phrase_turns)
    return out


def _output(out_dir: Path, uuid: str, part: str) -> Path:
    return out_dir / f"{uuid}.npz" if part == "proj" else out_dir / f"{uuid}.logit.json"


def read_run(backend, episodes_dir: str | Path, out_dir: str | Path, *, parts: Sequence[str] = ("proj",), vectors=None,
             vectors_path: str | Path | None = None, layers: Sequence[int] = (), names: Sequence[str] | None = None,
             saes: Mapping[int, Any] | None = None, phrases: dict | None = None, phrases_path: str | Path | None = None,
             phrase_turns: Sequence[int] | None = None, allow_other_model: bool = False, budget_s: float | None = None,
             limit: int | None = None, require_scan: bool = False, clock: Callable[[], float] = time.monotonic) -> dict:
    """Module docstring. Returns the status, also in `out_dir/status.json`."""
    from .monitors import scan_path

    episodes_dir, out_dir = Path(episodes_dir), Path(out_dir)
    if not parts or set(parts) - set(PARTS):
        raise ReadError(f"parts are {PARTS}, got {list(parts)}")
    if "proj" in parts and vectors is None:
        raise ReadError("the proj part needs a vector file")
    if "logit" in parts:
        if phrases is None:
            raise ReadError("the logit part needs a phrase file")
        if require_scan and scan_path(backend) != "continuation_scan":
            raise ReadError(f"this backend has no continuation_scan (it would use {scan_path(backend)}): one pass over the "
                            "prefix per turn and phrase. Use the HF backend for the logit part, or allow it explicitly")
    rows = _rows(episodes_dir)
    settings = _settings(backend, parts, vectors, vectors_path, layers, names, saes, phrases, phrases_path, phrase_turns)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = out_dir / "read.json"
    if state.exists() and json.loads(state.read_text()) != settings:
        raise ReadError(f"{out_dir} holds readings made with other settings: give another directory")
    _write_json(state, settings, indent=1, sort_keys=True)
    t0 = clock()
    failed: list[dict] = []
    stopped, tried, in_a_row = None, 0, 0
    for row, part in ((r, p) for r in rows for p in parts):
        if _output(out_dir, row["uuid"], part).exists():
            continue
        if limit is not None and tried >= limit:
            stopped = "limit"
            break
        if budget_s is not None and clock() - t0 >= budget_s:
            stopped = "budget"
            break
        if in_a_row >= MAX_FAILURES:
            stopped = f"{MAX_FAILURES} failures in a row"
            break
        tried += 1
        started = clock()
        try:
            ep = load_episode(episodes_dir / f"{row['uuid']}.json")
            if part == "proj":
                n = _read_proj(backend, ep, row, episodes_dir, out_dir, vectors, list(layers), names, saes, vectors_path,
                               allow_other_model)
            else:
                n = _read_logit(backend, ep, row, episodes_dir, out_dir, phrases, phrases_path, phrase_turns)
        except Exception as exc:
            failed.append({"uuid": row["uuid"], "part": part, "error": f"{type(exc).__name__}: {exc}"[:500]})
            in_a_row += 1
            continue
        in_a_row = 0
        with (out_dir / "timings.jsonl").open("a") as f:
            f.write(json.dumps({"uuid": row["uuid"], "part": part, "tokens": n, "seconds": round(clock() - started, 3),
                                "gpu_used_mib": gpu_used_mib(), "engine": backend.info.get("engine"), "at": now()}) + "\n")
    done = {p: sum(_output(out_dir, r["uuid"], p).exists() for r in rows) for p in parts}
    mismatches = {}
    for r in rows:
        p = out_dir / f"{r['uuid']}.summary.json"
        if p.exists() and (bad := json.loads(p.read_text()).get("prompt_mismatches")):
            mismatches[r["uuid"]] = bad
    status = {"episodes": len(rows), "parts": list(parts), "done": done, "failed": failed, "stopped": stopped,
              "complete": all(v == len(rows) for v in done.values()), "prompt_mismatches": mismatches,
              "backend": backend.info.get("engine"), "updated": now(), "git_commit": git_commit()}
    _write_json(out_dir / "status.json", status, indent=1, sort_keys=True)
    return status


def _post(url: str, body: dict) -> dict:
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def server_ids(episodes_dir: str | Path, url: str, *, post: Callable = _post) -> dict:
    """Asks the agent's server for the token IDs of each episode's last request, while it is up:
    `<uuid>.ids.json`. No forward pass and no change to any episode. Fetched once."""
    episodes_dir = Path(episodes_dir)
    base = url.rstrip("/").removesuffix("/v1")
    fetched = kept = 0
    for row in _rows(episodes_dir):
        out = episodes_dir / f"{row['uuid']}.ids.json"
        if out.exists():
            kept += 1
            continue
        wire = json.loads((episodes_dir / f"{row['uuid']}.json").read_text())["wire"]
        res = post(f"{base}/tokenize", {"model": wire["model"], "messages": wire["messages"], "tools": wire["tools"],
                                        "add_generation_prompt": True})
        if len(res["tokens"]) != res["count"]:
            raise ReadError(f"{row['uuid']}: the server returned {len(res['tokens'])} tokens but counted {res['count']}")
        _write_json(out, {"ids": res["tokens"], "count": res["count"], "url": base, "created": now()})
        fetched += 1
    return {"fetched": fetched, "kept": kept}
