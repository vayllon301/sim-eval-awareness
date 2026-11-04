"""A run's episodes for the reader: the Inspect adapter and the export of a log directory. Reference:
docs/INTERP.md, "Inspect episodes". Laptop only: this module needs Inspect, which the GPU venv hasn't
got, so an exported episode is plain JSON that `reader.load_episode` reads there.

The pilot's logs hold no token IDs: the server sends them only when asked (`return_token_ids`), and
the rawchat provider doesn't ask. They hold what rebuilds them, the exact request of every agent call
(`log_model_api`). So an episode comes from those requests, not from `sample.messages`:

- the last request's messages are everything the agent read, in the form the server received them;
- the last reply comes from `sample.messages`, as the provider would have replayed it;
- the tools come from the request. Tool outputs after the last call were never read, so they are
  dropped and counted, and failed calls (a connection error) are counted, never read.

Every check on the rebuild stays in the episode's metadata: the server's prompt and completion counts
per call, whether each request extends the one before it, and whether each earlier reply came back as
the provider's replay would send it. An episode with a failed check is still exported; the reading
records it (reader.prompt_checks, readrun.py).

`export_run` writes one `<uuid>.json` per episode and `manifest.json`. Which episodes: what `simtest
summarize` counts (coding.analysis.load_run): a sample run once, from its newest log. Scored and
flagged episodes are exported. A failed attempt has no model events (attempts/ keeps only messages
and the store), so it is listed as `not_read`, and an adapter error as `adapter_error`, with its
reason: every episode is reported, none dropped.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .meta import file_sha256, git_commit, now
from .reader import Episode, _arguments, save_episode

FORMAT = "simtest-interp-episodes"
FORMAT_VERSION = 1
CLEAN_STOPS = ("stop", "tool_calls", "tool_call")   # Inspect's names for a reply that wasn't cut


class AdapterError(ValueError):
    pass


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for p in content or []:
        if not (isinstance(p, dict) and p.get("type") == "text"):
            raise AdapterError(f"unsupported content in a request: {p!r:.80}")
        parts.append(p.get("text") or "")
    return "".join(parts)


def _message(m: dict) -> dict:
    """A request's message as the chat template takes it (the server's own renderer does this)."""
    role = m["role"]
    text = _text(m.get("content"))
    if text.startswith("attachment://"):
        raise AdapterError("the request still holds an attachment: read the sample with resolve_attachments=True")
    if role != "assistant":
        return {"role": role, "content": text}
    out = {"role": role, "content": text, "reasoning_content": m.get("reasoning") or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = [{"type": "function", "function": {"name": c["function"]["name"],
                                                               "arguments": _arguments(c["function"]["arguments"])}}
                             for c in m["tool_calls"]]
    return out


def _replayed(message) -> dict:
    """An assistant message of the sample as the rawchat provider sends it back."""
    from ..coding.provider import reasoning_text

    out = {"role": "assistant", "content": message.text, "reasoning_content": reasoning_text(message)}
    if message.tool_calls:
        out["tool_calls"] = [{"type": "function", "function": {"name": c.function, "arguments": _arguments(c.arguments)}}
                             for c in message.tool_calls]
    return out


def _is_agent_event(e) -> bool:
    return getattr(e, "event", None) == "model" and getattr(e, "role", None) is None


def agent_calls(sample) -> list:
    """The agent's model calls that returned, in order."""
    return [e for e in sample.events if _is_agent_event(e) and not e.error and e.call is not None
            and e.output is not None and e.output.usage is not None and "messages" in e.call.request]


def last_request(sample) -> dict:
    """The agent's last request as the server received it."""
    calls = agent_calls(sample)
    if not calls:
        raise AdapterError("no agent call with a saved request")
    return calls[-1].call.request


def from_inspect(sample, *, source: str | None = None) -> Episode:
    """One sample read with `resolve_attachments=True`. Module docstring."""
    calls = agent_calls(sample)
    if not calls:
        raise AdapterError("no agent call with a saved request")
    requests = [e.call.request for e in calls]
    last = requests[-1]
    msgs = [_message(m) for m in last["messages"]]
    replies = [m for m in sample.messages if m.role == "assistant"]
    if len(replies) < len(calls):
        raise AdapterError(f"{len(calls)} agent calls but {len(replies)} replies in the transcript")
    replies = replies[:len(calls)]
    prior = [m for m in msgs if m["role"] == "assistant"]
    msgs.append(_replayed(replies[-1]))
    turns: list[int] = []
    k = -1
    for m in msgs:
        k += m["role"] == "assistant"
        turns.append(k)
    seen_after = sample.messages[sample.messages.index(replies[-1]) + 1:]
    stop = calls[-1].output.stop_reason
    meta = {"adapter": "inspect", "source": source, "uuid": sample.uuid, "sample_id": sample.id, "epoch": sample.epoch,
            "n_turns": len(calls), "prompt_tokens": [e.output.usage.input_tokens for e in calls],
            "completion_tokens": [e.output.usage.output_tokens for e in calls],
            "failed_agent_calls": sum(_is_agent_event(e) and bool(e.error) for e in sample.events),
            "unread_outputs_dropped": sum(m.role == "tool" for m in seen_after),
            "requests_are_prefixes": all(a["messages"] == b["messages"][:len(a["messages"])]
                                         for a, b in zip(requests, requests[1:])),
            "replay_differs": [k for k, (m, r) in enumerate(zip(prior, replies)) if m != _replayed(r)],
            "partial_last_turn": stop not in CLEAN_STOPS}
    return Episode(messages=msgs, tools=last.get("tools") or [], turns=turns, meta=meta)


def _read_sample(e: dict):
    from inspect_ai.log import read_eval_log_sample

    return read_eval_log_sample(e["log_path"], id=e["sample_id"], epoch=e["epoch"], resolve_attachments=True)


def _load_run(log_dir: Path):
    from ..coding import analysis as A

    return A.load_run(log_dir)


def _scored(e: dict) -> bool:
    from ..coding import analysis as A

    return not A.is_excluded(e)


def export_run(log_dir: str | Path, out: str | Path, *, load: Callable = _load_run, read: Callable = _read_sample) -> dict:
    """Writes every readable episode of `log_dir` to `out`, and returns the manifest (also in
    `out/manifest.json`). A second run on the same directory replaces it."""
    log_dir, out = Path(log_dir), Path(out)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()).get("run") != log_dir.name:
        raise AdapterError(f"{out} holds another run's episodes")
    episodes, design = load(log_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in sorted(episodes, key=lambda x: (x["condition"], x["scenario"], x["epoch"], str(x["uuid"]))):
        row = {k: e.get(k) for k in ("uuid", "condition", "simulator", "scenario", "variant", "epoch", "end_reason",
                                     "log", "sample_id")}
        row["key"] = f"{log_dir.name}::{e['uuid']}" if e["uuid"] else None
        row["scored"] = _scored(e)
        if e["end_reason"] == "error" or not e["uuid"] or not e.get("log_path"):
            row.update(status="not_read", reason=(e.get("error") or "no log record")[:300])
            rows.append(row)
            continue
        try:
            sample = read(e)
            ep = from_inspect(sample, source=str(e["log_path"]))
            wire = {k: v for k, v in last_request(sample).items() if k in ("model", "messages", "tools")}
        except (AdapterError, ValueError) as exc:
            row.update(status="adapter_error", reason=str(exc)[:300])
            rows.append(row)
            continue
        path = out / f"{e['uuid']}.json"
        save_episode(ep, path, wire=wire)
        row.update(status="exported", sha256=file_sha256(path), n_turns=ep.meta["n_turns"],
                   tokens=ep.meta["prompt_tokens"][-1] + ep.meta["completion_tokens"][-1])
        rows.append(row)
    manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "run": log_dir.name, "log_dir": str(log_dir),
                "design": design, "created": now(), "git_commit": git_commit(),
                "counts": dict(sorted(Counter(r["status"] for r in rows).items())), "episodes": rows}
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest
