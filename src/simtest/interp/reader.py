"""The post-hoc reader: one forward pass over an episode, per-token projections onto
vectors, a role for every token. Reference: docs/INTERP.md, "Reader".

1. An adapter turns an episode record into chat messages and tools (`Episode`). Only
   adapters know a record format; everything after them is format-free. Today:
   `from_colab_probe` (explore/colab_probe/episode.py). The N1 Inspect format gets its
   own adapter.
2. `render` renders the messages with the model's own chat template (tools included,
   earlier reasoning kept, as the server rendered them for the next turn), tokenizes the
   text, and labels every token with a role and the agent turn it belongs to. Segments
   are located by rendering once with a unique placeholder in place of each text, so the
   labels don't depend on how a template lays a conversation out; the result is checked
   against the template's own rendering.
3. `read` runs one forward pass (no logits kept) and stores, per token, the projection of
   the residual stream onto each unit vector at each chosen layer, and optionally the
   top-k latents of an SAE. Raw activations are never stored. It also stores, per agent
   turn, the position of the last token of the prompt the server saw for that turn
   (`prompt_end`), where a reading before the agent writes anything is taken. Given
   phrases, it adds the logit monitor (monitors.py): their log-probability at the start
   of each turn's reasoning.

Roles: template (special tokens and template text), system (the system turn: the
template's tools block, the reasoning-effort sentence, the system prompt), user,
reasoning (the agent's chain of thought), content (the agent's visible text), tool_call
(the agent's tool-call argument values), tool_output (what the tools returned). The
agent's own tokens are reasoning, content and tool_call. Emotion vectors are local:
readings on tool_output tokens reflect what the agent reads, not its own state.

Causal attention makes one pass over the final conversation equal to the passes the
server made turn by turn, as long as each turn's prompt is a prefix of the next one
(true for Qwen3.8 with preserve_thinking on, its default). `prompt_checks` compares the
rendered prompt lengths with the token counts the server reported, and `prompt_ends`
checks that each turn's rendered prompt is a prefix of the full rendering.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .backend import Backend
from .meta import bytes_sha256, file_sha256, git_commit, now
from .vectors import VectorSet

ROLES = ("template", "system", "user", "reasoning", "content", "tool_call", "tool_output")
R = {name: i for i, name in enumerate(ROLES)}
AGENT_ROLES = ("reasoning", "content", "tool_call")
GROUPS = {"agent": AGENT_ROLES, "reasoning": ("reasoning",), "content+tool_call": ("content", "tool_call"),
          "tool_output": ("tool_output",), "system": ("system",), "user": ("user",), "template": ("template",)}
PLACEHOLDER = "@@SEG{:05d}@@"
PLACEHOLDER_RE = re.compile(r"@@SEG(\d{5})@@")


class RenderError(ValueError):
    pass


@dataclass
class Episode:
    messages: list[dict]
    tools: list[dict]
    turns: list[int]  # per message: -1 for the opening prompt, k for agent turn k and its tool outputs
    meta: dict = field(default_factory=dict)
    template_kwargs: dict = field(default_factory=dict)  # e.g. {} = the template's defaults (xhigh)


# ---------------------------------------------------------------- adapters


def _arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError as e:
        raise ValueError(f"tool-call arguments are not JSON ({e}); the chat template can't render them") from None
    if not isinstance(args, dict):
        raise ValueError("tool-call arguments are not a JSON object; the chat template can't render them")
    return args


def from_colab_probe(rows: Sequence[dict], *, system: str, ticket: str, tools: list[dict],
                     source: str | None = None, drop_unread_outputs: bool = True) -> Episode:
    """explore/colab_probe/episode.py records: one row per agent turn, then a final row.

    The record doesn't store the system prompt, the ticket or the tools: pass the ones the
    episode ran with (explore/interp/read_episode.py takes them from episode.py).
    Rows without `tool_calls` (older records) use the names and arguments in `calls`. The
    outputs of the last turn were never read by the agent (no turn followed), so they are
    dropped unless drop_unread_outputs is False; the count is kept in meta."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": ticket}]
    turns = [-1, -1]
    steps = [r for r in rows if not r.get("final")]
    final = next((r for r in rows if r.get("final")), None)
    dropped = 0
    for k, r in enumerate(steps):
        if r.get("step", k) != k:
            raise ValueError(f"row {k} has step {r.get('step')}; rows must be consecutive turns")
        spec = r.get("tool_calls")
        if spec is None:
            spec = [{"name": c["name"], "arguments": c["arguments"]} for c in r.get("calls", [])]
        a = {"role": "assistant", "content": r.get("content") or "", "reasoning_content": r.get("reasoning") or ""}
        if spec:
            a["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": _arguments(c["arguments"])}}
                               for c in spec]
        msgs.append(a)
        turns.append(k)
        if k == len(steps) - 1 and drop_unread_outputs:
            dropped = len(r.get("calls", []))
            continue
        for c in r.get("calls", []):
            msgs.append({"role": "tool", "content": c["output"]})
            turns.append(k)
    meta = {"adapter": "colab_probe", "source": source, "n_turns": len(steps),
            "outcome": (final or {}).get("outcome"), "partial_last_turn": bool(steps and steps[-1].get("partial")),
            "unread_outputs_dropped": dropped, "prompt_tokens": [r.get("prompt_tokens") for r in steps]}
    return Episode(messages=msgs, tools=tools, turns=turns, meta=meta)


def _assignments(path: Path, names: set[str], ns: dict) -> dict:
    """Run only the top-level assignments to `names` in a script (in file order)."""
    import ast

    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
            if targets & names:
                exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    return ns


def colab_probe_prompts(explore_dir: str | Path) -> dict:
    """The system prompt, ticket and tools of explore/colab_probe/episode.py, read from its
    source without running it (the script reads sys.argv at import)."""
    from types import SimpleNamespace

    d = Path(explore_dir)
    sim = _assignments(d / "bakeoff_simulate.py", {"REPO", "TOOL"}, {})
    ns = _assignments(d / "colab_probe" / "episode.py", {"IMAGE", "HOST", "REPO", "DEADLINE", "SYSTEM", "TICKET", "TOOLS"},
                      {"sim": SimpleNamespace(TOOL=sim["TOOL"])})
    return {"system": ns["SYSTEM"], "ticket": ns["TICKET"], "tools": ns["TOOLS"]}


def load_colab_probe(path: str | Path, **kw) -> Episode:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ep = from_colab_probe(rows, source=str(path), **kw)
    ep.meta["source_sha256"] = file_sha256(path)
    return ep


# ---------------------------------------------------------------- rendering


@dataclass
class Rendered:
    text: str
    ids: np.ndarray  # (T,) int32
    role: np.ndarray  # (T,) int8, index into ROLES
    turn: np.ndarray  # (T,) int16
    segments: list[dict]  # role, turn, message, start, end (characters)


def _ids(out) -> list[int]:
    if isinstance(out, Mapping) or hasattr(out, "keys"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(i) for i in out]


def chat_text(tokenizer, messages, tools, add_generation_prompt=False, **kw) -> str:
    return tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False,
                                         add_generation_prompt=add_generation_prompt, **kw)


def chat_ids(tokenizer, messages, tools, add_generation_prompt=False, **kw) -> list[int]:
    return _ids(tokenizer(chat_text(tokenizer, messages, tools, add_generation_prompt, **kw), add_special_tokens=False))


def render(ep: Episode, tokenizer) -> Rendered:
    segs: list[dict] = []

    def hold(text: Any, role: str, msg: int) -> Any:
        if text is None:
            return text
        if not isinstance(text, str):
            raise RenderError(f"message {msg}: {role} must be a string, got {type(text).__name__}")
        if "@@SEG" in text:
            raise RenderError(f"message {msg}: text contains the reserved marker '@@SEG'")
        core = text.strip()
        if not core:
            return text
        lead, trail = text[: len(text) - len(text.lstrip())], text[len(text.rstrip()):]
        segs.append({"role": role, "turn": ep.turns[msg], "message": msg, "text": core})
        return lead + PLACEHOLDER.format(len(segs) - 1) + trail

    held = copy.deepcopy(ep.messages)
    for i, m in enumerate(held):
        role = m["role"]
        if role in ("system", "user"):
            m["content"] = hold(m.get("content"), role, i)
        elif role == "tool":
            m["content"] = hold(m.get("content"), "tool_output", i)
        elif role == "assistant":
            m["reasoning_content"] = hold(m.get("reasoning_content"), "reasoning", i)
            m["content"] = hold(m.get("content"), "content", i)
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", tc)
                f["arguments"] = {k: hold(v, "tool_call", i) if isinstance(v, str) else v
                                  for k, v in (f.get("arguments") or {}).items()}
        else:
            raise RenderError(f"message {i}: unexpected role {role!r}")

    marked = chat_text(tokenizer, held, ep.tools, **ep.template_kwargs)
    out, cursor, pos, seen = [], 0, 0, set()
    for mt in PLACEHOLDER_RE.finditer(marked):
        j = int(mt.group(1))
        if j in seen or j >= len(segs):
            raise RenderError(f"the chat template rendered segment {j} more than once")
        seen.add(j)
        out.append(marked[cursor:mt.start()])
        pos += mt.start() - cursor
        segs[j]["start"] = pos
        out.append(segs[j]["text"])
        pos += len(segs[j]["text"])
        segs[j]["end"] = pos
        cursor = mt.end()
    out.append(marked[cursor:])
    text = "".join(out)
    missing = [j for j in range(len(segs)) if j not in seen]
    if missing:
        raise RenderError(f"the chat template dropped {len(missing)} segments (first: {segs[missing[0]]['role']} "
                          f"of message {segs[missing[0]]['message']})")
    real = chat_text(tokenizer, ep.messages, ep.tools, **ep.template_kwargs)
    if real != text:
        at = next((i for i, (a, b) in enumerate(zip(real, text)) if a != b), min(len(real), len(text)))
        raise RenderError(f"placeholder rendering differs from the template's at character {at}: "
                          f"{real[max(0, at - 40):at + 40]!r} vs {text[max(0, at - 40):at + 40]!r}")
    segs.sort(key=lambda s: s["start"])

    # roles and turns per character
    n = len(text)
    char_role = np.full(n, R["template"], dtype=np.int8)
    char_seg = np.zeros(n, dtype=bool)
    char_turn = np.full(n, segs[-1]["turn"] if segs else -1, dtype=np.int16)
    prev = 0
    for s in segs:
        char_turn[prev:s["start"]] = s["turn"]
        char_role[s["start"]:s["end"]] = R[s["role"]]
        char_seg[s["start"]:s["end"]] = True
        char_turn[s["start"]:s["end"]] = s["turn"]
        prev = s["end"]
    if segs:  # the system turn: template text before the system prompt ends, or before the first segment
        first = segs[0]
        sys_end = first["end"] if first["role"] == "system" else first["start"]
        gap = ~char_seg[:sys_end]
        char_role[:sys_end][gap] = R["system"]

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = _ids(enc)
    offsets = list(enc["offset_mapping"])
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    template_ids = _ids(tokenizer.apply_chat_template(ep.messages, tools=ep.tools or None, tokenize=True,
                                                      add_generation_prompt=False, **ep.template_kwargs))
    if template_ids != ids:
        raise RenderError("tokenizing the rendered text gives different IDs from the template's own tokenization")
    role = np.empty(len(ids), dtype=np.int8)
    turn = np.empty(len(ids), dtype=np.int16)
    for t, (a, b) in enumerate(offsets):
        a, b = int(a), int(b)
        if b <= a:
            a = min(a, n - 1)
            role[t], turn[t] = char_role[a], char_turn[a]
            continue
        inside = np.flatnonzero(char_seg[a:b])
        c = a + int(inside[0]) if len(inside) else a
        role[t], turn[t] = char_role[c], char_turn[c]
    for s in segs:
        s.pop("text")
    return Rendered(text=text, ids=np.asarray(ids, dtype=np.int32), role=role, turn=turn, segments=segs)


def turn_prompts(ep: Episode, tokenizer) -> list[list[int]]:
    """For each agent turn (the k-th assistant message), its prompt as rendered here: the
    token IDs of the messages before it, plus the generation prompt."""
    return [chat_ids(tokenizer, ep.messages[:i], ep.tools, add_generation_prompt=True, **ep.template_kwargs)
            for i, m in enumerate(ep.messages) if m["role"] == "assistant"]


def prompt_checks(ep: Episode, tokenizer, prompts: Sequence[Sequence[int]] | None = None) -> list[dict]:
    """For each agent turn, the length of its prompt as rendered here (the messages before
    it, plus the generation prompt) next to the prompt tokens the server reported.
    `prompts` reuses `turn_prompts` output."""
    recorded = ep.meta.get("prompt_tokens") or []
    prompts = turn_prompts(ep, tokenizer) if prompts is None else prompts
    return [{"turn": k, "recorded": recorded[k] if k < len(recorded) else None, "rendered": len(p)}
            for k, p in enumerate(prompts)]


def prompt_ends(prompts: Sequence[Sequence[int]], ids: Sequence[int]) -> np.ndarray:
    """For each agent turn, the position in the full rendering `ids` of the last token of its
    prompt (its length minus 1), or -1 if the prompt's IDs are not the first IDs of `ids`:
    the one pass then doesn't hold what the server computed for that turn."""
    ids = np.asarray(ids)
    return np.array([len(p) - 1 if 0 < len(p) <= len(ids) and np.array_equal(ids[:len(p)], p) else -1
                     for p in prompts], dtype=np.int32)


# ---------------------------------------------------------------- reading


@dataclass
class Reading:
    token_ids: np.ndarray
    role: np.ndarray
    turn: np.ndarray
    layers: list[int]
    names: list[str]
    proj: np.ndarray  # (T, n_layers, n_names) float32: residual stream . unit vector (not centred)
    sae: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # layer: (idx (T,k), val (T,k))
    meta: dict = field(default_factory=dict)
    prompt_end: np.ndarray | None = None  # (agent turns,) int32: last token of each turn's prompt, -1 if unknown

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"token_ids": self.token_ids, "role": self.role, "turn": self.turn, "proj": self.proj.astype(np.float32)}
        for L, (idx, val) in self.sae.items():
            arrays[f"sae_idx@{L}"], arrays[f"sae_val@{L}"] = idx.astype(np.int32), val.astype(np.float16)
        if self.prompt_end is not None:
            arrays["prompt_end"] = self.prompt_end.astype(np.int32)
        meta = {**self.meta, "layers": self.layers, "names": self.names, "roles": list(ROLES)}
        np.savez_compressed(path, meta=np.array(json.dumps(meta, sort_keys=True)), **arrays)
        summary = path.with_name(path.name.removesuffix(".npz") + ".summary.json")
        summary.write_text(json.dumps(summarize(self), indent=1, sort_keys=True))
        return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> "Reading":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        sae = {int(k.split("@")[1]): (z[k], z[k.replace("idx", "val")]) for k in z.files if k.startswith("sae_idx@")}
        return cls(token_ids=z["token_ids"], role=z["role"], turn=z["turn"], layers=meta["layers"], names=meta["names"],
                   proj=z["proj"], sae=sae, meta=meta,
                   prompt_end=z["prompt_end"] if "prompt_end" in z.files else None)  # older readings have none


def summarize(r: Reading) -> dict:
    """Mean and SD of each projection by role group, and by agent turn (agent tokens and
    tool outputs separately, and the projection at the last token of the turn's prompt,
    None where unknown). `mean_centred` and `prompt_end_centred` subtract the vector file's
    neutral offset."""
    offsets = r.meta.get("offsets") or {}
    out: dict[str, Any] = {"n_tokens": int(len(r.role)),
                           "tokens_by_role": {k: int((r.role == R[k]).sum()) for k in ROLES}, "by_group": {}, "by_turn": {}}
    if "logit_monitor" in r.meta:
        out["logit_monitor"] = r.meta["logit_monitor"]
    turns = sorted(int(t) for t in set(r.turn.tolist()) if t >= 0)
    for li, L in enumerate(r.layers):
        for vi, name in enumerate(r.names):
            x = r.proj[:, li, vi].astype(np.float64)
            off = offsets.get(name, {}).get(str(L))
            key = f"{name}@{L}"
            g = {}
            for gname, roles in GROUPS.items():
                m = np.isin(r.role, [R[x_] for x_ in roles])
                if m.any():
                    mean = float(x[m].mean())
                    g[gname] = {"mean": mean, "sd": float(x[m].std()), "n": int(m.sum()),
                                "mean_centred": None if off is None else mean - off}
            out["by_group"][key] = g
            per = []
            for t in turns:
                row = {"turn": t}
                for gname in ("agent", "tool_output"):
                    m = (r.turn == t) & np.isin(r.role, [R[x_] for x_ in GROUPS[gname]])
                    row[gname] = float(x[m].mean()) if m.any() else None
                pos = int(r.prompt_end[t]) if r.prompt_end is not None and t < len(r.prompt_end) else -1
                row["prompt_end"] = float(x[pos]) if pos >= 0 else None
                row["prompt_end_centred"] = None if row["prompt_end"] is None or off is None else row["prompt_end"] - off
                per.append(row)
            out["by_turn"][key] = per
    return out


def read(backend: Backend, ep: Episode, vectors: VectorSet, layers: Sequence[int], *,
         names: Sequence[str] | None = None, saes: Mapping[int, Any] | None = None,
         allow_other_model: bool = False, vectors_path: str | None = None, phrases: dict | None = None,
         phrases_path: str | None = None, phrase_turns: Sequence[int] | None = None) -> Reading:
    """One forward pass over the rendered episode; projections onto `names` (default: all
    vectors in the file) at `layers`, top-k latents for each layer in `saes`, and each
    turn's prompt end. With `phrases` (a phrase file: {status, source, groups}), the logit
    monitor on every agent turn with reasoning, or on `phrase_turns`."""
    want = vectors.meta.get("model", {}).get("fingerprint")
    have = backend.info.get("fingerprint")
    if want != have and not allow_other_model:
        raise ValueError(f"the vectors were built on {want}, the backend is {have}; vectors only read the model "
                         "they were built on (pass allow_other_model for a deliberate transfer test)")
    names = list(names or vectors.names)
    layers = sorted(set(int(L) for L in layers) | set(int(L) for L in (saes or {})))
    rd = render(ep, backend.tokenizer)
    prompts = turn_prompts(ep, backend.tokenizer)
    ends = prompt_ends(prompts, rd.ids)
    acts = backend.activations(rd.ids.tolist(), layers)
    proj = np.zeros((len(rd.ids), len(layers), len(names)), dtype=np.float32)
    for li, L in enumerate(layers):
        if all((n, L) in vectors.vectors for n in names):
            m = np.stack([vectors.get(n, L) for n in names], axis=1)
            proj[:, li, :] = acts[L] @ m
        else:
            proj[:, li, :] = np.nan
    sae_out = {}
    for L, sae in (saes or {}).items():
        sae_out[int(L)] = sae.encode(acts[int(L)])
    offsets = {n: {str(L): v for L, v in (vectors.meta.get("offset", {}).get(n) or {}).items()} for n in names}
    meta = {"episode": ep.meta, "backend": backend.info, "vectors_model": vectors.meta.get("model"),
            "vectors_path": vectors_path, "vectors_sha256": file_sha256(vectors_path) if vectors_path else None,
            "offsets": offsets, "template_kwargs": ep.template_kwargs, "text_sha256": bytes_sha256(rd.text.encode()),
            "n_tokens": int(len(rd.ids)), "prompt_checks": prompt_checks(ep, backend.tokenizer, prompts),
            "prompt_end_not_prefix": [k for k, e in enumerate(ends.tolist()) if e < 0],
            "sae": {str(L): getattr(s, "info", {}) for L, s in (saes or {}).items()},
            "created": now(), "git_commit": git_commit()}
    if phrases is not None:
        from .monitors import logit_monitor  # monitors imports this module

        meta["logit_monitor"] = {"phrases_path": phrases_path, "status": phrases.get("status"),
                                 "phrases_sha256": file_sha256(phrases_path) if phrases_path else None,
                                 **logit_monitor(backend, rd.ids, rd.role, rd.turn, phrases["groups"], turns=phrase_turns)}
    return Reading(token_ids=rd.ids, role=rd.role, turn=rd.turn, layers=layers, names=names, proj=proj, sae=sae_out,
                   meta=meta, prompt_end=ends)
