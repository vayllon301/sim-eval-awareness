"""A tiny numpy model and character tokenizer that honour the backend interface, for
offline tests. Not a language model: it only has to be deterministic, causal and
sensitive to its input and to additions in its residual stream.

The tokenizer is character level (token ID = code point, capped at 255) with a chat
template laid out like Qwen's (system turn with a tools block, <think> blocks kept for
every assistant turn, <tool_call> blocks with one <parameter=...> per argument, tool
outputs grouped into one user turn as <tool_response> blocks, contents trimmed).
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np

from .backend import Add, Replace, check_adds, check_layers

EOS = 0  # "\x00" ends a turn


class FakeTokenizer:
    eos_token_ids = frozenset({EOS})

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        ids = [min(ord(c), 255) for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False, **kwargs):
        text = render_fake_chat(messages, tools, add_generation_prompt)
        return self(text)["input_ids"] if tokenize else text


def _content(m) -> str:
    c = m.get("content")
    return "" if c is None else str(c).strip()


def render_fake_chat(messages, tools=None, add_generation_prompt=False) -> str:
    out = []
    sys_text = _content(messages[0]) if messages and messages[0]["role"] == "system" else ""
    if tools:
        block = "# Tools\n<tools>\n" + "\n".join(json.dumps(t, sort_keys=True) for t in tools) + "\n</tools>"
        out.append("<|im_start|>system\n" + block + ("\n\n" + sys_text if sys_text else "") + "<|im_end|>\n")
    elif sys_text:
        out.append("<|im_start|>system\n" + sys_text + "<|im_end|>\n")
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "system":
            if i:
                raise ValueError("system message must be first")
        elif role == "user":
            out.append("<|im_start|>user\n" + _content(m) + "<|im_end|>\n")
        elif role == "assistant":
            reasoning = (m.get("reasoning_content") or "").strip()
            content = _content(m)
            s = "<|im_start|>assistant\n<think>\n" + reasoning + "\n</think>\n\n" + content
            for j, tc in enumerate(m.get("tool_calls") or []):
                f = tc.get("function", tc)
                s += ("\n\n" if j == 0 and content else ("" if j == 0 else "\n")) + f"<tool_call>\n<function={f['name']}>\n"
                for k, v in (f.get("arguments") or {}).items():
                    s += f"<parameter={k}>\n" + (v if isinstance(v, str) else json.dumps(v)) + "\n</parameter>\n"
                s += "</function>\n</tool_call>"
            out.append(s + "<|im_end|>\n")
        elif role == "tool":
            prev = messages[i - 1]["role"] if i else None
            nxt = messages[i + 1]["role"] if i + 1 < len(messages) else None
            out.append(("<|im_start|>user" if prev != "tool" else "") + "\n<tool_response>\n" + _content(m)
                       + "\n</tool_response>" + ("<|im_end|>\n" if nxt != "tool" else ""))
        else:
            raise ValueError(f"unexpected role {role!r}")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n<think>\n")
    return "".join(out)


class FakeBackend:
    """h0 = E[ids]; each layer adds 0.5 * tanh(causal_mean(h) @ A + h @ B); logits = h @ U."""

    def __init__(self, n_layers: int = 6, d_model: int = 16, vocab: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_layers, self.d_model, self.vocab = n_layers, d_model, vocab
        self.E = rng.standard_normal((vocab, d_model)).astype(np.float32)
        self.A = [(rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)).astype(np.float32) for _ in range(n_layers)]
        self.B = [(rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)).astype(np.float32) for _ in range(n_layers)]
        self.U = rng.standard_normal((d_model, vocab)).astype(np.float32)
        self.tokenizer = FakeTokenizer()
        self.eos_token_ids = self.tokenizer.eos_token_ids
        self.info = {"engine": "fake", "model": f"fake-{n_layers}x{d_model}", "fingerprint": f"fake:{seed}:{n_layers}x{d_model}",
                     "quantization": "none (float32)", "n_layers": n_layers, "d_model": d_model}

    def _forward(self, ids, adds=None, replace=None, stop=None):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError("ids must be a non-empty 1-D sequence")
        h = self.E[ids]
        steps = np.arange(1, len(ids) + 1, dtype=np.float32)[:, None]
        out = {}
        for L in range(self.n_layers if stop is None else stop + 1):
            ctx = np.cumsum(h, 0) / steps
            h = h + 0.5 * np.tanh(ctx @ self.A[L] + h @ self.B[L])
            if replace and L in replace:
                h = np.asarray(replace[L](h.copy(), 0), dtype=np.float32)
            if adds and L in adds:
                h = h + adds[L]
            out[L] = h
        return out, h

    def activations(self, ids, layers, *, steer: Sequence[Add] = ()):
        layers = check_layers(layers, self.n_layers)
        hs, _ = self._forward(ids, check_adds(steer, self.n_layers, self.d_model), stop=layers[-1])
        return {L: hs[L].astype(np.float32) for L in layers}

    def token_losses(self, ids, *, replace: Mapping[int, Replace] | None = None):
        _, h = self._forward(ids, replace=dict(replace or {}))
        logits = (h @ self.U).astype(np.float64)
        logits -= logits.max(1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(1, keepdims=True))
        tgt = np.asarray(ids[1:])
        return -logp[np.arange(len(tgt)), tgt]

    def generate(self, ids, *, max_tokens: int, steer: Sequence[Add] = ()):
        adds = check_adds(steer, self.n_layers, self.d_model)
        seq, new = list(ids), []
        for _ in range(max_tokens):
            _, h = self._forward(seq, adds)
            t = int(np.argmax(h[-1] @ self.U))
            new.append(t)
            if t in self.eos_token_ids:
                break
            seq.append(t)
        return new
