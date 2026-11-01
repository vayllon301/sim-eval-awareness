"""The backend interface on PyTorch with Hugging Face transformers (GPU, or CPU for tests).

Loads what `AutoModelForCausalLM` loads: bf16 weights, or a pre-quantized checkpoint such as
the official Qwen3.8 FP8 one (transformers' FineGrainedFP8: the checkpoint's 128x128 weight
scales, activations quantized per token in groups of 128; on a GPU it needs the `kernels`
package). A checkpoint's vision tower and MTP head are skipped (Qwen3_5ForCausalLM).

On the GPU this is the cross-check and fallback for the agent, not the reference backend:
the reference is vLLM (vllm_backend.py, docs/INTERP.md). The FP8 checkpoint is the same
files on both, but the kernels differ (CUTLASS in vLLM, Triton here), so numbers are close,
not identical. It can't read the simulator's FP8 weights at all: vLLM quantizes those at
load, per tensor, with merged projections, which transformers doesn't reproduce.

Hooks: forward hooks on the decoder layers a call needs (`model.model.layers[L]`, whose
output is the residual stream after layer L), removed after the call: replace, then add,
then capture, then maybe stop. Readings run the decoder stack without `lm_head`. Long
sequences run in chunks with the model's own cache. `generate` copies the way
`model.generate` calls the model (one pass over the prompt with `logits_to_keep=1`, then
one token at a time, with an all-ones attention mask), so unhooked output is
token-identical to transformers' own greedy generation. `continuation_logprobs` runs the
prefix once and scores each continuation from a copy of its cache.
"""

from __future__ import annotations

import copy
import gc
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .backend import Add, Replace, check_adds, check_layers
from .meta import hf_revision, load_fingerprint, quantization, tokenizer_info


class _Stop(Exception):
    pass


class _Run:
    def __init__(self, capture=(), add=None, replace=None, stop=None):
        self.capture, self.add, self.replace, self.stop = set(capture), add or {}, replace or {}, stop
        self.store = {L: [] for L in self.capture}
        self.start = 0  # absolute position of the current chunk's first token


def decoder(model):
    """The decoder stack (the module with `.layers`) of a causal LM or a multimodal wrapper."""
    inner = getattr(model, "model", model)
    inner = getattr(inner, "language_model", inner)
    if not hasattr(inner, "layers"):
        raise ValueError(f"can't find the decoder layers of {type(model).__name__}")
    return inner


def eos_ids(model, tokenizer) -> frozenset[int]:
    """End-of-turn tokens: the generation config's and the tokenizer's."""
    out: set[int] = set()
    for v in (getattr(getattr(model, "generation_config", None), "eos_token_id", None),
              getattr(tokenizer, "eos_token_id", None)):
        if v is not None:
            out |= {int(i) for i in (v if isinstance(v, (list, tuple, set, frozenset)) else [v])}
    return frozenset(out or (int(i) for i in getattr(tokenizer, "eos_token_ids", ())))


def versions(*pkgs: str) -> dict[str, str | None]:
    out = {}
    for p in pkgs:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


class HFBackend:
    def __init__(self, path: str | Path, *, tokenizer=None, dtype="auto", device: str | None = None,
                 quantization_config=None, **load_kwargs):
        """`quantization_config` quantizes at load (transformers' own scheme, e.g.
        FineGrainedFP8Config); the scheme goes into the fingerprint, since those weights are
        not the files' (meta.load_fingerprint)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.path = Path(path).expanduser()
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        if device != "cpu":  # device_map needs accelerate; on the CPU the default is the CPU
            load_kwargs["device_map"] = device
        self.model = AutoModelForCausalLM.from_pretrained(str(self.path), dtype=dtype, **load_kwargs).eval()
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(str(self.path))
        self.inner = decoder(self.model)
        self.layers = self.inner.layers
        self.n_layers = len(self.layers)
        emb = self.inner.embed_tokens.weight
        self.device, self.dtype = emb.device, emb.dtype
        self.d_model = int(emb.shape[-1])
        self.eos_token_ids = eos_ids(self.model, self.tokenizer)
        label = None
        if quantization_config is not None:
            block = getattr(quantization_config, "weight_block_size", None)
            label = f"hf:{quantization_config.quant_method}" + (f"-block{'x'.join(map(str, block))}" if block else "")
        tok = tokenizer_info(self.tokenizer, self.path)
        self.info = {"engine": "hf", "model": str(self.path), "fingerprint": load_fingerprint(self.path, label),
                     "revision": hf_revision(self.path),
                     "quantization": quantization(self.path) if label is None else f"{label} at load",
                     "tokenizer_sha256": tok["tokenizer_sha256"], "chat_template_sha256": tok["chat_template_sha256"],
                     "n_layers": self.n_layers, "d_model": self.d_model, "dtype": str(self.dtype),
                     "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
                     "device": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else str(self.device),
                     "versions": {**versions("torch", "kernels", "flash-linear-attention", "causal-conv1d"),
                                  **tok["versions"]}}

    # -------------------------------------------------------------- plumbing

    def _hook(self, i: int, run: _Run):
        import torch

        def fn(module, args, out):
            h = out[0] if isinstance(out, tuple) else out
            if i in run.replace:
                new = run.replace[i](h[0].float().cpu().numpy(), run.start)
                h = torch.as_tensor(np.asarray(new, dtype=np.float32), device=h.device)[None].to(h.dtype)
            if i in run.add:
                h = h + run.add[i]
            if i in run.capture:
                run.store[i].append(h[0].float().cpu().numpy())
            if run.stop is not None and i >= run.stop:
                raise _Stop
            return (h, *out[1:]) if isinstance(out, tuple) else h

        return fn

    @contextmanager
    def _hooks(self, run: _Run):
        import torch

        idx = sorted(run.capture | set(run.add) | set(run.replace) | ({run.stop} if run.stop is not None else set()))
        handles = [self.layers[i].register_forward_hook(self._hook(i, run)) for i in idx]
        try:
            with torch.inference_mode():
                yield run
        finally:
            for h in handles:
                h.remove()

    def _adds(self, steer: Sequence[Add]):
        import torch

        return {L: torch.as_tensor(d, device=self.device).to(self.dtype)
                for L, d in check_adds(steer, self.n_layers, self.d_model).items()}

    def _ids(self, ids):
        import torch

        return torch.tensor([[int(i) for i in ids]], dtype=torch.long, device=self.device)

    def _prefill(self, ids: Sequence[int], chunk: int):
        """The model over `ids` in chunks with its cache; returns (cache, the last position's
        logits as float32)."""
        cache, out = None, None
        for a in range(0, len(ids), chunk):
            out = self.model(input_ids=self._ids(ids[a:a + chunk]), past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = out.past_key_values
        return cache, out.logits[0, -1].float()

    # -------------------------------------------------------------- interface

    def activations(self, ids, layers, *, steer: Sequence[Add] = (), chunk: int = 4096):
        layers = check_layers(layers, self.n_layers)
        ids = [int(i) for i in ids]
        one = len(ids) <= chunk
        # Stopping early leaves later layers' caches unfilled, so only a single pass may stop.
        run = _Run(capture=layers, add=self._adds(steer), stop=layers[-1] if one else None)
        with self._hooks(run):
            cache = None
            for a in range(0, len(ids), chunk):
                try:
                    out = self.inner(input_ids=self._ids(ids[a:a + chunk]), past_key_values=cache, use_cache=not one)
                    cache = out.past_key_values
                except _Stop:
                    pass
        return {L: np.concatenate(run.store[L]) for L in layers}

    def token_losses(self, ids, *, replace: Mapping[int, Replace] | None = None, chunk: int = 1024):
        import torch

        ids = [int(i) for i in ids]
        run = _Run(replace=dict(replace or {}))
        losses = []
        with self._hooks(run):
            cache = None
            for a in range(0, len(ids) - 1, chunk):
                run.start = a
                x = ids[a:a + chunk]
                out = self.model(input_ids=self._ids(x), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                tgt = ids[a + 1:a + 1 + len(x)]
                logits = out.logits[0, : len(tgt)].float()
                ce = torch.logsumexp(logits, -1) - logits[torch.arange(len(tgt)), torch.tensor(tgt, device=logits.device)]
                losses.append(ce.cpu().numpy().astype(np.float64))
        return np.concatenate(losses)

    def generate(self, ids, *, max_tokens: int, steer: Sequence[Add] = ()):
        """Greedy, called the way `model.generate` calls the model (module docstring)."""
        import torch

        ids = [int(i) for i in ids]
        run = _Run(add=self._adds(steer))
        new: list[int] = []
        with self._hooks(run):
            n = len(ids)
            out = self.model(input_ids=self._ids(ids), attention_mask=torch.ones((1, n), dtype=torch.long, device=self.device),
                             use_cache=True, logits_to_keep=1)
            for _ in range(max_tokens):
                t = int(torch.argmax(out.logits[0, -1].float()))
                new.append(t)
                if t in self.eos_token_ids:
                    break
                n += 1
                out = self.model(input_ids=self._ids([t]), past_key_values=out.past_key_values, use_cache=True,
                                 attention_mask=torch.ones((1, n), dtype=torch.long, device=self.device), logits_to_keep=1)
        return new

    def continuation_logprobs(self, prefix_ids, continuations, *, chunk: int = 4096) -> list[np.ndarray]:
        """Per-token log-probs (nats) of each continuation after one prefix: the prefix runs
        once, and each continuation runs from a copy of its cache."""
        import torch

        prefix = [int(i) for i in prefix_ids]
        out = []
        with torch.inference_mode():
            cache, last = self._prefill(prefix, chunk)
            first = torch.log_softmax(last, -1)
            for c in continuations:
                c = [int(i) for i in c]
                lp = [float(first[c[0]])]
                if len(c) > 1:
                    o = self.model(input_ids=self._ids(c[:-1]), past_key_values=copy.deepcopy(cache), use_cache=True)
                    ls = torch.log_softmax(o.logits[0].float(), -1)
                    lp += ls[torch.arange(len(c) - 1), torch.tensor(c[1:], device=ls.device)].cpu().tolist()
                out.append(np.asarray(lp, dtype=np.float64))
        return out

    def close(self) -> None:
        """Drop the weights and free the GPU memory."""
        import torch

        self.model = self.inner = self.layers = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
