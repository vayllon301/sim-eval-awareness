"""The backend interface on vLLM with vllm-lens: the reference backend on the GPU.

Both models run in FP8 on vLLM in every phase (LOG.md, 28-10-2026): episodes, vectors,
steering and reading. This backend reads and steers inside the engine the episodes run on,
so the reading path and the steering engine compute with the same FP8 weights by
construction: the agent's official FP8 checkpoint, and the simulator's bf16 files quantized
by vLLM at load (`quantization="fp8_per_channel"`, from the pilot on), which no other library
reproduces.

vllm-lens (a vLLM plugin, docs/INTERP.md "Backend interface"):
- Reading: `output_residual_stream` captures `hidden_states + residual` after decoder layer
  L (resid_post) at every prompt position, in the model's dtype: one request, max_tokens=1.
- Steering: one `SteeringVector` per request, the per-layer deltas stacked (`lens_steering`):
  scale 1, norm_match off, no positions (every position, prompt and generated). vllm-lens
  adds it to the layer's output in the model's dtype before its capture, so captured
  activations include it. A zero delta is still sent, so alpha 0 runs the hooked path.
- Post-hooks only: its pre-hooks raise on Qwen3.5-architecture layers (vllm-lens#52).
- A hook that fails is logged and skipped, not raised, so a steer that didn't apply looks
  like an unsteered run. The GPU smoke test checks that every steer moves the reading at its
  layer by its delta.

Engine settings (ENGINE): eager (vllm-lens forces it), prefix caching off (cached blocks skip
the forward pass, so they would have no activations, and an unsteered request could reuse
steered KV), and one request per call: vLLM isn't batch-invariant, and the simulator's
online FP8 scales MoE activations over the whole batch.

`token_losses` uses vLLM's prompt log-probs (the raw log-softmax). No `replace` (an SAE
splice needs a hook shipped to the worker process; the HF backend has it) and no
`continuation_logprobs`: vLLM skips the prefix cache for requests with prompt log-probs,
and batching the continuations would change the numbers, so the logit monitor falls back to
one `token_losses` call per phrase.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from .backend import Add, Replace, check_adds, check_layers
from .meta import hf_revision, load_fingerprint, tokenizer_info, versions
from .meta import quantization as files_quantization

# Every call's engine. The stock run it is compared with uses the same settings, with the
# plugin off (explore/interp/gpu_smoke.py). max_num_batched_tokens is vllm serve's default
# on a 96 GB card, so prompts are split into the same prefill chunks as the episodes'.
ENGINE = {"enforce_eager": True, "enable_prefix_caching": False, "max_model_len": 16384, "max_num_seqs": 8,
          "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.85, "language_model_only": True}


# What changes the engine besides its settings, recorded after it is built (vllm-lens sets
# VLLM_USE_V2_MODEL_RUNNER=0: its hooks need vLLM's V1 model runner).
ENV = ("VLLM_LENS_DISABLE", "VLLM_LENS_STRICT_COMPATIBILITY", "VLLM_USE_V2_MODEL_RUNNER")


def _api() -> SimpleNamespace:
    """vLLM, vllm-lens and torch, imported on first use (the offline tests swap in fakes)."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm_lens import SteeringVector

    return SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams, TokensPrompt=TokensPrompt,
                           SteeringVector=SteeringVector, tensor=lambda a: torch.from_numpy(np.ascontiguousarray(a)),
                           numpy=lambda t: t.float().cpu().numpy(),
                           device=lambda: torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                           tokenizer=lambda path: AutoTokenizer.from_pretrained(str(path)))


def lens_disabled() -> bool:
    return os.environ.get("VLLM_LENS_DISABLE", "").strip().lower() in ("1", "true", "yes")


def lens_steering(steer: Sequence[Add], n_layers: int, d_model: int) -> dict | None:
    """Additions (steering.resolve's output) as vllm-lens `SteeringVector` fields: one row
    per layer, layers in order, additions to the same layer summed, already scaled (scale 1,
    norm_match off), every position. None when there is nothing to add."""
    adds = check_adds(steer, n_layers, d_model)
    if not adds:
        return None
    layers = sorted(adds)
    return {"activations": np.stack([adds[L] for L in layers]).astype(np.float32), "layer_indices": layers,
            "scale": 1.0, "norm_match": False, "position_indices": None}


def eos_ids(model_dir: Path, tokenizer) -> frozenset[int]:
    """End-of-turn tokens: generation_config.json's and the tokenizer's."""
    out: set[int] = set()
    p = model_dir / "generation_config.json"
    for v in ((json.loads(p.read_text()).get("eos_token_id") if p.exists() else None),
              getattr(tokenizer, "eos_token_id", None)):
        if v is not None:
            out |= {int(i) for i in (v if isinstance(v, list) else [v])}
    return frozenset(out or (int(i) for i in getattr(tokenizer, "eos_token_ids", ())))


class VLLMBackend:
    def __init__(self, path: str | Path, *, quantization: str | None = None, engine: Mapping | None = None,
                 tokenizer=None, llm=None):
        """`quantization` is vLLM's flag ("fp8_per_channel" for the simulator, as `vllm serve` runs it);
        `engine` overrides ENGINE's settings; `llm` is an engine already built."""
        self.path = Path(path).expanduser()
        self.api = _api()
        self.engine = {**ENGINE, **(engine or {})}
        cfg = json.loads((self.path / "config.json").read_text())
        text = cfg.get("text_config", cfg)
        self.n_layers, self.d_model = int(text["num_hidden_layers"]), int(text["hidden_size"])
        self.tokenizer = tokenizer = tokenizer if tokenizer is not None else self.api.tokenizer(self.path)
        self.eos_token_ids = eos_ids(self.path, tokenizer)
        self.llm = llm if llm is not None else self.api.LLM(model=str(self.path), quantization=quantization, **self.engine)
        label = f"vllm:{quantization}" if quantization else None
        tok = tokenizer_info(tokenizer, self.path)
        files = files_quantization(self.path)
        self.info = {"engine": "vllm", "model": str(self.path), "fingerprint": load_fingerprint(self.path, label),
                     "revision": hf_revision(self.path),
                     "quantization": files if label is None else f"vLLM {quantization} at load (files: {files})",
                     "engine_settings": {**self.engine, "quantization": quantization},
                     "lens": "disabled" if lens_disabled() else "enabled",
                     "env": {k: os.environ.get(k) for k in ENV},
                     "tokenizer_sha256": tok["tokenizer_sha256"], "chat_template_sha256": tok["chat_template_sha256"],
                     "n_layers": self.n_layers, "d_model": self.d_model, "device": self.api.device(),
                     "versions": {**versions("vllm", "vllm-lens", "torch", "flashinfer-python"), **tok["versions"]}}

    # -------------------------------------------------------------- plumbing

    def _steering(self, steer: Sequence[Add]) -> dict:
        sv = lens_steering(steer, self.n_layers, self.d_model)
        if sv is None:
            return {}
        return {"apply_steering_vectors": [self.api.SteeringVector(**{**sv, "activations": self.api.tensor(sv["activations"])})]}

    def _one(self, ids: Sequence[int], **params):
        """One request alone (module docstring)."""
        ids = [int(i) for i in ids]
        if not ids:
            raise ValueError("ids must be non-empty")
        outs = self.llm.generate([self.api.TokensPrompt(prompt_token_ids=ids)], self.api.SamplingParams(**params),
                                 use_tqdm=False)
        return outs[0]

    # -------------------------------------------------------------- interface

    def activations(self, ids, layers, *, steer: Sequence[Add] = ()):
        layers = check_layers(layers, self.n_layers)
        ids = [int(i) for i in ids]
        out = self._one(ids, max_tokens=1, temperature=0.0,
                        extra_args={"output_residual_stream": list(layers), **self._steering(steer)})
        rs = (getattr(out, "activations", None) or {}).get("residual_stream")
        if rs is None:
            raise RuntimeError("vllm-lens returned no activations: is it installed, and VLLM_LENS_DISABLE unset?")
        rs = np.asarray(self.api.numpy(rs), dtype=np.float32)
        want = (len(layers), len(ids), self.d_model)
        if rs.shape != want:
            raise RuntimeError(f"vllm-lens returned activations of shape {rs.shape}, expected {want}")
        return {L: rs[i] for i, L in enumerate(layers)}

    def token_losses(self, ids, *, replace: Mapping[int, Replace] | None = None):
        if replace:
            raise NotImplementedError("the vLLM backend can't splice the residual stream (replace); use the HF backend")
        ids = [int(i) for i in ids]
        pl = self._one(ids, max_tokens=1, temperature=0.0, prompt_logprobs=0).prompt_logprobs
        if pl is None or len(pl) != len(ids):
            raise RuntimeError(f"vLLM returned {None if pl is None else len(pl)} prompt log-probs for {len(ids)} tokens")
        return np.array([-pl[i][ids[i]].logprob for i in range(1, len(ids))], dtype=np.float64)

    def generate(self, ids, *, max_tokens: int, steer: Sequence[Add] = ()):
        out = self._one(ids, max_tokens=int(max_tokens), temperature=0.0, stop_token_ids=sorted(self.eos_token_ids),
                        extra_args=self._steering(steer) or None)
        return [int(t) for t in out.outputs[0].token_ids]

    def close(self) -> None:
        self.llm = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
