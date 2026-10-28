"""The backend interface on MLX (Apple Silicon), for development on the laptop.

Local 8-bit and 4-bit weights are never evidence: the experiment's reference backend is
bf16 on a GPU (one backend per experiment, AGENTS.md). Use this to build and test the
pipeline, not to report results.

Hooks: each decoder layer the call needs is wrapped for the duration of the call (and
restored after), so the residual stream after it can be captured, have a vector added, or
be replaced. Long sequences run in chunks with the model's own cache, as mlx-lm's
prefill does, so a 30K-token episode doesn't need a 30K x 30K attention pass at once.
Activations-only calls stop after the last layer they need.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .backend import Add, Replace, check_adds, check_layers
from .meta import hf_revision, quantization, tokenizer_info, weights_fingerprint


class _Stop(Exception):
    pass


class _Hook:
    """Wraps one decoder layer: replace, then add, then capture, then maybe stop."""

    def __init__(self, layer, idx: int, run: "_Run"):
        self._layer, self._idx, self._run = layer, idx, run

    def __call__(self, x, *args, **kwargs):
        import mlx.core as mx

        out = self._layer(x, *args, **kwargs)
        h = out[0] if isinstance(out, tuple) else out
        r, i = self._run, self._idx
        if i in r.replace:
            new = r.replace[i](np.array(h[0].astype(mx.float32)), r.start)
            h = mx.array(np.asarray(new, dtype=np.float32))[None].astype(h.dtype)
        if i in r.add:
            h = h + r.add[i]
        if i in r.capture:
            r.store[i].append(h[0])
        if r.stop is not None and i >= r.stop:
            r.last = h
            raise _Stop
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def __getattr__(self, name):
        return getattr(self._layer, name)


class _Run:
    def __init__(self, capture=(), add=None, replace=None, stop=None):
        self.capture, self.add, self.replace, self.stop = set(capture), add or {}, replace or {}, stop
        self.store = {L: [] for L in self.capture}
        self.last = None
        self.start = 0  # absolute position of the current chunk's first token


class MLXBackend:
    def __init__(self, path: str | Path):
        import mlx.core as mx
        import mlx_lm
        from mlx_lm import load

        self.path = Path(path).expanduser()
        self.model, wrapper = load(str(self.path))
        self.tokenizer = getattr(wrapper, "_tokenizer", wrapper)  # the Hugging Face tokenizer
        self.eos_token_ids = frozenset(int(i) for i in (getattr(wrapper, "eos_token_ids", None) or
                                                         [self.tokenizer.eos_token_id]))
        self.lm = getattr(self.model, "language_model", self.model)
        self.inner = self.lm.model
        self.layers = self.inner.layers
        self.n_layers = len(self.layers)
        probe = self.inner.embed_tokens(mx.array([[0]]))
        self.dtype = probe.dtype
        self.d_model = int(probe.shape[-1])
        tok = tokenizer_info(self.tokenizer, self.path)
        self.info = {"engine": "mlx", "model": str(self.path), "fingerprint": weights_fingerprint(self.path),
                     "revision": hf_revision(self.path), "quantization": quantization(self.path),
                     "tokenizer_sha256": tok["tokenizer_sha256"], "chat_template_sha256": tok["chat_template_sha256"],
                     "n_layers": self.n_layers, "d_model": self.d_model, "dtype": str(self.dtype),
                     "versions": {"mlx": mx.__version__, "mlx_lm": mlx_lm.__version__, **tok["versions"]}}

    # -------------------------------------------------------------- plumbing

    @contextmanager
    def _hooks(self, run: _Run):
        idx = sorted(run.capture | set(run.add) | set(run.replace) | ({run.stop} if run.stop is not None else set()))
        saved = {i: self.layers[i] for i in idx}
        try:
            for i in idx:
                self.layers[i] = _Hook(saved[i], i, run)
            yield run
        finally:
            for i, layer in saved.items():
                self.layers[i] = layer

    def _adds(self, steer: Sequence[Add]):
        import mlx.core as mx

        return {L: mx.array(d).astype(self.dtype) for L, d in check_adds(steer, self.n_layers, self.d_model).items()}

    def _states(self, cache, upto=None):
        cs = cache if upto is None else cache[: upto + 1]
        return [s for c in cs for s in (c.state if isinstance(c.state, (list, tuple)) else [c.state]) if s is not None]

    # -------------------------------------------------------------- interface

    def activations(self, ids, layers, *, steer: Sequence[Add] = (), chunk: int = 1024):
        import mlx.core as mx

        layers = check_layers(layers, self.n_layers)
        run = _Run(capture=layers, add=self._adds(steer), stop=layers[-1])
        cache = self.model.make_cache()
        ids = [int(i) for i in ids]
        out = {L: [] for L in layers}
        with self._hooks(run):
            for a in range(0, len(ids), chunk):
                try:
                    self.inner(mx.array(ids[a:a + chunk])[None], cache=cache)
                except _Stop:
                    pass
                mx.eval([run.store[L][-1] for L in layers] + self._states(cache, layers[-1]))
                for L in layers:
                    out[L].append(np.array(run.store[L].pop().astype(mx.float32)))
        return {L: np.concatenate(v) for L, v in out.items()}

    def token_losses(self, ids, *, replace: Mapping[int, Replace] | None = None, chunk: int = 512):
        import mlx.core as mx

        ids = [int(i) for i in ids]
        run = _Run(replace=dict(replace or {}))
        cache = self.model.make_cache()
        losses = []
        with self._hooks(run):
            for a in range(0, len(ids) - 1, chunk):
                run.start = a
                x = ids[a:a + chunk]
                logits = self.lm(mx.array(x)[None], cache=cache)[0].astype(mx.float32)
                tgt = ids[a + 1:a + 1 + len(x)]
                logits = logits[: len(tgt)]
                lse = mx.logsumexp(logits, axis=-1)
                pick = mx.take_along_axis(logits, mx.array(tgt)[:, None], axis=-1)[:, 0]
                ce = lse - pick
                mx.eval(ce, self._states(cache))
                losses.append(np.array(ce))
        return np.concatenate(losses)

    def generate(self, ids, *, max_tokens: int, steer: Sequence[Add] = (), chunk: int = 2048):
        """Greedy. The prompt is processed in mlx-lm's order (all but the last token in
        chunks of 2048, then the last token alone), so unhooked output is token-identical
        to mlx-lm's own greedy generation; another order changes bf16 rounding, which can
        flip a near-tie."""
        import mlx.core as mx

        ids = [int(i) for i in ids]
        run = _Run(add=self._adds(steer))
        cache = self.model.make_cache()
        new: list[int] = []
        with self._hooks(run):
            for a in range(0, len(ids) - 1, chunk):
                self.lm(mx.array(ids[a:min(a + chunk, len(ids) - 1)])[None], cache=cache)
                mx.eval(self._states(cache))
            logits = self.lm(mx.array(ids[-1:])[None], cache=cache)
            for _ in range(max_tokens):
                t = int(mx.argmax(logits[0, -1]).item())
                new.append(t)
                if t in self.eos_token_ids:
                    break
                logits = self.lm(mx.array([[t]]), cache=cache)
        return new

    def close(self) -> None:
        """Drop the weights and clear MLX's buffer cache."""
        import gc

        import mlx.core as mx

        self.model = self.lm = self.inner = self.layers = None
        gc.collect()
        mx.clear_cache()
