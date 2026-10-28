"""The backend interface: what a model engine provides to the interp tools.

A backend wraps one model: its weights, quantization, engine and chat template. Every
number we report is tied to one backend (one backend per experiment, AGENTS.md), so its
`info` goes into the metadata of every vector file and reading.

Conventions every backend honours (docs/INTERP.md, "Backend interface"):
- Layers are 0-based decoder layers. "Layer L" is the residual stream right after decoder
  layer L (`resid_post`, the output of `model.layers[L]`), the hook point of Qwen-Scope.
- Arrays crossing the interface are numpy float32, shape (tokens, d_model).
- Token IDs are plain ints from the model's own tokenizer. Rendering chat messages is the
  reader's job (reader.py); a backend only sees token IDs.
- An `Add(layer, delta)` adds `delta` to the residual stream after decoder layer `layer`
  at every position (prompt and generated tokens), in the model's compute dtype.
- `replace` maps a layer to a function f(h, start) -> h' that substitutes the residual
  stream after that layer (used to splice an SAE reconstruction in). h is (tokens, d) for
  a run of consecutive positions starting at absolute position `start` (backends may call
  it once per chunk).
- `generate` is greedy (argmax) and stops at an end-of-turn token or `max_tokens`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

Replace = Callable[[np.ndarray, int], np.ndarray]  # (h (tokens, d), start position) -> h


@dataclass(frozen=True)
class Add:
    """h <- h + delta after decoder layer `layer`, at every position."""

    layer: int
    delta: np.ndarray  # (d_model,) float32

    def __post_init__(self):
        if self.delta.ndim != 1:
            raise ValueError(f"delta must be a vector, got shape {self.delta.shape}")


class Backend(Protocol):
    info: dict[str, Any]  # engine, model path, fingerprint, quantization, versions
    n_layers: int
    d_model: int
    tokenizer: Any  # Hugging Face style: __call__, decode, apply_chat_template
    eos_token_ids: frozenset[int]

    def activations(self, ids: Sequence[int], layers: Sequence[int], *,
                    steer: Sequence[Add] = ()) -> dict[int, np.ndarray]:
        """Residual stream after each of `layers`: {layer: (len(ids), d_model) float32}."""
        ...

    def token_losses(self, ids: Sequence[int], *, replace: Mapping[int, Replace] | None = None) -> np.ndarray:
        """Next-token cross-entropy in nats for positions 1.. (shape len(ids) - 1)."""
        ...

    def generate(self, ids: Sequence[int], *, max_tokens: int, steer: Sequence[Add] = ()) -> list[int]:
        """Greedy continuation of `ids` (the new tokens only)."""
        ...


def check_layers(layers: Sequence[int], n_layers: int) -> list[int]:
    out = sorted(set(int(L) for L in layers))
    if not out or out[0] < 0 or out[-1] >= n_layers:
        raise ValueError(f"layers must be in 0..{n_layers - 1}, got {list(layers)}")
    return out


def check_adds(steer: Sequence[Add], n_layers: int, d_model: int) -> dict[int, np.ndarray]:
    """Sum the additions per layer, after checking layers and shapes."""
    out: dict[int, np.ndarray] = {}
    for a in steer:
        if not 0 <= a.layer < n_layers:
            raise ValueError(f"steering layer {a.layer} is not in 0..{n_layers - 1}")
        if a.delta.shape != (d_model,):
            raise ValueError(f"steering delta has shape {a.delta.shape}, expected ({d_model},)")
        out[a.layer] = out.get(a.layer, 0) + a.delta.astype(np.float32)
    return out


def encode(tokenizer: Any, text: str) -> list[int]:
    """Plain text to token IDs, without special tokens."""
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])
