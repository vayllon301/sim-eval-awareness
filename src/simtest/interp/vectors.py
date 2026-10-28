"""The vector file: unit directions per layer, with full metadata.

One safetensors file per model and vector set. Tensors are float32 unit vectors named
"<name>@<layer>" (for example "desperate@42", "topic:bakery@42", "eval_awareness@42").
The header's metadata holds one key, "simtest_interp", a JSON object:

  format, format_version   "simtest-interp-vectors", 1
  model                    backend info: engine, path, fingerprint, revision,
                           quantization, versions (backend.info)
  layers, d_model, names
  resid_norm_mean          {layer: mean L2 norm of the residual stream} on the reference
                           texts (`norm_reference` says which); steering strengths are
                           fractions of it
  offset                   {name: {layer: mean projection of the neutral texts}}, so a
                           reading can be centred (reading - offset)
  vector_info              {name: {method, dataset, ...}} per vector
  datasets                 {dataset: {path, sha256, n, generator, ...}}
  eval                     validation numbers per layer (held-out, leave-one-out)
  created, git_commit

Only metadata that is JSON goes in; arrays are tensors. Nothing here needs torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT, FORMAT_VERSION = "simtest-interp-vectors", 1
META_KEY = "simtest_interp"


def key(name: str, layer: int) -> str:
    if "@" in name:
        raise ValueError(f"vector names can't contain '@': {name!r}")
    return f"{name}@{int(layer)}"


@dataclass
class VectorSet:
    vectors: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(dict.fromkeys(n for n, _ in self.vectors))

    @property
    def layers(self) -> list[int]:
        return sorted({L for _, L in self.vectors})

    @property
    def d_model(self) -> int:
        return int(next(iter(self.vectors.values())).shape[0])

    def get(self, name: str, layer: int) -> np.ndarray:
        try:
            return self.vectors[(name, int(layer))]
        except KeyError:
            raise KeyError(f"no vector {name!r} at layer {layer} (names {self.names}, layers {self.layers})") from None

    def add(self, name: str, layer: int, v: np.ndarray) -> None:
        v = np.asarray(v, dtype=np.float64)
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or abs(n - 1) > 1e-3:
            raise ValueError(f"{name}@{layer}: vectors are stored as unit vectors (norm {n})")
        if self.vectors and v.shape != (self.d_model,):
            raise ValueError(f"{name}@{layer}: shape {v.shape}, expected ({self.d_model},)")
        self.vectors[(name, int(layer))] = (v / n).astype(np.float32)

    def resid_norm(self, layer: int) -> float:
        norms = self.meta.get("resid_norm_mean") or {}
        if str(layer) not in norms:
            raise KeyError(f"no mean residual norm for layer {layer} in the vector file")
        return float(norms[str(layer)])

    def save(self, path: str | Path) -> None:
        from safetensors.numpy import save_file

        meta = {**self.meta, "format": FORMAT, "format_version": FORMAT_VERSION, "layers": self.layers,
                "names": self.names, "d_model": self.d_model}
        tensors = {key(n, L): np.ascontiguousarray(v, dtype=np.float32) for (n, L), v in self.vectors.items()}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(path), metadata={META_KEY: json.dumps(meta, sort_keys=True)})

    @classmethod
    def load(cls, path: str | Path) -> "VectorSet":
        from safetensors import safe_open

        with safe_open(str(path), framework="numpy") as f:
            header = f.metadata() or {}
            if META_KEY not in header:
                raise ValueError(f"{path}: not a simtest interp vector file (no '{META_KEY}' metadata)")
            meta = json.loads(header[META_KEY])
            if meta.get("format") != FORMAT or meta.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"{path}: format {meta.get('format')} v{meta.get('format_version')}, "
                                 f"expected {FORMAT} v{FORMAT_VERSION}")
            vectors = {}
            for k in f.keys():
                name, _, layer = k.rpartition("@")
                vectors[(name, int(layer))] = f.get_tensor(k).astype(np.float32)
        order = {n: i for i, n in enumerate(meta.get("names") or [])}  # safetensors sorts its keys
        vectors = dict(sorted(vectors.items(), key=lambda kv: (order.get(kv[0][0], len(order)), kv[0][0], kv[0][1])))
        return cls(vectors=vectors, meta=meta)
