"""The steering spec: one file format that every engine reads (MLX now; EasySteer or
vllm-lens on the GPU later). Reference: docs/INTERP.md, "Steering spec".

A spec is a small YAML (or JSON) file:

    spec_version: 1
    vectors: ../vectors/agent.safetensors   # vector file, relative to the spec file
    vectors_sha256: 3f1c...                 # optional; checked if given
    direction: desperate                    # a vector name in the file, or "random"
    seed: null                              # required for "random", else null
    layers: [39, 42, 45]                    # 0-based decoder layers (resid_post)
    alpha: 0.05                             # strength, a fraction of the mean residual norm
    sign: 1                                 # 1 or -1

Resolution (`resolve`): at each layer L, delta_L = sign * alpha * norm_L * u_L, where u_L
is the unit vector `direction@L` from the file (or `random_unit(d, seed, L)`), and norm_L
is the file's `resid_norm_mean[L]`, measured on the same backend as the vectors. The
engine adds delta_L to the residual stream after decoder layer L at every position,
prompt and generated tokens alike. Strength 0 must leave the model's output unchanged.

Random and unrelated-concept controls are specs too: "random" with a seed, or a concept
vector from the same file ("topic:bakery"). All have unit direction, so at the same alpha
they have the same norm as the target vector by construction.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .backend import Add
from .directions import random_unit
from .meta import file_sha256
from .vectors import VectorSet

SPEC_VERSION = 1
RANDOM = "random"


@dataclass(frozen=True)
class SteeringSpec:
    vectors: str
    direction: str
    layers: tuple[int, ...]
    alpha: float
    sign: int = 1
    seed: int | None = None
    vectors_sha256: str | None = None
    spec_version: int = SPEC_VERSION

    def __post_init__(self):
        if self.spec_version != SPEC_VERSION:
            raise ValueError(f"spec_version {self.spec_version} is not supported (expected {SPEC_VERSION})")
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be 1 or -1, got {self.sign!r}")
        if not (isinstance(self.alpha, (int, float)) and np.isfinite(self.alpha) and self.alpha >= 0):
            raise ValueError(f"alpha must be a finite number >= 0 (the sign goes in `sign`), got {self.alpha!r}")
        if not self.layers or len(set(self.layers)) != len(self.layers) or any(int(L) != L or L < 0 for L in self.layers):
            raise ValueError(f"layers must be distinct non-negative integers, got {self.layers!r}")
        if (self.direction == RANDOM) != (self.seed is not None):
            raise ValueError("seed is required for direction 'random' and must be null otherwise")
        object.__setattr__(self, "layers", tuple(int(L) for L in self.layers))

    @property
    def label(self) -> str:
        d = f"random:{self.seed}" if self.direction == RANDOM else self.direction
        return f"{d}{'+' if self.sign > 0 else '-'}{self.alpha:g}@{','.join(map(str, self.layers))}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["layers"] = list(self.layers)
        return {"spec_version": d.pop("spec_version"), **d}

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringSpec":
        known = {"spec_version", "vectors", "vectors_sha256", "direction", "seed", "layers", "alpha", "sign"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown steering spec fields: {sorted(unknown)}")
        missing = {"vectors", "direction", "layers", "alpha"} - set(d)
        if missing:
            raise ValueError(f"steering spec is missing: {sorted(missing)}")
        return cls(**{**d, "layers": tuple(d["layers"])})

    @classmethod
    def load(cls, path: str | Path) -> "SteeringSpec":
        text = Path(path).read_text()
        if str(path).endswith(".json"):
            d = json.loads(text)
        else:
            import yaml

            d = yaml.safe_load(text)
        spec = cls.from_dict(d)
        # vector paths are relative to the spec file
        p = Path(spec.vectors)
        if not p.is_absolute():
            spec = replace(spec, vectors=str((Path(path).parent / p).resolve()))
        return spec

    def save(self, path: str | Path) -> None:
        import yaml

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("# simtest interp steering spec (docs/INTERP.md)\n" + yaml.safe_dump(self.to_dict(), sort_keys=False))


def resolve(spec: SteeringSpec, vectors: VectorSet | None = None) -> list[Add]:
    """The spec as per-layer additions. Loads the vector file unless given (then the
    sha256 is only checked if the spec has one and `vectors` came from a file)."""
    if vectors is None:
        if spec.vectors_sha256 and file_sha256(spec.vectors) != spec.vectors_sha256:
            raise ValueError(f"{spec.vectors}: sha256 does not match the spec's vectors_sha256")
        vectors = VectorSet.load(spec.vectors)
    adds = []
    for L in spec.layers:
        u = random_unit(vectors.d_model, spec.seed, L) if spec.direction == RANDOM else vectors.get(spec.direction, L)
        coef = spec.sign * spec.alpha * vectors.resid_norm(L)
        adds.append(Add(L, (coef * u.astype(np.float64)).astype(np.float32)))
    return adds


def controls(target: SteeringSpec, *, random_seeds: Iterable[int] = (), concepts: Iterable[str] = (),
             signs: Sequence[int] = (1, -1)) -> list[SteeringSpec]:
    """Same-layer, same-strength control specs for `target`: random directions (one per
    seed) and unrelated-concept vectors, each with every sign in `signs`."""
    out = []
    for s in signs:
        out += [replace(target, direction=RANDOM, seed=int(seed), sign=s) for seed in random_seeds]
        out += [replace(target, direction=c, seed=None, sign=s) for c in concepts]
    return out


def dose_curve(target: SteeringSpec, alphas: Iterable[float]) -> list[SteeringSpec]:
    """The target at each signed strength (negative alphas flip the sign)."""
    return [replace(target, alpha=abs(float(a)), sign=(1 if a >= 0 else -1) * target.sign) for a in alphas]
