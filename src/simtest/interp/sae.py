"""TopK SAEs (Qwen-Scope) on numpy: a torch-free .pt reader, encode, decode, FVU.

Qwen-Scope (arXiv:2605.11887, HF collection Qwen/qwen-scope) ships one `layer{L}.sae.pt`
per layer: a dict of float32 tensors W_enc (d_sae, d_model), b_enc (d_sae,), W_dec
(d_model, d_sae), b_dec (d_model,). L is the 0-based decoder layer whose output
(resid_post) it was trained on. The encoder follows the repos' app.py:
pre = relu(x @ W_enc.T + b_enc), keep the k largest, no b_dec subtracted first and no
input scaling. The decoder isn't in their code; we use the standard
x_hat = acts @ W_dec.T + b_dec.

`read_pt` parses PyTorch's zip format with a restricted unpickler that only rebuilds
plain tensors (anything else is refused) and memory-maps their storage, so a 3.4 GB
layer opens instantly and nothing in the file can run code.
"""

from __future__ import annotations

import pickle
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
           "BFloat16Storage": "bfloat16", "LongStorage": np.int64, "IntStorage": np.int32,
           "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_}


@dataclass
class _StorageType:
    name: str


@dataclass
class _Storage:
    dtype: object
    key: str
    numel: int


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.astype(np.uint32) << 16).view(np.float32)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from a .pt file")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        _, stype, key, _location, numel = pid
        if not isinstance(stype, _StorageType):
            raise pickle.UnpicklingError(f"unexpected storage type {stype!r}")
        return _Storage(_DTYPES[stype.name], str(key), int(numel))


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    return ("tensor", storage, int(offset), tuple(size), tuple(stride))


def read_pt(path: str | Path) -> dict[str, np.ndarray]:
    """A torch.save'd dict of tensors as numpy arrays (memory-mapped, read only)."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        pkl = next((n for n in z.namelist() if n.endswith("/data.pkl")), None)
        if pkl is None:
            raise ValueError(f"{path}: not a PyTorch zip checkpoint")
        root = pkl[: -len("data.pkl")]
        obj = _Unpickler(z.open(pkl)).load()
        infos = {i.filename: i for i in z.infolist()}
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a dict of tensors, got {type(obj).__name__}")
    out = {}
    with open(path, "rb") as f:
        for name, t in obj.items():
            if not (isinstance(t, tuple) and t and t[0] == "tensor"):
                raise ValueError(f"{path}: entry {name!r} is not a tensor")
            _, st, offset, size, stride = t
            info = infos[f"{root}data/{st.key}"]
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{path}: storage {st.key} is compressed")
            f.seek(info.header_offset)
            head = f.read(30)
            data_start = info.header_offset + 30 + int.from_bytes(head[26:28], "little") + int.from_bytes(head[28:30], "little")
            bf16 = st.dtype == "bfloat16"
            dt = np.dtype(np.uint16 if bf16 else st.dtype)
            raw = np.memmap(path, dtype=dt, mode="r", offset=data_start, shape=(st.numel,))
            arr = np.lib.stride_tricks.as_strided(raw[offset:], shape=size, strides=tuple(s * dt.itemsize for s in stride))
            out[name] = _bf16_to_f32(arr) if bf16 else arr
    return out


@dataclass
class TopKSAE:
    W_enc: np.ndarray  # (d_sae, d_model)
    b_enc: np.ndarray  # (d_sae,)
    W_dec: np.ndarray  # (d_model, d_sae)
    b_dec: np.ndarray  # (d_model,)
    k: int
    info: dict = field(default_factory=dict)
    _rows: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        d_sae, d = self.W_enc.shape
        if self.W_dec.shape != (d, d_sae) or self.b_enc.shape != (d_sae,) or self.b_dec.shape != (d,):
            raise ValueError("inconsistent SAE shapes")
        if not 0 < self.k <= d_sae:
            raise ValueError(f"k must be in 1..{d_sae}")

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[1]

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[0]

    def encode(self, x: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """(tokens, d_model) -> (indices (tokens, k) int32, values (tokens, k) float32),
        each row sorted by value, largest first."""
        x = np.asarray(x, dtype=np.float32)
        idx = np.empty((len(x), self.k), dtype=np.int32)
        val = np.empty((len(x), self.k), dtype=np.float32)
        for a in range(0, len(x), chunk):
            pre = x[a:a + chunk] @ self.W_enc.T + self.b_enc
            np.maximum(pre, 0, out=pre)
            top = np.argpartition(pre, -self.k, axis=1)[:, -self.k:]
            v = np.take_along_axis(pre, top, axis=1)
            order = np.argsort(-v, axis=1)
            idx[a:a + chunk] = np.take_along_axis(top, order, axis=1)
            val[a:a + chunk] = np.take_along_axis(v, order, axis=1)
        return idx, val

    def decode(self, idx: np.ndarray, val: np.ndarray, chunk: int = 256) -> np.ndarray:
        if self._rows is None:  # decoder columns as contiguous rows, copied once (gathers from a memmap are slow)
            self._rows = np.ascontiguousarray(self.W_dec.T, dtype=np.float32)
        out = np.empty((len(idx), self.d_model), dtype=np.float32)
        for a in range(0, len(idx), chunk):
            out[a:a + chunk] = np.einsum("tk,tkd->td", val[a:a + chunk], self._rows[idx[a:a + chunk]]) + self.b_dec
        return out

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        return self.decode(*self.encode(x))


def load_qwen_scope(path: str | Path, k: int | None = None) -> TopKSAE:
    """One layer file (layer{L}.sae.pt); k from the repo's config.json next to it."""
    import json

    path = Path(path)
    cfg_path = path.parent / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    k = k or cfg.get("k")
    if not k:
        raise ValueError("k is not given and there is no config.json with k next to the SAE file")
    w = read_pt(path)
    return TopKSAE(W_enc=w["W_enc"], b_enc=w["b_enc"], W_dec=w["W_dec"], b_dec=w["b_dec"], k=int(k),
                   info={"path": str(path), "repo": cfg.get("hf_repo_id"), "base_model": cfg.get("base_model"),
                         "hook_point": cfg.get("hook_point"), "k": int(k), "file_size": path.stat().st_size,
                         "revision": path.parent.name if path.parent.parent.name == "snapshots" else None})


def fvu(x: np.ndarray, x_hat: np.ndarray) -> float:
    """Fraction of variance unexplained: sum |x - x_hat|^2 / sum |x - mean(x)|^2."""
    x, x_hat = np.asarray(x, np.float64), np.asarray(x_hat, np.float64)
    return float(((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
