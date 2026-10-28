"""What every interp output records about where it came from.

The weights fingerprint is fast, not a full hash: sha256 over config.json, the safetensors
index, and for each weight file its name, size and its first and last 4 MiB. A full hash
of 29 GB would take a minute per run. Two different weight sets with the same config,
sizes and edges are not a realistic risk for our use (telling apart quantizations,
conversions and revisions). If the directory is a Hugging Face cache snapshot, the
revision is recorded too.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

EDGE = 4 << 20  # bytes hashed at each end of a weight file


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def weights_fingerprint(model_dir: str | Path) -> str:
    """Fast fingerprint of a model directory (see the module docstring). 'fp1:' + hex."""
    d = Path(model_dir).expanduser().resolve()
    h = hashlib.sha256(b"simtest-weights-fp1")
    for name in ("config.json", "model.safetensors.index.json"):
        p = d / name
        if p.exists():
            h.update(name.encode() + b"\0" + p.read_bytes())
    files = sorted(p for p in d.iterdir() if p.suffix in (".safetensors", ".bin", ".gguf", ".pt"))
    if not files:
        raise FileNotFoundError(f"no weight files in {d}")
    for p in files:
        size = p.stat().st_size
        h.update(f"{p.name}\0{size}\0".encode())
        with open(p, "rb") as f:
            h.update(f.read(EDGE))
            if size > 2 * EDGE:
                f.seek(size - EDGE)
                h.update(f.read(EDGE))
    return "fp1:" + h.hexdigest()


def hf_revision(model_dir: str | Path) -> str | None:
    """The commit of a Hugging Face cache snapshot (…/snapshots/<sha>), else None."""
    parts = Path(model_dir).expanduser().resolve().parts
    if len(parts) >= 2 and parts[-2] == "snapshots":
        return parts[-1]
    return None


def quantization(model_dir: str | Path) -> str:
    """A short description of the quantization in config.json ('none' if absent)."""
    cfg = json.loads((Path(model_dir).expanduser() / "config.json").read_text())
    q = cfg.get("quantization") or cfg.get("quantization_config")
    if not q:
        return "none (" + str(cfg.get("text_config", cfg).get("dtype") or cfg.get("torch_dtype") or "?") + ")"
    overrides = sum(1 for v in q.values() if isinstance(v, dict))
    base = f"{q.get('mode', q.get('quant_method', '?'))} {q.get('bits', '?')}-bit, group {q.get('group_size', '?')}"
    return base + (f", {overrides} per-layer overrides" if overrides else "")


INTERP_PATHS = ("src/simtest/interp", "explore/interp")


def git_commit(repo: str | Path | None = None) -> str | None:
    """HEAD of the experiment repo, with '+dirty' if the interp code has uncommitted changes.

    Only the interp paths count: other work in the same checkout doesn't change these tools."""
    cwd = Path(repo or Path(__file__).resolve().parents[3])
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *INTERP_PATHS], cwd=cwd, capture_output=True,
                               text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
