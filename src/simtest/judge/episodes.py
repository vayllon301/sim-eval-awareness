"""Which episodes the awareness judge reads (PLAN.md, Measures; N2).

Two sets, kept apart:

- **The test set**: a hand-label manifest's drawn episodes (pool.py), each with its opaque
  label id. The judge runs on them once, with a frozen prompt version, for the kappa pass.
- **Tuning sets**: anything else, listed in a small YAML spec kept next to the manifest,
  outside this repo:

      name: judge-tune
      root: ../..                          # paths below are relative to this
      test_set: labels/awareness-v1/manifest.json
      from_test_manifest: [not_drawn, skipped]   # optional: those lists of the manifest
      dirs: [experiment/logs/colab-n0-c0-c3]     # optional: every coding sample in them
      exclude: []                          # optional: episode keys to leave out

`TestSetGuard` refuses any tuning episode that is in the test set: by log, sample id and
epoch, by sample uuid, by transcript hash, or because one of its agent turns has exactly the
words (reasoning and visible text, however short) of an agent turn in a drawn episode. The last catches an
earlier attempt of a drawn sample that began the same way. Refused episodes are listed with
the reason, never dropped silently.

An episode's key is `<log>::<sample id>::<epoch>`, with the log path relative to the root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from . import transcript as T
from ..runqueue import summary_never_started
from .pool import _path

SPEC_KEYS = {"name", "root", "test_set", "from_test_manifest", "dirs", "exclude"}
MANIFEST_LISTS = ("not_drawn", "skipped")
MIN_FINGERPRINT = 1    # any agent turn with words; one with tool calls only has none


class EpisodeError(ValueError):
    pass


class TestSetError(EpisodeError):
    """A tuning source asked for an episode of the test set."""


def episode_key(log: str, sample_id, epoch) -> str:
    return f"{log}::{sample_id}::{epoch}"


def turn_fingerprints(steps: list[dict]) -> set[str]:
    """Hashes of the agent's own words, one per agent turn that has any."""
    out = set()
    for s in steps:
        if s["kind"] != "turn":
            continue
        body = (s["reasoning"] or "").strip() + "\x00" + (s["text"] or "").strip()
        if len(body) - 1 >= MIN_FINGERPRINT:
            out.add(hashlib.sha256(body.encode()).hexdigest())
    return out


def read_sample(root: Path, ep: dict):
    from inspect_ai.log import read_eval_log_sample

    s = read_eval_log_sample(str(Path(root) / ep["log"]), id=ep["sample_id"], epoch=ep["epoch"])
    if ep.get("uuid") and s.uuid != ep["uuid"]:
        raise EpisodeError(f"{ep['key']}: uuid {s.uuid}, expected {ep['uuid']}")
    return s


def _manifest(path: Path) -> tuple[dict, Path]:
    path = Path(path)
    m = json.loads(path.read_text())
    if not isinstance(m, dict) or not isinstance(m.get("episodes"), list):
        raise EpisodeError(f"{path}: not a hand-label manifest")
    return m, (path.parent / m["root"]).resolve()


def _ep(e: dict, **extra) -> dict:
    return {"key": episode_key(e["log"], e["sample_id"], e["epoch"]), "log": e["log"],
            "sample_id": e["sample_id"], "epoch": e["epoch"], "uuid": e.get("uuid"), **extra}


class TestSetGuard:
    """Knows the test set's episodes and refuses them. The drawn episodes' agent-turn
    fingerprints are computed (by reading their samples) only when first needed; their text
    is never kept or shown."""

    __test__ = False   # not a pytest class

    def __init__(self, manifest_path: Path, read=None):
        m, self.root = _manifest(manifest_path)
        self.manifest_path = Path(manifest_path)
        self.drawn = [_ep(e, label_id=e["label_id"], transcript_sha256=e.get("transcript_sha256"))
                      for e in m["episodes"]]
        self.keys = {e["key"] for e in self.drawn}
        self.uuids = {e["uuid"] for e in self.drawn if e["uuid"]}
        self.shas = {e["transcript_sha256"] for e in self.drawn if e["transcript_sha256"]}
        self._read = read or (lambda ep: read_sample(self.root, ep))
        self._fps: set[str] | None = None

    def fingerprints(self) -> set[str]:
        if self._fps is None:
            fps: set[str] = set()
            for ep in self.drawn:
                fps |= turn_fingerprints(T.episode_steps(self._read(ep)))
            self._fps = fps
        return self._fps

    def reason(self, ep: dict, steps: list[dict] | None = None, uuid: str | None = None) -> str | None:
        """Why `ep` is (part of) the test set, or None."""
        if ep["key"] in self.keys:
            return "drawn in the test-set manifest"
        if (u := uuid or ep.get("uuid")) and u in self.uuids:
            return "same sample uuid as a drawn episode"
        if steps is not None:
            if T.transcript_sha256(steps) in self.shas:
                return "same transcript as a drawn episode"
            if shared := turn_fingerprints(steps) & self.fingerprints():
                return f"{len(shared)} agent turn(s) with the same words as a drawn episode"
        return None

    def check(self, ep: dict, steps: list[dict] | None = None, uuid: str | None = None) -> None:
        if why := self.reason(ep, steps, uuid):
            raise TestSetError(f"{ep['key']}: {why}; the test set is never used for tuning")


def test_set(manifest_path: Path) -> tuple[list[dict], Path]:
    """The drawn episodes (with label ids and the transcript hash they were labelled on)."""
    guard = TestSetGuard(manifest_path)
    return guard.drawn, guard.root


def load_tuning_spec(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise EpisodeError(f"{path}: not a mapping")
    if unknown := set(raw) - SPEC_KEYS:
        raise EpisodeError(f"{path}: unknown keys {sorted(unknown)}")
    if "test_set" not in raw or "name" not in raw:
        raise EpisodeError(f"{path}: needs name and test_set (the manifest whose drawn episodes are refused)")
    raw.setdefault("root", ".")
    raw.setdefault("from_test_manifest", [])
    raw.setdefault("dirs", [])
    raw.setdefault("exclude", [])
    if bad := set(raw["from_test_manifest"]) - set(MANIFEST_LISTS):
        raise EpisodeError(f"{path}: from_test_manifest takes {list(MANIFEST_LISTS)}, not {sorted(bad)}")
    if not raw["from_test_manifest"] and not raw["dirs"]:
        raise EpisodeError(f"{path}: no episodes (from_test_manifest and dirs are both empty)")
    return raw


def _dir_episodes(root: Path, d: str) -> list[dict]:
    """Every coding sample in a log directory; a sample run in several attempt logs (same
    uuid) counts once, from the newest log (log files are named by their start time)."""
    from inspect_ai.log import list_eval_logs, read_eval_log, read_eval_log_sample_summaries

    log_dir = root / d
    if not log_dir.is_dir():
        raise EpisodeError(f"no log directory {log_dir}")
    paths = sorted((_path(i.name) for i in list_eval_logs(str(log_dir))), key=lambda p: p.name)
    by_uuid: dict[str, dict] = {}
    out = []
    for p in paths:
        head = read_eval_log(str(p), header_only=True)
        if head.eval.task_registry_name and not head.eval.task_registry_name.endswith("coding"):
            continue
        rel = p.resolve().relative_to(root.resolve()).as_posix()
        for s in read_eval_log_sample_summaries(str(p)):
            if summary_never_started(p, head, s):   # the run queue never started it: not an episode
                continue
            ep = _ep({"log": rel, "sample_id": s.id, "epoch": s.epoch, "uuid": s.uuid}, source=d)
            if s.uuid:
                by_uuid[s.uuid] = ep      # a newer log replaces an older attempt
            else:
                out.append(ep)
    return out + list(by_uuid.values())


def tuning_set(spec_path: Path, guard: TestSetGuard | None = None, read=None) -> tuple[list[dict], list[dict], Path]:
    """(episodes, refused, root). Each episode is checked against the test set twice: by key
    and uuid when listed, and by transcript and agent-turn fingerprints when its sample is
    read (`load`)."""
    spec_path = Path(spec_path)
    spec = load_tuning_spec(spec_path)
    root = (spec_path.parent / spec["root"]).resolve()
    guard = guard or TestSetGuard(root / spec["test_set"], read=read)
    m, _ = _manifest(guard.manifest_path)
    listed: list[dict] = []
    for name in spec["from_test_manifest"]:
        listed += [_ep(e, source=f"manifest {name}") for e in m.get(name, [])]
    for d in spec["dirs"]:
        listed += _dir_episodes(root, d)
    seen, episodes, refused = set(), [], []
    exclude = set(spec["exclude"])
    for ep in listed:
        if ep["key"] in seen:
            continue
        seen.add(ep["key"])
        if ep["key"] in exclude:
            refused.append({**ep, "reason": "excluded in the tuning spec"})
        elif why := guard.reason(ep):
            refused.append({**ep, "reason": why})
        else:
            episodes.append(ep)
    return sorted(episodes, key=lambda e: e["key"]), refused, root
