"""The hand-label sample (PLAN.md, Measures; N2): the pool of episodes, a seeded draw, and
the manifest that records it.

The pool spec is a small YAML file kept next to the manifest, outside this repo:

    name: awareness-v1
    root: ../..          # log paths are relative to this, itself relative to the spec file
    seed: 20261101
    n: 100
    take_all: C3         # every scored episode of this condition
    fill: C0             # the rest from this condition, stratified by scenario
    scenarios: [...]     # optional: only these scenario ids
    dirs: [experiment/logs/colab-g4-s5-c3r, ...]

Each directory is read with simtest's own loader (`coding.analysis.load_run`, as `simtest
summarize` does): a sample run that appears in several attempt logs (same uuid) counts once,
from the newest log, and an episode whose end reason isn't valid (`analysis.is_excluded`:
an infrastructure error, an output error, an interrupted attempt, lost metrics) is not
drawn. Excluded episodes, other conditions and scenarios, duplicates of a counted sample,
and log files without samples are each listed in the manifest with the reason, so nothing
is dropped silently.

The draw: every pool episode of `take_all`, then `n` minus that many of `fill`, split
across scenarios as equally as the pool allows (a scenario with too few gives all it has
and the others share the rest; a last remainder goes to scenarios picked with the seed),
drawn at random within each scenario. Each drawn episode gets an opaque label id
(L001-L100, assigned at random) and a place in a separately shuffled order. The same pool
and seed give the same draw. The manifest records the condition of every episode: never
open it while labelling.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

from . import transcript as T

SPEC_KEYS = {"name", "root", "seed", "n", "take_all", "fill", "scenarios", "dirs"}
REQUIRED = {"name", "seed", "n", "take_all", "fill", "dirs"}
REPO = Path(__file__).resolve().parents[3]


class PoolError(ValueError):
    pass


def load_spec(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise PoolError(f"{path}: not a mapping")
    if unknown := set(raw) - SPEC_KEYS:
        raise PoolError(f"{path}: unknown keys {sorted(unknown)}")
    if missing := REQUIRED - set(raw):
        raise PoolError(f"{path}: missing keys {sorted(missing)}")
    if not isinstance(raw["n"], int) or raw["n"] < 1 or not isinstance(raw["seed"], int):
        raise PoolError(f"{path}: n and seed must be integers (n >= 1)")
    if not raw["dirs"] or len(set(raw["dirs"])) != len(raw["dirs"]):
        raise PoolError(f"{path}: dirs must be a non-empty list without repeats")
    raw.setdefault("root", ".")
    raw.setdefault("scenarios", None)
    return raw


def _path(info_name: str) -> Path:
    u = urlparse(info_name)
    return Path(unquote(u.path)) if u.scheme == "file" else Path(info_name)


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _last_line(text: str | None) -> str:
    lines = (text or "").strip().splitlines()
    return lines[-1][:200] if lines else ""


def scan(spec: dict, root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """(pool, skipped, logs). Every sample of every log file ends up in exactly one of: the
    pool, `skipped` (with a reason), or a log's `duplicates` count (the same sample run as
    one counted from a newer log)."""
    from inspect_ai.log import list_eval_logs, read_eval_log, read_eval_log_sample_summaries

    from ..coding import analysis as A

    pool, skipped, logs = [], [], []
    keep = {spec["take_all"], spec["fill"]}
    scenarios = set(spec["scenarios"]) if spec["scenarios"] else None
    for d in spec["dirs"]:
        log_dir = root / d
        if not log_dir.is_dir():
            raise PoolError(f"no log directory {log_dir}")
        eps, _ = A.load_run(log_dir)
        files: dict[str, dict] = {}
        index: dict[tuple, object] = {}
        for info in list_eval_logs(str(log_dir)):
            path = _path(info.name)
            head = read_eval_log(str(path), header_only=True)
            if head.eval.task_registry_name and not head.eval.task_registry_name.endswith("coding"):
                continue
            sums = read_eval_log_sample_summaries(str(path))
            files[path.name] = {"path": _rel(path, root), "status": head.status,
                                "condition": head.eval.task_args.get("condition"), "samples": len(sums),
                                "counted": 0, "duplicates": 0}
            for s in sums:
                key = (path.name, (s.metadata or {}).get("scenario", s.id), s.epoch)
                if key in index:
                    raise PoolError(f"{d}: two samples for {key}")
                index[key] = s
        counted: dict[str, str] = {}   # uuid -> the file it is counted from
        for ep in eps:
            s = index.get((ep["log"], ep["scenario"], ep["epoch"]))
            if s is None:
                raise PoolError(f"{d}: no sample summary for {ep['log']} {ep['scenario']} epoch {ep['epoch']}")
            if s.uuid:
                counted[s.uuid] = ep["log"]
            files[ep["log"]]["counted"] += 1
            rec = {"log": files[ep["log"]]["path"], "sample_id": s.id, "epoch": s.epoch, "uuid": s.uuid,
                   "run": d, "condition": ep["condition"], "simulator": ep["simulator"], "agent": ep["agent"],
                   "model": ep["model"], "scenario": ep["scenario"], "variant": ep["variant"],
                   "end_reason": ep["end_reason"]}
            if A.is_excluded(ep):
                why = f"excluded by simtest summarize: end reason {ep['end_reason']}"
                if err := _last_line(ep.get("error")):
                    why += f" ({err})"
                skipped.append({**rec, "reason": why})
            elif ep["condition"] not in keep:
                skipped.append({**rec, "reason": f"condition {ep['condition']} is not drawn"})
            elif scenarios is not None and ep["scenario"] not in scenarios:
                skipped.append({**rec, "reason": f"scenario {ep['scenario']} is not in the spec"})
            else:
                pool.append(rec)
        for (fname, _, _), s in index.items():
            owner = counted.get(s.uuid)
            if owner is not None and owner != fname:
                files[fname]["duplicates"] += 1
        for f in files.values():
            if f["counted"] + f["duplicates"] != f["samples"]:
                raise PoolError(f"{f['path']}: {f['samples']} samples, but {f['counted']} counted and "
                                f"{f['duplicates']} duplicates")
            if f["samples"] == 0:
                f["note"] = "the log has no samples"
            logs.append(f)
    # A later attempt of a skipped sample that made the pool (same run, sample and epoch).
    drawn_keys = {(e["run"], e["condition"], e["sample_id"], e["epoch"]) for e in pool}
    for s in skipped:
        if (s["run"], s["condition"], s["sample_id"], s["epoch"]) in drawn_keys:
            s["reason"] += "; a later attempt of the same sample is in the pool"
    key = lambda e: (e["log"], str(e["sample_id"]), e["epoch"])  # noqa: E731
    return sorted(pool, key=key), sorted(skipped, key=key), sorted(logs, key=lambda f: f["path"])


def equal_quotas(sizes: dict[str, int], k: int, rng: random.Random) -> dict[str, int]:
    """Split k across strata as equally as their sizes allow."""
    if k > sum(sizes.values()):
        raise PoolError(f"asked for {k} episodes from strata holding {sum(sizes.values())}")
    quotas = {s: 0 for s in sizes}
    left = k
    open_ = sorted(s for s in sizes if sizes[s] > 0)
    while left > 0:
        share = left // len(open_)
        if share == 0:
            for s in rng.sample(open_, left):
                quotas[s] += 1
            break
        for s in open_:
            add = min(share, sizes[s] - quotas[s])
            quotas[s] += add
            left -= add
        open_ = [s for s in open_ if quotas[s] < sizes[s]]
    return quotas


def draw(pool: list[dict], n: int, seed: int, take_all: str, fill: str) -> tuple[list[dict], dict]:
    """(drawn episodes in labelling order, the quotas per fill scenario)."""
    rng = random.Random(seed)
    key = lambda e: (e["log"], str(e["sample_id"]), e["epoch"])  # noqa: E731
    pool = sorted(pool, key=key)
    first = [e for e in pool if e["condition"] == take_all]
    k = n - len(first)
    if k < 0:
        raise PoolError(f"{len(first)} {take_all} episodes, more than n = {n}")
    strata: dict[str, list[dict]] = defaultdict(list)
    for e in pool:
        if e["condition"] == fill:
            strata[e["scenario"]].append(e)
    quotas = equal_quotas({s: len(v) for s, v in strata.items()}, k, rng)
    chosen = list(first)
    for s in sorted(strata):
        chosen += sorted(rng.sample(strata[s], quotas[s]), key=key)
    ids = [f"L{i:03d}" for i in range(1, n + 1)]
    rng.shuffle(ids)
    order = list(range(1, n + 1))
    rng.shuffle(order)
    out = [{"label_id": i, "order": o, **e} for e, i, o in zip(chosen, ids, order)]
    return sorted(out, key=lambda e: e["order"]), dict(sorted(quotas.items()))


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


_CONDITION = re.compile(r"\bC[0-3]\b")


def leak_terms(ep: dict) -> list[str]:
    """Strings that would tell the labeller where an episode comes from if the agent's
    transcript happened to contain them (checked, not stripped: it is what the agent saw)."""
    terms = {Path(ep["run"]).name, Path(ep["log"]).stem, ep["scenario"], ep["model"], str(ep["agent"])}
    if ep.get("simulator") and ep["simulator"] != "none":
        terms.add(ep["simulator"])
    return sorted(t for t in terms if t and len(t) >= 4)


def leak_hits(text: str, ep: dict) -> dict[str, int]:
    hits = {t: len(re.findall(rf"(?<![\w-]){re.escape(t)}(?![\w-])", text)) for t in leak_terms(ep)}
    hits["C0-C3 label"] = len(_CONDITION.findall(text))
    return {t: c for t, c in hits.items() if c}


def read_sample(root: Path, ep: dict):
    from inspect_ai.log import read_eval_log_sample

    s = read_eval_log_sample(str(root / ep["log"]), id=ep["sample_id"], epoch=ep["epoch"])
    if ep.get("uuid") and s.uuid != ep["uuid"]:
        raise PoolError(f"{ep['log']} {ep['sample_id']} epoch {ep['epoch']}: uuid {s.uuid}, expected {ep['uuid']}")
    return s


def _counts(eps: list[dict]) -> dict:
    c = Counter((e["condition"], e["scenario"]) for e in eps)
    out: dict = defaultdict(dict)
    for (cond, sc), k in sorted(c.items()):
        out[cond][sc] = k
    return {cond: {"total": sum(v.values()), **v} for cond, v in out.items()}


def build_manifest(spec_path: Path) -> dict:
    spec_path = Path(spec_path)
    spec = load_spec(spec_path)
    root = (spec_path.parent / spec["root"]).resolve()
    pool, skipped, logs = scan(spec, root)
    drawn, quotas = draw(pool, spec["n"], spec["seed"], spec["take_all"], spec["fill"])
    leaks = []
    for ep in drawn:
        steps = T.episode_steps(read_sample(root, ep))
        text = T.render_text(steps)
        ep["n_turns"] = len(T.turn_ids(steps))
        ep["transcript_sha256"] = T.transcript_sha256(steps)
        if hits := leak_hits(text, ep):
            leaks.append({"label_id": ep["label_id"], "hits": hits})
    drawn_keys = {(e["log"], str(e["sample_id"]), e["epoch"]) for e in drawn}
    return {
        "name": spec["name"],
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": Path(spec["root"]).as_posix(),
        "spec": spec,
        "code": {"simtest_commit": _git("rev-parse", "--short", "HEAD"),
                 "dirty": bool(_git("status", "--porcelain")), "render_version": T.RENDER_VERSION},
        "rule": (f"every scored {spec['take_all']} episode in the pool, then {spec['fill']} episodes "
                 "stratified by scenario, as equally as the pool allows; seed "
                 f"{spec['seed']}"),
        "counts": {"pool": _counts(pool), "drawn": _counts(drawn), "fill_quotas": quotas,
                   "skipped": dict(sorted(Counter(s["reason"].split(" (")[0].split(";")[0]
                                                  for s in skipped).items())),
                   "not_drawn": len(pool) - len(drawn)},
        "leak_check": leaks,
        "episodes": drawn,
        "not_drawn": [{k: e[k] for k in ("log", "sample_id", "epoch", "condition", "scenario")}
                      for e in pool if (e["log"], str(e["sample_id"]), e["epoch"]) not in drawn_keys],
        "skipped": skipped,
        "logs": logs,
    }


def write_manifest(manifest: dict, out: Path, force: bool = False) -> None:
    out = Path(out)
    if out.exists() and not force:
        raise PoolError(f"{out} exists: labels refer to its label ids. Use --force to replace it.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")


def format_counts(manifest: dict) -> str:
    c = manifest["counts"]
    lines = [f"{manifest['name']}: {len(manifest['episodes'])} drawn ({manifest['rule']})"]
    for what in ("pool", "drawn"):
        for cond, v in c[what].items():
            lines.append(f"  {what} {cond}: " + ", ".join(f"{k} {n}" for k, n in v.items()))
    lines.append(f"  not drawn (in the pool): {c['not_drawn']}")
    lines.append(f"  skipped: {sum(c['skipped'].values())}")
    for reason, n in c["skipped"].items():
        lines.append(f"    {n} x {reason}")
    dup = sum(f["duplicates"] for f in manifest["logs"])
    empty = [f["path"] for f in manifest["logs"] if f["samples"] == 0]
    lines.append(f"  log files: {len(manifest['logs'])}; samples counted once from a newer log: {dup}")
    for p in empty:
        lines.append(f"    no samples: {p}")
    lines.append(f"  leak check: {len(manifest['leak_check'])} transcript(s) with hits")
    for x in manifest["leak_check"]:
        lines.append(f"    {x['label_id']}: {x['hits']}")
    return "\n".join(lines)
