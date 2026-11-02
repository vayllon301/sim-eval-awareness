"""The hand-label sample (judge/pool.py): the pool read with simtest's loader, exclusions and
duplicates accounted for, a seeded stratified draw, and a deterministic manifest."""

from __future__ import annotations

import json
import random
import shutil

import pytest
import yaml

from simtest.judge import pool as P
from judge_helpers import Truncating, patch_machine, run_episodes

BENIGN = ["pricing_cap", "pricing_cap_benign", "churn_format", "churn_format_benign"]


def fake_pool(n_c3=3, c0=None):
    c0 = c0 or {"a": 5, "a_benign": 5, "b": 2, "b_benign": 4}
    out = [{"log": "r3/x.eval", "sample_id": f"s{i}", "epoch": 1, "condition": "C3", "scenario": "a"}
           for i in range(n_c3)]
    for sc, k in c0.items():
        out += [{"log": f"r0/{sc}.eval", "sample_id": sc, "epoch": e, "condition": "C0", "scenario": sc}
                for e in range(1, k + 1)]
    return out


def test_equal_quotas_fill_small_strata_first():
    q = P.equal_quotas({"a": 17, "b": 17, "c": 10, "d": 10, "e": 17, "f": 16}, 81, random.Random(1))
    assert sum(q.values()) == 81 and q["c"] == q["d"] == 10
    assert sorted(q[s] for s in "abef") == [15, 15, 15, 16]
    with pytest.raises(P.PoolError):
        P.equal_quotas({"a": 1}, 2, random.Random(1))


def test_draw_takes_every_c3_and_stratifies_c0():
    drawn, quotas = P.draw(fake_pool(), n=12, seed=7, take_all="C3", fill="C0")
    assert sum(e["condition"] == "C3" for e in drawn) == 3
    assert sum(quotas.values()) == 9 and quotas["b"] == 2          # b gives all it has
    assert sorted(quotas[s] for s in ("a", "a_benign", "b_benign")) == [2, 2, 3]
    assert sorted(e["label_id"] for e in drawn) == [f"L{i:03d}" for i in range(1, 13)]
    assert [e["order"] for e in drawn] == list(range(1, 13))


def test_draw_is_deterministic_for_a_seed_and_ignores_pool_order():
    pool = fake_pool()
    a, _ = P.draw(pool, n=12, seed=7, take_all="C3", fill="C0")
    b, _ = P.draw(list(reversed(pool)), n=12, seed=7, take_all="C3", fill="C0")
    c, _ = P.draw(pool, n=12, seed=8, take_all="C3", fill="C0")
    assert a == b
    assert [(e["sample_id"], e["epoch"], e["label_id"]) for e in a] != \
        [(e["sample_id"], e["epoch"], e["label_id"]) for e in c]


def test_draw_refuses_more_c3_than_n():
    with pytest.raises(P.PoolError):
        P.draw(fake_pool(n_c3=5), n=4, seed=1, take_all="C3", fill="C0")


@pytest.fixture
def logs(monkeypatch, tmp_path):
    """Three log directories of offline episodes: C0 (4 scenarios x 2 epochs), C3 (2 scored),
    and C3 with one episode excluded (simulator truncated)."""
    patch_machine(monkeypatch, [("REAL", 0)])
    run_episodes(tmp_path / "logs" / "c0", "C0", BENIGN, epochs=2)
    run_episodes(tmp_path / "logs" / "c3", "C3", ["pricing_cap", "pricing_cap_benign"])
    run_episodes(tmp_path / "logs" / "c3t", "C3", ["churn_format"], sim=Truncating())
    spec = {"name": "test-v1", "root": "..", "seed": 11, "n": 6, "take_all": "C3", "fill": "C0",
            "dirs": ["logs/c0", "logs/c3", "logs/c3t"]}
    (tmp_path / "labels").mkdir()
    path = tmp_path / "labels" / "pool.yaml"
    path.write_text(yaml.safe_dump(spec))
    return tmp_path, path


def test_manifest_from_logs(logs):
    root, spec = logs
    m = P.build_manifest(spec)
    eps = m["episodes"]
    assert len(eps) == 6
    assert sum(e["condition"] == "C3" for e in eps) == 2                       # every scored C3
    assert {e["scenario"] for e in eps if e["condition"] == "C0"} == set(BENIGN)  # one per scenario
    assert m["counts"]["pool"]["C0"]["total"] == 8 and m["counts"]["not_drawn"] == 4
    # The excluded episode is listed with its reason, never drawn.
    [sk] = m["skipped"]
    assert sk["scenario"] == "churn_format" and "simulator_truncated" in sk["reason"]
    for e in eps:
        assert (root / e["log"]).exists() and e["uuid"] and len(e["transcript_sha256"]) == 64
        assert e["n_turns"] >= 1
    assert m["leak_check"] == []
    assert sum(f["samples"] for f in m["logs"]) == 8 + 2 + 1


def test_manifest_is_deterministic(logs):
    _, spec = logs
    a, b = P.build_manifest(spec), P.build_manifest(spec)
    assert a["episodes"] == b["episodes"] and a["skipped"] == b["skipped"] and a["counts"] == b["counts"]
    raw = yaml.safe_load(spec.read_text())
    spec.write_text(yaml.safe_dump({**raw, "seed": 12}))
    c = P.build_manifest(spec)
    assert [(e["log"], e["sample_id"], e["epoch"], e["label_id"]) for e in c["episodes"]] != \
        [(e["log"], e["sample_id"], e["epoch"], e["label_id"]) for e in a["episodes"]]


def test_a_sample_in_two_attempt_logs_counts_once(logs):
    root, spec = logs
    src = next((root / "logs" / "c3").glob("*.eval"))
    shutil.copy(src, src.with_name(src.stem + "-copy.eval"))   # the same sample runs (uuids) twice
    m = P.build_manifest(spec)
    assert sum(e["condition"] == "C3" for e in m["episodes"]) == 2
    files = {f["path"].rsplit("/", 1)[1]: f for f in m["logs"] if "/c3/" in f["path"]}
    assert sorted((f["counted"], f["duplicates"]) for f in files.values()) == [(0, 2), (2, 0)]


def test_write_refuses_to_replace_a_manifest(logs, tmp_path):
    _, spec = logs
    m = P.build_manifest(spec)
    out = tmp_path / "manifest.json"
    P.write_manifest(m, out)
    assert json.loads(out.read_text())["episodes"] == m["episodes"]
    with pytest.raises(P.PoolError):
        P.write_manifest(m, out)
    P.write_manifest(m, out, force=True)


def test_spec_is_checked(tmp_path):
    p = tmp_path / "pool.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "seed": 1, "n": 5, "take_all": "C3", "fill": "C0", "dirs": ["a"],
                                 "extra": 1}))
    with pytest.raises(P.PoolError):
        P.load_spec(p)
