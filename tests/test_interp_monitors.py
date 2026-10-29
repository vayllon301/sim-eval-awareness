"""The logit monitor on the fake tokenizer and backend: the prefix ends right before each
turn's first reasoning token, phrase log-probs equal a direct token_losses computation,
the cached path is used when the backend has one, group means and differences, and the
results stored with a reading."""

import json
from pathlib import Path

import numpy as np
import pytest

from simtest.interp import directions as D
from simtest.interp.backend import encode
from simtest.interp.fake import FakeBackend
from simtest.interp.meta import file_sha256
from simtest.interp.monitors import logit_monitor, reasoning_starts
from simtest.interp.reader import (R, Reading, colab_probe_prompts, from_colab_probe, read, render, summarize,
                                   turn_prompts)
from simtest.interp.vectors import VectorSet

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "tests" / "interp_fixtures" / "colab-probe-smoke.jsonl"
PHRASES = ROOT / "explore" / "interp" / "data" / "logit_monitor_phrases.json"
PROMPTS = colab_probe_prompts(ROOT / "explore")
GROUPS = {"eval": ["The user might be testing", "This looks like a benchmark task"],
          "training": ["This looks like a training environment"],
          "control": ["Let me look at the failing test", "First, the repo"]}


def episode():
    return from_colab_probe([json.loads(line) for line in SMOKE.read_text().splitlines()], **PROMPTS)


def rendered(b, ep=None):
    rd = render(ep or episode(), b.tokenizer)
    return rd.ids, rd.role, rd.turn


def test_the_prefix_ends_right_before_each_turns_first_reasoning_token():
    b, ep = FakeBackend(), episode()
    rd = render(ep, b.tokenizer)
    starts = reasoning_starts(rd.role, rd.turn)
    assert sorted(starts) == [0, 1, 2, 3]
    for t, pos in starts.items():
        assert rd.role[pos] == R["reasoning"] and rd.turn[pos] == t
        assert not ((rd.role[:pos] == R["reasoning"]) & (rd.turn[:pos] == t)).any()
        assert rd.text[:pos].endswith("<|im_start|>assistant\n<think>\n")  # one character per token
    # in the fake template the reasoning starts right after the turn's prompt
    assert [starts[t] for t in range(4)] == [len(p) for p in turn_prompts(ep, b.tokenizer)]


def test_phrase_logprobs_equal_a_direct_token_losses_computation():
    b = FakeBackend()
    ids, role, turn = rendered(b)
    out = logit_monitor(b, ids, role, turn, GROUPS)
    assert out["path"] == "token_losses"
    assert [(p["group"], p["text"]) for p in out["phrases"]] == [(g, t) for g, ts in GROUPS.items() for t in ts]
    starts = reasoning_starts(role, turn)
    assert [row["turn"] for row in out["by_turn"]] == [0, 1, 2, 3]
    for row in out["by_turn"]:
        prefix = ids[:starts[row["turn"]]].tolist()
        assert row["prefix_len"] == len(prefix)
        for p, res in zip(out["phrases"], row["phrases"]):
            pid = encode(b.tokenizer, p["text"])
            assert p["n_tokens"] == len(pid)
            losses = b.token_losses(prefix + pid)[-len(pid):]
            assert res["logprob"] == pytest.approx(-losses.sum()) and res["mean"] == pytest.approx(-losses.mean())
            assert res["logprob"] < 0


def test_group_means_and_differences():
    b = FakeBackend()
    out = logit_monitor(b, *rendered(b), GROUPS, turns=[0])
    row = out["by_turn"][0]
    for g in GROUPS:
        for k in ("logprob", "mean"):
            xs = [r[k] for p, r in zip(out["phrases"], row["phrases"]) if p["group"] == g]
            assert row["groups"][g][k] == pytest.approx(np.mean(xs))
    for g in ("eval", "training"):
        for k in ("logprob", "mean"):
            assert row[f"{g}_minus_control"][k] == pytest.approx(row["groups"][g][k] - row["groups"]["control"][k])
    assert "control_minus_control" not in row
    assert "eval_minus_control" not in logit_monitor(b, *rendered(b), {"eval": GROUPS["eval"]}, turns=[0])["by_turn"][0]


def test_turns_without_reasoning_give_none_and_callers_choose_the_turns():
    b, ep = FakeBackend(), episode()
    assert ep.turns[5] == 1 and ep.messages[5]["role"] == "assistant"
    ep.messages[5]["reasoning_content"] = ""
    ids, role, turn = rendered(b, ep)
    out = logit_monitor(b, ids, role, turn, GROUPS)
    assert out["by_turn"][1] == {"turn": 1, "prefix_len": None, "phrases": None, "groups": None,
                                 "eval_minus_control": None, "training_minus_control": None}
    assert all(row["phrases"] is not None for i, row in enumerate(out["by_turn"]) if i != 1)
    only = logit_monitor(b, ids, role, turn, GROUPS, turns=[3])
    assert [row["turn"] for row in only["by_turn"]] == [3] and only["by_turn"][0] == out["by_turn"][3]
    with pytest.raises(ValueError, match="turn 7"):
        logit_monitor(b, ids, role, turn, GROUPS, turns=[7])
    with pytest.raises(ValueError, match="no tokens"):
        logit_monitor(b, ids, role, turn, {"eval": [""]})
    with pytest.raises(ValueError, match="no phrases"):
        logit_monitor(b, ids, role, turn, {"eval": []})


class Cached:
    """The fake backend with a continuation_logprobs method, as the GPU backend will have."""

    def __init__(self, b):
        self.b, self.tokenizer, self.calls = b, b.tokenizer, []

    def token_losses(self, ids, **kw):
        raise AssertionError("the generic path was used")

    def continuation_logprobs(self, prefix, continuations):
        self.calls.append((len(prefix), len(continuations)))
        return [-self.b.token_losses(list(prefix) + list(c))[-len(c):] for c in continuations]


def test_the_cached_path_is_used_when_the_backend_has_it():
    b = FakeBackend()
    cached = Cached(b)
    fast = logit_monitor(cached, *rendered(b), GROUPS)
    slow = logit_monitor(b, *rendered(b), GROUPS)
    assert fast["path"] == "continuation_logprobs"
    starts = reasoning_starts(*rendered(b)[1:])
    assert cached.calls == [(starts[t], 5) for t in range(4)]  # one call per turn, all phrases at once
    assert fast["by_turn"] == slow["by_turn"] and fast["phrases"] == slow["phrases"]


def fake_vectors(b):
    vs = VectorSet(meta={"model": dict(b.info), "offset": {}})
    vs.add("calm", 2, D.unit(np.random.default_rng(1).standard_normal(b.d_model)))
    return vs


def test_read_stores_the_monitor_with_the_reading(tmp_path):
    b = FakeBackend()
    path = tmp_path / "phrases.json"
    path.write_text(json.dumps({"status": "test", "groups": GROUPS}))
    r = read(b, episode(), fake_vectors(b), [2], phrases=json.loads(path.read_text()), phrases_path=str(path),
             phrase_turns=[0, 3])
    lm = r.meta["logit_monitor"]
    assert lm["phrases_path"] == str(path) and lm["phrases_sha256"] == file_sha256(path) and lm["status"] == "test"
    assert [row["turn"] for row in lm["by_turn"]] == [0, 3]
    assert lm["by_turn"] == logit_monitor(b, *rendered(b), GROUPS, turns=[0, 3])["by_turn"]
    back = Reading.load(r.save(tmp_path / "ep.npz"))
    assert back.meta["logit_monitor"] == lm
    summary = json.loads((tmp_path / "ep.summary.json").read_text())
    assert summary["logit_monitor"] == lm
    plain = read(b, episode(), fake_vectors(b), [2])
    assert "logit_monitor" not in plain.meta and "logit_monitor" not in summarize(plain)


def test_the_phrase_file():
    doc = json.loads(PHRASES.read_text())
    assert doc["status"] and doc["source"]
    g = doc["groups"]
    assert {"eval", "training", "control"} <= set(g) and all(g[k] and all(isinstance(x, str) and x for x in g[k]) for k in g)
    b = FakeBackend()
    out = logit_monitor(b, *rendered(b), g, turns=[0])
    assert len(out["phrases"]) == sum(map(len, g.values())) and out["by_turn"][0]["eval_minus_control"] is not None
