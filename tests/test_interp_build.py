"""The vector-building pipeline end to end on the fake backend: every vector at every
layer, validation numbers, metadata, and a file that loads back."""

import json
from pathlib import Path

import numpy as np
import pytest

from simtest.interp.build import build_all, choose_layer, slug
from simtest.interp.fake import FakeBackend
from simtest.interp.vectors import VectorSet

ROOT = Path(__file__).resolve().parents[1]
WORDS = {"desperate": "please no time left begging last chance", "calm": "slow quiet still easy breath rest",
         "angry": "slam shout furious rage fists glare"}
TOPICS = ["a bakery morning", "a mountain trail", "a train journey", "a school exam"]


def rows():
    out = []
    for e, words in WORDS.items():
        for ti, topic in enumerate(TOPICS):
            for k in range(3):
                text = f"Story {k} about {topic}. " + " ".join([words] * 6) + f" The end of {topic}."
                out.append({"id": f"{e}-{ti}-{k}", "kind": "emotion", "emotion": e, "topic": topic,
                            "split": "test" if ti == 3 else "train", "text": text, "generator": "test"})
    for ti, topic in enumerate(TOPICS):
        for k in range(3):
            out.append({"id": f"n-{ti}-{k}", "kind": "neutral", "emotion": None, "topic": topic, "split": "train",
                        "text": f"Human: How long is {topic}? Assistant: It is {k + 3} hours long. " * 3, "generator": "test"})
    return out


def test_build_all_on_the_fake_backend(tmp_path):
    b = FakeBackend()
    pairs_path = ROOT / "explore" / "interp" / "data" / "eval_awareness_pairs.json"
    pairs = json.loads(pairs_path.read_text())
    vs, rep = build_all(b, rows(), [2, 3], texts_path=None, pairs=pairs, pairs_path=str(pairs_path),
                        concept_topics=["a bakery morning"])
    assert vs.names == ["desperate", "calm", "angry", "topic:bakery-morning", "eval_awareness"]
    assert vs.layers == [2, 3]
    for L in (2, 3):
        r = rep["layers"][str(L)]
        assert r["heldout"]["n"] == 9 and r["loo"]["n"] == 27
        assert r["heldout"]["accuracy"] == 1.0  # the fake stories differ by their words
        assert r["n_neutral_pcs"] >= 1
        ea = rep["eval_awareness"][str(L)]
        assert ea["loo_pairs"]["n"] == 16 and ea["heldout_pairs"]["n"] == 4 and ea["generalisation"]["n_pairs"] == 8
        for name in vs.names:
            assert str(L) in vs.meta["offset"][name]
        assert vs.resid_norm(L) > 0
    assert vs.meta["chosen_layer"] == rep["chosen_layer"] in (2, 3)
    assert vs.meta["model"]["fingerprint"] == b.info["fingerprint"]
    assert vs.meta["datasets"]["texts"]["n_emotion_test"] == 9 and vs.meta["datasets"]["eval_pairs"]["n_pairs"] == 16
    assert vs.meta["vector_info"]["eval_awareness"]["sign"] == "positive = evaluation"
    vs.save(tmp_path / "v.safetensors")
    back = VectorSet.load(tmp_path / "v.safetensors")
    assert back.names == vs.names and np.allclose(back.get("calm", 3), vs.get("calm", 3))
    assert back.meta["eval"]["layers"]["2"]["heldout"]["accuracy"] == 1.0


def test_short_texts_are_refused():
    rs = rows()
    rs[0]["text"] = "Too short."
    with pytest.raises(ValueError, match="need more than 50"):
        build_all(FakeBackend(), rs, [2])


def test_choose_layer_breaks_ties_towards_two_thirds():
    rep = {"layers": {str(L): {"heldout": {"accuracy": a}} for L, a in [(36, 0.9), (42, 0.95), (45, 0.95), (48, 0.8)]}}
    assert choose_layer(rep, 64) == 42  # 42.67 is two-thirds of 64
    rep["layers"]["48"]["heldout"]["accuracy"] = 0.99
    assert choose_layer(rep, 64) == 48


def test_slug():
    assert slug("A baker prepares for the busiest morning of the year") == "baker-prepares"
    assert slug("A neighbor starts a renovation project") == "neighbor-renovation"
