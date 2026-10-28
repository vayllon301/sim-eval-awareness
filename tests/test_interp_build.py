"""The vector-building pipeline end to end on the fake backend: every vector at every
layer, validation numbers, metadata, and a file that loads back."""

import copy
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
            out.append({"id": f"n-{ti}-{k}", "kind": "neutral", "emotion": None, "topic": topic,
                        "split": "test" if ti == 3 else "train",
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


def test_heldout_neutral_rows_do_not_change_what_is_fitted():
    """Review B1: PCA, offsets and reference norms come from training rows only."""
    b = FakeBackend()
    pairs = json.loads((ROOT / "explore" / "interp" / "data" / "eval_awareness_pairs.json").read_text())
    base = rows()
    changed = copy.deepcopy(base)
    for r in changed:
        if r["kind"] == "neutral" and r["split"] == "test":
            r["text"] = "Human: List four prime numbers. Assistant: 2, 3, 5 and 7, in order. " * 3
    v1, r1 = build_all(b, base, [2, 3], pairs=pairs, concept_topics=["a bakery morning"])
    v2, r2 = build_all(b, changed, [2, 3], pairs=pairs, concept_topics=["a bakery morning"])
    for k, v in v1.vectors.items():
        assert np.array_equal(v, v2.vectors[k]), k
    assert v1.meta["offset"] == v2.meta["offset"]
    assert v1.meta["resid_norm_mean"] == v2.meta["resid_norm_mean"]
    for L in ("2", "3"):
        assert r1["layers"][L]["variants"] == r2["layers"][L]["variants"]


def test_short_texts_are_refused():
    rs = rows()
    rs[0]["text"] = "Too short."
    with pytest.raises(ValueError, match="need more than 50"):
        build_all(FakeBackend(), rs, [2])


def test_backend_provenance_covers_tokenizer_and_template(tmp_path):
    """Review B3: tokenizer files and the template in effect change the recorded provenance."""
    from simtest.interp.meta import tokenizer_info

    class Tok:
        chat_template = "{{ messages }}"

    (tmp_path / "tokenizer.json").write_text('{"model": "a"}')
    a = tokenizer_info(Tok(), tmp_path)
    assert set(a["versions"]) == {"transformers", "tokenizers"} and a["chat_template_sha256"]
    other = Tok()
    other.chat_template = "{{ messages }}\n"
    assert tokenizer_info(other, tmp_path)["chat_template_sha256"] != a["chat_template_sha256"]
    assert tokenizer_info(other, tmp_path)["tokenizer_sha256"] == a["tokenizer_sha256"]
    (tmp_path / "tokenizer.json").write_text('{"model": "b"}')
    assert tokenizer_info(Tok(), tmp_path)["tokenizer_sha256"] != a["tokenizer_sha256"]


def test_choose_layer_breaks_ties_towards_two_thirds():
    rep = {"layers": {str(L): {"heldout": {"accuracy": a}} for L, a in [(36, 0.9), (42, 0.95), (45, 0.95), (48, 0.8)]}}
    assert choose_layer(rep, 64) == 42  # 42.67 is two-thirds of 64
    rep["layers"]["48"]["heldout"]["accuracy"] = 0.99
    assert choose_layer(rep, 64) == 48


def test_slug():
    assert slug("A baker prepares for the busiest morning of the year") == "baker-prepares"
    assert slug("A neighbor starts a renovation project") == "neighbor-renovation"
