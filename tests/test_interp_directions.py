"""Interp math on small synthetic arrays: difference of means, neutral PCs, accuracy,
contrast vectors, random controls. No models, no MLX."""

import numpy as np
import pytest

from simtest.interp import directions as D


def clusters(seed=0, d=32, n=12, labels=("a", "b", "c"), spread=0.3):
    rng = np.random.default_rng(seed)
    centres = {k: rng.standard_normal(d) * 3 for k in labels}
    return {k: c + spread * rng.standard_normal((n, d)) for k, c in centres.items()}, centres


def test_story_mean_from_token_50():
    acts = np.zeros((60, 4))
    acts[50:] = 1.0
    acts[:50] = 100.0
    assert np.allclose(D.story_mean(acts), 1.0)
    with pytest.raises(ValueError, match="need more than 50"):
        D.story_mean(np.zeros((50, 4)))


def test_diff_of_means_subtracts_the_mean_of_class_means():
    groups = {"a": np.array([[1.0, 0], [3.0, 0]]), "b": np.array([[0, 2.0]]), "c": np.array([[0, 0.0], [0, 0], [0, 0]])}
    v = D.diff_of_means(groups)
    centre = (np.array([2.0, 0]) + np.array([0, 2.0]) + np.zeros(2)) / 3  # class means, not pooled stories
    assert np.allclose(v["a"], [2, 0] - centre)
    assert np.allclose(sum(v.values()), 0)


def test_neutral_pcs_reach_the_variance_fraction_and_are_orthonormal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((200, 10)) * np.array([10, 5, 1, 1, 1, 1, 1, 1, 1, 1])
    pcs = D.neutral_pcs(x, 0.5)
    assert pcs.shape[0] == 1  # the first axis alone carries 100 / (100 + 25 + 8) of the variance
    assert np.allclose(pcs @ pcs.T, np.eye(len(pcs)), atol=1e-8)
    assert D.neutral_pcs(x, 0.9).shape[0] >= 2
    with pytest.raises(ValueError):
        D.neutral_pcs(x, 1.0)


def test_project_out_removes_the_span():
    rng = np.random.default_rng(2)
    pcs = np.linalg.qr(rng.standard_normal((8, 3)))[0].T  # 3 orthonormal rows
    v = rng.standard_normal(8)
    p = D.project_out(v, pcs)
    assert np.allclose(pcs @ p, 0, atol=1e-10)
    assert np.allclose(D.project_out(p, pcs), p)


def test_emotion_vectors_are_unit_and_orthogonal_to_neutral_pcs():
    groups, _ = clusters()
    rng = np.random.default_rng(3)
    neutral = rng.standard_normal((100, 32)) * np.r_[np.full(4, 8.0), np.ones(28)]
    vecs, info = D.emotion_vectors(groups, neutral, 0.5)
    for v in vecs.values():
        assert np.isclose(np.linalg.norm(v), 1)
        assert np.allclose(info["pcs"] @ v, 0, atol=1e-8)
    assert info["n_neutral_pcs"] >= 1
    assert set(info["norms"]) == set(groups)


def test_heldout_and_loo_accuracy_on_separable_clusters():
    train, centres = clusters(seed=4)
    rng = np.random.default_rng(5)
    test = {k: c + 0.3 * rng.standard_normal((5, 32)) for k, c in centres.items()}
    vecs, info = D.emotion_vectors(train, None)
    acc = D.accuracy(test, vecs, info["grand_mean"])
    assert acc["accuracy"] == 1.0 and acc["n"] == 15
    assert acc["confusion"]["a"] == {"a": 5, "b": 0, "c": 0}
    assert D.loo_accuracy(train)["accuracy"] == 1.0


def test_accuracy_is_at_chance_on_noise():
    rng = np.random.default_rng(6)
    train = {k: rng.standard_normal((30, 64)) for k in "abcd"}
    test = {k: rng.standard_normal((200, 64)) for k in "abcd"}
    vecs, info = D.emotion_vectors(train, None)
    assert 0.15 < D.accuracy(test, vecs, info["grand_mean"])["accuracy"] < 0.35
    assert D.loo_accuracy(train)["accuracy"] < 0.45


def test_loo_matches_a_manual_rebuild():
    train, _ = clusters(seed=7, n=4, spread=3.0)  # noisy enough that some stories fail
    manual = 0
    for k, g in train.items():
        for i in range(len(g)):
            rest = {j: (np.delete(train[j], i, 0) if j == k else train[j]) for j in train}
            vecs, info = D.emotion_vectors(rest, None)
            manual += D.classify(g[i], vecs, info["grand_mean"])[0] == k
    assert D.loo_accuracy(train)["correct"] == manual


def test_contrast_vector_and_paired_loo():
    rng = np.random.default_rng(8)
    direction = D.unit(rng.standard_normal(16))
    base = rng.standard_normal((10, 16))
    pos, neg = base + 2 * direction, base.copy()
    v = D.contrast_vector(pos, neg)
    assert np.isclose(v @ direction, 1)
    r = D.paired_loo(pos, neg)
    assert r["accuracy"] == 1.0 and r["n"] == 10
    pos[0] = base[0] - 2 * direction  # one pair points the other way: only it fails when held out
    r = D.paired_loo(pos, neg)
    assert r["correct"] == 9 and r["margins"][0] < 0


def test_auroc():
    assert D.auroc([3, 4], [1, 2]) == 1.0
    assert D.auroc([1, 2], [3, 4]) == 0.0
    assert D.auroc([1, 1], [1, 1]) == 0.5


def test_random_unit_is_reproducible_per_seed_and_layer():
    a = D.random_unit(5120, seed=3, layer=42)
    assert a.dtype == np.float32 and np.isclose(np.linalg.norm(a), 1, atol=1e-6)
    assert np.array_equal(a, D.random_unit(5120, seed=3, layer=42))
    assert not np.allclose(a, D.random_unit(5120, seed=3, layer=43))
    assert not np.allclose(a, D.random_unit(5120, seed=4, layer=42))
    # pinned: the GPU side must get the same vector from (seed, layer)
    assert np.allclose(D.random_unit(4, seed=0, layer=0), [0.18652, -0.19597, 0.95005, 0.15562], atol=1e-4)
