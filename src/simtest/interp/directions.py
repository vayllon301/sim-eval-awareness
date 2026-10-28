"""Directions in the residual stream: the math, on numpy arrays only.

Emotion vectors follow Sofroniew et al. (2026), docs/INTERP.md "Emotion vectors":
1. One mean activation per story, over its tokens from token 50 on (`story_mean`).
2. Each emotion's vector is its mean over training stories minus the mean of the emotion
   means (`diff_of_means`).
3. The top principal components of neutral-text activations that explain 50% of their
   variance are projected out of each vector (`neutral_pcs`, `project_out`).
4. Vectors are stored as unit vectors; readings are dot products with them.

Classification (for held-out and leave-one-out accuracy): a story is assigned the
emotion whose unit vector has the largest dot product with the story's mean minus the
training grand mean (the mean over all training stories).

Contrast vectors (evaluation awareness, unrelated-concept controls): mean of the positive
set minus mean of the negative set, unit length.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

TOKEN_START = 50  # Sofroniew et al.: average from token 50 on


def story_mean(acts: np.ndarray, start: int = TOKEN_START) -> np.ndarray:
    """Mean over positions start.. of one text's activations (tokens, d)."""
    if acts.ndim != 2:
        raise ValueError(f"expected (tokens, d), got {acts.shape}")
    if acts.shape[0] <= start:
        raise ValueError(f"text has {acts.shape[0]} tokens; need more than {start} to average from token {start}")
    return acts[start:].astype(np.float64).mean(0)


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0 or not np.isfinite(n):
        raise ValueError("cannot normalise a zero or non-finite vector")
    return v / n


def neutral_pcs(neutral: np.ndarray, variance: float = 0.5) -> np.ndarray:
    """Orthonormal rows: the fewest top principal components of `neutral` (n, d) whose
    explained variance reaches `variance` (a fraction). Shape (k, d)."""
    if not 0 < variance < 1:
        raise ValueError("variance must be a fraction in (0, 1)")
    x = neutral.astype(np.float64)
    x = x - x.mean(0)
    if x.shape[0] <= x.shape[1]:
        _, s, vt = np.linalg.svd(x, full_matrices=False)
        var = s**2
    else:  # many tokens: the eigenvectors of the d x d scatter matrix are the same components, faster
        w, v = np.linalg.eigh(x.T @ x)
        order = np.argsort(w)[::-1]
        var, vt = np.clip(w[order], 0, None), v[:, order].T
    explained = var / float(var.sum())
    k = int(np.searchsorted(np.cumsum(explained), variance) + 1)
    return vt[:k]


def project_out(v: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """v minus its projection on the span of the orthonormal rows of pcs."""
    if pcs.size == 0:
        return v
    return v - pcs.T @ (pcs @ v)


def diff_of_means(groups: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """{label: (n_label, d)} -> {label: mean(label) - mean over labels of the label means}."""
    means = {k: g.astype(np.float64).mean(0) for k, g in groups.items()}
    centre = np.mean(list(means.values()), axis=0)
    return {k: m - centre for k, m in means.items()}


def emotion_vectors(train: Mapping[str, np.ndarray], neutral: np.ndarray | None,
                    variance: float = 0.5) -> tuple[dict[str, np.ndarray], dict]:
    """Unit emotion vectors from training story means, neutral PCs projected out.

    Returns (vectors, info): info has the number of neutral PCs removed, each vector's
    norm before and after the projection, and the training grand mean."""
    raw = diff_of_means(train)
    pcs = neutral_pcs(neutral, variance) if neutral is not None else np.zeros((0, next(iter(raw.values())).shape[0]))
    vecs, norms = {}, {}
    for k, v in raw.items():
        p = project_out(v, pcs)
        norms[k] = {"raw": float(np.linalg.norm(v)), "after_pcs": float(np.linalg.norm(p))}
        vecs[k] = unit(p)
    grand = np.concatenate([g.astype(np.float64) for g in train.values()]).mean(0)
    return vecs, {"n_neutral_pcs": int(pcs.shape[0]), "norms": norms, "grand_mean": grand, "pcs": pcs}


def scores(x: np.ndarray, vectors: Mapping[str, np.ndarray], centre: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Dot products of (x - centre) with each unit vector: (labels, (n, n_labels))."""
    labels = list(vectors)
    m = np.stack([vectors[k] for k in labels], axis=1)
    return labels, (np.atleast_2d(x).astype(np.float64) - centre) @ m


def classify(x: np.ndarray, vectors: Mapping[str, np.ndarray], centre: np.ndarray) -> list[str]:
    labels, s = scores(x, vectors, centre)
    return [labels[i] for i in s.argmax(1)]


def accuracy(groups: Mapping[str, np.ndarray], vectors: Mapping[str, np.ndarray], centre: np.ndarray) -> dict:
    """Accuracy and confusion counts of `classify` on {true label: (n, d)}."""
    confusion = {t: {p: 0 for p in vectors} for t in groups}
    for t, x in groups.items():
        if len(x):
            for p in classify(x, vectors, centre):
                confusion[t][p] += 1
    n = sum(sum(r.values()) for r in confusion.values())
    hit = sum(confusion[t].get(t, 0) for t in confusion)
    return {"accuracy": hit / n if n else float("nan"), "correct": hit, "n": n, "confusion": confusion}


def loo_accuracy(train: Mapping[str, np.ndarray], pcs: np.ndarray | None = None) -> dict:
    """Leave-one-out over training stories: rebuild the vectors without each story, classify it.
    The neutral PCs (k, d) don't depend on the stories, so they are passed in."""
    d = next(iter(train.values())).shape[1]
    pcs = np.zeros((0, d)) if pcs is None else pcs
    sums = {k: g.astype(np.float64).sum(0) for k, g in train.items()}
    counts = {k: len(g) for k, g in train.items()}
    total, n_all = sum(sums.values()), sum(counts.values())
    hit = n = 0
    for k, g in train.items():
        for i in range(len(g)):
            x = g[i].astype(np.float64)
            means = {j: (sums[j] - x) / (counts[j] - 1) if j == k else sums[j] / counts[j] for j in train}
            centre = np.mean(list(means.values()), axis=0)
            vecs = {j: unit(project_out(m - centre, pcs)) for j, m in means.items()}
            grand = (total - x) / (n_all - 1)
            hit += classify(x, vecs, grand)[0] == k
            n += 1
    return {"accuracy": hit / n, "correct": int(hit), "n": n}


def contrast_vector(pos: np.ndarray, neg: np.ndarray, pcs: np.ndarray | None = None) -> np.ndarray:
    """Unit vector mean(pos) - mean(neg), optionally with PCs projected out."""
    v = pos.astype(np.float64).mean(0) - neg.astype(np.float64).mean(0)
    return unit(project_out(v, pcs) if pcs is not None else v)


def paired_loo(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Leave-one-pair-out for a contrast vector built from pairs (pos[i], neg[i]):
    the held-out pair counts as correct if pos[i] projects higher than neg[i]."""
    if pos.shape != neg.shape:
        raise ValueError("pairs must have the same shape")
    n, hit, margins = len(pos), 0, []
    for i in range(n):
        keep = np.arange(n) != i
        v = contrast_vector(pos[keep], neg[keep])
        m = float((pos[i] - neg[i]) @ v)
        margins.append(m)
        hit += m > 0
    return {"accuracy": hit / n, "correct": int(hit), "n": n, "margins": margins}


def auroc(pos_scores: Sequence[float], neg_scores: Sequence[float]) -> float:
    """Probability that a random positive outscores a random negative (ties count half)."""
    p, q = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    gt = (p[:, None] > q[None, :]).sum() + 0.5 * (p[:, None] == q[None, :]).sum()
    return float(gt / (len(p) * len(q)))


def random_unit(d: int, seed: int, layer: int) -> np.ndarray:
    """A reproducible random unit vector: numpy's default_rng([seed, layer]), standard
    normal, normalised. Any engine (laptop or GPU) gets the same vector from (seed, layer)."""
    rng = np.random.default_rng([int(seed), int(layer)])
    return unit(rng.standard_normal(d)).astype(np.float32)
