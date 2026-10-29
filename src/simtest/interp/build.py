"""Build a vector file on a backend: emotion vectors, unrelated-concept controls, an
evaluation-awareness vector and its training twin, each at every scanned layer, with their
validation.

Inputs:
- texts: rows {id, kind: emotion|neutral, emotion, topic, split: train|test, text}
  (explore/interp/gen_texts.py). Texts are fed raw, without a chat template.
- eval pairs: {pairs: [{real, eval}], heldout: [indices], generalisation: {pairs}}
  (explore/interp/data/eval_awareness_pairs.json).
- train pairs: {pairs: [{real, train}], heldout: [indices]}, the same real statements with
  the evaluation side reworded as training (explore/interp/data/train_awareness_pairs.json).
  Its generalisation check uses the eval pairs' coding pairs.

Layer choice, fixed before any run: the scanned layer with the highest held-out emotion
accuracy; ties go to the layer closest to two-thirds of the depth. Every scanned layer is
saved either way; the choice is recorded as `chosen_layer`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np

from . import directions as D
from .backend import Backend, encode
from .meta import file_sha256, git_commit, now
from .reader import R, Episode, render
from .vectors import VectorSet

METHOD = {"emotions": "emotion-diffmeans-v1: story means from token 50, minus the mean of emotion means, "
                      "neutral-text PCs (50% of variance, per token from token 50) projected out, unit length",
          "topics": "topic-diffmeans-v1: mean of one premise's stories (all emotions) minus the mean of the other "
                    "training premises' stories, neutral PCs projected out, unit length",
          "eval_awareness": "contrast-v1 (Hua et al.): mean(eval) - mean(real) at the colon ending each prompt, "
                            "rendered as a user message with the chat template, unit length, no PC projection",
          "train_awareness": "contrast-v1, training twin (after Aranguri): mean(train) - mean(real) at the colon ending "
                             "each prompt, rendered as a user message with the chat template, unit length, no PC "
                             "projection; the eval pairs with the evaluation side reworded as training"}
# contrast vectors: the positive side's key in the pairs file, the dataset name, the sign
CONTRASTS = {"eval_awareness": ("eval", "eval_pairs", "positive = evaluation"),
             "train_awareness": ("train", "train_pairs", "positive = training")}


def slug(topic: str, words: int = 2) -> str:
    stop = {"a", "an", "the", "of", "on", "for", "to", "their", "with", "someone", "two", "starts", "finds"}
    w = [x for x in re.findall(r"[a-z]+", topic.lower()) if x not in stop]
    return "-".join(w[:words])


def collect(backend: Backend, texts: Sequence[str], layers: Sequence[int], start: int = D.TOKEN_START,
            keep_tokens: Sequence[bool] | None = None, progress=None):
    """Per text: the mean activation from `start` on at each layer, optionally its token
    activations from `start` on, and its token norms (for the mean residual norm)."""
    means = {L: [] for L in layers}
    tokens = {L: [] for L in layers}
    norms = {L: [] for L in layers}
    n_tokens = []
    for i, t in enumerate(texts):
        ids = encode(backend.tokenizer, t)
        n_tokens.append(len(ids))
        acts = backend.activations(ids, layers)
        for L in layers:
            means[L].append(D.story_mean(acts[L], start))
            norms[L].append(np.linalg.norm(acts[L][start:].astype(np.float64), axis=1))
            if keep_tokens is not None and keep_tokens[i]:
                tokens[L].append(acts[L][start:])
        if progress:
            progress(i + 1, len(texts))
    return {L: np.stack(v) for L, v in means.items()}, tokens, norms, n_tokens


def build_emotions(backend: Backend, rows: Sequence[dict], layers: Sequence[int], *, variance: float = 0.5,
                   concept_topics: Sequence[str] = (), progress=None) -> tuple[VectorSet, dict, dict]:
    """Returns (vectors, report, training neutral text means per layer). Everything fitted
    (vectors, neutral PCs, offsets, reference norms) uses training rows only; held-out rows
    are only scored."""
    layers = sorted(int(L) for L in layers)
    emotions = list(dict.fromkeys(r["emotion"] for r in rows if r["kind"] == "emotion"))
    train_neutral = [r["kind"] == "neutral" and r["split"] == "train" for r in rows]
    means, tokens, norms, n_tokens = collect(backend, [r["text"] for r in rows], layers, keep_tokens=train_neutral,
                                             progress=progress)
    idx = {"train": {e: [] for e in emotions}, "test": {e: [] for e in emotions}}
    for i, r in enumerate(rows):
        if r["kind"] == "emotion":
            idx[r["split"]][r["emotion"]].append(i)
    neutral_idx = [i for i, keep in enumerate(train_neutral) if keep]
    train_idx = [i for i, r in enumerate(rows) if r["split"] == "train"]
    vs = VectorSet(meta={})
    report = {"layers": {}, "n_tokens": {"min": int(min(n_tokens)), "median": float(np.median(n_tokens)),
                                         "max": int(max(n_tokens))}}
    offsets: dict[str, dict[str, float]] = {}
    resid = {}
    for L in layers:
        m = means[L]
        train = {e: m[idx["train"][e]] for e in emotions}
        test = {e: m[idx["test"][e]] for e in emotions if idx["test"][e]}
        neutral_tok = np.concatenate(tokens[L]) if tokens[L] else None
        vecs, info = D.emotion_vectors(train, neutral_tok, variance)
        rep = {"n_neutral_pcs": info["n_neutral_pcs"], "norms": info["norms"]}
        if test:
            rep["heldout"] = D.accuracy(test, vecs, info["grand_mean"])
        rep["loo"] = D.loo_accuracy(train, info["pcs"])
        # variants, for comparison only: no PC projection, and PCs of per-text neutral means
        raw, raw_info = D.emotion_vectors(train, None)
        means_pcs = D.neutral_pcs(m[neutral_idx], variance) if len(neutral_idx) > 2 else np.zeros((0, m.shape[1]))
        on_means = {e: D.unit(D.project_out(v, means_pcs)) for e, v in D.diff_of_means(train).items()}
        if test:
            rep["variants"] = {"no_pcs": D.accuracy(test, raw, raw_info["grand_mean"])["accuracy"],
                               "pcs_on_neutral_means": D.accuracy(test, on_means, info["grand_mean"])["accuracy"]}
        for e, v in vecs.items():
            vs.add(e, L, v)
        # unrelated-concept controls: one premise against the other training premises
        train_emotion = [i for e in emotions for i in idx["train"][e]]
        for topic in concept_topics:
            pos = [i for i in train_emotion if rows[i]["topic"] == topic]
            neg = [i for i in train_emotion if rows[i]["topic"] != topic]
            if not pos or not neg:
                raise ValueError(f"concept topic {topic!r} has no training stories")
            vs.add(f"topic:{slug(topic)}", L, D.contrast_vector(m[pos], m[neg], info["pcs"]))
        resid[str(L)] = float(np.concatenate([norms[L][i] for i in train_idx]).mean())
        neutral_means = m[neutral_idx]
        for name in [*emotions, *[f"topic:{slug(t)}" for t in concept_topics]]:
            offsets.setdefault(name, {})[str(L)] = float((neutral_means @ vs.get(name, L)).mean()) if len(neutral_idx) else 0.0
        report["layers"][str(L)] = rep
    report["chosen_layer"] = choose_layer(report, backend.n_layers)
    vs.meta.update({"resid_norm_mean": resid, "offset": offsets,
                    "norm_reference": "training texts (stories and neutral dialogues), tokens from 50 on",
                    "params": {"token_start": D.TOKEN_START, "neutral_pc_variance": variance, "pca_on": "neutral tokens"},
                    "vector_info": {**{e: {"method": "emotions", "dataset": "texts"} for e in emotions},
                                    **{f"topic:{slug(t)}": {"method": "topics", "dataset": "texts", "topic": t}
                                       for t in concept_topics}}})
    return vs, report, {L: means[L][neutral_idx] for L in layers}


def choose_layer(report: dict, n_layers: int) -> int | None:
    accs = {int(L): r["heldout"]["accuracy"] for L, r in report["layers"].items() if "heldout" in r}
    if not accs:
        return None
    best = max(accs.values())
    target = 2 * n_layers / 3
    return min((L for L, a in accs.items() if a == best), key=lambda L: (abs(L - target), L))


def _user_render(backend: Backend, text: str):
    ep = Episode(messages=[{"role": "user", "content": text}], tools=[], turns=[-1])
    return render(ep, backend.tokenizer)


def build_contrast(backend: Backend, name: str, doc: dict, layers: Sequence[int], vs: VectorSet, *,
                   generalisation: Sequence[dict] = ()) -> dict:
    """Adds `name` (a key of CONTRASTS) at each layer to `vs`: mean(positive) - mean(real) at
    the colon ending each prompt of the training pairs. Returns its validation; the
    `generalisation` pairs {real, eval} are read as the mean over the user message's tokens."""
    positive, dataset, sign = CONTRASTS[name]
    layers = sorted(int(L) for L in layers)
    pairs = doc["pairs"]
    held = set(doc.get("heldout") or [])
    train_i = [i for i in range(len(pairs)) if i not in held]

    def colon_acts(text):
        rd = _user_render(backend, text)
        user = np.flatnonzero(rd.role == R["user"])
        pos = int(user[-1])
        tok = backend.tokenizer.decode([int(rd.ids[pos])])
        if not tok.rstrip().endswith(":"):
            raise ValueError(f"the last user token of {text!r} is {tok!r}, not the colon")
        acts = backend.activations(rd.ids.tolist(), layers)
        return {L: acts[L][pos].astype(np.float64) for L in layers}

    def mean_user_acts(text):
        rd = _user_render(backend, text)
        user = rd.role == R["user"]
        acts = backend.activations(rd.ids.tolist(), layers)
        return {L: acts[L][user].astype(np.float64).mean(0) for L in layers}

    real = [colon_acts(p["real"]) for p in pairs]
    fake = [colon_acts(p[positive]) for p in pairs]
    gen = list(generalisation)
    g_real = [mean_user_acts(p["real"]) for p in gen]
    g_fake = [mean_user_acts(p["eval"]) for p in gen]
    out = {}
    for L in layers:
        R_, F_ = np.stack([a[L] for a in real]), np.stack([a[L] for a in fake])
        v = D.contrast_vector(F_[train_i], R_[train_i])
        vs.add(name, L, v)
        rep = {"loo_pairs": D.paired_loo(F_, R_)}
        rep["loo_pairs"].pop("margins")
        if held:
            h = sorted(held)
            margins = (F_[h] - R_[h]) @ v
            rep["heldout_pairs"] = {"accuracy": float((margins > 0).mean()), "n": len(h)}
            rep["heldout_auroc"] = D.auroc(F_[h] @ v, R_[h] @ v)
        if gen:
            gr, gf = np.stack([a[L] for a in g_real]) @ v, np.stack([a[L] for a in g_fake]) @ v
            rep["generalisation"] = {"pair_accuracy": float((gf > gr).mean()), "auroc": D.auroc(gf, gr), "n_pairs": len(gen)}
        out[str(L)] = rep
    vs.meta.setdefault("vector_info", {})[name] = {"method": name, "dataset": dataset, "sign": sign}
    return out


def build_all(backend: Backend, rows: Sequence[dict], layers: Sequence[int], *, texts_path: str | None = None,
              pairs: dict | None = None, pairs_path: str | None = None, train_pairs: dict | None = None,
              train_pairs_path: str | None = None, concept_topics: Sequence[str] = (),
              extra_datasets: dict | None = None, progress=None) -> tuple[VectorSet, dict]:
    vs, report, neutral_means = build_emotions(backend, rows, layers, concept_topics=concept_topics, progress=progress)
    gen = ((pairs or {}).get("generalisation") or {}).get("pairs") or []  # the eval file's coding pairs, for both
    contrasts = [(n, d, p) for n, d, p in (("eval_awareness", pairs, pairs_path),
                                           ("train_awareness", train_pairs, train_pairs_path)) if d is not None]
    for name, doc, _ in contrasts:
        report[name] = build_contrast(backend, name, doc, layers, vs, generalisation=gen)
        for L in layers:  # the neutral offset, for centring readings
            nm = neutral_means[int(L)]
            vs.meta["offset"].setdefault(name, {})[str(L)] = float((nm @ vs.get(name, L)).mean()) if len(nm) else 0.0
    if pairs is not None and train_pairs is not None:
        for L in layers:
            report["train_awareness"][str(L)]["cosine_with_eval_awareness"] = float(
                vs.get("eval_awareness", L).astype(np.float64) @ vs.get("train_awareness", L))
    n = lambda kind, split=None: sum(1 for r in rows if r["kind"] == kind and (split is None or r["split"] == split))
    datasets = {"texts": {"path": texts_path, "sha256": file_sha256(texts_path) if texts_path else None,
                          "n_emotion_train": n("emotion", "train"), "n_emotion_test": n("emotion", "test"),
                          "n_neutral": n("neutral"), "generators": sorted({r.get("generator") for r in rows} - {None})}}
    for name, doc, path in contrasts:
        datasets[CONTRASTS[name][1]] = {"path": path, "sha256": file_sha256(path) if path else None,
                                        "n_pairs": len(doc["pairs"]), "heldout": doc.get("heldout")}
    datasets.update(extra_datasets or {})
    vs.meta.update({"model": backend.info, "method": METHOD, "datasets": datasets, "eval": report,
                    "chosen_layer": report["chosen_layer"], "layer_rule": "highest held-out emotion accuracy; ties to "
                    "the layer closest to two-thirds depth", "created": now(), "git_commit": git_commit()})
    return vs, report


def load_rows(path: str | Path) -> list[dict]:
    import json

    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
