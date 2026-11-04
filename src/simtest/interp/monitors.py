"""The logit monitor: how likely the agent is to start a turn's reasoning with a given
sentence. Reference: docs/INTERP.md, "Reader".

After Aranguri ("Different eval awareness may track different graders", LessWrong,
25-10-2026): the log-probability that the model writes a short awareness sentence ("The
user might be testing") at the start of its chain of thought.

- For each agent turn with reasoning tokens, the prefix is the episode's token IDs up to
  (not including) the turn's first reasoning token. Each phrase is encoded as plain text
  (`backend.encode`) and scored after it: total log-prob (nats), mean per token, tokens.
- Phrases come in groups (eval, training, control). Each group's mean is compared with the
  control group's, so a condition that makes every sentence more or less likely isn't
  read as awareness.
- The scan: a backend with `continuation_scan` (HF, MLX; backend.py) makes one pass over the
  episode and forks its cache at each turn's stop. This is the monitor's path: its cost is
  one pass plus a short pass per turn and phrase.
- Generic path: `backend.token_losses(prefix + phrase)`, the last len(phrase) losses. One
  forward pass over the whole prefix per turn and phrase (the vLLM backend), so meant for
  checks: on the pilot it would take about 430M tokens against 2.3M (docs/INTERP.md).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .backend import Backend, check_scan, encode
from .reader import R

FIELDS = ("logprob", "mean")  # total log-prob of a phrase (nats), mean per token


def continuation_logprobs(backend: Backend, prefix: Sequence[int],
                          continuations: Sequence[Sequence[int]]) -> list[np.ndarray]:
    """Per-token log-probs (nats) of each continuation after `prefix`: the backend's own
    `continuation_logprobs` if it has one, else one `token_losses` pass per continuation."""
    prefix = [int(i) for i in prefix]
    conts = [[int(i) for i in c] for c in continuations]
    if not prefix or not all(conts):
        raise ValueError("the prefix and every continuation need at least one token")
    fast = getattr(backend, "continuation_logprobs", None)
    if fast is not None:
        out = [np.asarray(x, dtype=np.float64) for x in fast(prefix, conts)]
    else:
        out = [-np.asarray(backend.token_losses(prefix + c), dtype=np.float64)[-len(c):] for c in conts]
    if [x.shape for x in out] != [(len(c),) for c in conts]:
        raise ValueError("the backend returned log-probs of the wrong shape")
    return out


def continuation_scan(backend: Backend, ids: Sequence[int], stops: Sequence[int],
                      continuations: Sequence[Sequence[int]]) -> dict[int, list[np.ndarray]]:
    """{stop: per-token log-probs of each continuation after ids[:stop]}: the backend's own
    `continuation_scan` if it has one, else one prefix at a time."""
    ids, stops, conts = check_scan(ids, stops, continuations)
    fast = getattr(backend, "continuation_scan", None)
    if fast is not None:
        out = {int(p): [np.asarray(x, dtype=np.float64) for x in row] for p, row in fast(ids, stops, conts).items()}
    else:
        out = {p: continuation_logprobs(backend, ids[:p], conts) for p in stops}
    if sorted(out) != stops or any([x.shape for x in row] != [(len(c),) for c in conts] for row in out.values()):
        raise ValueError("the backend's scan returned the wrong stops or shapes")
    return out


def scan_path(backend: Backend) -> str:
    for name in ("continuation_scan", "continuation_logprobs"):
        if getattr(backend, name, None) is not None:
            return name
    return "token_losses"


def reasoning_starts(role: np.ndarray, turn: np.ndarray) -> dict[int, int | None]:
    """For each agent turn, the position of its first reasoning token (None if it has none)."""
    role, turn = np.asarray(role), np.asarray(turn)
    out = {}
    for t in sorted(int(x) for x in set(turn.tolist()) if x >= 0):
        at = np.flatnonzero((turn == t) & (role == R["reasoning"]))
        out[t] = int(at[0]) if len(at) else None
    return out


def logit_monitor(backend: Backend, ids: Sequence[int], role: np.ndarray, turn: np.ndarray,
                  groups: Mapping[str, Sequence[str]], *, turns: Sequence[int] | None = None,
                  control: str = "control") -> dict:
    """Scores every phrase in `groups` ({group: [phrases]}) at the start of each agent turn's
    reasoning (all turns, or `turns`). Per turn: each phrase's log-prob, each group's mean,
    and each other group minus `control`; None for a turn without reasoning tokens."""
    phrases = [(g, text) for g, texts in groups.items() for text in texts]
    for g, texts in groups.items():
        if not texts:
            raise ValueError(f"phrase group {g!r} has no phrases")
    pids = [encode(backend.tokenizer, text) for _, text in phrases]
    for (_, text), p in zip(phrases, pids):
        if not p:
            raise ValueError(f"phrase {text!r} has no tokens")
    starts = reasoning_starts(role, turn)
    turns = sorted(starts) if turns is None else [int(t) for t in turns]
    diffs = [g for g in groups if g != control] if control in groups else []
    for t in turns:
        if t not in starts:
            raise ValueError(f"turn {t} is not in the episode (agent turns {sorted(starts)})")
    stops = [starts[t] for t in turns if starts[t] is not None]
    scores = continuation_scan(backend, ids, stops, pids) if stops else {}
    by_turn = []
    for t in turns:
        pos = starts[t]
        row: dict = {"turn": t, "prefix_len": pos, "phrases": None, "groups": None,
                     **{f"{g}_minus_control": None for g in diffs}}
        if pos is not None:
            lp = scores[pos]
            row["phrases"] = [{"logprob": float(x.sum()), "mean": float(x.mean())} for x in lp]
            row["groups"] = {g: {k: float(np.mean([r[k] for r, (g_, _) in zip(row["phrases"], phrases) if g_ == g]))
                                 for k in FIELDS} for g in groups}
            for g in diffs:
                row[f"{g}_minus_control"] = {k: row["groups"][g][k] - row["groups"][control][k] for k in FIELDS}
        by_turn.append(row)
    return {"path": scan_path(backend), "phrases": [{"group": g, "text": text, "n_tokens": len(p)} for (g, text), p in zip(phrases, pids)],
            "by_turn": by_turn}
