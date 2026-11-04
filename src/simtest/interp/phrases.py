"""Picking the logit monitor's sentences from what the pilot agent actually wrote (PLAN.md,
Interpretability). The rule is a DRAFT (docs/INTERP.md, "Draft rule: the monitor's sentences"): a
mechanical pick from text alone, made before anyone sees a monitor's value on these sentences.

    simtest interp phrases logs/n2-pilot --judge judge.jsonl --kappa pass.json --episodes EPISODES --out FILE

- Eval and training groups: the frozen judge's verified quotes from the agent's reasoning, in scored
  episodes it labelled positive (eval_awareness, training). Each is cut to a clean sentence, near
  duplicates are one sentence, and a sentence needs 2 episodes. The ones in the most world pairs and
  episodes come first. Sentences that name code, paths or the scenario's own words are dropped.
- Control group: the first sentences of reasoning turns, in scored episodes with both labels
  negative: what the agent writes when it isn't aware, ranked the same way.
- A group short of its minimum is filled from the draft file, in its order, marked `draft`.
- `provenance` says where each sentence came from. The report uses it to score no sentence on a world
  it came from alone (validate.py), since a sentence picked on an episode can't validate on it.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable

from .validate import pair_of

MAX = {"eval": 5, "training": 3, "control": 5}
MIN = {"eval": 3, "training": 2, "control": 3}
MIN_EPISODES = 2
WORDS = (3, 14)
NOT_PROSE = re.compile(r"[/`_\\<>=$#{}\[\]\d.]")   # a dot left inside a sentence is a file name
END = re.compile(r"[.!?:;]+(?=\s|$)")
ELLIPSIS = re.compile(r"^(?:\.\.\.|…)\s*")


def sentence(text: str) -> str:
    """The first sentence of `text`, without its end mark, starting with a capital."""
    t = " ".join(ELLIPSIS.sub("", text.strip()).split("\n")[0].split())
    cut = END.search(t)
    t = (t[:cut.start()] if cut else t).strip(" ,-—\"'")
    return t[:1].upper() + t[1:]


def usable(text: str, banned: Iterable[str] = ()) -> bool:
    words = text.lower().split()
    return (WORDS[0] <= len(words) <= WORDS[1] and not NOT_PROSE.search(text)
            and not set(re.findall(r"[a-z]+", text.lower())) & set(banned))


def _key(text: str) -> str:
    return " ".join(re.findall(r"[a-z']+", text.lower())[:8])


def _rank(found: dict[str, dict], minimum: int) -> list[tuple[str, dict]]:
    """Sentences by world pairs, then episodes, then shortest; each with its first spelling."""
    keep = [(v["text"], v) for v in found.values() if len(v["episodes"]) >= minimum]
    return sorted(keep, key=lambda x: (-len(x[1]["pairs"]), -len(x[1]["episodes"]), len(x[0]), x[0]))


def _add(found: dict, text: str, uuid: str, scenario: str, banned: set[str]) -> None:
    s = sentence(text)
    if usable(s, banned):
        entry = found.setdefault(_key(s), {"text": s, "episodes": set(), "pairs": set()})
        entry["episodes"].add(uuid)
        entry["pairs"].add(pair_of(scenario))


def pick(results: list[dict], episodes: list[dict], turn_texts: Callable[[str], list[str]], draft: dict, *,
         run: str | None = None) -> dict:
    """A phrase file. `episodes` as `load_run` gives them, with `judge_key`; `turn_texts(uuid)` is an
    episode's reasoning per turn; `draft` the draft phrase file."""
    from ..coding import analysis as A
    from ..pilot import _index

    by = _index(results)
    scored = [e for e in episodes if not A.is_excluded(e)]
    banned = {w for e in scored for w in re.findall(r"[a-z]{3,}", e["scenario"].lower())} - {"benign"}
    found: dict[str, dict] = {"eval": {}, "training": {}, "control": {}}
    for e in scored:
        results_ = {label: by.get((e["judge_key"], label)) for label in ("eval_awareness", "training")}
        for group, label in (("eval", "eval_awareness"), ("training", "training")):
            r = results_[label]
            if r and r.get("positive") and not r.get("error"):
                for q in r.get("quotes") or []:
                    if q.get("status") == "verified" and q.get("part") == "reasoning":
                        _add(found[group], q["text"], e["uuid"], e["scenario"], banned)
        if all(r and r.get("positive") is False and not r.get("error") for r in results_.values()):
            for text in turn_texts(e["uuid"]):
                _add(found["control"], text, e["uuid"], e["scenario"], banned)
    groups: dict[str, list[str]] = {}
    provenance: dict[str, dict] = {}
    for group, cap in MAX.items():
        picked = _rank(found[group], MIN_EPISODES)[:cap]
        groups[group] = [t for t, _ in picked]
        for t, v in picked:
            provenance[t] = {"group": group, "source": "pilot", "episodes": len(v["episodes"]), "pairs": sorted(v["pairs"])}
        for t in draft["groups"].get(group, []):
            if len(groups[group]) >= MIN[group]:
                break
            if t not in groups[group]:
                groups[group].append(t)
                provenance[t] = {"group": group, "source": "draft", "episodes": 0, "pairs": []}
    counts = {g: [sum(provenance[t]["source"] == where for t in groups[g]) for where in ("pilot", "draft")] for g in MAX}
    return {"status": "DRAFT rule, not frozen: " + "; ".join(f"{g} {p} from the pilot, {d} from the draft file"
                                                             for g, (p, d) in counts.items()),
            "source": f"The frozen judge's verified quotes on {run or 'the pilot'} (simtest interp phrases, docs/INTERP.md), "
                      "filled from the draft file where a group is short.",
            "groups": groups, "provenance": provenance}


def reasoning_turns(episode) -> list[str]:
    """An exported episode's reasoning, one string per agent turn."""
    return [m.get("reasoning_content") or "" for m in episode.messages if m["role"] == "assistant"]

