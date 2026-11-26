"""The online leak guard (observers.format_leaked) on every logged coding episode: how many real
bash outputs it would flag (false positives: must be 0) and how many simulated observations the
agent saw that it flags now, by reason.

Analysis only: reads .eval logs, calls no model.

    uv run --no-sync python explore/leak_guard_corpus.py logs experiments [--show 20]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def bash_steps(roots: list[Path]):
    """(path, sample id, step, command, real output as C0 renders it, observation) for every
    executed bash step with an output, in every .eval log under `roots`."""
    from inspect_ai.log import read_eval_log

    from simtest.coding.machine import tool_output

    for root in roots:
        for path in sorted(root.rglob("*.eval")):
            try:
                log = read_eval_log(str(path))
            except Exception as exc:  # a log being written, or another format
                print(f"skipped {path}: {exc!r}", file=sys.stderr)
                continue
            for s in log.samples or []:
                steps = (s.store or {}).get("CodingEpisode:steps")
                for st in steps if isinstance(steps, list) else []:
                    res = st.get("result") or {}
                    if st.get("tool") != "bash" or not st.get("executed") or "output" not in res:
                        continue
                    yield (str(path), str(s.id), st.get("step"), st.get("command") or "",
                           tool_output(res.get("output") or ""), st.get("observation"))


def main(argv=None) -> int:
    from simtest.coding.observers import format_leaked

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--show", type=int, default=10, help="examples to print per class")
    args = p.parse_args(argv)
    n_real = n_sim = 0
    false_pos: list[tuple] = []
    caught: Counter = Counter()
    examples: list[tuple] = []
    for path, sid, step, command, real, seen in bash_steps(args.roots):
        n_real += 1
        if (why := format_leaked(real, command)) is not None:
            false_pos.append((why, path, sid, step, command[:120], real[:160]))
        if seen is not None and seen != real:
            n_sim += 1
            if (why := format_leaked(seen, command)) is not None:
                caught[why] += 1
                examples.append((why, path, sid, step, seen[:160]))
    print(f"real bash outputs: {n_real}; flagged (false positives): {len(false_pos)}")
    for x in false_pos[:args.show]:
        print("  FP", *x)
    print(f"observations that differ from the real output: {n_sim}; flagged: {sum(caught.values())} {dict(caught)}")
    for x in examples[:args.show]:
        print("  ", *x)
    return 1 if false_pos else 0


if __name__ == "__main__":
    sys.exit(main())
