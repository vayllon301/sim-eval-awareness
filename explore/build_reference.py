"""Build C2's reference transcripts from a run's C0 episodes: scenarios/coding/<id>.reference.json.

The rule, fixed before any C2 episode used them: epoch 1 of C0, per scenario, commands and the outputs
as the agent saw them, nothing else. Only an episode the agent ended itself counts, unflagged. The
transcript stops before the first command outside the exact read-only allowlist: with the whole
episode the simulator showed the agent's fix as if it had happened (LOG.md, 05-11-2026). The six
Python scenarios' allowlists end with the agent's first, unedited test or schema check, the one
output that runs code, audited before any edit (review of 99d2c4c); the flows ones run no test.

    uv run python explore/build_reference.py logs/n2-pilot
    uv run python explore/build_reference.py logs/latent-qwen-hopper-only --scenarios pricing_cap billing_rounding

`--scenarios` writes only those scenarios' references (default: every C0 epoch-1 episode in the log
directory); each must be in the log and in the allowlist.
"""

from __future__ import annotations

import glob
import json
import sys
import zipfile
from pathlib import Path

ALLOWLIST = Path(__file__).resolve().parents[1] / "configs/coding/c2_reference_readonly.json"


def readonly_commands(scenario: str) -> set[str]:
    """Exact commands audited in the six development references; new forms need an audit."""
    return set(json.loads(ALLOWLIST.read_text())[scenario])


def prefix(steps: list[dict], allowed: set[str]) -> list[dict]:
    """Stop before the first command outside the audited allowlist. No shell inference."""
    for i, step in enumerate(steps):
        if step["command"] not in allowed:
            return steps[:i]
    return steps


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("log_dir")
    p.add_argument("--scenarios", nargs="+", default=None, help="only these scenario ids (default: all in log_dir)")
    p.add_argument("--out", default="scenarios/coding", help="where the <id>.reference.json files go")
    args = p.parse_args(argv)
    wanted = set(args.scenarios) if args.scenarios else None
    built: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(f"{args.log_dir}/*.eval")):
        z = zipfile.ZipFile(path)
        if json.loads(z.read("_journal/start.json"))["eval"]["task_args"]["condition"] != "C0":
            continue
        for name in sorted(n for n in z.namelist() if n.startswith("samples/") and n.endswith("_epoch_1.json")):
            s = json.loads(z.read(name))
            if wanted is not None and s["id"] not in wanted:
                continue
            meta = s["scores"]["coding_scorer"]["metadata"]
            if meta["end_reason"] != "no_tool_call" or meta.get("flagged"):
                raise SystemExit(f"{s['id']}: epoch 1 didn't end cleanly; the rule has no fallback, decide by hand")
            steps = [{"command": x["command"], "output": x.get("observation") or ""}
                     for x in s["store"]["CodingEpisode:steps"] if x.get("tool") == "bash" and x.get("executed", True)]
            built[s["id"]] = prefix(steps, readonly_commands(s["id"]))
    if wanted is not None and (missing := sorted(wanted - set(built))):
        raise SystemExit(f"no C0 epoch-1 episode in {args.log_dir} for: {', '.join(missing)}")
    for sid, steps in built.items():      # nothing is written unless every scenario was found
        out = Path(args.out) / f"{sid}.reference.json"
        out.write_text(json.dumps(steps, indent=1, ensure_ascii=False) + "\n")
        print(out, len(steps), "steps", out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
