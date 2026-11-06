"""Build C2's reference transcripts from a run's C0 episodes: scenarios/coding/<id>.reference.json.

The rule, fixed before any C2 episode used them: epoch 1 of C0, per scenario, commands and the outputs
as the agent saw them, nothing else. Only an episode the agent ended itself counts, unflagged. The
transcript stops before the first command outside the exact read-only allowlist: with the whole
episode the simulator showed the agent's fix as if it had happened (LOG.md, 05-11-2026).

    uv run python explore/build_reference.py logs/n2-pilot
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


def main(log_dir: str) -> None:
    for path in glob.glob(f"{log_dir}/*.eval"):
        z = zipfile.ZipFile(path)
        if json.loads(z.read("_journal/start.json"))["eval"]["task_args"]["condition"] != "C0":
            continue
        for name in sorted(n for n in z.namelist() if n.startswith("samples/") and n.endswith("_epoch_1.json")):
            s = json.loads(z.read(name))
            meta = s["scores"]["coding_scorer"]["metadata"]
            if meta["end_reason"] != "no_tool_call" or meta.get("flagged"):
                raise SystemExit(f"{s['id']}: epoch 1 didn't end cleanly; the rule has no fallback, decide by hand")
            steps = [{"command": x["command"], "output": x.get("observation") or ""}
                     for x in s["store"]["CodingEpisode:steps"] if x.get("tool") == "bash" and x.get("executed", True)]
            steps = prefix(steps, readonly_commands(s["id"]))
            out = Path("scenarios/coding") / f"{s['id']}.reference.json"
            out.write_text(json.dumps(steps, indent=1, ensure_ascii=False) + "\n")
            print(out, len(steps), "steps", out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
