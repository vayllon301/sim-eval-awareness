"""Exploratory (Colab sessions; LOG.md). Not part of the testbed.

Progress of a running coding run, read from Inspect's files rather than its display: every
attempt log in the run's directory (a task retry is a new attempt log), and each attempt's live
sample buffer (agent turns, simulator calls, last activity per episode). In session 5 a stopper
that watched the display's retry banner never fired (LOG.md, 01-11-2026).

    uv run python -I explore/colab_probe/progress.py logs/<run-name>
"""
import json, sqlite3, sys
from collections import defaultdict
from pathlib import Path

from inspect_ai.log import read_eval_log

BUFFERS = [Path.home() / "Library/Application Support/inspect_ai/samplebuffer",
           Path.home() / ".local/share/inspect_ai/samplebuffer"]


def buffer_for(log: Path) -> Path | None:
    for root in BUFFERS:
        found = sorted(root.glob(f"*/{log.name}.*.db"), key=lambda p: p.stat().st_mtime)
        if found:
            return found[-1]
    return None


def live(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    agent, sim, last = defaultdict(set), defaultdict(set), {}
    for sid, epoch, eid, data in con.execute("select sample_id, sample_epoch, event_id, data from events"):
        e = json.loads(data)
        if e.get("event") != "model":
            continue
        key = (sid, epoch)
        (sim if "simulator" in e.get("model", "") else agent)[key].add(eid)
        last[key] = max(last.get(key, ""), e.get("timestamp", "")[11:19])
    return {k: (len(agent[k]), len(sim[k]), last[k]) for k in last}


def main(run_dir: str) -> None:
    logs = sorted(Path(run_dir).glob("*.eval"))
    print(f"{run_dir}: {len(logs)} attempt log(s); a task retry adds one")
    for log in logs:
        header = read_eval_log(str(log), header_only=True)
        samples = read_eval_log(str(log)).samples or []
        done = sum(1 for s in samples if not s.error)
        errors = [f"{s.id} e{s.epoch}: {s.error.message[:80]}" for s in samples if s.error]
        print(f"\n{log.name}  status={header.status}  finished={done}  errored={len(errors)}")
        for e in errors:
            print(f"  error  {e}")
        db = buffer_for(log) if header.status == "started" else None
        for (sid, epoch), (turns, calls, at) in sorted((live(db) if db else {}).items()):
            print(f"  live   {sid} e{epoch}: agent turns {turns}, simulator calls {calls}, last {at} UTC")


if __name__ == "__main__":
    main(sys.argv[1])
