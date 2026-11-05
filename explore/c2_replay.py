"""Does the check against the real output stop the C2 takeover? Replays pilot C2 episodes on the laptop.

For each episode where the simulator left the real output, take the first such step and ask the
simulator again, with the history as the agent saw it up to there: up to 3 samples, as the observer
does. Then take the next step, with the real output in the history, as the fix leaves it. Development
only, on MLX, so never evidence (AGENTS.md, Rules).

    uv run python explore/c2_replay.py ../TFM/experiment/logs/n2-pilot --out logs/c2-replay.jsonl

Re-running with the same --out skips the episodes already there. The turn is sent as the observer sends
it; --no-reminder takes GROUNDED_REMINDER off, to compare. As the observer, an empty real output calls no
simulator and a cut reply never passes.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
import urllib.request
import zipfile
from pathlib import Path

from simtest.coding import prompts
from simtest.coding.machine import ExecResult, tool_output
from simtest.coding.observers import RESAMPLES, grounded_mismatch, make_observer, parse
from simtest.coding.scenario import load_scenario

URL = "http://localhost:1234/v1/chat/completions"


def episodes(log_dir: str):
    for path in sorted(glob.glob(f"{log_dir}/*.eval")):
        z = zipfile.ZipFile(path)
        if json.loads(z.read("_journal/start.json"))["eval"]["task_args"]["condition"] != "C2":
            continue
        for name in sorted(n for n in z.namelist() if n.startswith("samples/")):
            yield json.loads(z.read(name))


def ask(model: str, messages: list[dict]) -> tuple[str, dict]:
    body = {"model": model, "messages": messages, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16000}
    request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        reply = json.load(response)
    choice = reply["choices"][0]
    return tool_output(parse(choice["message"].get("content") or "")), {
        "seconds": round(time.monotonic() - t0, 1), "stop": choice.get("finish_reason"), "usage": reply.get("usage")}


def step(obs, model: str, command: str, real: str, reminder: bool = True) -> dict:
    """Up to 1 + RESAMPLES samples, stopping at the first that passes, as SimulatedBash.observe."""
    messages = [{"role": m.role, "content": m.text} for m in obs.messages(command, ExecResult(real, 0, False, 0.0))]
    if not reminder:
        messages[-1]["content"] = messages[-1]["content"].removesuffix("\n" + prompts.GROUNDED_REMINDER)
    truth = tool_output(real)
    if not truth.strip():
        return {"command": command[:200], "real_chars": 0, "samples": [], "passed_on": None, "real_empty": True}
    samples = []
    for _ in range(1 + RESAMPLES):
        text, info = ask(model, messages)
        reason = "incomplete" if info["stop"] != "stop" else grounded_mismatch(text, truth)
        samples.append({"reason": reason, "head": text[:300], **info})
        if reason is None:
            break
    return {"command": command[:200], "real_chars": len(truth), "samples": samples,
            "passed_on": next((i for i, s in enumerate(samples) if s["reason"] is None), None)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="qwen-agentworld-35b-a3b-mlx")
    parser.add_argument("--no-reminder", action="store_true")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(args.out).exists():
        done = {(r["id"], r["epoch"]) for r in map(json.loads, open(args.out))}
    with open(args.out, "a") as out:
        for s in episodes(args.log_dir):
            if (s["id"], s["epoch"]) in done:
                continue
            steps = s["store"]["CodingEpisode:steps"]
            bash = [x for x in steps if x.get("tool") == "bash" and x.get("result") is not None]
            first = next((i for i, x in enumerate(bash) if tool_output(x["result"].get("output") or "").strip()
                          and grounded_mismatch(x.get("observation") or "", tool_output(x["result"]["output"]))), None)
            if first is None:
                continue
            obs = make_observer("C2", "laptop", load_scenario(s["id"]), s["store"]["CodingEpisode:session_start"])
            obs.history = [(x["command"], x.get("observation") or "") for x in bash[:first]]
            row = {"id": s["id"], "epoch": s["epoch"], "first_bad_step": first, "n_steps": len(bash)}
            row["bad_step"] = step(obs, args.model, bash[first]["command"], bash[first]["result"].get("output") or "", not args.no_reminder)
            if first + 1 < len(bash):                  # the fix leaves the real output in the history
                obs.history.append((bash[first]["command"], tool_output(bash[first]["result"].get("output") or "")))
                nxt = bash[first + 1]
                row["next_step"] = step(obs, args.model, nxt["command"], nxt["result"].get("output") or "", not args.no_reminder)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(s["id"], s["epoch"], "first bad", first, "passed on", row["bad_step"]["passed_on"],
                  "next", row.get("next_step", {}).get("passed_on"), flush=True)


if __name__ == "__main__":
    main()
