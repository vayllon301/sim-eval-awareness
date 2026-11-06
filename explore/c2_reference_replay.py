"""Does the reference move fidelity? Replays pilot C3 steps on the laptop, with and without it.

For each scenario's C3 epoch 1, steps 1, 3, 5, 7 and 9, with the history as the agent saw it, one reply
from C3's prompt and one from C2's. Each is scored against the real output, as a word multiset. Development
only, on MLX, so never evidence.

    uv run python explore/c2_reference_replay.py logs/n2-pilot --out logs/c2-reference/replay.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from simtest.coding.machine import ExecResult, tool_output
from simtest.coding.observers import make_observer, parse
from simtest.coding.scenario import load_scenario

URL = "http://localhost:1234/v1/chat/completions"
STEPS = (1, 3, 5, 7, 9)


def similarity(a: str, b: str) -> float:
    x, y = Counter(a.split()), Counter(b.split())
    if not x and not y:
        return 1.0
    shared = sum((x & y).values())
    return min(shared / max(1, sum(x.values())), shared / max(1, sum(y.values())))


def ask(model: str, messages: list[dict]) -> tuple[str, dict]:
    body = {"model": model, "messages": messages, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16000}
    request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(request, timeout=3600) as response:
        reply = json.load(response)
    choice = reply["choices"][0]
    return tool_output(parse(choice["message"].get("content") or "")), {
        "seconds": round(time.monotonic() - t0, 1), "stop": choice.get("finish_reason"), "usage": reply.get("usage")}


def one(job: dict, model: str) -> dict:
    text, info = ask(model, job.pop("messages"))
    return job | {"sim": round(similarity(text, job["real"]), 3), "exact": text.strip() == job["real"].strip(),
                  "head": text[:200], **info} | {"real": job["real"][:200]}


def jobs(log_dir: str):
    for path in glob.glob(f"{log_dir}/*.eval"):
        z = zipfile.ZipFile(path)
        if json.loads(z.read("_journal/start.json"))["eval"]["task_args"]["condition"] != "C3":
            continue
        for name in sorted(n for n in z.namelist() if n.startswith("samples/") and n.endswith("_epoch_1.json")):
            s = json.loads(z.read(name))
            bash = [x for x in s["store"]["CodingEpisode:steps"] if x.get("tool") == "bash" and x.get("result") is not None]
            scenario = load_scenario(s["id"])
            for i in STEPS:
                if i >= len(bash):
                    continue
                real = tool_output(bash[i]["result"].get("output") or "")
                for condition in ("C3", "C2"):
                    obs = make_observer(condition, "laptop", scenario, s["store"]["CodingEpisode:session_start"])
                    obs.history = [(x["command"], x.get("observation") or "") for x in bash[:i]]
                    messages = [{"role": m.role, "content": m.text}
                                for m in obs.messages(bash[i]["command"], ExecResult("", 0, False, 0.0))]
                    yield {"id": s["id"], "step": i, "condition": condition, "command": bash[i]["command"][:160],
                           "real": real, "messages": messages}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="qwen-agentworld-35b-a3b-mlx")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as out, cf.ThreadPoolExecutor(4) as pool:
        for fut in cf.as_completed([pool.submit(one, j, args.model) for j in jobs(args.log_dir)]):
            row = fut.result()
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(row["id"], row["step"], row["condition"], row["sim"], row["seconds"], flush=True)


if __name__ == "__main__":
    main()
