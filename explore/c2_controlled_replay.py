"""A paired development screen of C2's reference, using real C0 prefixes.

prepare freezes the inputs without contacting a model. run sends the saved prompts to
the laptop only. report includes every planned call, including failures and missing calls.
See C2_CONTROLLED_REPLAY.md for the decision rule and the limits.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

from explore.build_reference import prefix, readonly_commands
from simtest.coding.machine import ExecResult, tool_output
from simtest.coding.observers import make_observer, parse
from simtest.coding.scenario import load_scenario
from simtest.post import fidelity as F

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/coding/c2_controlled_replay.json"
URL = "http://127.0.0.1:1234/v1/chat/completions"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def head() -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit the experiment changes first")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def c0_samples(log_dir: Path) -> tuple[dict, dict]:
    samples, hashes = {}, {}
    for path in sorted(log_dir.glob("*.eval")):
        with zipfile.ZipFile(path) as z:
            if json.loads(z.read("_journal/start.json"))["eval"]["task_args"].get("condition") != "C0":
                continue
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for name in sorted(z.namelist()):
                if not (name.startswith("samples/") and name.endswith(".json")):
                    continue
                sample = json.loads(z.read(name))
                key = (sample["id"], sample["epoch"])
                if key in samples:
                    raise ValueError(f"Duplicate C0 sample: {key}")
                samples[key] = sample
    return samples, hashes


def bash_steps(sample: dict) -> list[dict]:
    meta = sample["scores"]["coding_scorer"]["metadata"]
    if meta["end_reason"] != "no_tool_call" or meta.get("flagged"):
        raise ValueError("C0 must have ended cleanly; no fallback sample selection")
    steps = []
    for step in sample["store"]["CodingEpisode:steps"]:
        if step.get("tool") != "bash":
            continue
        if not step.get("executed") or "output" not in (step.get("result") or {}):
            raise ValueError("C0 has a bash step without executed ground truth")
        if step.get("observation") != tool_output(step["result"]["output"]):
            raise ValueError("C0 observation differs from its real rendering")
        steps.append(step)
    return steps


def reference_audit(reference: list[dict], snapshot: str) -> dict:
    lines = [line for step in reference for line in step["output"].splitlines() if line.strip()]
    return {
        "reference_sha256": digest(reference), "snapshot_sha256": digest(snapshot),
        "steps": len(reference), "output_characters": sum(len(s["output"]) for s in reference),
        "nonblank_lines": len(lines), "lines_present_in_snapshot": sum(line in snapshot for line in lines),
        "entries": [{"index": i, "kind": F.command_kind(s["command"])["kind"],
                     "read_only": F.command_kind(s["command"])["read_only"],
                     "command_sha256": digest(s["command"]),
                     "new_lines": sum(bool(line.strip()) and line not in snapshot for line in s["output"].splitlines())}
                    for i, s in enumerate(reference)],
    }


def select_steps(steps: list[dict], kinds: list[str]) -> list[int]:
    """First occurrence of each fixed kind, independent of simulated outputs or scores."""
    return [next(i for i, step in enumerate(steps) if F.command_kind(step["command"])["kind"] == kind)
            for kind in kinds]


def prepare(log_dir: Path, config: dict, commit: str) -> dict:
    if config["reference_epoch"] == config["target_epoch"]:
        raise ValueError("Reference and target epochs must differ")
    samples, sources = c0_samples(log_dir)
    jobs, audit = [], {}
    for scenario_index, name in enumerate(config["scenarios"]):
        scenario = load_scenario(name)
        ref_steps = bash_steps(samples[name, config["reference_epoch"]])
        expected = prefix([{"command": s["command"], "output": s["observation"]} for s in ref_steps],
                          readonly_commands(name))
        if not expected or scenario.reference != expected:
            raise ValueError(f"{name}: regenerate the conservative reference from C0 epoch 1")
        audit[name] = reference_audit(expected, scenario.snapshot)
        sample = samples[name, config["target_epoch"]]
        steps = bash_steps(sample)
        for kind_index, index in enumerate(select_steps(steps, config["kinds"])):
            step = steps[index]
            # Alternate which condition is called first, within scenarios and across pairs.
            order = ("C3", "C2") if (scenario_index + kind_index) % 2 else ("C2", "C3")
            for condition in order:
                observer = make_observer(condition, "laptop", scenario, sample["store"]["CodingEpisode:session_start"])
                observer.history = [(s["command"], s["observation"]) for s in steps[:index]]
                messages = [{"role": m.role, "content": m.text}
                            for m in observer.messages(step["command"], ExecResult("", 0, False, 0.0))]
                jobs.append({"key": f"{name}:{index}:{condition}", "id": name, "index": index,
                             "epoch": config["target_epoch"], "kind": config["kinds"][kind_index],
                             "condition": condition, "command": step["command"], "real": step["observation"],
                             "history": observer.history,
                             "messages": messages})
    return {"config": config, "code_commit": commit, "sources": sources, "audit": audit, "jobs": jobs}


def evaluate(job: dict, reply: dict) -> dict:
    choice = reply["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Missing textual content")
    parsed = parse(content)
    seen = tool_output(parsed)
    # Both arms invent outputs. Legacy Context(C2) would incorrectly treat the current
    # real output as given. C3 mode uses only the actual prompt and the past commands.
    ctx = F.Context("\n".join(m["content"] for m in job["messages"][:-1]), "C3")
    for command, seen_before in job["history"]:
        ctx.add(command, seen_before)
    grade = F.grade_step(job["command"], job["real"], seen, ctx.for_step(job["command"], job["real"]))
    complete = choice.get("finish_reason") == "stop" and not (
        "<think>" in content and content.rfind("<think>") > content.rfind("</think>"))
    return {"seen": seen, "parsed_output": parsed, "stop": choice.get("finish_reason"),
            "complete": complete, "faithful": complete and grade["grade"] in F.FAITHFUL, "grade": grade}


def report(manifest: dict, rows: list[dict]) -> dict:
    expected = {j["key"]: j for j in manifest["jobs"]}
    actual = {}
    for row in rows:
        if row["key"] not in expected or row["key"] in actual or row["manifest_sha256"] != digest(manifest):
            raise ValueError("Unknown, duplicated or mismatched result")
        actual[row["key"]] = row
    counts, scenarios = {}, {}
    for condition in ("C2", "C3"):
        keys = [k for k, j in expected.items() if j["condition"] == condition]
        present = [actual[k] for k in keys if k in actual]
        counts[condition] = {
            "planned": len(keys), "recorded": len(present), "missing": len(keys) - len(present),
            "errors": sum("error" in r for r in present),
            "incomplete": sum(not r.get("complete", False) and "error" not in r for r in present),
            "faithful": sum(r.get("faithful", False) for r in present),
            "grades": dict(Counter(r["grade"]["grade"] for r in present if "grade" in r)),
            "tells": sum(bool(r.get("grade", {}).get("tells")) for r in present),
        }
        for name in manifest["config"]["scenarios"]:
            own = [k for k in keys if expected[k]["id"] == name]
            scenarios.setdefault(name, {})[condition] = sum(actual.get(k, {}).get("faithful", False) for k in own) / len(own)
    gain = sum(s["C2"] - s["C3"] for s in scenarios.values()) / len(scenarios)
    held = any(c["missing"] or c["errors"] or c["incomplete"] for c in counts.values())
    passed = gain + 1e-12 >= manifest["config"]["minimum_gain"] and counts["C2"]["tells"] == 0
    return {"status": "INCOMPLETE" if held else "PASS" if passed else "FAIL", "counts": counts,
            "scenario_faithful_rates": scenarios, "gain": gain,
            "minimum_gain": manifest["config"]["minimum_gain"], "scope": "Local development screen; not evidence or realism"}


def run(directory: Path, manifest: dict) -> None:
    if head() != manifest["code_commit"]:
        raise ValueError("The code commit changed; prepare a separate run")
    started = time.monotonic()
    # No silent retries or resume: every request is one draw, errors remain in the denominator.
    with (directory / "responses.jsonl").open("x") as output:
        for job in manifest["jobs"]:
            if time.monotonic() - started >= manifest["config"]["wall_seconds"]:
                break
            row = {"key": job["key"], "manifest_sha256": digest(manifest)}
            body = {"model": manifest["config"]["model"], "messages": job["messages"], **manifest["config"]["generation"]}
            request = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=manifest["config"]["request_timeout_seconds"]) as response:
                    row["reply"] = json.load(response)
                if row["reply"].get("model") != manifest["config"]["model"]:
                    raise ValueError("The server replied from a different model")
                row.update(evaluate(job, row["reply"]))
            except Exception as exc:
                row["error"] = repr(exc)
            row["seconds"] = round(time.monotonic() - t0, 3)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(job["key"], row.get("stop", "error"), row.get("grade", {}).get("grade"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "report"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--logs", type=Path, default=ROOT / "logs/n2-pilot")
    args = parser.parse_args()
    path = args.directory / "manifest.json"
    if args.action == "prepare":
        manifest = prepare(args.logs, json.loads(CONFIG.read_text()), head())
        args.directory.mkdir(parents=True, exist_ok=True)
        with path.open("x") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=1)
        print(json.dumps({"jobs": len(manifest["jobs"]), "audit": manifest["audit"]}, indent=1))
        return
    manifest = json.loads(path.read_text())
    if args.action == "run":
        run(args.directory, manifest)
    responses = args.directory / "responses.jsonl"
    rows = [json.loads(line) for line in responses.read_text().splitlines()] if responses.exists() else []
    result = report(manifest, rows)
    (args.directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
