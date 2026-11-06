"""Explicit, one-use continuation of the interrupted controlled C2 screen.

See C2_CONTROLLED_REPLAY.md's amendment. Original inputs and responses stay intact.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from explore.c2_controlled_replay import ROOT, URL, digest, evaluate, head, report

BASE = "2f1bcde57650b56957323efdfff04819519a3262"
ALLOWED_CHANGES = {"explore/c2_controlled_resume.py", "tests/test_c2_controlled_resume.py",
                   "explore/C2_CONTROLLED_REPLAY.md"}
LMS = "/Users/victorayllon/.lmstudio/bin/lms"
MODEL_FIELDS = ("identifier", "modelKey", "path", "format", "sizeBytes", "quantization",
                "contextLength", "parallel", "engineConfigFileEnabled")


def pending_jobs(manifest, rows):
    report(manifest, rows)  # Reject duplicate, unknown or stale identities.
    if [r["key"] for r in rows] != [j["key"] for j in manifest["jobs"][:len(rows)]]:
        raise ValueError("Saved rows must be an ordered prefix")
    for job, row in zip(manifest["jobs"], rows):
        if "error" in row or not row.get("complete"):
            raise ValueError("This amendment covers only the complete saved prefix")
        for key, value in evaluate(job, row["reply"]).items():
            if row[key] != value:
                raise ValueError("Saved reply or grade changed")
    return manifest["jobs"][len(rows):]


def continue_run(original: Path, output: Path):
    commit = head()
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only", BASE, commit], cwd=ROOT, text=True).splitlines())
    if changed - ALLOWED_CHANGES:
        raise ValueError("Original generation/scoring code or inputs changed")
    manifest_bytes = (original / "manifest.json").read_bytes()
    original_bytes = (original / "responses.jsonl").read_bytes()
    if not original_bytes.endswith(b"\n"):
        raise ValueError("Torn original response file; do not discard its tail")
    manifest = json.loads(manifest_bytes)
    if manifest["code_commit"] != BASE:
        raise ValueError("Unexpected original code commit")
    rows = [json.loads(line) for line in original_bytes.splitlines()]
    pending = pending_jobs(manifest, rows)
    if len(rows) != 12 or len(pending) != 12:
        raise ValueError("This amendment covers exactly the remaining 12 calls")
    previous_model = json.loads((original / "lms-loaded.json").read_text())
    current_model = json.loads(subprocess.check_output([LMS, "ps", "--json"], text=True))
    signature = lambda models: [{k: m[k] for k in MODEL_FIELDS} for m in models]
    if signature(previous_model) != signature(current_model):
        raise ValueError("Loaded model configuration changed")
    # Reserve a full request timeout for a possibly lost in-flight call. Downtime
    # between segments is excluded by the explicit amendment, never by result.
    spent = sum(r["seconds"] for r in rows) + manifest["config"]["request_timeout_seconds"]
    remaining = manifest["config"]["wall_seconds"] - spent
    if remaining <= 0:
        raise ValueError("No admission budget remains")
    output.mkdir(exist_ok=False)
    (output / "manifest.json").write_bytes(manifest_bytes)
    metadata = {"runner_commit": commit, "original_commit": BASE,
                "original_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "original_responses_sha256": hashlib.sha256(original_bytes).hexdigest(),
                "remaining_admission_seconds": remaining, "reserved_lost_call_seconds":
                manifest["config"]["request_timeout_seconds"], "model": current_model,
                "pending_keys": [j["key"] for j in pending]}
    (output / "continuation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    started = time.monotonic()
    new_rows = []
    with (output / "responses.jsonl").open("x") as stream:
        for job in pending:
            if time.monotonic() - started >= remaining:
                break
            row = {"key": job["key"], "manifest_sha256": digest(manifest)}
            body = {"model": manifest["config"]["model"], "messages": job["messages"],
                    **manifest["config"]["generation"]}
            request = urllib.request.Request(URL, json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=manifest["config"]["request_timeout_seconds"]) as response:
                    row["reply"] = json.load(response)
                if row["reply"].get("model") != manifest["config"]["model"]:
                    raise ValueError("Server replied from a different model")
                row.update(evaluate(job, row["reply"]))
            except Exception as exc:
                row["error"] = repr(exc)
            row["seconds"] = round(time.monotonic() - t0, 3)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            new_rows.append(row)
            print(job["key"], row.get("stop", "error"), row.get("grade", {}).get("grade"), flush=True)
    if (original / "responses.jsonl").read_bytes() != original_bytes:
        raise ValueError("Original response file changed during continuation")
    result = report(manifest, rows + new_rows)
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    continue_run(args.original, args.output)
