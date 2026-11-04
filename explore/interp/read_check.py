"""Exploratory: the first GPU run's check of the reading paths (docs/INTERP.md, "What the first GPU run
measures"). One step per process and engine, since vLLM and transformers can't share the GPU:

    python -I explore/interp/read_check.py run MODEL --engine vllm --episodes E --vectors V --phrases P --out OUT
    python -I explore/interp/read_check.py run MODEL --engine hf   --episodes E --vectors V --phrases P --out OUT
    python -I explore/interp/read_check.py run MODEL --engine hf --chunk 2048 --label hf-2048 ...
    python -I explore/interp/read_check.py compare --out OUT

`run` takes the longest episode and the shortest (OOM first, then the short one) and writes
OUT/check-<label>-<uuid>.json. On vLLM the phrases go through prompt log-probs at three turns: slow and
meant for this check only. `compare` needs every path to have read the same episodes: if not, it exits 1
and writes nothing, so a retry runs the check. Then it pairs every check with vllm's on the same episode,
writes the numbers to OUT/read-check-diagnostics.json and applies the bars (readrun.path_rule):
- all met: OUT/read-check.json, the accepted check, and exit 0. read_run.sh skips the check once it exists.
- one missed: no read-check.json, OUT/read-check-held.txt with the held parts, one a line, and exit 4.
  read_run.sh holds those parts and doesn't pay for the check again. No setting overrides a hold: what
  comes next is decided with Victor. To run the check again, delete the file.
Development numbers: none of this is evidence. Pass rules: docs/INTERP.md.
"""
import argparse, json, sys, time
from pathlib import Path

PATHS = ("vllm", "hf", "hf-2048")     # read_run.sh's three runs
MISSED = 4                            # a bar missed; read_run.sh passes it on


def main():
    from simtest.interp.reader import load_episode
    from simtest.interp.readrun import ReadError, compare_paths, make_backend, path_check, path_rule
    from simtest.interp.vectors import VectorSet

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model"), r.add_argument("--engine", required=True, choices=["vllm", "hf", "mlx"])
    r.add_argument("--episodes", required=True), r.add_argument("--vectors", required=True)
    r.add_argument("--phrases", required=True), r.add_argument("--out", required=True)
    r.add_argument("--chunk", type=int, help="the scan's chunk size: a second run at another size gives the chunk noise")
    r.add_argument("--label"), r.add_argument("--allow-other-model", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "compare":
        checks = [json.loads(p.read_text()) for p in sorted(out.glob("check-*.json"))]
        read = {label: {x["uuid"] for x in checks if x["label"] == label} for label in PATHS}
        short = [label for label, eps in read.items() if not eps or eps != set().union(*read.values())]
        if short:
            sys.exit(f"read_check: {', '.join(short)} didn't read every episode: no read-check.json, so a retry runs the check")
        rows = []
        for ref in (x for x in checks if x["label"] == "vllm"):
            for other in (x for x in checks if x["label"] != "vllm" and x["uuid"] == ref["uuid"]):
                rows.append(compare_paths(ref, other))
        for ref in (x for x in checks if x["label"] == "hf"):
            for other in (x for x in checks if x["label"].startswith("hf-") and x["uuid"] == ref["uuid"]):
                rows.append(compare_paths(ref, other))
        rule = path_rule(rows)
        (out / "read-check-diagnostics.json").write_text(json.dumps({**rule, "rows": rows}, indent=1))
        for row in rows:
            print({k: v for k, v in row.items() if k != "monitor"})
        accepted, held = out / "read-check.json", out / "read-check-held.txt"
        if rule["missed"]:
            accepted.unlink(missing_ok=True)
            held.write_text("".join(f"{part}\n" for part in rule["held"]))
            for x in rule["missed"]:
                print(f"read_check: missed: {x}")
            print(f"read_check: no read-check.json; held: {', '.join(rule['held'])} (docs/INTERP.md says what comes next)")
            sys.exit(MISSED)
        held.unlink(missing_ok=True)
        accepted.write_text(json.dumps(rows, indent=1))
        return
    out.mkdir(parents=True, exist_ok=True)
    vs = VectorSet.load(a.vectors)
    groups = json.loads(Path(a.phrases).read_text())["groups"]
    rows = [r for r in json.loads((Path(a.episodes) / "manifest.json").read_text())["episodes"] if r["status"] == "exported"]
    rows.sort(key=lambda r: -(r.get("tokens") or 0))
    t = time.time()
    b = make_backend(a.engine, a.model)
    print(f"{a.engine} loaded in {time.time() - t:.0f}s", flush=True)
    if a.chunk:
        scan = b.continuation_scan
        b.continuation_scan = lambda ids, stops, conts: scan(ids, stops, conts, chunk=a.chunk)
    label = a.label or a.engine
    for row in (rows[0], rows[-1]):
        try:
            res = path_check(b, load_episode(Path(a.episodes) / f"{row['uuid']}.json"), vs, vs.meta["chosen_layer"], groups, label=label)
        except ReadError as exc:
            sys.exit(str(exc))
        (out / f"check-{label}-{row['uuid']}.json").write_text(json.dumps(res))
        print({k: v for k, v in res.items() if k not in ("projection", "logprobs", "monitor")}, flush=True)


if __name__ == "__main__":
    main()
