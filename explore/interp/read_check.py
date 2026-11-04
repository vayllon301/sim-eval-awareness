"""Exploratory: the first GPU run's check of the reading paths (docs/INTERP.md, "What the first GPU run
measures"). One step per process and engine, since vLLM and transformers can't share the GPU:

    python -I explore/interp/read_check.py run MODEL --engine vllm --episodes E --vectors V --phrases P --out OUT
    python -I explore/interp/read_check.py run MODEL --engine hf   --episodes E --vectors V --phrases P --out OUT
    python -I explore/interp/read_check.py run MODEL --engine hf --chunk 2048 --label hf-2048 ...
    python -I explore/interp/read_check.py compare --out OUT

`run` takes the longest episode and the shortest (OOM first, then the short one) and writes
OUT/check-<label>-<uuid>.json. On vLLM the phrases go through prompt log-probs at three turns: slow and
meant for this check only. `compare` pairs every check with vllm's on the same episode and writes
OUT/read-check.json. Development numbers: none of this is evidence. Pass rules: docs/INTERP.md.
"""
import argparse, json, sys, time
from pathlib import Path


def main():
    from simtest.interp.reader import load_episode
    from simtest.interp.readrun import ReadError, compare_paths, make_backend, path_check
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
        rows = []
        for ref in (x for x in checks if x["label"] == "vllm"):
            for other in (x for x in checks if x["label"] != "vllm" and x["uuid"] == ref["uuid"]):
                rows.append(compare_paths(ref, other))
        for ref in (x for x in checks if x["label"] == "hf"):
            for other in (x for x in checks if x["label"].startswith("hf-") and x["uuid"] == ref["uuid"]):
                rows.append(compare_paths(ref, other))
        (out / "read-check.json").write_text(json.dumps(rows, indent=1))
        for row in rows:
            print({k: v for k, v in row.items() if k != "monitor"})
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
