"""Exploratory: read a run's exported episodes from the inside (docs/INTERP.md, "Reading a run"). The
logic is simtest.interp.readrun; this is its command line for the GPU venv, which has no Inspect and
so no `simtest` command. `simtest interp export` makes the episodes on the laptop first.

    python -I explore/interp/read_run.py MODEL_DIR --engine vllm --episodes EPISODES --out OUT \\
        --vectors VECTORS --parts proj [--layers 36,42] [--sae-dir DIR --sae-layers 42] [--budget-min 20]
    python -I explore/interp/read_run.py MODEL_DIR --engine hf --episodes EPISODES --out OUT \\
        --parts logit --phrases explore/interp/data/logit_monitor_phrases.json

--layers defaults to the vector file's chosen layer. An episode without the server's token IDs (`simtest
interp server-ids`) is refused unless --no-server-ids is given, which each reading records: for reads
with no server behind them, such as the laptop's. Exit: 0 when every episode is read, 3 when some are
left (run it again: it resumes), 1 after an error or a failed episode.
"""
import argparse, json, sys, time
from pathlib import Path


def main():
    from simtest.interp.readrun import PARTS, ReadError, make_backend, read_run
    from simtest.interp.sae import load_qwen_scope
    from simtest.interp.vectors import VectorSet

    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--engine", required=True, choices=["vllm", "hf", "mlx"])
    ap.add_argument("--quantization", help="vLLM's flag, for the simulator (the agent's checkpoint is FP8 already)")
    ap.add_argument("--episodes", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="proj", help=f"comma-separated: {', '.join(PARTS)}")
    ap.add_argument("--vectors"), ap.add_argument("--layers", help="comma-separated (default: the file's chosen layer)")
    ap.add_argument("--names", help="comma-separated vector names (default: every vector in the file)")
    ap.add_argument("--sae-dir", help="a Qwen-Scope snapshot: adds top-k latents at --sae-layers")
    ap.add_argument("--sae-layers", default=""), ap.add_argument("--sae-device", help="torch device for the encoder (cuda)")
    ap.add_argument("--phrases"), ap.add_argument("--logit-turns", help="comma-separated (default: every turn)")
    ap.add_argument("--allow-slow-logit", action="store_true", help="let the logit part run without continuation_scan")
    ap.add_argument("--allow-other-model", action="store_true", help="vectors from other weights: a transfer test only")
    ap.add_argument("--no-server-ids", action="store_true", help="read episodes that have no server IDs: each reading says so")
    ap.add_argument("--budget-min", type=float), ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    parts = a.parts.split(",")
    vs = VectorSet.load(a.vectors) if a.vectors else None
    layers = [int(x) for x in a.layers.split(",")] if a.layers else ([vs.meta["chosen_layer"]] if vs else [])
    saes = {}
    for L in (int(x) for x in a.sae_layers.split(",") if x):
        saes[L] = load_qwen_scope(Path(a.sae_dir).expanduser() / f"layer{L}.sae.pt")
        saes[L].device = a.sae_device
    phrases = json.loads(Path(a.phrases).read_text()) if a.phrases else None
    t = time.time()
    b = make_backend(a.engine, a.model, quantization=a.quantization)
    print(f"{a.engine} loaded in {time.time() - t:.0f}s", flush=True)
    try:
        st = read_run(b, a.episodes, a.out, parts=parts, vectors=vs, vectors_path=a.vectors, layers=layers,
                      names=a.names.split(",") if a.names else None, saes=saes, phrases=phrases, phrases_path=a.phrases,
                      phrase_turns=[int(x) for x in a.logit_turns.split(",")] if a.logit_turns else None,
                      allow_other_model=a.allow_other_model, budget_s=a.budget_min * 60 if a.budget_min else None,
                      limit=a.limit, require_scan=not a.allow_slow_logit, no_server_ids=a.no_server_ids)
    except ReadError as exc:
        sys.exit(str(exc))
    print(json.dumps(st, indent=1, sort_keys=True))
    sys.exit(1 if st["failed"] else 0 if st["complete"] else 3)


if __name__ == "__main__":
    main()
