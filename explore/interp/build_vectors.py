"""Exploratory: build a vector file on a local MLX model (emotion vectors, unrelated-concept
controls, the evaluation-awareness vector) at several layers, with validation.
Reference: docs/INTERP.md. The logic is simtest.interp.build; this is its command line.

Development only: local 8-bit/4-bit weights are never evidence (one backend per
experiment). Layer choice, fixed before running: the highest held-out emotion accuracy,
ties to the layer closest to two-thirds depth (simtest.interp.build).

Run (free, local; one model at a time, LM Studio unloaded):
    uv run --extra interp python -I explore/interp/build_vectors.py \\
        ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit \\
        explore/interp/data/texts-qwen3.8-27b.jsonl logs/interp/vectors/agent-mlx8.safetensors \\
        --layers 36,39,42,45,48
"""
import argparse, json, sys, time
from pathlib import Path

from simtest.interp.build import build_all, load_rows, slug
from simtest.interp.mlx_backend import MLXBackend

CONCEPTS = ["A baker prepares for the busiest morning of the year", "A family drives across the country to a new home",
            "A neighbor starts a renovation project"]  # training premises used as unrelated-concept controls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"), ap.add_argument("texts"), ap.add_argument("out")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--pairs", default="explore/interp/data/eval_awareness_pairs.json")
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"{a.out} exists; refusing to overwrite")
    layers = [int(x) for x in a.layers.split(",")]
    t = time.time()
    b = MLXBackend(a.model)
    print(f"loaded {b.info['model']} ({b.info['quantization']}, {b.n_layers} layers, d {b.d_model}) in {time.time() - t:.0f}s",
          flush=True)
    rows = load_rows(a.texts)
    meta_path = Path(a.texts.removesuffix(".jsonl") + ".meta.json")
    gen_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    t = time.time()
    tick = lambda i, n: (i % 50 == 0 or i == n) and print(f"  {i}/{n} texts [{time.time() - t:.0f}s]", flush=True)
    vs, rep = build_all(b, rows, layers, texts_path=a.texts, pairs=json.loads(Path(a.pairs).read_text()),
                        pairs_path=a.pairs, concept_topics=CONCEPTS, progress=tick,
                        extra_datasets={"texts_generation": {"generator": gen_meta.get("generator"), "note": gen_meta.get("note"),
                                                             "providers": sorted({c.get("provider") or "?" for c in gen_meta.get("calls", [])})}})
    vs.save(a.out)
    Path(a.out + ".report.json").write_text(json.dumps(rep, indent=1, sort_keys=True))
    b.close()
    print(f"built in {time.time() - t:.0f}s; tokens per text {rep['n_tokens']}", flush=True)
    print(f"\n{'layer':>5} {'held-out':>9} {'LOO':>9} {'PCs':>4} {'no PCs':>7} {'PCs/means':>9} | "
          f"{'EA LOO':>7} {'EA held':>7} {'EA gen AUROC':>12} {'gen pairs':>9}  resid norm")
    for L in layers:
        r, e = rep["layers"][str(L)], rep.get("eval_awareness", {}).get(str(L), {})
        h, lo = r["heldout"], r["loo"]
        print(f"{L:>5} {h['correct']:>4}/{h['n']:<4} {lo['correct']:>4}/{lo['n']:<4} {r['n_neutral_pcs']:>4} "
              f"{r['variants']['no_pcs']:>7.2f} {r['variants']['pcs_on_neutral_means']:>9.2f} | "
              f"{e.get('loo_pairs', {}).get('correct', '-'):>4}/{e.get('loo_pairs', {}).get('n', '-'):<2} "
              f"{e.get('heldout_pairs', {}).get('accuracy', float('nan')):>7.2f} "
              f"{e.get('generalisation', {}).get('auroc', float('nan')):>12.2f} "
              f"{e.get('generalisation', {}).get('pair_accuracy', float('nan')):>9.2f}  {vs.resid_norm(L):.1f}")
    L = rep["chosen_layer"]
    print(f"\nchosen layer {L}; held-out confusion there (rows: true, columns: predicted):")
    conf = rep["layers"][str(L)]["heldout"]["confusion"]
    print(" " * 11 + "".join(f"{p[:9]:>10}" for p in conf))
    for t_, row in conf.items():
        print(f"{t_[:10]:>10} " + "".join(f"{row[p]:>10}" for p in conf))
    print(f"\nwrote {a.out} ({len(vs.names)} vectors x {len(layers)} layers: {', '.join(vs.names)})")


if __name__ == "__main__":
    main()
