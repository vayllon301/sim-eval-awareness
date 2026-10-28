"""Exploratory: the steering prototype's checks on a local MLX model.
Reference: docs/INTERP.md, "Steering spec" and "Checks".

1. Strength 0: greedy output with zero additions at the steering layers, and with a spec
   at alpha 0, is token-identical to the unhooked model (mlx-lm's own generator).
2. Dose curve: teacher-forced on a fixed text, the mean projection onto the target
   vector, at a steered layer and at a later layer, moves monotonically with the signed
   strength. The same strengths with random and unrelated-concept controls (same
   layers, same norm) show how much of the move is specific to the target.
3. A short greedy sample at -alpha, 0 and +alpha for the target and one more vector.

Every spec used is written to <out-dir>/specs/. Development only on local weights.

Run (free, local):
    uv run --extra interp python -I explore/interp/steer_check.py \\
        ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit logs/interp/vectors/agent-mlx8.safetensors \\
        --layers 39,42,45 --read-layer 48 --out-dir logs/interp/steer/agent-mlx8
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

from simtest.interp.backend import Add
from simtest.interp.build import load_rows
from simtest.interp.meta import file_sha256, git_commit, now
from simtest.interp.mlx_backend import MLXBackend
from simtest.interp.reader import chat_ids
from simtest.interp.steering import SteeringSpec, controls, dose_curve, resolve
from simtest.interp.vectors import VectorSet

ALPHAS = [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]
PROMPT = "Write three sentences about how your afternoon is going."
N_NEW = 48


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"), ap.add_argument("vectors")
    ap.add_argument("--layers", required=True), ap.add_argument("--read-layer", type=int, required=True)
    ap.add_argument("--target", default="desperate"), ap.add_argument("--second", default="calm")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--texts", default="explore/interp/data/texts-qwen3.8-27b.jsonl")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    if (out / "checks.json").exists():
        sys.exit(f"{out}/checks.json exists; refusing to overwrite")
    layers = [int(x) for x in a.layers.split(",")]
    vs = VectorSet.load(a.vectors)
    sha = file_sha256(a.vectors)
    target = SteeringSpec(vectors=str(Path(a.vectors).resolve()), vectors_sha256=sha, direction=a.target,
                          layers=tuple(layers), alpha=a.alpha)
    t0 = time.time()
    b = MLXBackend(a.model)
    if vs.meta["model"]["fingerprint"] != b.info["fingerprint"]:
        sys.exit("the vectors were built on another model")
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    res = {"model": b.info, "vectors": a.vectors, "vectors_sha256": sha, "target": target.to_dict(), "created": now(),
           "git_commit": git_commit()}

    # 1. strength 0
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    prompt = chat_ids(b.tokenizer, [{"role": "user", "content": PROMPT}], [], add_generation_prompt=True,
                      enable_thinking=False)
    stock = [r.token for r in stream_generate(b.model, b.tokenizer, prompt, max_tokens=N_NEW, sampler=make_sampler(temp=0.0))]
    ours = b.generate(prompt, max_tokens=N_NEW)
    zero = b.generate(prompt, max_tokens=N_NEW, steer=[Add(L, np.zeros(b.d_model, np.float32)) for L in layers])
    spec0 = b.generate(prompt, max_tokens=N_NEW, steer=resolve(dose_curve(target, [0.0])[0], vs))
    stock = [t for t in stock if t not in b.eos_token_ids] + [t for t in stock if t in b.eos_token_ids][:1]
    trim = lambda s: s[: len(stock)]
    res["strength_zero"] = {"n_tokens": len(stock), "ours_equals_stock": trim(ours) == stock,
                            "zero_add_equals_stock": trim(zero) == stock, "alpha0_spec_equals_stock": trim(spec0) == stock,
                            "text": b.tokenizer.decode(stock)}
    print("strength 0:", {k: v for k, v in res["strength_zero"].items() if k != "text"}, f"[{time.time() - t0:.0f}s]", flush=True)

    # 2. dose curve and controls, teacher-forced on a held-out neutral dialogue
    text = next(r["text"] for r in load_rows(a.texts) if r["kind"] == "neutral" and r["split"] == "test")
    ids = b.tokenizer(text, add_special_tokens=False)["input_ids"][:400]
    read_layers = sorted({layers[len(layers) // 2], a.read_layer})
    u = {L: vs.get(a.target, L).astype(np.float64) for L in read_layers}
    specs = {"target": target, **{c.label: c for c in controls(target, random_seeds=[0, 1], concepts=["topic:baker-prepares"],
                                                                 signs=(1,))}}
    curves = {}
    (out / "specs").mkdir(parents=True, exist_ok=True)
    for name, base in specs.items():
        curve = {str(L): [] for L in read_layers}
        for s in dose_curve(base, ALPHAS):
            acts = b.activations(ids, read_layers, steer=resolve(s, vs))
            for L in read_layers:
                curve[str(L)].append(float((acts[L][1:] @ u[L]).mean()))
        curves[name] = curve
        base.save(out / "specs" / f"{base.label.replace(':', '_').replace(',', '-')}.yaml")
    res["dose_curve"] = {"alphas": ALPHAS, "read_layers": read_layers, "projection_onto": a.target, "text_tokens": len(ids),
                         "curves": curves,
                         "monotonic": {n: {L: bool(np.all(np.diff(c[L]) > 0)) for L in c} for n, c in curves.items()},
                         "slope_per_alpha": {n: {L: float(np.polyfit(ALPHAS, c[L], 1)[0]) for L in c} for n, c in curves.items()}}
    print("dose curve, slope of the mean projection per unit alpha:", flush=True)
    for n, sl in res["dose_curve"]["slope_per_alpha"].items():
        print(f"  {n:>34}: " + ", ".join(f"layer {L} {v:+9.1f} (monotonic {res['dose_curve']['monotonic'][n][L]})"
                                        for L, v in sl.items()))
    print(f"[{time.time() - t0:.0f}s]", flush=True)

    # 3. short samples
    samples = {}
    for direction in (a.target, a.second):
        for alpha in (-a.alpha, a.alpha):
            s = dose_curve(SteeringSpec(**{**target.to_dict(), "direction": direction, "layers": tuple(layers)}), [alpha])[0]
            samples[s.label] = b.tokenizer.decode(b.generate(prompt, max_tokens=N_NEW + 32, steer=resolve(s, vs)))
    samples["unsteered"] = b.tokenizer.decode(b.generate(prompt, max_tokens=N_NEW + 32))
    res["samples"] = {"prompt": PROMPT, "enable_thinking": False, "texts": samples}
    for k, v in samples.items():
        print(f"\n[{k}] {v!r}")
    (out / "checks.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out}/checks.json [{time.time() - t0:.0f}s]")
    b.close()


if __name__ == "__main__":
    main()
