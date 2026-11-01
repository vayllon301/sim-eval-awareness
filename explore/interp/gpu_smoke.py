"""Exploratory: the N1 GPU smoke test of the interp tools on the reference backend (vLLM
0.32, both models in FP8). Reference: docs/INTERP.md, "GPU smoke test".

Runs on the Colab G4 through explore/interp/gpu_smoke.sh: one step per process, one model at
a time, each writing <out-dir>/<name>/<step>.json (and refusing to overwrite it):

  stock   greedy tokens of the fixed prompts on stock vLLM (vllm-lens off), each prompt
          twice. `--mode eager`: the lens steps' engine settings and model runner (V1).
          `--mode served`: vllm serve's (compiled, CUDA graphs, prefix caching, vLLM's
          default runner), as the episodes ran.
  lens    on vLLM with vllm-lens: (i) the vector file built on the FP8 weights (the texts,
          pairs and layer rule of build_vectors.py); (ii) greedy tokens before the hooks are
          installed, then unsteered, with zero additions and with an alpha-0 spec; (iii) a
          steer at one layer against its delta there, the dose curve with random and
          unrelated controls, short samples; (iv) prompt log-probs. Saves the unsteered
          activations of the dose text for the hf step.
  hf      the agent's FP8 checkpoint on transformers: alpha 0 against model.generate, its
          activations against vLLM's on the same tokens, a steer against its delta.
  report  every check into <out-dir>/checks.json (CPU only): PASS, FAIL, or INCOMPLETE when its
          step failed or was skipped; it exits 1 unless every check passed.

Pass rules, fixed before running. Smoke level: they say the tools work on these weights;
the numbers are not results.
- vectors: at the chosen layer, emotion held-out and leave-one-out accuracy >= 0.5 (chance
  1/6), eval-awareness leave-one-pair-out >= 14 of 16.
- alpha0: for every fixed prompt, the lens engine's greedy tokens before the hooks,
  unsteered after them, with zero additions and with an alpha-0 spec all equal stock vLLM's
  with the same settings (stock-eager).
- steer_applied: alpha 0.1 at one layer moves the mean projection onto that layer's unit
  vector, at that layer, by 0.1 x its mean residual norm, within 5%.
- dose: the target's mean projection rises strictly with the signed strength at the middle
  steered layer.
- hf_alpha0 and hf_steer_applied: the same two checks on the HF backend (agent).
Reported, not pass/fail: stock-served against stock-eager (does the episodes' compiled
engine give the eager engine's tokens?), stock repeats, the later layer's dose curve, the
controls' slopes, HF against vLLM.

Run (on the VM; the runner does this):
    python -I explore/interp/gpu_smoke.py stock /content/models/agent --name agent --mode eager --out-dir OUT
    python -I explore/interp/gpu_smoke.py lens /content/models/agent --name agent --out-dir OUT
    python -I explore/interp/gpu_smoke.py hf /content/models/agent --name agent --out-dir OUT
    python -I explore/interp/gpu_smoke.py report --out-dir OUT
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
# Per model: texts, scanned layers (around two-thirds depth, docs/INTERP.md), steered layers
# and the later read layer (as steer_check.py), vLLM's flags as explore/colab_probe/serve.sh.
PRESETS = {"agent": {"texts": DATA / "texts-qwen3.8-27b.jsonl", "layers": [36, 39, 42, 45, 48], "steer": [39, 42, 45],
                     "read": 48, "quantization": None, "engine": {}},
           "simulator": {"texts": DATA / "texts-qwen3.5-35b-a3b.jsonl", "layers": [22, 24, 27, 30, 32], "steer": [24, 27, 30],
                         "read": 32, "quantization": "fp8", "engine": {"trust_remote_code": True}}}
# vllm serve's settings in serve.sh on a 96 GB card; unset ones keep vLLM's defaults.
SERVED = {"enforce_eager": False, "enable_prefix_caching": None, "max_model_len": 131072, "max_num_seqs": 64,
          "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.47, "language_model_only": True}
PROMPTS = [  # (messages, chat-template options, new tokens)
    ([{"role": "user", "content": "Write three sentences about how your afternoon is going."}], {"enable_thinking": False}, 96),
    ([{"role": "user", "content": "The test suite fails with `AssertionError: expected 3, got 4` in tests/test_pricing.py "
                                  "after my change to the discount rule. What do you check first?"}], {}, 128),
]
TARGET, SECOND, ALPHA = "desperate", "calm", 0.1
ALPHAS = [-0.1, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1]
CONTROL_SEEDS, CONTROL_CONCEPT = [0, 1], "topic:baker-prepares"
TEXT_TOKENS, SAMPLE_TOKENS = 400, 80
RULES = {"vectors_min_accuracy": 0.5, "eval_loo_min_correct": 14, "steer_rel_tol": 0.05}
# Every pass/fail check, per model. A check whose step failed or was skipped is INCOMPLETE (null),
# never left out, and the report exits 1 unless every check passed.
REQUIRED = {"agent": ["vectors", "steer_applied", "dose", "token_losses", "alpha0", "hf_alpha0", "hf_steer_applied"],
            "simulator": ["vectors", "steer_applied", "dose", "token_losses", "alpha0"]}


def out_file(a, step: str) -> Path:
    p = Path(a.out_dir) / a.name / f"{step}.json"
    if p.exists():
        sys.exit(f"{p} exists; refusing to overwrite")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write(p: Path, res: dict) -> None:
    p.write_text(json.dumps(res, indent=1))
    print(f"wrote {p}", flush=True)


def header(b, t0: float) -> dict:
    from simtest.interp.meta import git_commit, now

    return {"created": now(), "git_commit": git_commit(), "argv": sys.argv, "backend": b.info,
            "load_seconds": round(time.time() - t0, 1)}


def prompt_ids(tokenizer) -> list[list[int]]:
    from simtest.interp.reader import chat_ids

    return [chat_ids(tokenizer, m, [], add_generation_prompt=True, **kw) for m, kw, _ in PROMPTS]


def greedy(b, ids, steer=()) -> list[list[int]]:
    return [b.generate(x, max_tokens=n, steer=steer) for x, (_, _, n) in zip(ids, PROMPTS)]


def dose_text(b, texts: Path) -> list[int]:
    """The first held-out neutral dialogue, as steer_check.py."""
    from simtest.interp.backend import encode
    from simtest.interp.build import load_rows

    text = next(r["text"] for r in load_rows(texts) if r["kind"] == "neutral" and r["split"] == "test")
    return encode(b.tokenizer, text)[:TEXT_TOKENS]


def steer_applied(b, vs, ids, layer: int) -> dict:
    """A steer at one layer, read at that layer: the mean projection onto its unit vector
    must move by alpha x the layer's mean residual norm. A steer vllm-lens skipped (it logs
    hook errors instead of raising) would move it by 0."""
    from simtest.interp.steering import SteeringSpec, resolve

    spec = SteeringSpec(vectors="(loaded)", direction=TARGET, layers=(layer,), alpha=ALPHA)
    adds = resolve(spec, vs, fingerprint=b.info["fingerprint"])
    u = vs.get(TARGET, layer).astype(np.float64)
    clean = b.activations(ids, [layer])[layer].astype(np.float64)
    steered = b.activations(ids, [layer], steer=adds)[layer].astype(np.float64)
    want = ALPHA * vs.resid_norm(layer)
    moved = float(((steered - clean) @ u).mean())
    resid = np.linalg.norm((steered - clean) - adds[0].delta, axis=1) / np.linalg.norm(adds[0].delta)
    return {"layer": layer, "alpha": ALPHA, "expected": want, "moved": moved, "rel_err": abs(moved - want) / want,
            "delta_rel_err_mean": float(resid.mean()), "delta_rel_err_max": float(resid.max())}


# ---------------------------------------------------------------- steps


def stock(a, p):
    os.environ["VLLM_LENS_DISABLE"] = "1"  # before vLLM loads its plugins
    # vllm-lens moves vLLM to its V1 model runner (V2 is vLLM's default for dense models), so the
    # eager baseline runs there too; served keeps vLLM's default, as serve.sh.
    if a.mode == "eager":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    else:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    from simtest.interp.vllm_backend import VLLMBackend

    out = out_file(a, f"stock-{a.mode}")
    t0 = time.time()
    b = VLLMBackend(a.model, quantization=p["quantization"], engine={**(SERVED if a.mode == "served" else {}), **p["engine"]})
    res = header(b, t0)
    ids = prompt_ids(b.tokenizer)
    first, again = greedy(b, ids), greedy(b, ids)
    res.update(mode=a.mode, prompt_ids=ids, outputs=first, repeat=again, repeat_equal=first == again,
               texts=[b.tokenizer.decode(x) for x in first], seconds=round(time.time() - t0, 1))
    print(f"stock {a.mode}: repeat equal {first == again} [{time.time() - t0:.0f}s]", flush=True)
    write(out, res)


def lens(a, p):
    os.environ.pop("VLLM_LENS_DISABLE", None)
    os.environ.setdefault("VLLM_LENS_STRICT_COMPATIBILITY", "1")  # vllm-lens 1.3.0 is validated on vLLM 0.30 only
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"  # what vllm-lens sets; explicit, so it is recorded
    from simtest.interp.backend import Add
    from simtest.interp.build import CONCEPTS, build_all, load_rows
    from simtest.interp.meta import file_sha256
    from simtest.interp.steering import SteeringSpec, controls, dose_curve, resolve
    from simtest.interp.vllm_backend import VLLMBackend

    out = out_file(a, "lens")
    vec_path = out.parent / "vectors.safetensors"
    if vec_path.exists():
        sys.exit(f"{vec_path} exists; refusing to overwrite")
    t0 = time.time()
    b = VLLMBackend(a.model, quantization=p["quantization"], engine=p["engine"])
    if b.info["lens"] != "enabled":
        sys.exit("vllm-lens is disabled (VLLM_LENS_DISABLE)")
    res = header(b, t0)
    print(f"loaded {b.info['model']} ({b.info['quantization']}) in {res['load_seconds']}s", flush=True)
    ids = prompt_ids(b.tokenizer)
    res["prompt_ids"] = ids
    res["identity"] = {"before_hooks": greedy(b, ids)}  # vllm-lens installs its hooks on the first request needing them
    stamp = lambda what: print(f"{what} [{time.time() - t0:.0f}s]", flush=True)
    stamp("greedy before hooks")

    # (i) vectors
    texts = Path(p["texts"])
    rows = load_rows(texts)
    meta_path = Path(str(texts).removesuffix(".jsonl") + ".meta.json")
    gen_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pairs_path, train_path = DATA / "eval_awareness_pairs.json", DATA / "train_awareness_pairs.json"
    tick = lambda i, n: (i % 50 == 0 or i == n) and stamp(f"  {i}/{n} texts")
    vs, rep = build_all(b, rows, p["layers"], texts_path=str(texts), pairs=json.loads(pairs_path.read_text()),
                        pairs_path=str(pairs_path), train_pairs=json.loads(train_path.read_text()),
                        train_pairs_path=str(train_path), concept_topics=CONCEPTS, progress=tick,
                        extra_datasets={"texts_generation": {"generator": gen_meta.get("generator"), "note": gen_meta.get("note"),
                                                             "providers": sorted({c.get("provider") or "?" for c in gen_meta.get("calls", [])})}})
    vs.save(vec_path)
    (out.parent / "vectors.safetensors.report.json").write_text(json.dumps(rep, indent=1, sort_keys=True))
    L = rep["chosen_layer"]
    em, ea = rep["layers"][str(L)], rep["eval_awareness"][str(L)]
    res["vectors"] = {"path": str(vec_path), "sha256": file_sha256(vec_path), "chosen_layer": L,
                      "heldout": em["heldout"]["accuracy"], "loo": em["loo"]["accuracy"],
                      "eval_loo_correct": ea["loo_pairs"]["correct"], "eval_loo_n": ea["loo_pairs"]["n"],
                      "by_layer": {l: {"heldout": r["heldout"]["accuracy"], "loo": r["loo"]["accuracy"],
                                       "eval_loo": rep["eval_awareness"][l]["loo_pairs"]["correct"]}
                                   for l, r in rep["layers"].items()},
                      "resid_norm_mean": vs.meta["resid_norm_mean"]}
    stamp(f"vectors: chosen layer {L}, held-out {em['heldout']['accuracy']:.2f}, LOO {em['loo']['accuracy']:.2f}, "
          f"eval LOO {ea['loo_pairs']['correct']}/{ea['loo_pairs']['n']}")

    # (ii) alpha 0, with the hooks installed
    fp = b.info["fingerprint"]
    target = SteeringSpec(vectors=str(vec_path.resolve()), vectors_sha256=res["vectors"]["sha256"], direction=TARGET,
                          layers=tuple(p["steer"]), alpha=ALPHA)
    zero = [Add(l, np.zeros(b.d_model, np.float32)) for l in p["steer"]]
    res["identity"].update(unsteered=greedy(b, ids), zero_add=greedy(b, ids, zero),
                           alpha0_spec=greedy(b, ids, resolve(dose_curve(target, [0.0])[0], vs, fingerprint=fp)))
    stamp("greedy after hooks")

    # (iii) steering
    text = dose_text(b, texts)
    mid = p["steer"][len(p["steer"]) // 2]
    res["steer_applied"] = steer_applied(b, vs, text, mid)
    stamp(f"steer at layer {mid}: moved {res['steer_applied']['moved']:.2f}, expected {res['steer_applied']['expected']:.2f}")
    read_layers = sorted({mid, p["read"]})
    u = {l: vs.get(TARGET, l).astype(np.float64) for l in read_layers}
    specs = {"target": target, **{c.label: c for c in controls(target, random_seeds=CONTROL_SEEDS,
                                                                 concepts=[CONTROL_CONCEPT], signs=(1,))}}
    curves = {}
    (out.parent / "specs").mkdir(exist_ok=True)
    for name, base in specs.items():
        curve = {str(l): [] for l in read_layers}
        for s in dose_curve(base, ALPHAS):
            acts = b.activations(text, read_layers, steer=resolve(s, vs, fingerprint=fp))
            for l in read_layers:
                curve[str(l)].append(float((acts[l][1:] @ u[l]).mean()))
        curves[name] = curve
        base.save(out.parent / "specs" / f"{base.label.replace(':', '_').replace(',', '-')}.yaml")
    res["dose_curve"] = {"alphas": ALPHAS, "read_layers": read_layers, "steered_layer_read": mid, "projection_onto": TARGET,
                         "text_tokens": len(text), "curves": curves,
                         "monotonic": {n: {l: bool(np.all(np.diff(c[l]) > 0)) for l in c} for n, c in curves.items()},
                         "slope_per_alpha": {n: {l: float(np.polyfit(ALPHAS, c[l], 1)[0]) for l in c} for n, c in curves.items()}}
    stamp("dose curve")
    samples = {"unsteered": b.tokenizer.decode(b.generate(ids[0], max_tokens=SAMPLE_TOKENS))}
    for direction in (TARGET, SECOND):
        for alpha in (-ALPHA, ALPHA):
            s = dose_curve(SteeringSpec(**{**target.to_dict(), "direction": direction, "layers": tuple(p["steer"])}), [alpha])[0]
            samples[s.label] = b.tokenizer.decode(b.generate(ids[0], max_tokens=SAMPLE_TOKENS, steer=resolve(s, vs, fingerprint=fp)))
    res["samples"] = {"prompt": PROMPTS[0][0][0]["content"], "texts": samples}
    stamp("samples")

    # (iv) prompt log-probs, and the unsteered activations for the hf step
    losses = b.token_losses(text)
    res["token_losses"] = {"n": len(losses), "finite": bool(np.isfinite(losses).all()), "mean_nats": float(losses.mean())}
    acts = b.activations(text, read_layers)
    np.savez_compressed(out.parent / "lens-acts.npz", ids=np.asarray(text, np.int64), layers=np.asarray(read_layers),
                        **{f"acts@{l}": acts[l] for l in read_layers})
    res["seconds"] = round(time.time() - t0, 1)
    write(out, res)


def hf(a, p):
    import torch

    from simtest.interp.backend import Add
    from simtest.interp.hf_backend import HFBackend
    from simtest.interp.steering import SteeringSpec, dose_curve, resolve
    from simtest.interp.vectors import VectorSet

    out = out_file(a, "hf")
    d = out.parent
    lens_res = json.loads((d / "lens.json").read_text())
    vs = VectorSet.load(d / "vectors.safetensors")
    t0 = time.time()
    b = HFBackend(a.model)
    res = header(b, t0)
    print(f"loaded {b.info['model']} ({b.info['quantization']}, {b.info['attn_implementation']}) in {res['load_seconds']}s",
          flush=True)
    ids = prompt_ids(b.tokenizer)
    res["prompt_ids_equal_lens"] = ids == lens_res["prompt_ids"]
    stock_out = []
    for x, (_, _, n) in zip(ids, PROMPTS):
        g = b.model.generate(torch.tensor([x], device=b.device), attention_mask=torch.ones((1, len(x)), dtype=torch.long, device=b.device),
                             max_new_tokens=n, do_sample=False, eos_token_id=sorted(b.eos_token_ids))
        stock_out.append(g[0, len(x):].tolist())
    same_weights = vs.meta["model"]["fingerprint"] == b.info["fingerprint"]
    zero = [Add(l, np.zeros(b.d_model, np.float32)) for l in p["steer"]]
    res["identity"] = {"stock": stock_out, "ours": greedy(b, ids), "zero_add": greedy(b, ids, zero)}
    res["vectors_fingerprint_equal"] = same_weights  # the same FP8 files as vLLM's
    if same_weights:
        spec0 = SteeringSpec(vectors="(loaded)", direction=TARGET, layers=tuple(p["steer"]), alpha=0.0)
        res["identity"]["alpha0_spec"] = greedy(b, ids, resolve(spec0, vs, fingerprint=b.info["fingerprint"]))
    div = lambda x, y: next((i for i, (s, t) in enumerate(zip(x, y)) if s != t), min(len(x), len(y)))
    res["vs_vllm_greedy_first_difference"] = [div(x, y) for x, y in zip(stock_out, lens_res["identity"]["unsteered"])]
    print(f"greedy [{time.time() - t0:.0f}s]", flush=True)

    z = np.load(d / "lens-acts.npz")
    text = z["ids"].tolist()
    layers = z["layers"].tolist()
    mine = b.activations(text, layers)
    agree = {}
    for l in layers:
        v, h = z[f"acts@{l}"][1:].astype(np.float64), mine[l][1:].astype(np.float64)
        cos = (v * h).sum(1) / (np.linalg.norm(v, axis=1) * np.linalg.norm(h, axis=1))
        u = vs.get(TARGET, l).astype(np.float64)
        agree[str(l)] = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
                         "rel_err": float(np.linalg.norm(h - v) / np.linalg.norm(v)),
                         "proj_corr": float(np.corrcoef(v @ u, h @ u)[0, 1]),
                         "proj_mean_vllm": float((v @ u).mean()), "proj_mean_hf": float((h @ u).mean())}
    res["activations_vs_vllm"] = agree
    if same_weights:
        res["steer_applied"] = steer_applied(b, vs, text, p["steer"][len(p["steer"]) // 2])
    res["seconds"] = round(time.time() - t0, 1)
    write(out, res)


def report(a):
    D = Path(a.out_dir)
    load = lambda name, step: json.loads((D / name / f"{step}.json").read_text()) if (D / name / f"{step}.json").exists() else None
    checks, notes = {}, {}
    for name in PRESETS:
        st, sv, ln, h = load(name, "stock-eager"), load(name, "stock-served"), load(name, "lens"), load(name, "hf")
        c, n = {}, {}
        if ln:
            v = ln["vectors"]
            c["vectors"] = (v["heldout"] >= RULES["vectors_min_accuracy"] and v["loo"] >= RULES["vectors_min_accuracy"]
                            and v["eval_loo_correct"] >= RULES["eval_loo_min_correct"])
            n["vectors"] = {k: v[k] for k in ("chosen_layer", "heldout", "loo", "eval_loo_correct", "eval_loo_n")}
            c["steer_applied"] = ln["steer_applied"]["rel_err"] <= RULES["steer_rel_tol"]
            n["steer_applied"] = ln["steer_applied"]
            mid = str(ln["dose_curve"]["steered_layer_read"])
            c["dose"] = ln["dose_curve"]["monotonic"]["target"][mid]
            n["dose"] = {"monotonic": ln["dose_curve"]["monotonic"], "slope_per_alpha": ln["dose_curve"]["slope_per_alpha"]}
            c["token_losses"] = ln["token_losses"]["finite"]
            n["token_losses"] = ln["token_losses"]
            idn = ln["identity"]
            lens_same = all(idn[k] == idn["before_hooks"] for k in ("unsteered", "zero_add", "alpha0_spec"))
            n["alpha0"] = {"lens_runs_agree": lens_same}
            if st:
                c["alpha0"] = lens_same and st["prompt_ids"] == ln["prompt_ids"] and idn["before_hooks"] == st["outputs"]
                n["alpha0"]["equals_stock_eager"] = idn["before_hooks"] == st["outputs"]
        if st:
            n["stock_eager_repeat_equal"] = st["repeat_equal"]
        if st and sv:
            n["served_equals_eager"] = sv["outputs"] == st["outputs"]
            n["served_repeat_equal"] = sv["repeat_equal"]
        if h:
            idn = h["identity"]
            # without the alpha-0 spec (vectors from other weights) the check can't pass
            c["hf_alpha0"] = "alpha0_spec" in idn and all(idn[k] == idn["stock"] for k in ("ours", "zero_add", "alpha0_spec"))
            if "steer_applied" in h:
                c["hf_steer_applied"] = h["steer_applied"]["rel_err"] <= RULES["steer_rel_tol"]
            n["hf"] = {"activations_vs_vllm": h["activations_vs_vllm"], "vs_vllm_greedy_first_difference":
                       h["vs_vllm_greedy_first_difference"], "prompt_ids_equal_lens": h["prompt_ids_equal_lens"],
                       "vectors_fingerprint_equal": h["vectors_fingerprint_equal"], "steer_applied": h.get("steer_applied")}
        n["steps_run"] = [s for s in ("stock-eager", "lens", "stock-served", "hf") if load(name, s)]
        checks[name] = {k: (bool(c[k]) if k in c else None) for k in REQUIRED[name]}
        notes[name] = n
    incomplete = {name: [k for k, ok in c.items() if ok is None] for name, c in checks.items()}
    failed = {name: [k for k, ok in c.items() if ok is False] for name, c in checks.items()}
    out = {"rules": RULES, "checks": checks, "incomplete": incomplete, "failed": failed, "notes": notes}
    (D / "checks.json").write_text(json.dumps(out, indent=1))
    word = {True: "PASS", False: "FAIL", None: "INCOMPLETE"}
    for name, c in checks.items():
        print(f"{name}: " + ", ".join(f"{k} {word[ok]}" for k, ok in c.items()))
        for k in ("served_equals_eager", "stock_eager_repeat_equal", "served_repeat_equal"):
            if k in notes[name]:
                print(f"  {k}: {notes[name][k]}")
    print(f"wrote {D / 'checks.json'}")
    if any(incomplete.values()) or any(failed.values()):
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="step", required=True)
    for step in ("stock", "lens", "hf"):
        s = sub.add_parser(step)
        s.add_argument("model"), s.add_argument("--name", choices=sorted(PRESETS), required=True)
        s.add_argument("--out-dir", required=True)
        if step == "stock":
            s.add_argument("--mode", choices=["eager", "served"], required=True)
    sub.add_parser("report").add_argument("--out-dir", required=True)
    a = ap.parse_args()
    if a.step == "report":
        return report(a)
    if a.step == "hf" and a.name != "agent":
        sys.exit("the hf step is for the agent: transformers can't reproduce vLLM's online FP8 for the simulator")
    p = PRESETS[a.name]
    if not set(p["steer"]) | {p["read"]} <= set(p["layers"]):
        sys.exit(f"{a.name}: the steered and read layers must be scanned layers (vectors exist only there)")
    {"stock": stock, "lens": lens, "hf": hf}[a.step](a, p)


if __name__ == "__main__":
    main()
