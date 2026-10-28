"""Exploratory: how well Qwen-Scope SAEs reconstruct a local model's residual stream.
Reference: docs/INTERP.md, "SAEs".

On a fixed sample (one smoke episode rendered with the model's chat template, about 7K
tokens, and 2,048 tokens of held-out stories and neutral dialogues as raw text), for each
layer:
- FVU: fraction of variance unexplained by the SAE reconstruction, overall and by token
  group (the agent's own tokens, tool outputs, the prompt, prose); also without b_dec, to
  check the decoder convention, and the mean cosine between x and x_hat;
- spliced loss: next-token cross-entropy with the layer's output replaced by the
  reconstruction, against the clean model and against zero ablation; "recovered" is
  (CE_zero - CE_spliced) / (CE_zero - CE_clean). The headline is on tokens the model is
  trained to predict: the agent's own tokens and prose. Tool-output tokens are reported
  apart: the clean model gives many of them almost no probability (it was not trained to
  predict them), so a blurrier reconstruction can "improve" their loss.
The token IDs and roles are built once, with the first model's tokenizer, and saved
(--sample); later runs reuse them, so every model reads exactly the same tokens (the
tokenizer hashes are recorded). Position 0 of each sequence is left out of FVU and never
replaced (its residual norm is an outlier). Qwen-Scope publishes no FVU or loss numbers,
so the comparison is the same measurement on Qwen3.5-27B (the SAEs' own model) at the
same quantization.

Development only: 8-bit and 4-bit weights shift activations, so FVU here is not the bf16
number. Compare models at the same quantization only.

Run (free, local; one model at a time):
    uv run --extra interp python -I explore/interp/sae_transfer.py \\
        ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit \\
        ~/.cache/huggingface/hub/models--Qwen--SAE-Res-Qwen3.5-27B-W80K-L0_50/snapshots/<rev> \\
        --layers 32,42,48 --out logs/interp/sae/qwen3.8-27b-mlx8.json
"""
import argparse, gc, json, sys, time
from pathlib import Path

import numpy as np

from simtest.interp.backend import encode
from simtest.interp.build import load_rows
from simtest.interp.meta import file_sha256, git_commit, now
from simtest.interp.mlx_backend import MLXBackend
from simtest.interp.reader import AGENT_ROLES, R, colab_probe_prompts, load_colab_probe, render
from simtest.interp.sae import fvu, load_qwen_scope

EPISODE = "logs/colab-probe/smoke-or-20261027.jsonl"
TEXTS = "explore/interp/data/texts-qwen3.8-27b.jsonl"
SEQ = 2048


PROSE = -1  # role code for the prose sequence
GROUPS = {"agent": [R[r] for r in AGENT_ROLES], "tool_output": [R["tool_output"]],
          "prompt": [R["system"], R["user"], R["template"]], "prose": [PROSE]}


def sample(tok):
    """The whole agent transcript with its token roles, then SEQ tokens of held-out prose."""
    ep = load_colab_probe(EPISODE, **colab_probe_prompts("explore"))
    rd = render(ep, tok)
    prose = [r["text"] for r in load_rows(TEXTS) if r["split"] == "test"]
    b = encode(tok, "\n\n".join(prose))[:SEQ]
    if len(b) < SEQ:
        sys.exit("sample too short")
    return {"episode": rd.ids.tolist(), "prose": b}, {"episode": rd.role.tolist(), "prose": [PROSE] * len(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"), ap.add_argument("sae_dir")
    ap.add_argument("--layers", required=True), ap.add_argument("--out", required=True)
    ap.add_argument("--fvu-only", action="store_true")
    ap.add_argument("--sample", default="logs/interp/sae/sample.json")
    a = ap.parse_args()
    if Path(a.out).exists():
        sys.exit(f"{a.out} exists; refusing to overwrite")
    layers = [int(x) for x in a.layers.split(",")]
    t0 = time.time()
    b = MLXBackend(a.model)
    print(f"loaded {b.info['model']} ({b.info['quantization']}) in {time.time() - t0:.0f}s", flush=True)
    tok_sha = file_sha256(Path(a.model).expanduser() / "tokenizer.json")
    if Path(a.sample).exists():
        saved = json.loads(Path(a.sample).read_text())
        seqs, roles = saved["ids"], saved["roles"]
        if max(max(v) for v in seqs.values()) >= len(b.tokenizer):
            sys.exit("the saved sample has token IDs outside this model's vocabulary")
    else:
        seqs, roles = sample(b.tokenizer)
        saved = {"tokenizer_sha256": tok_sha, "tokenizer_model": str(a.model), "ids": seqs, "roles": roles,
                 "episode": EPISODE, "texts": TEXTS}
        Path(a.sample).parent.mkdir(parents=True, exist_ok=True)
        Path(a.sample).write_text(json.dumps(saved))
    acts = {name: b.activations(ids, layers) for name, ids in seqs.items()}
    clean = {} if a.fvu_only else {name: b.token_losses(ids) for name, ids in seqs.items()}
    print(f"activations{'' if a.fvu_only else ' and clean losses'} in {time.time() - t0:.0f}s", flush=True)
    roles = {k: np.asarray(v) for k, v in roles.items()}
    names = list(seqs)
    res = {"model": b.info, "layers": {}, "sample": {k: len(v) for k, v in seqs.items()}, "sample_file": a.sample,
           "sample_tokenizer_sha256": saved["tokenizer_sha256"], "model_tokenizer_sha256": tok_sha,
           "tokens_by_group": {g: int(sum(np.isin(roles[n][1:], c).sum() for n in names)) for g, c in GROUPS.items()},
           "episode": EPISODE, "texts": TEXTS, "created": now(), "git_commit": git_commit()}
    pos = lambda codes: {n: np.isin(roles[n][1:], codes) for n in names}  # positions 1.. (FVU) = CE targets
    for L in layers:
        sae = load_qwen_scope(Path(a.sae_dir).expanduser() / f"layer{L}.sae.pt")
        if sae.d_model != b.d_model:
            sys.exit(f"SAE d_model {sae.d_model} != model d_model {b.d_model}")
        x = {n: acts[n][L][1:] for n in names}
        rec = {n: sae.reconstruct(x[n]) for n in names}
        cat = lambda d, m=None: np.concatenate([d[n] if m is None else d[n][m[n]] for n in names])
        X, Rc = cat(x), cat(rec)
        cos = (X * Rc).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Rc, axis=1))
        row = {"sae": sae.info, "fvu": fvu(X, Rc), "fvu_without_b_dec": fvu(X, Rc - sae.b_dec), "cosine_mean": float(cos.mean()),
               "fvu_by_group": {g: fvu(cat(x, m), cat(rec, m)) for g, c in GROUPS.items() if (m := pos(c)) and cat(x, m).size},
               "norm_by_group": {g: float(np.linalg.norm(cat(x, m), axis=1).mean()) for g, c in GROUPS.items()
                                 if (m := pos(c)) and cat(x, m).size},
               "first_token_norm_ratio": float(np.mean([np.linalg.norm(acts[n][L][0]) / np.linalg.norm(x[n], axis=1).mean()
                                                        for n in names]))}
        del rec
        if not a.fvu_only:
            def splice(h, start, sae=sae):
                out = sae.reconstruct(h)
                if start == 0:
                    out[0] = h[0]
                return out

            def zero(h, start):
                out = np.zeros_like(h)
                if start == 0:
                    out[0] = h[0]
                return out

            sp = {n: b.token_losses(seqs[n], replace={L: splice}) for n in names}
            zr = {n: b.token_losses(seqs[n], replace={L: zero}) for n in names}
            ce = {}
            for g, codes in {**GROUPS, "agent+prose": GROUPS["agent"] + GROUPS["prose"]}.items():
                m = pos(codes)
                if not sum(v.sum() for v in m.values()):
                    continue
                c, s_, z = (float(cat(d, m).mean()) for d in (clean, sp, zr))
                ce[g] = {"clean": c, "spliced": s_, "zero": z, "delta": s_ - c, "recovered": (z - s_) / (z - c)}
            row["ce"] = ce
        res["layers"][str(L)] = row
        g = row.get("ce", {}).get("agent+prose")
        print(f"layer {L}: FVU {row['fvu']:.3f} (by group " + ", ".join(f"{k} {v:.3f}" for k, v in row["fvu_by_group"].items())
              + f"; without b_dec {row['fvu_without_b_dec']:.3f}; cos {row['cosine_mean']:.3f})"
              + ("" if not g else "\n  CE on agent+prose tokens: clean {clean:.3f}, spliced {spliced:.3f}, zero {zero:.3f}, "
                 "delta {delta:+.3f}, recovered {recovered:.3f}".format(**g))
              + ("" if not g else "\n  delta by group: " + ", ".join(f"{k} {v['delta']:+.3f}" for k, v in row["ce"].items()))
              + f"  [{time.time() - t0:.0f}s]", flush=True)
        del sae
        gc.collect()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")
    b.close()


if __name__ == "__main__":
    main()
