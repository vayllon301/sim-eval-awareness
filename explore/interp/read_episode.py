"""Exploratory: read colab-probe episodes from the inside (post hoc, one forward pass each)
and print the agent's own readings next to the readings on tool outputs.
Reference: docs/INTERP.md, "Reader". The logic is simtest.interp.reader.

Writes <out-dir>/<episode>.npz (token IDs, roles, turns, per-token projections, each
turn's prompt end) and <episode>.summary.json. Development only on local weights.

--logit-phrases adds the logit monitor (simtest.interp.monitors): one forward pass over
the prefix per turn and phrase on MLX, so slow on long episodes; --logit-turns limits it
to some turns.

Run (free, local; the vectors must come from the same model):
    uv run --extra interp python -I explore/interp/read_episode.py \\
        ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit logs/interp/vectors/agent-mlx8.safetensors \\
        --layers 42 --out-dir logs/interp/readings logs/colab-probe/*.jsonl \\
        --logit-phrases explore/interp/data/logit_monitor_phrases.json --logit-turns 0
"""
import argparse, json, sys, time
from pathlib import Path

from simtest.interp.mlx_backend import MLXBackend
from simtest.interp.reader import ROLES, colab_probe_prompts, load_colab_probe, read
from simtest.interp.vectors import VectorSet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"), ap.add_argument("vectors"), ap.add_argument("episodes", nargs="+")
    ap.add_argument("--layers", required=True), ap.add_argument("--out-dir", required=True)
    ap.add_argument("--logit-phrases", help="a phrase file (explore/interp/data/logit_monitor_phrases.json)")
    ap.add_argument("--logit-turns", help="comma-separated agent turns for the logit monitor (default: all)")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    phrases = json.loads(Path(a.logit_phrases).read_text()) if a.logit_phrases else None
    phrase_turns = [int(x) for x in a.logit_turns.split(",")] if a.logit_turns else None
    vs = VectorSet.load(a.vectors)
    prompts = colab_probe_prompts("explore")
    t = time.time()
    b = MLXBackend(a.model)
    print(f"loaded in {time.time() - t:.0f}s", flush=True)
    for path in a.episodes:
        out = Path(a.out_dir) / (Path(path).stem + ".npz")
        if out.exists():
            sys.exit(f"{out} exists; refusing to overwrite")
        ep = load_colab_probe(path, **prompts)
        t = time.time()
        r = read(b, ep, vs, layers, vectors_path=a.vectors, phrases=phrases, phrases_path=a.logit_phrases,
                 phrase_turns=phrase_turns)
        r.save(out)
        s = json.loads(out.with_name(out.stem + ".summary.json").read_text())
        print(f"\n{path}: {s['n_tokens']} tokens in {time.time() - t:.1f}s; outcome {ep.meta['outcome']}; "
              f"unread outputs dropped {ep.meta['unread_outputs_dropped']}")
        print("  tokens by role: " + ", ".join(f"{k} {v}" for k, v in s["tokens_by_role"].items() if v))
        print("  prompt length per turn, rendered here vs reported by the server: "
              + ", ".join(f"{c['rendered']}/{c['recorded']}" for c in r.meta["prompt_checks"])
              + (f"; not a prefix of the episode (no prompt-end reading): turns {r.meta['prompt_end_not_prefix']}"
                 if r.meta["prompt_end_not_prefix"] else ""))
        for L in r.layers:
            print(f"  layer {L}, mean projection minus the neutral offset (agent's own tokens | tool outputs | system+user):")
            for name in r.names:
                g = s["by_group"][f"{name}@{L}"]
                cell = lambda k: f"{g[k]['mean_centred']:+7.2f}" if k in g and g[k]["mean_centred"] is not None else "      -"
                print(f"    {name:>22}  {cell('agent')} | {cell('tool_output')} | {cell('system')} {cell('user')}")
        if "logit_monitor" in s:
            diff = lambda row, g: f"{row[g]['mean']:+.2f}" if row.get(g) else "-"
            print("  logit monitor, mean log-prob per token minus control (eval | training), by turn: "
                  + ", ".join(f"{row['turn']} {diff(row, 'eval_minus_control')} | {diff(row, 'training_minus_control')}"
                              for row in s["logit_monitor"]["by_turn"]))
    b.close()


if __name__ == "__main__":
    main()
