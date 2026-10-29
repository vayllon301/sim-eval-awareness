"""Interpretability tools: emotion and evaluation-awareness vectors, a post-hoc episode
reader with a logit monitor, Qwen-Scope SAEs and activation steering. Reference:
docs/INTERP.md.

Optional: `uv sync --extra interp` installs MLX (Apple Silicon), safetensors and
transformers. Nothing here is imported by the testbed, and importing this package does not
import MLX; only `simtest.interp.mlx_backend` does.

Modules:
  backend      the interface a model engine provides (activations, losses, greedy
               generation with additive residual hooks)
  mlx_backend  that interface on MLX (laptop, development only)
  fake         a tiny numpy model and tokenizer for offline tests
  directions   difference-of-means vectors, neutral principal components, accuracy
  vectors      the vector file format (safetensors and JSON metadata)
  steering     the steering spec format and its resolution to per-layer additions
  reader       episode adapters, rendering with role masks, per-token projections
  monitors     the logit monitor: log-probs of short sentences at the start of each
               turn's reasoning
  sae          TopK SAEs (Qwen-Scope), a torch-free .pt reader, FVU
  meta         fingerprints, hashes and versions recorded in every output
"""
