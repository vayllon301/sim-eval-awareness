"""Interpretability tools: emotion and evaluation-awareness vectors, a post-hoc episode
reader with a logit monitor, Qwen-Scope SAEs and activation steering. Reference:
docs/INTERP.md.

Optional: `uv sync --extra interp` installs MLX (Apple Silicon), safetensors and
transformers; `--extra torch` installs torch for hf_backend's tests. The GPU install is in
explore/interp/gpu_smoke.sh. Nothing here is imported by the testbed, and importing this
package does not import MLX, torch or vLLM; each backend imports its engine when built.

Modules:
  backend      the interface a model engine provides (activations, losses, greedy
               generation with additive residual hooks)
  mlx_backend  that interface on MLX (laptop, development only)
  vllm_backend that interface on vLLM with vllm-lens (GPU, the reference backend), and the
               steering spec as a vllm-lens SteeringVector
  hf_backend   that interface on PyTorch with transformers (GPU cross-check, splicing)
  fake         a tiny numpy model and tokenizer for offline tests, and a fake vLLM engine
  directions   difference-of-means vectors, neutral principal components, accuracy
  vectors      the vector file format (safetensors and JSON metadata)
  steering     the steering spec format and its resolution to per-layer additions
  reader       episode adapters, rendering with role masks, per-token projections
  monitors     the logit monitor: log-probs of short sentences at the start of each
               turn's reasoning, in one pass over the episode where the backend can
  episodes     the Inspect adapter and a run's export (laptop: it needs Inspect)
  readrun      reading a run, resumable; the server's token IDs; the path check
  validate     each monitor against the judge's labels and behaviour; the DRAFT rule
  phrases      the monitor's sentences from the pilot's verbalisations (DRAFT rule)
  command      `simtest interp`
  sae          TopK SAEs (Qwen-Scope), a torch-free .pt reader, FVU
  meta         fingerprints, hashes and versions recorded in every output
"""
