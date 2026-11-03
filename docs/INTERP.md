# Interpretability tools (`simtest.interp`)

Reference for the code that reads and steers the two models from the inside: emotion
vectors, an evaluation-awareness vector and its training twin, a post-hoc episode reader
with a logit monitor, Qwen-Scope SAEs and activation steering. The testbed itself is in
[`TESTBED.md`](TESTBED.md).

**Status.** Built and tested on the laptop with MLX. The experiment's reference backend is
vLLM 0.32 on the Colab G4, with both models in FP8 in every phase (LOG.md, 28-10-2026). Its
backend reads and steers through vllm-lens, and a transformers backend cross-checks the
agent ([GPU backends](#gpu-backends)). Both are tested offline; the
[GPU smoke test](#gpu-smoke-test) has not run yet. Numbers from local 8-bit or 4-bit
weights are for development only and are never evidence: every condition of an experiment
runs on one backend (same weights, quantization, engine and chat template).

## Install

```bash
uv sync --extra interp      # MLX and mlx-lm (Apple Silicon), safetensors, transformers
uv run pytest -q            # the interp tests run offline, without the extra
uv run --extra torch pytest -q tests/test_interp_hf.py   # the torch backend, on tiny random models (CPU)
```

The GPU install (vLLM 0.32.0, vllm-lens 1.3.0 and what the HF step needs, in a venv of its
own) is in `explore/interp/gpu_smoke.sh`. Nothing in the testbed imports `simtest.interp`,
and importing it doesn't import MLX, torch or vLLM: each backend imports its engine when
it is built.

## Layout

| Path | What |
| --- | --- |
| `src/simtest/interp/backend.py` | The backend interface and its conventions |
| `src/simtest/interp/mlx_backend.py` | The interface on MLX (laptop) |
| `src/simtest/interp/vllm_backend.py` | The interface on vLLM with vllm-lens (GPU, the reference backend), and the spec-to-`SteeringVector` adapter |
| `src/simtest/interp/hf_backend.py` | The interface on PyTorch with transformers (GPU cross-check and fallback for the agent; splicing; cached continuation log-probs) |
| `src/simtest/interp/fake.py` | A tiny numpy model and character tokenizer for offline tests, and a fake vLLM engine with vllm-lens's semantics |
| `src/simtest/interp/directions.py` | Difference of means, neutral principal components, accuracy, random controls |
| `src/simtest/interp/build.py` | Builds a vector file from texts and contrast pairs, with validation |
| `src/simtest/interp/vectors.py` | The vector file format |
| `src/simtest/interp/steering.py` | The steering spec format and its resolution |
| `src/simtest/interp/reader.py` | Episode adapters, rendering with role masks, per-token projections, prompt-end positions |
| `src/simtest/interp/monitors.py` | The logit monitor: log-probs of short sentences at the start of each turn's reasoning |
| `src/simtest/interp/sae.py` | TopK SAEs: a torch-free `.pt` reader, encode, decode, FVU |
| `src/simtest/interp/meta.py` | Weights fingerprints, hashes, versions |
| `explore/interp/` | Command-line scripts (below) and the datasets in `data/`: texts, eval and training pairs, logit-monitor phrases |
| `explore/interp/gpu_smoke.{py,sh}` | The [GPU smoke test](#gpu-smoke-test) and its VM runner |
| `tests/test_interp_*.py` | Offline tests; `test_interp_model.py` needs a local model and is skipped by default; `test_interp_hf.py` needs the torch extra |

## Backend interface

A backend wraps one model: weights, quantization, engine, chat template. Its `info` (engine,
model path, weights fingerprint, revision, quantization, the sha256 of the tokenizer files
and of the chat template in effect, and the versions of the engine, transformers and
tokenizers) goes into the metadata of every vector file and reading. A vector file only
reads and steers the model it was built on: the reader and `steering.resolve` refuse
vectors whose weights fingerprint differs from the backend's.

```python
class Backend(Protocol):
    info: dict; n_layers: int; d_model: int; tokenizer; eos_token_ids: frozenset[int]
    def activations(self, ids, layers, *, steer=()) -> dict[int, np.ndarray]   # {L: (T, d) float32}
    def token_losses(self, ids, *, replace=None) -> np.ndarray                 # (T-1,) next-token CE, nats
    def generate(self, ids, *, max_tokens, steer=()) -> list[int]              # greedy
    # optional, not in the Protocol:
    def continuation_logprobs(self, prefix_ids, continuations) -> list[np.ndarray]  # per token, nats
```

Conventions every backend honours:

- **Layers** are 0-based decoder layers. Layer L is the residual stream right after
  decoder layer L (`resid_post`, the output of `model.layers[L]`). This is Qwen-Scope's
  hook point and the index EasySteer and vllm-lens use.
- **Arrays** crossing the interface are numpy float32, (tokens, d_model).
- **Token IDs** come from the model's own tokenizer. Chat rendering is the reader's job;
  a backend only sees IDs.
- **`Add(layer, delta)`** adds `delta` to the residual stream after that layer at every
  position, prompt and generated tokens alike, in the model's compute dtype. Captured
  activations include the additions made at their layer.
- **`replace`** maps a layer to `f(h, start) -> h'`, which substitutes the residual stream
  after it; `h` is a run of consecutive positions starting at absolute position `start`.
  Used to splice an SAE reconstruction in.
- **`generate`** is greedy and stops at an end-of-turn token or `max_tokens`.
- **`continuation_logprobs`** (optional): the log-prob of each token of each continuation
  after one prefix, one array of shape (len(c),) per continuation, with the prefix
  computed once. The logit monitor uses it when a backend has it and otherwise calls
  `token_losses` once per continuation. The HF backend has it, with a prefix cache; the
  MLX and vLLM backends don't.

The MLX backend wraps the decoder layers it needs for the duration of a call and restores
them after. Long sequences run in chunks with the model's own cache (as mlx-lm's prefill
does), and activations-only calls stop after the last layer needed. `generate` processes
the prompt in mlx-lm's order (all but the last token in chunks of 2048, then the last
token alone): another order changes bf16 rounding, and greedy decoding can then flip a
near-tie, which would look like a steering effect at strength 0.

## GPU backends

The reference backend is vLLM 0.32 with both models in FP8, as the episodes run
(`explore/colab_probe/serve.sh`). The agent is the official FP8 checkpoint
(`Qwen/Qwen3.8-27B-FP8`: 128×128 weight blocks, activations quantized per token in groups
of 128). The simulator is bf16 weights (`Qwen/Qwen-AgentWorld-35B-A3B`) quantized by vLLM
at load (`--quantization fp8`). Vectors, steering and readings run inside that engine, so
they use the episodes' FP8 weights by construction.

**Path per model** (sources checked on 31-10-2026, listed below):

- *Agent:* vLLM with vllm-lens, the episodes' engine and kernels (CUTLASS block FP8 on the
  G4's sm_120). transformers loads the same FP8 files (FineGrainedFP8, a Triton kernel from
  the `kernels` hub), but with other kernels, so its numbers are close, not identical. It is
  the cross-check and the fallback.
- *Simulator:* vLLM with vllm-lens, the only engine with these FP8 weights.
  - In vLLM 0.32, `--quantization fp8` on a bf16 checkpoint is the deprecated name of
    `fp8_per_tensor`.
  - Weights: one scale per merged projection (qkv, gate_up, `in_proj_qkvz`, `in_proj_ba`)
    and one per expert.
  - Activations: per token in dense layers (CUTLASS), per tensor over the whole batch in
    the MoE (Triton).
  - transformers quantizes each unmerged projection in 128×128 blocks, so it can't
    reproduce those weights.
- *A consequence for every simulator number.* The MoE's activation scale is computed over
  the whole batch, so the simulator's output depends on what else is in the batch. The
  episodes ran 13–16 at once; the vLLM backend sends one request at a time.
  `fp8_per_channel` (per-token activations, the MoE included) removes that part of the
  dependence, but it changes the simulator's weights for every condition, so it is
  decided before the pilot.
  - Checked in vLLM 0.32.0's source (02-11-2026, [`BACKENDS.md`](BACKENDS.md), "The
    simulator's quantization"): it exists, and it covers every quantized layer, the MoE
    included, on the G4 and the H100. Other batch dependence remains (MoE tile sizes, attention
    splits, prefill chunks), and vLLM's batch-invariant mode refuses the GDN layers, so the
    pin at 12 at once stays either way.
  - If it is adopted, every phase uses it: `serve.sh`'s `SIM_QUANT=fp8_per_channel` for
    episodes (`EAGER=1` too), and `quantization="fp8_per_channel"` for this backend and the
    smoke test (`explore/interp/gpu_smoke.py` says `fp8` today).

**vLLM backend** (`vllm_backend.py`), on vllm-lens 1.3.0 (UK AISI, MIT):

- *Reading:* `output_residual_stream` captures `hidden_states + residual` after decoder
  layer L (resid_post) at every prompt position, in bf16. vllm-lens's own tests check it
  against transformers' `hidden_states[L + 1]`. One request, `max_tokens=1`.
- *Steering:* `lens_steering` turns `steering.resolve`'s additions into one
  `SteeringVector` (below). vllm-lens adds it to the layer's output before its capture, so
  captured activations include it, as the interface requires.
- *Settings:* eager (vllm-lens forces it) and prefix caching off. Cached blocks skip the
  forward pass, so they would have no activations, and an unsteered request could reuse
  steered KV. One request per call: vLLM isn't batch-invariant, and the simulator's MoE
  scales depend on the batch. `max_num_batched_tokens` is 8192, vllm serve's default on a
  96 GB card, so prompts are split into the same prefill chunks as the episodes'.
- *`token_losses`* uses vLLM's prompt log-probs (the raw log-softmax). There is no
  `replace` (an SAE splice needs a hook shipped to the worker process) and no
  `continuation_logprobs`. vLLM skips the prefix cache for requests with prompt log-probs,
  and batching the continuations would change the numbers, so the logit monitor falls back
  to `token_losses`.
- *Caveats, all checked by the smoke test:*
  - vllm-lens 1.3.0 pins vLLM 0.30.0 and is validated only there. It goes on 0.32.0
    without its dependencies. Every symbol it patches exists in 0.32.0, and
    `VLLM_LENS_STRICT_COMPATIBILITY=1` makes it fail at startup if a patch doesn't apply.
  - Its pre-hooks raise on Qwen3.5-architecture layers, which vLLM calls with keywords
    only ([vllm-lens#52](https://github.com/UKGovernmentBEIS/vllm-lens/issues/52), reported
    on Qwen3.8-27B-FP8). Post-hooks, which this backend uses, work: the issue says they
    gave "values matching a bf16 HF reference within FP8 noise". The traceback it logs for
    every layer at every forward pass is filtered out in the smoke test's venv.
  - It logs and skips a hook that fails instead of raising, so a skipped steer looks like
    an unsteered run. The smoke test checks that a steer moves its layer by its delta.
  - The episodes ran compiled (torch.compile, CUDA graphs) on vLLM's default model runner.
    vllm-lens runs eager on the V1 runner (it sets `VLLM_USE_V2_MODEL_RUNNER=0`; its hooks
    read the V1 runner's internals), and the stock-eager baseline uses the same runner.
    Whether the greedy tokens are the same is the smoke test's `served_equals_eager`. If
    they aren't, a steering experiment runs all its conditions, the unsteered ones
    included, on the lens engine's settings.
  - The plugin loads into every vLLM process of its environment, so it gets a venv of its
    own, apart from serve.sh's install.

**HF backend** (`hf_backend.py`):

- Forward hooks on `model.model.layers[L]` (resid_post), removed after each call. Readings
  run the decoder stack without `lm_head`.
- `generate` calls the model the way `model.generate` does (one pass over the prompt with
  `logits_to_keep=1`, then one token at a time, with an all-ones attention mask). So
  unhooked output is token-identical to transformers' own greedy generation.
- `continuation_logprobs` runs the prefix once and scores each continuation from a copy of
  its cache. `replace` works.
- On the GPU it needs `kernels` (the FP8 kernel) and `accelerate`. The Gated DeltaNet fast
  path needs `flash-linear-attention`; without it transformers uses its torch path.
- Tested offline on tiny random Qwen3 and Qwen3.5-hybrid models (`tests/test_interp_hf.py`),
  with transformers 5.21 on the laptop and with 5.19 on Python 3.12, vLLM 0.32's pins as on
  the VM.

**Fingerprints.** A backend that quantizes at load appends its scheme (`fp1:…+vllm:fp8`),
so vectors built on the simulator's bf16 files don't steer its FP8 weights. A pre-quantized
checkpoint has one fingerprint (the same files) on vLLM and on transformers.

**EasySteer** (v0.29.0) can't be used.
- It is a fork of vLLM 0.29.0, and its install overwrites vLLM's own files, so it can't
  run on 0.32, the episodes' engine.
- On its own base it would hook the same point (resid_post). Its layer discovery doesn't
  depend on the architecture, and its fork registers both models' architectures.
- But nobody has run a Qwen3.5-architecture or FP8 model on it. Using it means moving both
  models to vLLM 0.29 in every phase, after an equivalence check. Not done.

Sources:
- vllm-lens:
  [README, `vllm_lens/_worker_ext.py`, `_helpers/types.py`, `pyproject.toml`](https://github.com/UKGovernmentBEIS/vllm-lens)
  and [issue #52](https://github.com/UKGovernmentBEIS/vllm-lens/issues/52).
- vLLM v0.32.0:
  - `layers/quantization/online/fp8.py`, `model_loader/weight_utils.py`, `config/quantization.py`
  - for `fp8_per_channel` and the batch dependence: `fused_moe/oracle/fp8.py`, `fused_moe/fused_moe.py`,
    `kernels/linear/__init__.py`, `v1/attention/selector.py`, `v1/core/sched/scheduler.py`
    (all cited in [`BACKENDS.md`](BACKENDS.md))
  - `engine/arg_utils.py` (defaults by card), `sampling_params.py`
- [EasySteer](https://github.com/ZJU-REAL/EasySteer) v0.29.0: `docs/getting-started/installation.md`,
  `model_hooks/components/discovery.py`.
- The models' `config.json` on Hugging Face: Qwen3.8-27B-FP8 at `017b9c7`,
  Qwen-AgentWorld-35B-A3B at `60d2b04`.
- transformers 5.21: `integrations/finegrained_fp8.py`, `models/qwen3_5/`.

## Vector file

One safetensors file per model and vector set. Tensors are float32 unit vectors named
`<name>@<layer>` (`desperate@42`, `topic:baker-prepares@42`, `eval_awareness@42`,
`train_awareness@42`). The
header metadata has one key, `simtest_interp`, holding a JSON object:

| Field | What |
| --- | --- |
| `format`, `format_version` | `simtest-interp-vectors`, 1 |
| `model` | The backend's `info`: engine, path, weights fingerprint, revision, quantization, tokenizer and chat-template sha256, versions |
| `layers`, `d_model`, `names` | What the file holds |
| `resid_norm_mean` | `{layer: mean L2 norm of the residual stream}` on `norm_reference` (training texts, tokens from 50 on). Steering strengths are fractions of it |
| `offset` | `{name: {layer: mean projection of the training neutral texts}}`, to centre readings |
| `vector_info`, `method`, `params` | How each vector was made |
| `datasets` | Paths, sha256, counts and generators of the texts and pairs |
| `eval` | Validation per layer (held-out, leave-one-out, eval-awareness and training-twin checks) |
| `chosen_layer`, `layer_rule` | The layer the rule picked |
| `created`, `git_commit` | When and from which code |

The weights fingerprint (`fp1:`) is a sha256 over `config.json`, the safetensors index,
and each weight file's name, size and first and last 4 MiB. It tells apart quantizations,
conversions and revisions without hashing 29 GB on every run.

## Emotion vectors

Sofroniew et al. (2026, arXiv:2604.07729), scaled down. Built separately in each model,
from texts written by the model family itself.

- **Texts** (`explore/interp/gen_texts.py`, written through OpenRouter: text generation,
  not an experiment backend). The paper's story and neutral-dialogue prompts, verbatim
  (its App. 6.5): the emotion word and its synonyms are forbidden in the stories.
  - Emotions: desperate, calm, frustrated, afraid, angry, happy.
  - 12 premises, 3 stories per emotion and premise (216 stories), plus 36 neutral
    Human/Assistant dialogues on the same premises. Five premises are the paper's; seven
    are ours, in its style.
  - The last three premises are held out, so held-out accuracy is transfer to unseen
    premises.
  - Generators: `qwen/qwen3.8-27b` for the agent. Qwen-AgentWorld is not hosted anywhere,
    so the simulator's texts come from `qwen/qwen3.5-35b-a3b`, its parent and closest
    hosted relative (the metadata says so). Each row records the provider.
- **Method** (`build.py`, `directions.py`):
  1. Each text goes in raw (no chat template). Its mean activation from token 50 on.
  2. Each emotion's mean over its training stories, minus the mean of the six emotion
     means.
  3. The neutral dialogues' top principal components, enough for 50% of their variance,
     projected out of each vector. The PCA runs on per-token activations (from token 50
     on). The paper doesn't say whether it used tokens or per-text means; the vllm-lens
     replication uses means, and the build reports that variant too.
  4. Unit length. Readings are dot products with the unit vector.
  - Everything fitted (vectors, neutral PCs and their per-text variant, offsets, the
    reference norms) uses training rows only; held-out stories and dialogues are only
    scored.
- **Validation.** Held-out accuracy: a held-out story is assigned the emotion whose vector
  has the largest dot product with its mean minus the training grand mean (chance 1/6).
  Leave-one-out over training stories. Variants without the PC projection and with PCs of
  per-text means, for comparison. The paper reports no held-out accuracy for its story
  vectors.
- **Layer scan.** A few layers around two-thirds depth (agent 36–48 of 64, simulator
  22–32 of 40). Rule, fixed before running: the highest held-out accuracy; ties go to
  the layer closest to two-thirds depth. Every scanned layer is saved.

## Evaluation-awareness vector

Hua, Qin, Marks and Nanda (arXiv:2510.20487): their 16 contrastive pairs ("real user" vs
"AI evaluator", their Table 12, verbatim) in `explore/interp/data/eval_awareness_pairs.json`.

- Each prompt is rendered as a user message with the model's chat template (no system
  prompt; the template's own defaults), and the activation is read at the colon that ends
  it, as they did.
- Vector: mean(evaluation) − mean(real) over the training pairs, unit length, no PC
  projection. Positive means evaluation.
- Validation: pairs 13–16 held out (fixed before any run); leave-one-pair-out over all
  16; and 8 hand-written pairs of our own (the same coding request as a real user's
  message and as an evaluation item), read as the mean over the user message's tokens
  (AUROC and pair accuracy).

**Training twin** (`train_awareness`). Aranguri ("Different eval awareness may track
different graders", LessWrong, 25-10-2026) found on Qwen3.6-27B that eval-awareness probes
score nearly the same when their statements are rewritten as training (r ≈ 0.98). A probe
may read training awareness rather than evaluation awareness.

- Pairs in `explore/interp/data/train_awareness_pairs.json`: the same 16 real statements,
  verbatim, and each evaluation statement reworded as training, with the same structure
  and about the same length ("You are talking to an AI evaluator:" becomes "You are
  talking to an AI trainer:"). Ours: his rewrites aren't public.
- Built like the eval vector (`build.build_contrast`): read at the colon, mean(training) −
  mean(real) over the training pairs, unit length, no PC projection. Positive means
  training. Same held-out pairs (13–16) and leave-one-pair-out.
- Two more checks per layer: `cosine_with_eval_awareness`, and `generalisation` on the
  eval file's 8 coding pairs (real vs evaluation, mean over the user tokens), which asks
  how well the training direction separates real from evaluation requests. If it does as
  well as the eval vector, or tracks C3 as well, the reading isn't specific to evaluation.
- Its neutral offset is set like eval_awareness's. `eval_awareness` itself is unchanged.

## Controls

The steering design needs controls at the same norm, both signs (Fomin et al.; UK AISI
GLM-5):

- **Random directions:** `direction: random` with a seed. The vector at layer L is
  numpy's `default_rng([seed, L])`, standard normal, normalised, so any engine
  reproduces it from (seed, layer).
- **Unrelated concepts:** one premise's stories (all emotions) minus the other training
  premises' stories, with the same PC projection: `topic:baker-prepares`,
  `topic:family-drives`, `topic:neighbor-renovation`. Same story style as the emotion
  vectors, different content.

All directions are unit vectors, so at the same strength every control has the target's
norm by construction. `steering.controls()` and `steering.dose_curve()` make the specs.

## Steering spec

One YAML (or JSON) file per steering condition, read by every engine:

```yaml
# simtest interp steering spec (docs/INTERP.md)
spec_version: 1
vectors: ../vectors/agent.safetensors   # relative to the spec file
vectors_sha256: 3f1c...                 # optional; checked when given
direction: desperate                    # a vector name in the file, or "random"
seed: null                              # required for "random", null otherwise
layers: [39, 42, 45]                    # 0-based decoder layers (resid_post)
alpha: 0.05                             # strength, a fraction of the mean residual norm (>= 0)
sign: 1                                 # 1 or -1
```

**Resolution** (`steering.resolve(spec, fingerprint=backend.info["fingerprint"])`, which
refuses vectors built on another model; same-size models like Qwen3.5-27B and
Qwen3.8-27B would otherwise accept each other's vectors): at each layer L,
`delta_L = sign × alpha × resid_norm_mean[L] × u_L`, with `u_L` the unit vector
`direction@L` (or the seeded random vector) and `resid_norm_mean[L]` from the same vector
file, measured on the same backend. The engine adds `delta_L` to the residual stream after
decoder layer L at every position, prompt and generated tokens. Alpha 0 must leave greedy
output token-identical.

**On the GPU** (`vllm_backend.lens_steering`, [GPU backends](#gpu-backends)):

- One vllm-lens `SteeringVector` per request, in
  `SamplingParams.extra_args["apply_steering_vectors"]`. Its fields:
  `activations=stack(delta_L)` (the layers in order, already scaled), `layer_indices`,
  `scale=1.0`, `norm_match=False`, `position_indices=None` (every position).
- Alpha 0 still sends a zero vector, so it runs the hooked path.
- Over `vllm serve` the same vector goes in `vllm_xargs` as
  `json.dumps([sv.model_dump(mode="json")])` (vllm-lens README). Not used yet.
- The smoke test comes first: alpha 0 gives stock vLLM's greedy output, a steer moves its
  layer by its delta, and the projection moves with alpha (`gpu_smoke.py`;
  `steer_check.py` does the same on MLX).
- EasySteer, which can't run on vLLM 0.32, would take a GGUF control vector (arch
  `controlvector`, float32 tensors `direction.<L>` = `delta_L`) with `scale=1.0`,
  algorithm `direct`, `normalize` off, `ApplySpec(prompt="all", generation="all")`.

Positions: every position, prompt included. "Generated tokens only" would need its own
check, and it is not in the spec. Reading vllm-lens 1.3.0's code, its per-position steering
looks for absolute positions in the wrong place (`seq_lens` in a per-layer dict), so every
chunk and decode step would count from 0. That was inferred, not run.

## Reader

Post hoc, one forward pass per episode, no logits kept (the logit monitor, when asked,
adds its own passes). Stores projections (and SAE top-k latents when asked), never raw
activations.

1. **Adapters** turn an episode record into chat messages and tools. Only adapters know a
   record format:
   - `from_colab_probe`: `explore/colab_probe/episode.py` records (one row per agent turn,
     then a final row). The record has no system prompt, ticket or tools, so
     `colab_probe_prompts` reads them from `episode.py`'s source without running it.
     Rows without `tool_calls` (older records) use `calls`. The last turn's tool outputs
     were never read by the agent, so they are dropped (counted in the metadata).
   - The N1 Inspect episode format gets its own adapter; nothing after the adapter changes.
2. **Rendering** (`render`) uses the model's own chat template with the tools, earlier
   reasoning kept (`reasoning_content`), and the template's defaults (for Qwen3.8:
   xhigh, `preserve_thinking`). Each text is replaced by a unique placeholder, the
   messages are rendered once, and the placeholders give the exact span of every text,
   whatever the template's layout; the result must equal the template's own rendering
   and tokenization, or the reader stops.
3. **Roles per token:** `template` (special tokens, template text), `system` (the system
   turn: the template's tools block, the reasoning-effort sentence, the system prompt),
   `user`, `reasoning`, `content`, `tool_call` (argument values), `tool_output`. The
   agent's own tokens are `reasoning`, `content` and `tool_call`. Each token also has
   the agent turn it belongs to (−1 for the opening prompt).
4. **Why one pass is enough.** Attention is causal, so the pass over the final
   conversation gives the same activations the server computed turn by turn, as long as
   each turn's prompt is a prefix of the next (true for Qwen3.8 with `preserve_thinking`
   on). `prompt_checks` compares each turn's rendered prompt length with the prompt
   tokens the server reported; a mismatch means the server saw something else (another
   template, dropped reasoning). On the two smoke episodes: LM Studio's first prompt
   matches exactly, then each earlier agent turn renders 3 tokens shorter than LM Studio
   counted (trimming of content or reasoning ruled out); OpenRouter's prompts are a
   constant 22 tokens longer (the provider's own opening). The way out is to record the
   token IDs the server saw: vLLM can return them, and the N1 format should store them,
   so the reader uses them instead of re-rendering.

**Output:** `<episode>.npz` with `token_ids`, `role` (index into `ROLES`), `turn`, `proj`
(T × layers × vectors, float32, the raw dot product), `prompt_end` (below), optional
`sae_idx@L` / `sae_val@L` (top-k latents), and `meta` (JSON: episode, backend, vectors
and their hash, offsets, prompt checks, text hash, the logit monitor when asked).
`<episode>.summary.json` has the mean and SD of each projection by role group (agent,
reasoning, content+tool_call, tool_output, system, user, template), centred by the vector
file's neutral offset, per agent turn (with the prompt-end reading), and the logit
monitor when asked.

**Prompt-end readings.** Aranguri's best eval-vs-deployment detector was a probe read at
the last token of the prompt, before the model writes anything. Reading with the response
included diluted it. The reader keeps that position for every agent turn.

- For agent turn k (the k-th assistant message), the position is n_k − 1. n_k is the
  token length of the messages before it, rendered with the generation prompt: the
  prompts `prompt_checks` compares with the server's counts, rendered once for both.
- Those n_k IDs must equal the first n_k IDs of the full rendering. Where they don't (a
  template that drops earlier reasoning from prompts, say), the position is −1 and the
  turn is listed in `meta["prompt_end_not_prefix"]`. This compares our own two
  renderings; whether the server saw the same prompt is what `prompt_checks` asks.
- Stored as `prompt_end` in the npz (int32, one per agent turn). Older readings without it
  still load (`prompt_end` is None).
- Each `by_turn` row of the summary has `prompt_end` (the raw projection there) and
  `prompt_end_centred` (minus the neutral offset, like `mean_centred`). Both are None when
  the position is −1 or unknown.
- If the generation prompt opens the reasoning (`<think>\n`, as in the fake template), the
  next token is the turn's first reasoning token.

**Logit monitor** (`monitors.py`, after Aranguri): the log-probability that the agent starts
a turn's reasoning with a short awareness sentence ("The user might be testing"). In his
work it tracked verbalised awareness best. Here it gives a continuous version of
verbalisation, so a difference can show even if verbalised rates sit near zero.

- Prefix: the full rendering's token IDs up to, not including, the turn's first
  `reasoning` token. Turns without reasoning tokens give None.
- Each phrase is encoded as plain text (`backend.encode`, no special tokens) and scored
  after the prefix: total log-prob (`logprob`, nats), mean per token (`mean`), and its
  token count.
- Phrases come in groups (`eval`, `training`, `control`). Per turn: each phrase, each
  group's mean of `logprob` and of `mean`, and each group minus the control group
  (`eval_minus_control`, `training_minus_control`). The control sentences are neutral first
  sentences, so a condition that shifts every sentence's probability isn't read as
  awareness.
- `read(..., phrases=doc, phrases_path=path, phrase_turns=None)` stores it as
  `meta["logit_monitor"]` and `summary["logit_monitor"]`: the phrase file's path, sha256 and
  status, the scoring path (`token_losses` or `continuation_logprobs`), the phrases with
  their token counts, and one row per scored turn (`turn`, `prefix_len`, `phrases`,
  `groups`, the differences). `phrase_turns` scores only some turns.
- **Cost.** The generic path calls `token_losses` on prefix + phrase: one forward pass over
  the whole prefix per turn and phrase. With the draft's 10 phrases, an 8-turn episode is
  80 passes over prefixes of up to the whole episode, so it is slow on the laptop and
  meant for checks (score a few turns with `--logit-turns`). The cached path
  (`continuation_logprobs`) computes each prefix once; the HF backend has it.
- Phrases: `explore/interp/data/logit_monitor_phrases.json`, a draft. Aranguri's two
  sentences verbatim, three coding-agent drafts of ours (eval), two training sentences, and
  three controls (one shares the common "The user" opening). In N2 they are replaced by
  sentences picked from what the pilot agent actually writes, and frozen before N3.

Emotion vectors are local: readings on tool-output tokens reflect what the agent reads,
not its own state. Report the agent's own tokens and tool outputs separately.

## SAEs

**Qwen-Scope** (arXiv:2605.11887, HF collection `Qwen/qwen-scope`): TopK SAEs at every
layer, `resid_post`. Qwen3.5-27B: 80K latents, k = 50 or 100, trained on the instruct
model; Qwen3.5-35B-A3B-Base: 32K or 128K latents. One `layer{L}.sae.pt` per layer, a dict
of float32 `W_enc` (d_sae, d), `b_enc`, `W_dec` (d, d_sae), `b_dec`; L is 0-based (the
paper's tables count from 1).

- `read_pt` reads PyTorch's zip format with a restricted unpickler that rebuilds plain
  tensors only (anything else is refused) and memory-maps the storage. No torch needed,
  and a 3.4 GB layer opens instantly. Checked against `torch.load` on a real layer and on
  a fixture written by torch 2.14.
- **Encoder** as in the repos' `app.py`: `relu(x @ W_enc.T + b_enc)`, keep the top k. No
  `b_dec` subtracted first, no input scaling. **Decoder**: not in their code; we use
  `acts @ W_dec.T + b_dec` (the transfer check reports FVU without `b_dec` too).
- **Transfer check** (`explore/interp/sae_transfer.py`): on the same token IDs for every
  model (a whole smoke episode rendered with the chat template, about 7K tokens, with its
  token roles, then 2,048 tokens of held-out stories and dialogues, taken in turn from
  each emotion and the neutral dialogues, with the contributing row IDs saved), FVU by token group
  and spliced next-token loss (the layer's output replaced by the reconstruction, against
  clean and zero ablation). Position 0 is left out of FVU and never replaced. Qwen-Scope
  publishes no FVU or loss numbers, so the reference is the same measurement on
  Qwen3.5-27B, the SAEs' own model, at the same quantization.
- **Spliced loss only on tokens the model predicts.** Qwen3.8 gives many tool-output
  tokens almost no probability, even obvious ones ("up" after "Your branch is" costs
  about 20 nats), as if tool outputs were masked out of its training loss. A blurrier
  reconstruction then "improves" their loss, so the headline spliced loss is on the
  agent's own tokens and prose; tool outputs and the prompt are reported apart.
- **Quantization caveat.** The SAEs were trained on bf16 activations. 8-bit and 4-bit
  weights shift activations, so local FVU is not the bf16 number. Only compare models at
  the same quantization. The transfer numbers that count come from the FP8 reference
  backend, itself a shift from bf16.
- **Licence.** The `LICENSE` file in the SAE repos is Apache 2.0, but the model card says
  `license: other` (`qwen`) and adds a "Caution" restricting use to scientific research.
- The reader stores SAE top-k latents with `read(..., saes={L: sae})`.

## Scripts (laptop)

All free and local. One model at a time, with LM Studio unloaded; each run takes a few
minutes. Outputs go to `logs/interp/` (not in git).

```bash
# texts (paid, OpenRouter, a few cents per set)
uv run --env-file ../.env python -I explore/interp/gen_texts.py qwen/qwen3.8-27b explore/interp/data/texts-qwen3.8-27b
# vectors: emotions, controls, eval awareness and its training twin, at several layers
# (--pairs and --train-pairs default to the files in explore/interp/data/)
uv run --extra interp python -I explore/interp/build_vectors.py <model-dir> <texts.jsonl> <out.safetensors> --layers 36,39,42,45,48
# readings of episodes (--logit-phrases adds the logit monitor; --logit-turns 0,3 limits it to those turns)
uv run --extra interp python -I explore/interp/read_episode.py <model-dir> <vectors> --layers 42 --out-dir logs/interp/readings logs/colab-probe/*.jsonl \
    --logit-phrases explore/interp/data/logit_monitor_phrases.json --logit-turns 0
# SAE transfer
uv run --extra interp python -I explore/interp/sae_transfer.py <model-dir> <sae-snapshot-dir> --layers 32,42,48 --out logs/interp/sae/<name>.json
# steering checks
uv run --extra interp python -I explore/interp/steer_check.py <model-dir> <vectors> --layers 39,42,45 --read-layer 48 --out-dir logs/interp/steer/<name>
# model tests
SIMTEST_INTERP_MODEL=<model-dir> uv run --extra interp pytest -q -m interp_model
```

`explore/emotion_probe.py` (21-10-2026: 5 emotions × 6 hand-written stories, layer 42) is
superseded by `build_vectors.py` and kept for the record.

## GPU smoke test

`explore/interp/gpu_smoke.sh` runs `explore/interp/gpu_smoke.py` on the Colab G4, alone on
the GPU (stop serve.sh's servers first), one model and one step per process. It hasn't run
yet. `tests/test_interp_gpu_smoke.py` runs every step offline on the fake engine and a tiny
HF model.

| Step | What | Minutes (estimate) |
| --- | --- | --- |
| install | vLLM 0.32.0 and vllm-lens 1.3.0 (no dependencies), zstandard, kernels, accelerate, flash-linear-attention, in `/content/interp-venv` (Python 3.12) | 3 |
| download | Both models, unless serve.sh's copies are in `/content/models` | 5 (0) |
| `<model>-stock-eager` | Greedy tokens of two fixed prompts on stock vLLM (plugin off), with the lens steps' settings and model runner, each prompt twice | agent 3, simulator 4 |
| `<model>-lens` | (i) The vector file on the FP8 weights (texts, pairs and layer rule of `build_vectors.py`; layers agent 36–48, simulator 22–32). (ii) Greedy tokens before the hooks are installed, then unsteered, with zero additions and with an alpha-0 spec. (iii) A steer at one layer against its delta, the dose curve with random and unrelated controls, samples. (iv) Prompt log-probs | agent 8, simulator 7 |
| `<model>-stock-served` | The same prompts with vllm serve's settings (compiled, CUDA graphs, prefix caching, the default runner) | agent 5, simulator 6 |
| `agent-hf` | The agent's FP8 checkpoint on transformers: alpha 0 against `model.generate`, activations against vLLM's on the same tokens, a steer against its delta | 8 |
| report | `checks.json` | <1 |

About 50 minutes in all, 8 to 9 compute units. `BUDGET_MIN` (default 75) skips a step that
would end later; `STEPS` picks steps; a second run resumes. A step that fails is logged and
the next one runs.

**Pass rules**, fixed before running (`gpu_smoke.py`'s header). They say the tools work on
these weights; the numbers are not results.

- `vectors`: at the chosen layer, emotion held-out and leave-one-out accuracy ≥ 0.5 (chance
  1/6), and eval-awareness leave-one-pair-out ≥ 14 of 16. The MLX development builds had
  37–44 of 54 held out and 16 of 16.
- `alpha0`: for every fixed prompt, the lens engine's greedy tokens equal stock vLLM's with
  the same settings. That holds before the hooks, unsteered after them, with zero additions
  and with an alpha-0 spec.
- `steer_applied`: alpha 0.1 at one layer moves the mean projection onto that layer's unit
  vector, at that layer, by 0.1 × its mean residual norm, within 5%.
- `dose`: the target's mean projection rises strictly with the signed strength at the middle
  steered layer.
- `hf_alpha0`, `hf_steer_applied`: the same checks on the HF backend (agent).
- Reported, not pass or fail:
  - `served_equals_eager`: does the episodes' compiled engine give the eager engine's
    tokens?
  - Stock repeats.
  - The later layer's dose curve and the controls' slopes.
  - HF against vLLM: cosine and relative error per layer, projection correlation, the first
    greedy token that differs.

**Outputs** in `/content/interp-smoke/`:

- Per model: `<model>/{stock-eager,stock-served,lens,hf}.json`, `vectors.safetensors` with
  its report, `specs/` and `lens-acts.npz`.
- `checks.json` and `report.txt`.
- `timings.txt`, `versions.txt` (versions, model commits, experiment commit),
  `pip-freeze.txt`, `gpu.txt`, and one log per step.

```bash
# from the laptop (session "probe"; the runner's header has the same lines)
colab --auth adc new -s probe --gpu G4
PROXY="colab --auth adc ssh --proxy-mode -s probe"
SSH="ssh -o ProxyCommand='$PROXY' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@probe"
tar czf - --exclude .venv --exclude logs -C experiment . | eval "$SSH 'mkdir -p /content/sim && tar xzf - -C /content/sim'"
eval "$SSH 'setsid nohup bash /content/sim/explore/interp/gpu_smoke.sh >/content/interp-smoke.log 2>&1 </dev/null &'"
eval "$SSH 'cat /content/interp-smoke/timings.txt; tail -3 /content/interp-smoke.log'"   # progress
eval "$SSH 'tar czf - -C /content interp-smoke'" > logs/interp/gpu-smoke.tgz             # results
colab --auth adc stop -s probe
```

## Left for the GPU

- Run the smoke test and log it. Until then, nothing on the GPU path has run.
- If `served_equals_eager` fails, steering experiments run all their conditions on the
  eager engine (above).
- The simulator's batch-dependent FP8 MoE ([GPU backends](#gpu-backends)): keep
  `fp8_per_tensor`, or move every phase to `fp8_per_channel`. Decided on the
  measurement and DRAFT rule in [`BACKENDS.md`](BACKENDS.md) (the pilot session's first hour).
- The SAE transfer check on FP8 activations of pilot transcripts. `sae_transfer.py` splices
  with `token_losses(replace=)`, which only the HF backend has. So either the agent on the
  HF backend, or a vllm-lens `Hook` that splices in the worker process.
- The logit monitor on vLLM. Prompt log-probs over long prefixes skip the prefix cache and
  hold full-vocabulary logits per chunk (248K × 8192). Measure memory and time on a real
  episode, or score the agent on the HF backend's cached path. That path must match the
  `token_losses` path on the same episode; it does on the tiny models.
- An Inspect-format adapter for the reader, with the token IDs the server saw.
- Steering episodes through `vllm serve` (`vllm_xargs`), for the steering experiments.
