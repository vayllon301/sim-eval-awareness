# Interpretability tools (`simtest.interp`)

Reference for the code that reads and steers the two models from the inside: emotion
vectors, an evaluation-awareness vector and its training twin, a post-hoc episode reader
with a logit monitor, Qwen-Scope SAEs and activation steering. The testbed itself is in
[`TESTBED.md`](TESTBED.md).

**Status.** Built and tested on the laptop with MLX. The experiment's reference backend is
bf16 on a GPU (Colab), and its torch and steering-engine backends come later (see
[Left for the GPU](#left-for-the-gpu)). Numbers from local 8-bit or 4-bit weights are for
development only and are never evidence: every condition of an experiment runs on one
backend (same weights, quantization, engine and chat template).

## Install

```bash
uv sync --extra interp      # MLX and mlx-lm (Apple Silicon), safetensors, transformers
uv run pytest -q            # the interp tests run offline, without the extra
```

Nothing in the testbed imports `simtest.interp`, and importing it doesn't import MLX
(only `simtest.interp.mlx_backend` does).

## Layout

| Path | What |
| --- | --- |
| `src/simtest/interp/backend.py` | The backend interface and its conventions |
| `src/simtest/interp/mlx_backend.py` | The interface on MLX (laptop) |
| `src/simtest/interp/fake.py` | A tiny numpy model and character tokenizer for offline tests |
| `src/simtest/interp/directions.py` | Difference of means, neutral principal components, accuracy, random controls |
| `src/simtest/interp/build.py` | Builds a vector file from texts and contrast pairs, with validation |
| `src/simtest/interp/vectors.py` | The vector file format |
| `src/simtest/interp/steering.py` | The steering spec format and its resolution |
| `src/simtest/interp/reader.py` | Episode adapters, rendering with role masks, per-token projections, prompt-end positions |
| `src/simtest/interp/monitors.py` | The logit monitor: log-probs of short sentences at the start of each turn's reasoning |
| `src/simtest/interp/sae.py` | TopK SAEs: a torch-free `.pt` reader, encode, decode, FVU |
| `src/simtest/interp/meta.py` | Weights fingerprints, hashes, versions |
| `explore/interp/` | Command-line scripts (below) and the datasets in `data/`: texts, eval and training pairs, logit-monitor phrases |
| `tests/test_interp_*.py` | Offline tests; `test_interp_model.py` needs a local model and is skipped by default |

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
  `token_losses` once per continuation. The MLX backend doesn't have it; the GPU backend
  will, with a prefix cache.

The MLX backend wraps the decoder layers it needs for the duration of a call and restores
them after. Long sequences run in chunks with the model's own cache (as mlx-lm's prefill
does), and activations-only calls stop after the last layer needed. `generate` processes
the prompt in mlx-lm's order (all but the last token in chunks of 2048, then the last
token alone): another order changes bf16 rounding, and greedy decoding can then flip a
near-tie, which would look like a steering effect at strength 0.

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

**What a GPU engine must do with a spec:**

- *EasySteer* (v0.29.0, a vLLM fork): write a GGUF control vector (arch `controlvector`)
  with tensors `direction.<L>` = `delta_L` (float32, already scaled), and steer with
  `VectorSpec(..., scale=1.0, layers=[...], apply=ApplySpec(prompt="all", generation="all"))`,
  algorithm `direct`, `normalize` off.
- *vllm-lens* (v1.3.0): `SteeringVector(activations=stack(delta_L), layer_indices=layers,
  scale=1.0, norm_match=False)`, which is applied at every position.
- Either way, the smoke test first: alpha 0 gives the same greedy output as stock vLLM,
  and the projection moves with alpha (`explore/interp/steer_check.py` does both on MLX).

Positions: every position, prompt included. EasySteer can restrict positions and vllm-lens
can't, so "generated tokens only" would need its own check; it is not in the spec.

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
  (`continuation_logprobs`) computes each prefix once and comes with the GPU backend.
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
  the same quantization; the real transfer numbers come from bf16 on the GPU.
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

## Left for the GPU

- A torch/HF backend (bf16) honouring the interface, for building vectors and the reader:
  teacher-forced passes with hooks on `model.model.layers[L]`, no `lm_head` for readings.
  It must pass the offline tests' contract and an equivalence check against vLLM's
  tokenization of the same episodes.
- The steering-engine adapter (EasySteer first, vllm-lens as the fallback) that turns a
  spec into the engine's format (above), with the smoke test on Qwen3.8: alpha 0 equals
  stock vLLM greedy output, and the projection moves with alpha.
- Vectors rebuilt on the reference backend (the same texts and pairs), the SAE transfer
  check on bf16 activations of pilot transcripts, and an Inspect-format adapter for the
  reader.
- `continuation_logprobs` on the GPU backend, the logit monitor's cached path: each turn's
  prefix run once, its cache kept, then every phrase scored from it. It must match the
  generic `token_losses` path on the same episode, and its cost is measured there.
