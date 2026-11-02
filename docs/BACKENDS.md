# Backends: serving the models, and checking two backends agree

Every condition of an experiment runs on one backend: the same weights, quantization, engine,
chat template and settings. The simulator adds one more: its FP8 MoE scales activations per
tensor over the whole batch ([`INTERP.md`](INTERP.md), "GPU backends"), so its output depends on
what else shares the batch. How many episodes run at once (`max_connections`) is part of the
backend too.

This page covers how `explore/colab_probe/serve.sh` serves both models on each GPU we may get, how
`simtest equiv` checks whether two backends compute the same models before work moves between
them, and the measurement that decides between the simulator's two FP8 schemes.
How a whole session runs (the preflight, `session.sh`, the watchdog, Hopper's chunked jobs) is in
[`SESSIONS.md`](SESSIONS.md).

## Serving (`serve.sh`)

The script picks a profile from the GPUs it sees (`CUDA_VISIBLE_DEVICES`, or `GPUS`, chooses them):

| Profile | GPUs | Agent | Simulator | Notes |
| --- | --- | --- | --- | --- |
| `one-gpu-96g` | one 90–135 GB card (Colab G4) | 0.47 of the card, 128K | 0.47, 128K | `MIX=c3`: 0.42 / 0.52, for runs that are mostly C1–C3 |
| `one-gpu-141g` | one card of 135 GB or more (H200) | as the G4 | as the G4 | the same fractions leave each model more KV cache |
| `one-gpu-80g` | one 80 GB card (H100, A100) | 0.44, **64K** | 0.50, **64K** | a loud warning: not equivalent to the G4 (below) |
| `two-gpu` | two cards of 80 GB or more | its own card, 0.90, 128K | its own card, 0.90, 128K | `MIX` has no effect |

- vLLM is pinned at 0.32.0. The script refuses another version.
- `WORK` is the work directory. It defaults to `/content` on Colab and must be given anywhere else.
  It needs about 110 GB free: the downloads are 29 + 67 GB.
- Off Colab, vLLM goes into a venv (`$WORK/venv`, or `VENV`), and Hugging Face's cache goes into
  `$WORK/hf-home` unless `HF_HOME` is set. `VENV=` (empty) forces the system install that Colab
  uses.
- `EAGER=1` serves on the steering engine's settings (N4). `AGENT_MEM`, `SIM_MEM`, `AGENT_LEN`,
  `SIM_LEN` and `MAX_SEQS` override the profile.
- `SIM_QUANT` is `fp8` (the default, every episode so far) or `fp8_per_channel`; nothing else is
  accepted. `SIM_ONLY=1` restarts the simulator alone next to a running agent (it stops only the
  simulator of the same `WORK`). Both are for the
  measurement [below](#the-simulators-quantization-fp8-or-fp8_per_channel).
- **80 GB, one card.** The agent's weights and its KV cache at 128K don't fit next to the
  simulator, so both models get 64K. Long C3 episodes can pass 64K: session 5's largest simulator
  call was 69K tokens. Those calls then fail with a context-length error. An experiment on this
  profile runs every condition on it, and its episodes never mix with the G4's.
- `DRY_RUN=1` prints the profile and both `vllm serve` lines, and does nothing else. `GPU_MIB`
  and `GPU_COUNT` stand in for `nvidia-smi`. `tests/test_serve_script.py` pins the G4's lines to
  the ones the script used before Hopper support, and checks the dry run against what the script
  really starts.

```bash
# what a machine would serve (on the laptop: no GPU needed)
DRY_RUN=1 WORK=/content GPU_MIB=97887 GPU_COUNT=1 MIX=c3 bash explore/colab_probe/serve.sh
# Colab: see the script's header (colab new, ssh, the tunnel, colab stop)
# Hopper, on a GPU node (behind a login host), then the tunnel from the laptop
ssh <node> 'WORK=/scratch/$USER/sim MIX=c3 bash -s' < explore/colab_probe/serve.sh
SSH_TARGET=<node> SSH_JUMP=<login> explore/colab_probe/tunnel.sh >/tmp/tunnel.log 2>&1 &
```

The servers listen on the machine's loopback only. The tunnel brings them to the laptop's ports
18000 (agent) and 18001 (simulator), where the configs expect them. `tunnel.sh` reconnects at
once when the connection drops, for Colab and for a plain SSH host alike.

## The equivalence check (`simtest equiv`)

The check teacher-forces saved episodes on two backends and compares the numbers each backend
computes for the same tokens. Nothing is sampled. At every position it reads the log-probability
of the recorded token and the backend's top-1 token.

1. **Contexts** (offline, from `.eval` logs). For each episode it takes the agent's last call,
   whose prompt is the whole episode, and a few simulator calls spread over the episode, with the
   last one kept. A context is the exact request Inspect logged plus the reply, as the next turn
   sends it back (checked on session 5: all 156 agent replies equal the next request's copy).
   Errored samples are left out. A sample with several attempts counts once, from its newest log.
2. **Score** (on a running backend). Each server renders and tokenizes the context itself
   (`/tokenize`, with its own chat template). Then the token IDs go to `/v1/completions` with
   `prompt_logprobs`, `max_tokens` 1 and temperature 0. Each row records:
   - the IDs' hash;
   - where the reply starts.
   A context is misaligned if the prompt alone isn't a prefix of the full context, or has a
   different number of tokens from what the episode's server counted. It is never scored: it
   becomes an error row (`stage: align`). Any failed context is recorded as an error row, never
   dropped. `--top-k` must be at least 1, and a position without a rank-1 token is an error.
3. **Compare.** Two score files are compared per model, on contexts scored with identical token
   IDs:
   - the absolute difference in log-prob of the recorded token: mean, median, p99 and max;
   - top-1 agreement;
   - the same two over the reply's own tokens only.
   Different token IDs mean the backends render differently.

**The noise floor.** The same server, scored one request at a time and then N at a time, gives
different numbers, because requests share batches. That is the reference backend's own noise. It
is measured with the same contexts at `--concurrency 1 12`.

```bash
# contexts from saved episodes (logs/ is not in git: contexts hold transcripts)
uv run simtest equiv contexts logs/colab-g4-s5-c3r logs/colab-g4-s5-c3b2 logs/colab-g4-s4-flows --out logs/equiv/contexts.jsonl
# on the G4 (servers up, tunnel open, no episodes running): one request at a time, then 12
uv run simtest equiv score logs/equiv/contexts.jsonl --agent-url http://localhost:18000/v1 \
    --simulator-url http://localhost:18001/v1 --concurrency 1 12 --out logs/equiv/g4 --label "G4, session N"
# on the other backend, one at a time
uv run simtest equiv score logs/equiv/contexts.jsonl --agent-url http://localhost:18000/v1 \
    --simulator-url http://localhost:18001/v1 --out logs/equiv/hopper --label "Hopper, 2x H100"
# the gap, the noise floor and the pass rule
uv run simtest equiv compare logs/equiv/g4-c1.jsonl logs/equiv/hopper-c1.jsonl --noise logs/equiv/g4-c12.jsonl
```

**Pass rule (DRAFT, 02-11-2026; to be frozen in the log before the first check).** For each model,
the gap between the two backends, both scored one request at a time, is no bigger than the G4's own
batching noise (one at a time against 12 at once) on the same contexts:

- mean absolute difference in log-prob: the gap's is no larger than the noise's;
- top-1 agreement: the gap's is no lower than the noise's.

Both models must pass: a comparison that scored only one is INCOMPLETE. If any context has different token IDs, the check fails. If a context is
missing or errored in any file, the check is INCOMPLETE, never PASS. `compare` also prints the
ratio of the two means and a paired bootstrap interval of each difference over contexts. Those
are for reading, not part of the rule.

What it doesn't cover:
- It compares the models' arithmetic, not episodes. At temperature 1, two episodes differ anyway.
- The noise floor batches prefills only. The episodes also batched decoding, which the check
  doesn't reproduce.
- If the agent's noise floor is close to zero, only identical kernels pass, and a G4 (sm_120) and
  an H100 (sm_90) use different FP8 kernels. That case goes back to the rule.
- No KL: the servers return only the top-k log-probs.

Practicalities (unmeasured; check on the first run):
- **Never while episodes run.** The check's requests would join the episodes' batches.
- **Memory.** Prompt log-probs need the full vocabulary's logits for each prefill chunk, outside
  vLLM's own memory pool. That is 248K entries for each of up to 8,192 tokens: several GB. On the
  G4 the two servers hold 94% of the card. If a scoring request runs a server out of memory,
  restart both servers for the check with smaller fractions (`AGENT_MEM`, `SIM_MEM`). The
  fractions set the KV cache's size, not the arithmetic.
- **Size.** The 30 contexts from sessions 4 and 5 (12 agent, 18 simulator; prompts of 12K–78K
  tokens) hold about 1.1M tokens. A pass prefills each once, so it should take minutes. The score
  files are a few tens of MB.
- **64K backends.** Contexts longer than the server's context error, and the check is INCOMPLETE.
  `--max-tokens 60000` cuts every context to the same first 60K tokens.

Code: `src/simtest/equiv.py`. Tests: `tests/test_equiv.py` (mocked servers, including one whose
numbers drift with the number of requests in flight).

## The simulator's quantization: fp8 or fp8_per_channel

The simulator ships bf16 weights, and vLLM quantizes them at load. Which scheme is Victor's call
before the pilot (LOG.md, 02-11-2026): it changes the simulator in every condition, so the pilot
and N3 must use the same one. The rule is to adopt `fp8_per_channel` only if it is no worse.

**What each one does** (vLLM v0.32.0 source, checked 02-11-2026; paths under `vllm/`):

| | `fp8` (default) | `fp8_per_channel` |
| --- | --- | --- |
| Name | deprecated alias of `fp8_per_tensor` (`model_executor/model_loader/weight_utils.py:351-358`) | an online shorthand (`config/quantization.py:201-206`) |
| Dense layers (attention `qkv_proj`, `o_proj`; GDN `in_proj_qkvz`, `in_proj_ba`, `out_proj`; the shared expert) | one weight scale for the whole merged projection; activations per token (`model_executor/layers/quantization/online/fp8.py:204-262`) | one weight scale per output channel; activations per token (`online/fp8.py:391-454`) |
| Routed experts (MoE) | one weight scale per expert and projection; activations per tensor over the whole batch, both expert GEMMs (`online/fp8.py:517-523, 608-662`; `fused_moe/experts/triton_moe.py:340-351`) | weight scales per output channel; activations per token, both expert GEMMs (`online/fp8.py:763-846`; `fused_moe/prepare_finalize/no_dp_ep.py:27-32`; `triton_moe.py:522-526`; `fused_moe/utils.py:323-337`) |
| Not quantized, both | the router, the shared-expert gate, the GDN `conv1d`, embeddings and `lm_head` (`models/qwen3_next.py:191-201`; `layers/mamba/gdn/qwen_gdn_linear_attn.py:503-508`; `online/base.py:388-411`) | |
| Dense kernel, G4 (sm_120) and H100 (sm_90) | CUTLASS `scaled_mm` (`kernels/linear/__init__.py:443-452`; `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu:145-158`) | the same; Marlin, the weight-only fallback, is refused (`online/fp8.py:430-438`) |
| MoE kernel, G4 | Triton (the 29-10-2026 G4 log) | Triton (`fused_moe/experts/triton_moe.py:157-163`): the CUTLASS grouped GEMM isn't built for sm_120 (`scaled_mm_entry.cu:176-194`), and FlashInfer's and DeepGEMM's kernels don't take this scheme (`experts/flashinfer_cutlass_moe.py:146-181`, `experts/trtllm_fp8_moe.py:165-222`, `experts/deep_gemm_moe.py:178-188`) |
| MoE kernel, H100 (from the source; never run) | Triton: the CUTLASS MoE is left out for this scheme (`fused_moe/oracle/fp8.py:510-512`) | CUTLASS grouped GEMM by default (`oracle/fp8.py:86-102`; `experts/cutlass_moe.py:311-328`); `serve.sh` passes `--moe-backend triton` (`config/kernel.py:294`; `oracle/fp8.py:450-477`), so both GPUs run Triton, as with `fp8` |

- No batch-wide maximum is left in `fp8_per_channel`'s quantization: every activation scale is one
  token's (`_custom_ops.py:1880-1883`), and weight scales are fixed at load.
- vllm-lens and `EAGER=1` don't depend on the scheme: the hooks read decoder-layer outputs. The
  interp backend takes the scheme as an argument and puts it in the fingerprint
  (`src/simtest/interp/vllm_backend.py:104-123`), so vectors built on one scheme are never used on
  the other.

**What stays batch-dependent with either scheme.** So `max_connections` stays pinned at 12:
- The Triton MoE's tile sizes are chosen from the number of tokens in the batch
  (`fused_moe/fused_moe.py:1374-1398, 1419-1446`). vLLM's batch-invariant mode replaces them with
  one fixed tile (`fused_moe.py:1305-1312`).
- FlashAttention splits the KV by a heuristic that depends on the batch (`v1/attention/backends/flash_attn.py:829-863`).
- A long prompt's prefill chunks end where the step's token budget runs out, so their boundaries
  depend on the other requests (`v1/core/sched/scheduler.py:710-712`).
- `VLLM_BATCH_INVARIANT=1`, vLLM's fix for all three, refuses this model: the GDN layers' backend
  doesn't support it (`v1/attention/selector.py:242-246`; `v1/attention/backend.py:206-207`).
- No seed is sent, at temperature 0.6, so a rerun samples other tokens anyway.

What `fp8_per_channel` removes is the systematic part: under `fp8`, a bigger batch has a larger
maximum, so every token's MoE activations are rounded to a coarser grid.

### The measurement, in the pilot session's first hour

On the reference G4, before any episode, one setting after the other (both simulators don't fit
next to the agent). `fp8` goes first: if `fp8_per_channel` is adopted, no restart is needed.

```bash
# before the session, on the laptop (offline): the contexts from sessions 4 and 5
uv run simtest equiv contexts logs/colab-g4-s5-c3r logs/colab-g4-s5-c3b2 logs/colab-g4-s4-flows --out logs/equiv/contexts.jsonl
mkdir -p logs/simquant/fp8 logs/simquant/fp8_per_channel
# the session's usual setup: serve.sh with MIX=c3 (fp8), the tunnel; no episodes yet
SIM_LINES="grep -h -E \"Quantizing|Fp8 MoE backend|Selected .* for Fp8\" /content/probe/simulator.log; grep -- \"--served-model-name simulator\" /content/probe/versions.txt | tail -1"
# 1. fp8: the server's lines, the noise floor of both models, fidelity and speed
eval "$SSH '$SIM_LINES'" > logs/simquant/fp8/server.txt
uv run simtest equiv score logs/equiv/contexts.jsonl --agent-url http://localhost:18000/v1 \
    --simulator-url http://localhost:18001/v1 --concurrency 1 12 --out logs/simquant/equiv/fp8 --label "G4, session N, fp8"
uv run python -I explore/sim_quant.py fidelity fp8
# 2. fp8_per_channel: restart the simulator alone (the agent keeps serving), then the same
eval "$SSH 'MIX=c3 SIM_QUANT=fp8_per_channel SIM_ONLY=1 bash -s'" < explore/colab_probe/serve.sh
eval "$SSH '$SIM_LINES'" > logs/simquant/fp8_per_channel/server.txt
uv run simtest equiv score logs/equiv/contexts.jsonl --simulator-url http://localhost:18001/v1 \
    --concurrency 1 12 --out logs/simquant/equiv/fp8_per_channel --label "G4, session N, fp8_per_channel"
uv run python -I explore/sim_quant.py fidelity fp8_per_channel
# 3. the rule; if it keeps fp8 (KEEP or INCOMPLETE), restart the simulator as fp8 before the episodes
uv run python -I explore/sim_quant.py decide
eval "$SSH 'MIX=c3 SIM_ONLY=1 bash -s'" < explore/colab_probe/serve.sh
```

- **Fidelity:** the bake-off's forced mode (LOG.md, 26-10-2026): each of the 84 recorded steps
  predicted from the real history, one sample each, 12 requests at once. `sim_quant.py` scores it
  with `bakeoff_score.py`, and with the graded fidelity (`simtest.post.fidelity`) if the checkout
  has it; that one is reported, not in the rule.
- **Batching noise:** `simtest equiv`'s noise floor of the simulator, 1 against 12 at once, for each
  setting. The agent's floor under `fp8` is kept too: it is the reference for a later move to Hopper.
- **Speed:** generation tokens per second with 12 running and none waiting, from the server's
  `/metrics`, during each fidelity pass. The agent is idle then, so episodes will be slower.
- **Cost:** about 45 minutes of the G4, so about 7.5 units at 10.05 an hour (7–9): two forced
  passes of 14–18 minutes each (the 29-10 pass's 339K generated tokens replayed through 12 slots at
  400–500 tokens a second), a few minutes of `equiv` per setting, about 4 minutes per simulator
  restart, and 4 more if it reverts to `fp8`. An `equiv` request that runs a server out of memory
  (above) adds a restart of both servers, about 1 unit.
- **Not covered:** the noise floor batches prefills only, and decoding is where `fp8`'s shared scale
  differs most between 1 and 12 at once, so the measured reduction is a lower bound. One sample
  per step can catch a broken quantization, not a difference of a few thousandths. Triton's
  per-channel path has never run on our sm_120: the first startup is its test.

**Decision rule (DRAFT, 02-11-2026; frozen in LOG.md before the run).** `fp8_per_channel` is
adopted for the pilot and N3, in every condition and phase, if all of these hold. Otherwise `fp8`
is kept.
1. **Fidelity:** on the 81 deterministic forced steps, its mean similarity is at least `fp8`'s
   minus 0.02, and at most 3 of its 84 steps hit the 16K cap (the 29-10-2026 rule). 0.02 is about
   2 standard errors of the paired difference between two forced passes: 0.0097, from the MLX
   (26-10) and G4 (29-10) runs. That pair also differed in quantization and engine, so it is an
   upper bound on the run-to-run spread.
2. **Batching noise:** the simulator's mean |difference in log-prob|, 1 against 12 at once, is lower
   than under `fp8`, with the 95% paired bootstrap interval of the difference below zero, and its
   top-1 agreement is no lower.
3. **Complete evidence:** both passes predicted the bake-off's recording (pinned by its hash) and
   scored exactly its 84 steps, none with an error; the four score files come from the simulator
   server on vLLM 0.32.0, each `--label` ends with its setting, and they cover the same simulator
   contexts with the same token IDs; each server shows its scheme on every quantized layer and the
   Triton MoE. Otherwise the verdict is INCOMPLETE, and the pilot keeps `fp8`.

Speed never decides. If `fp8_per_channel` is adopted at under 0.9 of `fp8`'s tokens per second
(or its speed is missing), Victor confirms the rescaled budget before the episodes.

**The pin at 12, under each outcome:**
- `fp8` kept: 12 at once for every condition and session, as now.
- `fp8_per_channel` adopted: still 12 for the pilot and N3 (the batch dependence above remains).
  Another number at once, for example on Hopper, needs a `simtest equiv` check recorded in LOG.md
  first: the noise floor at 1 against N no larger than at 1 against 12, for both models. Under
  `fp8` that check isn't offered: a bigger batch coarsens the simulator's MoE.

Code: `explore/sim_quant.py`. Tests: `tests/test_sim_quant.py`, and the options in
`tests/test_serve_script.py`.
