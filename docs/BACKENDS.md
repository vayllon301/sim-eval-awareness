# Backends: serving the models, and checking two backends agree

Every condition of an experiment runs on one backend: the same weights, quantization, engine,
chat template and settings. The simulator adds one more: its output depends on what else shares
the batch ([below](#the-simulators-quantization-fp8_per_channel)). How many episodes run at once
(`max_connections`) is part of the backend too.

This page covers how `explore/colab_probe/serve.sh` serves both models on each GPU we may get, how
`simtest equiv` checks whether two backends compute the same models before work moves between
them, the simulator's FP8 scheme, and the setup check that runs before a run's first episode.
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
- Both models come down at fixed commits, the ones every session's `versions.txt` records (29-10 to
  01-11-2026): `Qwen/Qwen3.8-27B-FP8` at `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` and
  `Qwen/Qwen-AgentWorld-35B-A3B` at `60d2b0434a53d2e62a7c00a489586815d94ebffb`. They are not settings:
  a new upstream commit is another backend. `explore/interp/gpu_env.sh` downloads on its own, at the
  same two, so the served files, the vectors and the readings are one revision (a test keeps them
  equal). The pin changes no `vllm serve` line, and each commit was its repo's head through those
  sessions, so `SIM_QUANT=fp8` still gives the old sessions' lines on the old sessions' files.
  `read_run.sh` pins the SAEs the same way: `Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50` at
  `13d4221569f7ca5d3c1e605e3e3dc95117e4807c`, recorded in the readings' `versions.txt`.
- `WORK` is the work directory. It defaults to `/content` on Colab and must be given anywhere else.
  It needs about 110 GB free: the downloads are 29 + 67 GB.
- Off Colab, vLLM goes into a venv (`$WORK/venv`, or `VENV`), and Hugging Face's cache goes into
  `$WORK/hf-home` unless `HF_HOME` is set. `VENV=` (empty) forces the system install that Colab
  uses.
- `EAGER=1` serves on the steering engine's settings (N4). `AGENT_MEM`, `SIM_MEM`, `AGENT_LEN`,
  `SIM_LEN` and `MAX_SEQS` override the profile.
- `SIM_QUANT` is `fp8_per_channel` (the default, in every condition and phase from the pilot on) or
  `fp8` (by name only, with a warning: every session before the pilot, for reproducing those).
  Nothing else is accepted. [Below](#the-simulators-quantization-fp8_per_channel).
- `session.sh` and `submit_chunks.sh` always name `fp8_per_channel` to `serve.sh`. They refuse `fp8`
  for the setup check and for episodes, even if it is left over in the shell.
- `$WORK/probe/server.txt` holds what the setup check reads: the GPU, both servers' served lines,
  what vLLM says it quantized and the MoE kernel it chose.
- `SIM_ONLY=1` restarts the simulator alone next to a running agent (it stops only the simulator of
  the same `WORK`).
- **80 GB, one card.** The agent's weights and its KV cache at 128K don't fit next to the
  simulator, so both models get 64K. Long C3 episodes can pass 64K: session 5's largest simulator
  call was 69K tokens. Those calls then fail with a context-length error. An experiment on this
  profile runs every condition on it, and its episodes never mix with the G4's.
- `DRY_RUN=1` prints the profile and both `vllm serve` lines, and does nothing else. `GPU_MIB`
  and `GPU_COUNT` stand in for `nvidia-smi`. `tests/test_serve_script.py` pins `SIM_QUANT=fp8`'s G4
  lines to the ones the script used before Hopper support, byte for byte, pins the default as the
  same lines with the new scheme, and checks the dry run against what the script really starts.

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
is measured with the same contexts at `--concurrency 1 12`, in every setup check.

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
both backends on the same scheme, the gap between them, both scored one request at a time, is no
bigger than the reference's own batching noise (one at a time against 12 at once) on the same contexts:

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

## The simulator's quantization: fp8_per_channel

The simulator ships bf16 weights, and vLLM quantizes them at load. It runs `fp8_per_channel` in
every condition and phase from the pilot on: episodes (`EAGER=1` too), vectors, steering and
readings (Victor, LOG.md, 02-11-2026). Every session before the pilot ran `fp8`, and those numbers
stay labelled as development. The agent doesn't change: it is its own FP8 checkpoint.

**What each one does** (vLLM v0.32.0 source, checked 02-11-2026; paths under `vllm/`):

| | `fp8` (every session before the pilot) | `fp8_per_channel` (from the pilot on; `serve.sh`'s default) |
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
maximum, so every token's MoE activations are rounded to a coarser grid. Its expected cost, from the
source and not measured: about the same speed (same kernels and tile settings, only the scale loads
differ), and about 0.12 GB of scales.

### The setup check, before a run's first episode

`explore/sim_quant.py` checks that the session really serves the final scheme and that it works. It
isn't a comparison: no `fp8` pass runs. `session.sh`'s `setup` step and `run_chunks.sh` run it on the
laptop, through the tunnel, before any episode, in this order:

1. **The server's scheme lines.** `serve.sh` writes them to `$WORK/probe/server.txt`: the GPU, both
   servers' served lines, and what vLLM says it quantized and which MoE kernel it chose. They come
   first, so a wrong server costs no GPU time.
2. **The noise floors.** `simtest equiv score --concurrency 1 12`, both models, the 30 contexts of
   sessions 4 and 5.
3. **One forced bake-off pass at 12 at once.** Each of the 84 recorded steps is predicted from the
   real history (LOG.md, 26-10-2026), with `bakeoff_score.py`'s scores. While it runs, the server's
   `/metrics` gives the tokens per second with 12 running and none waiting. The graded fidelity
   (`simtest.post.fidelity`) is reported, and is not in the rule.
4. **The rule.** `sim_quant.py check` writes `check.json`. Exit 0 PASS, 1 FAIL, 2 INCOMPLETE.

Before each `simtest run`, on every VM and every Hopper job, `sim_quant.py gate` runs the last check.
A PASS on this commit must exist, this server's own lines must show the scheme, its GPU must be the
check's, and both servers must be served as the check saw them: memory, context length, engine and
every other flag. Only the card of the job, the model's directory, the host and the port may differ.
A pass isn't measured twice, so this is what holds it to the settings it measured. Then it appends one
line to the run's record, `logs/<name>/backend.jsonl`: the scheme, both served lines, the GPU, the
commit and the check's numbers. If the gate refuses, no episode starts.

```bash
# before the session, on the laptop (offline): the contexts from sessions 4 and 5
uv run simtest equiv contexts logs/colab-g4-s5-c3r logs/colab-g4-s5-c3b2 logs/colab-g4-s4-flows --out logs/equiv/contexts.jsonl
# the session runs the check itself, then the episodes
MIX=c3 ADMIT_FOR=14 explore/colab_probe/session.sh configs/coding/pilot.yaml 180 2>&1 | tee /tmp/session.log
```

It needs `logs/equiv/contexts.jsonl` and the pinned recording in `logs/bakeoff/`; the scripts refuse
to start without them. Its files stay in `logs/setup/<name>-<backend>-<commit>/` (`SETUP_DIR`), so a
pass is never used for another commit or another backend, and a resumed session doesn't pay for it
twice.

**Cost:** about 25 minutes of the G4, so about 4 units at 10.05 an hour: 14–18 minutes for the forced
pass (the 29-10 pass's 339K generated tokens through 12 slots at 400–500 tokens a second) and a few
minutes of `equiv`. This is an estimate, not a measurement. An `equiv` request that runs a server out
of memory (above) stops the session. The check can then run alone with `STEPS=setup AGENT_MEM=...
SIM_MEM=...`, but its PASS admits no episodes: they never run on those fractions, and the gate refuses
any others. What comes next is Victor's to decide.

**The rule (frozen in LOG.md before the pilot).** No `fp8` pass runs in the session, so the bars are
fixed numbers from earlier sessions. PASS needs all four:

1. **Scheme.** The served line says `--quantization fp8_per_channel --moe-backend triton`, every
   quantized layer kind says `fp8_per_channel_static`, and the MoE kernel is `TRITON`.
2. **Fidelity.** On the 81 deterministic forced steps, the mean similarity is at least 0.892. That is
   0.912 (the G4's forced pass of 29-10-2026, on `fp8`, 8 at once) minus 0.02, about 2 standard errors
   of the paired difference between two forced passes (0.0097). That pair also differed in backend,
   so it is an upper bound on the run-to-run spread.
3. **Truncation.** At most 3 of the 84 steps hit the 16K cap (the 29-10-2026 rule; that pass had 0).
4. **Complete evidence.** The pass predicted the bake-off's recording (pinned by its hash) and scored
   exactly its 84 steps, none with an error. Both `equiv` files exist, come from the agent and
   simulator servers on vLLM 0.32.0, have a label ending in the scheme, used the same contexts file at
   1 and at 12 at once, and score every context of both models on identical token IDs. The server's
   lines are all there.

The verdicts:
- **FAIL:** a line shows another scheme or kernel, or bar 2 or 3 is missed.
- **INCOMPLETE:** bar 4 is missed, so the evidence isn't there. A layer kind the check can't read, or kinds
  that don't add up to the layers quantized, count as a line that isn't there: never a pass, and never a
  FAIL for a good server in a format we didn't expect.
- Either way **no episode runs, and the session stops**: `session.sh` stops the VM, and `run_chunks.sh`
  cancels every job.

The noise floors have no bar on their size. The old bar was "lower than under `fp8`", and no `fp8`
floor was ever measured, so there is nothing to hold them against. They are recorded as this
backend's reference. A broken quantization shows in bars 1 and 2.

Speed never decides. Under 405 tokens per second at 12 at once, or with no speed at all, the episodes
wait until Victor confirms the rescaled budget. 405 is 0.9 of 450, the middle of the 400–500 the
02-11 estimate used for this pass. After he confirms, the same command with `SPEED_OK=1` starts them,
and the record keeps that. `KEEP=1` keeps the VM or the jobs for that restart.

**On a failure:**
- **It never falls back to `fp8`.** Going back would be a new decision in LOG.md, with the thesis's
  chapter 4 and the OSF text following it.
- **INCOMPLETE:** fix what is missing and run the same command. The floors are scored again, and the
  forced pass keeps its finished steps.
- **FAIL:** Victor decides. The check stays where it is, and the scripts refuse to run on it. A new
  check, after a fix, runs on the new commit or in a new `SETUP_DIR`, and both are reported.
- **No re-sampling past a bar.** Bar 2 is about 2 standard errors, so about 1 clean setup in 40 misses
  it by chance. With clean lines and no cause found, Victor may ask for one more pass, reported beside
  the first.

**The pin:** `max_connections` stays 12 in the pilot and N3, on every backend. Another number at once
would be a new decision, with a `simtest equiv` check recorded in LOG.md first: the noise floor at 1
against N no larger than at 1 against 12, for both models.

**What it can't show:**
- That `fp8_per_channel` lowered the batch noise. There is no `fp8` floor, so that claim rests on the
  source (no batch-wide maximum is left), not on a measurement.
- The noise floor batches prefills only, and decoding is where `fp8`'s shared scale differed most.
- One sample per step can catch a broken quantization, not a difference of a few thousandths.
- That fidelity equals `fp8`'s. Bar 2 crosses schemes and 8 against 12 at once: it is a floor against
  a broken setup.
- The episodes' speed. The forced prompts are short and the agent is idle, so episodes will be
  slower. 405 is a rough bar.
- Triton's per-channel path has never run on our sm_120 or sm_90: the first startup is its test.

### Hopper, and what a check against the G4 can claim

Everything above runs the same way on Hopper, with these differences:
- The first usable job runs the check; every job has its own gate. A job on another GPU model than
  the check's is refused, because that is another backend.
- The check takes about 25 minutes from the first job. `run_chunks.sh` works out the admission window
  after it, so the room for the longest episode stays. Add it to that job's `SETUP_HOURS` if you can.
- Bar 2 and the 405 tokens per second are G4 numbers. The first is a floor, as above. The second is
  about the plan: `simtest plan --durations` uses the G4's durations, so a lower rate means the
  chunks are optimistic, and Victor confirms a new plan.
- The MoE kernel on the H100 is read from the source and has never run (table above). If its log line
  differs from the G4's, the check says INCOMPLETE or FAIL, and nothing is lost but the job's time.

**What exists to compare with.** `simtest equiv` has never run on a GPU, so the G4 has no score files
on any scheme. The simulator's G4 numbers (the forced pass, the rates, session 4's vectors, the episodes)
are all on `fp8`. So a check of Hopper's simulator against them crosses the scheme as well as the
hardware.

| What the check is | What it can claim | What it can't claim |
| --- | --- | --- |
| Hopper's setup check alone | This backend serves the final scheme on the kernels named, fidelity isn't below the floor, and these are its own noise floors | That it computes what the G4 computes |
| Bar 2 against the G4's 0.912 | Hardware and scheme together didn't break the simulator by more than about 0.02, at one sample per step | Which of the two moved it, or that either moved it by nothing. A miss doesn't say which either |
| Token IDs on the same contexts | The same rendering (tokenizer and chat template), since this doesn't depend on the scheme | Anything about the arithmetic |
| `equiv compare`, simulator, Hopper `fp8_per_channel` against any G4 `fp8` file | Nothing as a backend check. The gap includes the scheme by construction, because the weights are rounded differently. A pass would be weak too: the bar would be `fp8`'s noise, which we expect to be the larger | That the hardware is equivalent |
| `equiv compare`, agent, Hopper against the G4 | The agent's scheme doesn't change, so this is a clean hardware check, once G4 scores exist. The near-zero floor caveat above still holds | Anything about the simulator |
| `equiv compare`, simulator, both on `fp8_per_channel` | A clean hardware check, under the pass rule above | |

**What each use needs:**
- **The pilot and N3 on Hopper, all conditions:** no G4 comparison. Every condition runs on one
  backend, so Hopper is the reference, and its setup check gives the floors. The pilot counts either
  way.
- **The simulator's G4 numbers:** none carry over. Session 4's vectors aren't reused (LOG.md,
  02-11-2026), and the development numbers keep their "development" label.
- **The agent's G4 numbers:** they carry over only after the clean agent check, which needs G4 scores
  that don't exist yet.
- **Vectors:** built on the backend that runs the episodes (PLAN.md: "Same FP8 weights for everything"),
  so session 4's file is not used on Hopper. `PARTS="vectors check proj"` on `submit_read.sh` builds them
  in the job, with the layer scan down to 30 (session 4 chose 36, the lowest it scanned), the training
  twin and the checks of `gpu_smoke.py`'s lens step, and they come back in `readings/vectors/`. Read
  `lens.json` against the smoke test's rules before any reading counts. `read()` compares the file's
  `model.device` with the backend's and refuses another card: the weights fingerprint can't tell, since
  both load the same files. Another engine on the same card passes, as the `hf` fallback needs. A file
  with no device is refused. `--allow-other-backend` (`ALLOW_OTHER_BACKEND=1`) reads them anyway, and
  each reading and the report say so.
- **Moving work between the G4 and Hopper** (a pilot on one and N3 on the other): first score both
  backends on `fp8_per_channel`. `STEPS=setup` on a G4 is about 25 minutes plus the VM's setup, so about
  6 units (an estimate). It spends units, so it needs Victor's OK and a reason in the log.

Code: `explore/sim_quant.py`. Tests: `tests/test_sim_quant.py`, and the scheme and the steps in
`tests/test_serve_script.py`, `tests/test_session_script.py` and `tests/test_hopper_scripts.py`.
