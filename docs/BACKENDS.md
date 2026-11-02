# Backends: serving the models, and checking two backends agree

Every condition of an experiment runs on one backend: the same weights, quantization, engine,
chat template and settings. The simulator adds one more: its FP8 MoE scales activations per
tensor over the whole batch ([`INTERP.md`](INTERP.md), "GPU backends"), so its output depends on
what else shares the batch. How many episodes run at once (`max_connections`) is part of the
backend too.

This page covers how `explore/colab_probe/serve.sh` serves both models on each GPU we may get, and
how `simtest equiv` checks whether two backends compute the same models before work moves between
them.

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
ssh -N -J <login> -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001 <node> &
```

The servers listen on the machine's loopback only. The tunnel brings them to the laptop's ports
18000 (agent) and 18001 (simulator), where the configs expect them.

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
   - where the reply starts;
   - whether the prompt has as many tokens as the episode's server counted (`prompt_matches_log`).
   A failed context is recorded as an error row, never dropped.
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

Both models must pass. If any context has different token IDs, the check fails. If a context is
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
