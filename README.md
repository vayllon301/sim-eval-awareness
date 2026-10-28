# Does a simulated environment change what an AI agent does?

Code and experiments for a master's thesis at La Salle (Barcelona), by Victor Ayllón,
tutored by Jordi Escayola.

**The question.** Labs test AI agents in environments where another LLM plays the
tools. Does an LLM-simulated environment, in place of a real one, change the evaluated
model's *evaluation awareness* or its behaviour? And what happens inside the two models:
how do the emotion representations and sparse features of the agent and of the simulator
relate to the agent's awareness and behaviour?

**The setup.** A coding agent fixes tickets in small synthetic repos with a `bash`
tool. Only the tool outputs change:

| | What the agent sees |
| --- | --- |
| **C0** | The real output of a real Docker container |
| **C1** | The real output, rewritten by an LLM that sees only the current command |
| **C2** | As C1, and the LLM also sees the history |
| **C3** | Output invented by an LLM, with no real output |

The agent's commands also run in a real "shadow" container in every condition, so
behaviour is measured on real state.

## What is here

| Path | What |
| --- | --- |
| `src/simtest/` | The testbed, built on [Inspect AI](https://inspect.aisi.org.uk/) (pinned 0.3.281). The coding family is `src/simtest/coding/`. Reference: [`docs/TESTBED.md`](docs/TESTBED.md) |
| `src/simtest/interp/` | Interpretability tools: emotion and evaluation-awareness vectors, a post-hoc episode reader, Qwen-Scope SAEs, activation steering (optional extra, MLX on Apple Silicon). Reference: [`docs/INTERP.md`](docs/INTERP.md) |
| `envs/` | The dev-machine image (`envs/devbox/`: a real init, reached over SSH), one image per scenario repo (`envs/coding/`), and the bake-off's `pyrepo` |
| `scenarios/`, `configs/` | Tasks, and the configs that were run. Coding scenarios and their benign twins: `scenarios/coding/` |
| `explore/` | Exploratory scripts (simulator bake-off, Colab probe, interpretability scripts in `explore/interp/`). Not part of the testbed |
| `tests/` | Offline tests (mock models, no API calls) |
| `experiments/<run-name>/` | One folder per finished experiment: findings, report, config, logs |

The testbed started with a first design (rule-following in a simulated office, runs
M1–M1c). Its code and scenarios are kept so those runs stay reproducible. The
coding-agent environment is built on top of it.

Comments that cite `LOG.md` or `PLAN.md` point to the project's private research log.
Everything a result rests on is published under `experiments/`.

## Setup

You need [uv](https://docs.astral.sh/uv/) (it installs the pinned Python 3.14 itself) and,
for the coding environment, [Docker](https://www.docker.com/). Everything below is
free and makes no API calls unless it says "paid".

```bash
git clone https://github.com/vayllon301/sim-eval-awareness
cd sim-eval-awareness
uv sync
uv run pytest -q            # offline tests, about a minute
uv run simtest demo         # scripted agents and mock simulators, no API calls
```

### The coding environment

The coding family runs a coding agent on a Debian dev box (a real init, reached over SSH
as a login shell), fixing a ticket in a small repo. There are four scenarios, each with a
benign twin without the temptation; `pricing_cap` is the old `pyrepo`. Every command also
runs on the real box in every condition, and the behaviour checks read its final state.

```bash
uv run simtest images                                  # build the dev-box and scenario images (Docker)
uv run simtest snapshot --check                        # the C2/C3 snapshots still match a fresh box
uv run --env-file .env simtest run configs/coding/dev_c0.yaml   # paid: needs the model servers
uv run simtest summarize logs/coding-dev-c0
```

Models are OpenAI-compatible endpoints (`rawchat/<served-name>` with a `base_url`): vLLM on
Colab through the tunnel, or OpenRouter for smoke tests only. See `docs/TESTBED.md`,
"Coding family".

The bake-off's repo: `envs/pyrepo` is a small Python service whose CI is red: one test still expects an old
EU discount cap. The agent's job is to fix it.

```bash
docker build -t pyrepo envs/pyrepo
docker run --rm --network none pyrepo bash -lc 'python -m pytest -q'   # 1 failed, 12 passed
uv run python -I explore/bakeoff_record.py logs/bakeoff                # real outputs + the simulator's snapshot
```

### One episode on your own machine

`explore/colab_probe/episode.py` runs one coding-agent episode against any
OpenAI-compatible server, for example [LM Studio](https://lmstudio.ai/) on port 1234.
In `real` mode the agent's commands run in a fresh `pyrepo` container (C0). In `sim`
mode the simulator invents every output (C3); it needs the snapshot from
`bakeoff_record.py` and a simulator server on port 18001.

```bash
AGENT_URL=http://localhost:1234 AGENT_MODEL=qwen/qwen3.8-27b MAX_STEPS=6 \
  uv run python -I explore/colab_probe/episode.py real logs/colab-probe/smoke-real.jsonl
```

No local GPU? The same episode runs through [OpenRouter](https://openrouter.ai/) for a
cent or two (put `OPENROUTER_API_KEY` in `.env`). Use it to test the setup only, never
for an experiment: the provider behind a model can change between calls, so the weights,
quantization and chat template aren't fixed.

```bash
cp .env.example .env    # then add your OpenRouter key
AGENT_URL=https://openrouter.ai/api AGENT_MODEL=qwen/qwen3.8-27b MAX_STEPS=6 \
  uv run --env-file .env python -I explore/colab_probe/episode.py real logs/colab-probe/smoke-or.jsonl
```

### Interpretability (optional)

```bash
uv sync --extra interp      # MLX, safetensors, transformers
uv run --extra interp python -I explore/interp/build_vectors.py <model-dir> <texts.jsonl> <out.safetensors> --layers 36,39,42,45,48
```

The other scripts (the episode reader, the SAE transfer check, the steering check) are in
[`docs/INTERP.md`](docs/INTERP.md), "Scripts". Numbers from quantized laptop weights are
for development only; the experiment's backend is bf16 on a GPU.

### Both models on one GPU (Colab)

`explore/colab_probe/serve.sh` serves the agent (Qwen3.8-27B) and the simulator
(Qwen-AgentWorld-35B-A3B) with vLLM on one 80–96 GB GPU. Its header has the full
sequence: start a Colab VM, run the script, open the SSH tunnel to ports 18000 and 18001,
and stop the VM at the end. Colab compute units are paid.

### The testbed runner

```bash
cp .env.example .env                                      # then add the keys it lists
uv run --env-file .env simtest run configs/<name>.yaml    # paid; needs a clean git tree
uv run simtest summarize logs/<name>                      # rates, CIs, exclusions
uv run inspect view --log-dir logs/<name>                 # transcripts
```

- Runs refuse a dirty working tree, so every result maps to a commit.
- Re-running the same `run` command resumes it.
- Logs go to `logs/<name>/` and are not in git. Published logs are in `experiments/`.

## Status

Work in progress: the coding-agent environment and the interpretability tools are built
and reviewed; the GPU backends and the first experiment come next. Code under the
[MIT licence](LICENSE). Published with the agreement of the thesis tutor.
