# Does a simulated environment change what an AI agent does?

Code and experiments for a master's thesis at La Salle (Barcelona), by Victor Ayllón,
tutored by Jordi Escayola.

**The question.** Labs test AI agents in environments where another LLM plays the
tools. Does an LLM-simulated environment, in place of a real one, change the evaluated
model's *evaluation awareness* or its behaviour? And what in the simulation drives it?

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
| `src/simtest/` | The testbed, built on [Inspect AI](https://inspect.aisi.org.uk/) (pinned 0.3.281). Reference: [`docs/TESTBED.md`](docs/TESTBED.md) |
| `envs/` | Synthetic repos and their Docker images |
| `scenarios/`, `configs/` | Tasks, and the configs that were run |
| `explore/` | Exploratory scripts (simulator bake-off, Colab probe). Not part of the testbed |
| `tests/` | Offline tests (mock models, no API calls) |
| `experiments/<run-name>/` | One folder per finished experiment: findings, report, config, logs |

The testbed started with a first design (rule-following in a simulated office, runs
M1–M1c). Its code and scenarios are kept so those runs stay reproducible. The
coding-agent environment is being built on top of it.

Comments that cite `LOG.md` or `PLAN.md` point to the project's private research log.
Everything a result rests on is published under `experiments/`.

## Quick start

```bash
uv sync
uv run pytest -q                                          # offline tests
uv run simtest demo                                       # scripted agents, no API calls
cp .env.example .env                                      # then add the keys it lists
uv run --env-file .env simtest run configs/<name>.yaml    # paid; needs a clean git tree
uv run simtest summarize logs/<name>                      # rates, CIs, exclusions
uv run inspect view --log-dir logs/<name>                 # transcripts
```

- Runs refuse a dirty working tree, so every result maps to a commit.
- Re-running the same `run` command resumes it.
- Logs go to `logs/<name>/` and are not in git. Published logs are in `experiments/`.

## Status

Work in progress. No licence has been chosen yet; until one is, all rights are reserved.
