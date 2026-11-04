# GPU sessions: preflight, session.sh, the watchdog, Hopper chunks, reading a run

GPU time is paid in Colab units (a G4 bills 10.05 an hour while the VM lives) or in Hopper's
queue. These scripts cut the time a GPU waits for nothing. They never change an episode: the
servers are `serve.sh`'s, and the episodes are `simtest run`'s. Every spending guard fails safe:
when unsure, it stops the VM or refuses to start.

Inspect, Docker and the logs run on the laptop. Only vLLM runs on the GPU machine
([`BACKENDS.md`](BACKENDS.md)).

## 1. The preflight

```bash
uv run simtest preflight configs/coding/pilot.yaml      # offline: mock models, the real Docker images
```

It runs the whole config with mock models and writes `logs/preflight/<name>.ok` with the commit it
passed on. `session.sh`, `submit_chunks.sh` and `run_chunks.sh` refuse to start unless that file
holds the current commit and the tree is clean. A config that fails on the laptop never reaches a
GPU.

## 2. Colab: `explore/colab_probe/session.sh`

```bash
DRY_RUN=1 MIX=c3 ADMIT_FOR=14 explore/colab_probe/session.sh configs/coding/pilot.yaml 180   # prints, does nothing
MIX=c3 ADMIT_FOR=14 explore/colab_probe/session.sh configs/coding/pilot.yaml 180 2>&1 | tee /tmp/session.log
```

One VM session does all of a config's GPU work, so the setup is paid once (741 s in session 4):

1. Checks first: the preflight, a clean tree, the env file, the equivalence contexts, the bake-off's
   recording, free ports 18000 and 18001, a readable balance. A session with episodes or the setup
   check refuses `SIM_QUANT=fp8`, and episodes without a passed check are refused. Any failure means
   no VM.
2. Creates the VM (or attaches session `SESSION` if it exists), and starts the watchdog with the
   cap (the second argument, in units).
3. Runs `serve.sh` once with the given `MIX` (required: `c3` or `default`) and the simulator's scheme
   by name, opens the tunnel and waits for both `/v1/models`.
4. Runs the steps in `STEPS`, always in this order (default: `setup run`; `equiv`, `ids`, `readings`
   and `read` run when named):
   - `setup`: the simulator's setup check (BACKENDS.md, "The setup check"), before any episode: the
     server's scheme lines, the `simtest equiv` noise floors of both models, one forced bake-off
     pass at 12 at once with its speed. Anything but PASS stops the session here. A PASS on this
     commit and backend stays in `logs/setup/`, so a resumed session doesn't pay for it twice.
   - `run`: the gate first (a PASS on this commit, and this server's lines show the scheme; it writes
     the scheme to `logs/<name>/backend.jsonl`), then `simtest run`, with `EPOCHS`, `ADMIT_FOR` and
     `CARRY_UNDER` passed on. If the run is incomplete (exit 3), it says so and goes on.
   - `equiv` (only when named): `simtest equiv score` on the contexts, one request at a time and 12
     at once (BACKENDS.md). The setup step already scores them for a run, so this is for a session
     without episodes. Build the contexts first (`simtest equiv contexts`), and keep them: every
     backend is scored on the same contexts.
   - `ids`: `simtest interp export`, then `simtest interp server-ids`: the run's episodes for the
     reader, and the token IDs of the agent server's own renderer for each one. It needs the servers
     up, no GPU time, and it changes no episode. It takes a minute or two.
   - `readings`: stops the servers to free the GPU, sends the code at HEAD (a git bundle), and runs
     `explore/interp/gpu_smoke.sh` (INTERP.md, "GPU smoke test"). It builds the vector files and
     takes about 50 GPU minutes.
   - `read`: alone on the GPU in the same way, runs `explore/interp/read_run.sh` on the run's
     episodes ([section 5](#5-reading-a-run-on-the-gpu)). `READ_PARTS` picks the parts.
5. Always stops the VM at the end, also after an error or Ctrl-C, unless `KEEP=1`. The VM's
   `$WORK/probe` and the smoke test's outputs come back first; a read's come back after each chunk.

Everything goes to `logs/sessions/<name>-<UTC time>/`: `watchdog.log`, `serve.log`, `tunnel.log`,
`server.txt`, `probe/`, `readings.log` and `interp-smoke/`. The setup check goes to
`logs/setup/<name>-<GPU>-<commit>/`. The script exits 0 when every step passed, 3 when the run is
incomplete and nothing failed, and 1 otherwise.

**The cap and `ADMIT_FOR`.** If the cap stops the VM in the middle of a run, the running episodes
are cut. They come back as re-runs, and since long episodes are cut most often, the re-runs favour
short ones. Set `ADMIT_FOR` so that admissions stop while the cap still leaves room for the longest
episode: (cap / 10.05) − setup − the setup check (about 0.4 h) − longest episode, in hours.

## 3. The watchdog: `explore/colab_probe/watchdog.sh`

`session.sh` starts it. It can also run alone: `watchdog.sh SESSION CAP`. It stops the VM when the
first of these holds:

- **Cap.** The units spent reach the cap. It reads the balance from `colab usage` and compares it
  with the balance before `colab new`, so setup counts too.
- **Unreadable balance.** The balance can't be read `BAD_READS` polls in a row (default 5). Spending
  it can't see is never counted as under the cap.
- **Idle.** Nothing has run for `IDLE_MIN` minutes (default 20). That means no running or waiting
  request on either server (vLLM's `/metrics` through the tunnel), and no `simtest
  run|preflight|equiv` process. `session.sh`'s steps on the VM carry the tag `sim-session-step`
  and count as busy too. If a metrics read fails, that counts as idle.
- **Max time.** `MAX_MIN` minutes have passed (optional).

It writes one line per minute. After a stop, it stops again at every poll until the session is
gone. A failed `colab sessions` is never taken as a gone session.

The watchdog runs on the laptop. If the laptop sleeps, the watchdog stops but the VM keeps
billing, so `session.sh` runs `caffeinate` while the watchdog lives. Keep the lid open.

## 4. Hopper: chunks of a run, one Slurm job each

```bash
LOGIN=<user@login> TIME_LIMIT=<h per job> SETUP_HOURS=<h> WORK=<scratch on the nodes> \
SBATCH_ARGS="--partition=... --account=..." explore/hopper/submit_chunks.sh CONFIG <durations file>
explore/hopper/run_chunks.sh logs/hopper/<name>-<time>.jobs
```

Build the setup check's contexts first (BACKENDS.md). `submit_chunks.sh` refuses without them, and
refuses `SIM_QUANT=fp8`.

- `submit_chunks.sh` runs on the laptop. It calls `simtest plan --json` to split the run into
  chunks of epochs, one per job, and submits `explore/hopper/serve.slurm` once per chunk with
  `--dependency=afterany:<previous>` and the simulator's scheme by name. If a submission fails, it
  cancels the jobs it already submitted.
- `serve.slurm` asks for `--gres=gpu:2`. It runs `serve.sh`'s two-gpu profile (one model per card,
  both at 128K) and refuses any other profile. It keeps the servers up until the job ends. It writes
  `status/<job id>` next to `serve.sh` on the cluster (the node, the state, and the job's end time),
  and `status/<job id>.server`, the server's scheme lines.
- `run_chunks.sh` runs on the laptop. For each chunk, it waits until the job is ready, opens the
  tunnel to the node through the login host, waits for both servers, and reads the job's scheme
  lines. On the first usable job it runs the setup check before any episode (BACKENDS.md), which
  scores `simtest equiv` as the pilot asks. Anything but PASS stops everything. Every chunk then
  passes the gate on its own job's lines, runs `simtest run
  --epochs <chunk> --admit-for <hours>`, and cancels the job so the next one starts. When it exits,
  it cancels every job in the file, unless `KEEP=1`.

The job's time limit, the login host, the partition and the scratch path are unknown, so they are
parameters only. A job killed at its time limit cuts the episodes still running, so `--admit-for`
must leave room for the longest one. The plan keeps `TIME_LIMIT − SETUP_HOURS − admit_for` hours of
room. `run_chunks.sh` keeps the same room before the job's real end: if the servers come up late,
or the setup check takes time from the first job, the window shrinks, and a chunk with no room left
is skipped and submitted again later.

## 5. Reading a run on the GPU

The steps come from `explore/interp/read_run.sh`, on a Colab VM or a Hopper node, alone on the GPU
(INTERP.md, "Reading a run"; every number below is an estimate until the first run):

| `READ_PARTS` (`PARTS`) | What | Engine | Pilot, minutes |
| --- | --- | --- | --- |
| `check` | The first GPU run: the reading paths on the longest and shortest episode | vLLM, then transformers | about 15 |
| `proj` | Projections, prompt ends, turn curves, SAE latents | vLLM (`READ_ENGINE=hf` to change) | about 15 |
| `logit` | The logit monitor. Needs `READ_PHRASES`, so the judge first | transformers | about 25 |
| `transfer` | The Qwen-Scope transfer check | transformers, then vLLM | about 15 |

```bash
# with the episodes: read the projections while the VM is up
STEPS="setup run ids read" READ_VECTORS=logs/interp/... MIX=c3 ADMIT_FOR=14 explore/colab_probe/session.sh configs/coding/pilot.yaml 180
# after the judge: a short second session
STEPS=read READ_PARTS="logit transfer" READ_PHRASES=<file> MIX=c3 explore/colab_probe/session.sh configs/coding/pilot.yaml 12
# Hopper
LOGIN=<user@login> TIME_LIMIT=<h> WORK=<scratch> SBATCH_ARGS="--partition=..." VECTORS=<file> PARTS="check proj" explore/hopper/submit_read.sh configs/coding/pilot.yaml
```

- **Vectors.** `READ_VECTORS` is a file on the laptop. Without it, `proj` uses the file the `readings`
  step builds on that VM, so name `readings` in `STEPS`.
- **Chunks and resuming.** Colab allows one SSH session, so a read can't copy results out while it
  runs. `session.sh` runs `read_run.sh` for `READ_CHUNK_MIN` minutes (20), copies the readings back to
  `logs/read/<name>/readings/`, and goes on until nothing is left. A VM lost mid-run loses one chunk.
  A second session sends the readings first and goes on from them. A chunk that finishes nothing more
  stops the read, so it can't loop on a billing VM. Each engine load costs about a minute per chunk.
- **Order of a session.** `ids` runs while the servers are up. `readings` and `read` share one stop of
  the servers. The sentences don't exist until the judge has read the episodes, so `logit` is a
  second, short session: about 45 minutes with setup. On Hopper it is a second job.
- **Hopper.** `submit_read.sh` sends the code, the episodes, the readings so far, the vectors and the
  sentences to `WORK`, and submits `read.slurm` (one GPU). The login host must see `WORK`. The time
  limit is a parameter; `CHUNK_MIN` ends 10 minutes before it. The job's exit code is the reader's: 3
  means episodes are left, and submitting again resumes. It prints the command that brings the
  readings back.

## Not covered yet

- **Reading on a GPU.** `read_run.sh` and `read.slurm` have run only against stubs
  (`tests/test_read_run_script.py`, `test_session_script.py`). `read_run.py` has run on the laptop's MLX
  on real logs.
- **Equivalence memory.** Prompt log-probs may need more memory than the servers leave free
  (BACKENDS.md, "Practicalities"). If the setup check's `equiv` runs out of memory, run the check
  alone: `STEPS=setup AGENT_MEM=... SIM_MEM=...` (`session.sh` passes both to `serve.sh`), then
  `STEPS=run` without them. Never run episodes with them: other fractions mean another KV cache.
- **Untested on the real systems.** None of these scripts has run against Colab or Slurm yet:
  only against stubs (`tests/test_watchdog_script.py`, `test_session_script.py`,
  `test_hopper_scripts.py`) and in dry runs.
