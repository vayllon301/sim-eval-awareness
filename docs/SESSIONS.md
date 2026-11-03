# GPU sessions: preflight, session.sh, the watchdog, Hopper chunks

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

1. Checks first: the preflight, a clean tree, the env file, the equivalence contexts, free ports
   18000 and 18001, a readable balance. Any failure means no VM.
2. Creates the VM (or attaches session `SESSION` if it exists), and starts the watchdog with the
   cap (the second argument, in units).
3. Runs `serve.sh` once with the given `MIX` (required: `c3` or `default`), opens the tunnel and
   waits for both `/v1/models`.
4. Runs the steps in `STEPS`, always in this order (default: `run equiv`):
   - `run`: `simtest run`, with `EPOCHS`, `ADMIT_FOR` and `CARRY_UNDER` passed on. If the run is
     incomplete (exit 3), it says so and goes on.
   - `equiv`: `simtest equiv score` on the contexts, one request at a time and 12 at once
     (BACKENDS.md). It runs after the episodes, so they don't share its batches. With no
     contexts file yet, it first builds one from this run's logs (`simtest equiv contexts`).
     Keep that file: every backend is scored on the same contexts.
   - `readings` (only when named): stops the servers to free the GPU, sends the code at HEAD (a
     git bundle), and runs `explore/interp/gpu_smoke.sh` (INTERP.md, "GPU smoke test"). It reads
     no episode yet and takes about 50 GPU minutes, so it is off by default.
5. Always stops the VM at the end, also after an error or Ctrl-C, unless `KEEP=1`. The VM's
   `$WORK/probe` and the readings come back first.

Everything goes to `logs/sessions/<name>-<UTC time>/`: `watchdog.log`, `serve.log`, `tunnel.log`,
`probe/`, `equiv-c1.jsonl` and `equiv-c12.jsonl`, `readings.log` and `interp-smoke/`. The script
exits 0 when every step passed, 3 when the run is incomplete and nothing failed, and 1 otherwise.

**The cap and `ADMIT_FOR`.** If the cap stops the VM in the middle of a run, the running episodes
are cut. They come back as re-runs, and since long episodes are cut most often, the re-runs favour
short ones. Set `ADMIT_FOR` so that admissions stop while the cap still leaves room for the longest
episode: (cap / 10.05) − setup − longest episode, in hours.

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

- `submit_chunks.sh` runs on the laptop. It calls `simtest plan --json` to split the run into
  chunks of epochs, one per job, and submits `explore/hopper/serve.slurm` once per chunk with
  `--dependency=afterany:<previous>`. If a submission fails, it cancels the jobs it already
  submitted.
- `serve.slurm` asks for `--gres=gpu:2`. It runs `serve.sh`'s two-gpu profile (one model per card,
  both at 128K) and refuses any other profile. It keeps the servers up until the job ends. It writes
  `status/<job id>` next to `serve.sh` on the cluster: the node, the state, and the job's end time.
- `run_chunks.sh` runs on the laptop. For each chunk, it waits until the job is ready, opens the
  tunnel to the node through the login host, waits for both servers, runs `simtest run --epochs
  <chunk> --admit-for <hours>`, then cancels the job so the next one starts. With `EQUIV=1`, it
  runs `simtest equiv score` on the first job's servers before any episode, as the pilot asks for
  a backend other than the G4. When it exits, it cancels every job in the file, unless `KEEP=1`.

The job's time limit, the login host, the partition and the scratch path are unknown, so they are
parameters only. A job killed at its time limit cuts the episodes still running, so `--admit-for`
must leave room for the longest one. The plan keeps `TIME_LIMIT − SETUP_HOURS − admit_for` hours of
room. `run_chunks.sh` keeps the same room before the job's real end: if the servers come up late,
the window shrinks, and a chunk with no room left is skipped and submitted again later.

## Not covered yet

- **Readings.** The `readings` step runs the GPU smoke test, the only GPU script in INTERP.md so
  far. Reading pilot episodes on vLLM needs the reader's Inspect adapter, which is not built yet
  (INTERP.md, "Left for the GPU").
- **Equivalence memory.** Prompt log-probs may need more memory than the servers leave free
  (BACKENDS.md, "Practicalities"). If `equiv` runs out of memory, run it again alone:
  `STEPS=equiv AGENT_MEM=... SIM_MEM=...` (`session.sh` passes both to `serve.sh`). Never run
  episodes with them: other fractions mean another KV cache.
- **Untested on the real systems.** None of these scripts has run against Colab or Slurm yet:
  only against stubs (`tests/test_watchdog_script.py`, `test_session_script.py`,
  `test_hopper_scripts.py`) and in dry runs.
