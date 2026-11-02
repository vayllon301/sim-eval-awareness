"""explore/hopper/: the Slurm job that serves both models on two GPUs (serve.slurm), the laptop script
that submits one job per chunk of a plan (submit_chunks.sh), and the one that runs each chunk on its
job's servers (run_chunks.sh).

Nothing reaches a cluster: serve.slurm runs under bash with a stub serve.sh, curl and hostname, and
the laptop scripts with stubs for ssh (the login host: sbatch, squeue, the status files, scancel),
uv (simtest plan and run) and the rest. Each stub records its calls.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SLURM = REPO / "explore" / "hopper" / "serve.slurm"
SUBMIT = REPO / "explore" / "hopper" / "submit_chunks.sh"
RUN = REPO / "explore" / "hopper" / "run_chunks.sh"
CONFIG = "configs/coding/pilot.yaml"   # name: n2-pilot
BASH, ZSH = shutil.which("bash"), shutil.which("zsh")
HEAD = "0123456789abcdef0123456789abcdef01234567"
REC = 'echo "$(basename "$0") $*" >> "$STATE/calls.log"\n'


def stub_dir(tmp_path: Path, stubs: dict[str, str]) -> Path:
    d = tmp_path / "stubs"
    d.mkdir(exist_ok=True)
    for name, body in stubs.items():
        (d / name).write_text("#!/bin/sh\n" + body)
        (d / name).chmod(0o755)
    return d


# --- serve.slurm ------------------------------------------------------------------------------------

FAKE_SERVE = """#!/bin/sh
# stands in for explore/colab_probe/serve.sh: the dry run prints a profile; the real run "serves"
if [ "${DRY_RUN:-}" = 1 ]; then echo "profile: ${FAKE_PROFILE:-two-gpu} (2 GPU of 81559 MiB), vllm 0.32.0, work $WORK"; exit 0; fi
echo "serve WORK=$WORK MIX=${MIX:-}" >> "$STATE/calls.log"
exit "${SERVE_RC:-0}"
"""
# healthy for HEALTHY_CHECKS calls, then down; it copies the status file the first time it is asked
SLURM_CURL = REC + """[ -e "$STATE/status-at-first-check" ] || cp "$STATUS_DIR/42" "$STATE/status-at-first-check"
n=$(grep -c '^curl' "$STATE/calls.log")
[ "$n" -le "${HEALTHY_CHECKS:-1000000}" ] && exit 0
exit 7
"""


def run_slurm(tmp_path: Path, env: dict | None = None, background=False):
    assert BASH
    state = tmp_path / "state"
    state.mkdir()
    stubs = stub_dir(tmp_path, {"curl": SLURM_CURL, "hostname": "echo login-node\n"})
    serve = tmp_path / "serve.sh"
    serve.write_text(FAKE_SERVE)
    full = {"PATH": f"{stubs}:/usr/bin:/bin", "STATE": str(state), "SLURM_JOB_ID": "42",
            "SLURMD_NODENAME": "gpu07", "WORK": str(tmp_path / "work"), "SERVE_SH": str(serve),
            "STATUS_DIR": str(tmp_path / "status"), "TIME_LIMIT_MIN": "720", "HEALTH_POLL": "0.05",
            **(env or {})}
    full = {k: v for k, v in full.items() if v is not None}
    if background:
        return subprocess.Popen([BASH, str(SLURM)], env=full, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True), state
    proc = subprocess.run([BASH, str(SLURM)], env=full, capture_output=True, text=True, timeout=60)
    return proc, state


def status(tmp_path: Path, path: Path | None = None) -> dict:
    text = (path or tmp_path / "status" / "42").read_text()
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def test_the_job_asks_for_two_gpus_and_leaves_the_time_limit_to_sbatch():
    head = [x for x in SLURM.read_text().splitlines() if x.startswith("#SBATCH")]
    assert "#SBATCH --gres=gpu:2" in head
    assert not [x for x in head if "--time" in x or "-t " in x]      # the limit is a parameter


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_the_job_serves_writes_ready_and_ends_when_a_server_dies(tmp_path):
    proc, state = run_slurm(tmp_path, {"HEALTHY_CHECKS": "4", "MIX": "c3"})
    assert proc.returncode == 1, proc.stderr
    first = status(tmp_path, state / "status-at-first-check")
    assert first["state"] == "ready" and first["node"] == "gpu07"
    assert int(first["end"]) - int(first["start"]) == 720 * 60
    last = status(tmp_path)
    assert last["state"] == "server-died"                       # after 3 failed checks in a row
    assert f"serve WORK={tmp_path / 'work'} MIX=c3" in (state / "calls.log").read_text()


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_a_failed_serve_is_written_as_failed(tmp_path):
    proc, state = run_slurm(tmp_path, {"SERVE_RC": "1"})
    assert proc.returncode == 1
    assert status(tmp_path)["state"] == "failed"


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_another_profile_is_refused_before_serving(tmp_path):
    # e.g. one 80 GB card: 64K contexts, not equivalent; the job must not serve it
    proc, state = run_slurm(tmp_path, {"FAKE_PROFILE": "one-gpu-80g"})
    assert proc.returncode == 1
    assert status(tmp_path)["state"] == "failed"
    assert "one-gpu-80g" in proc.stdout + proc.stderr
    assert "serve WORK" not in (state / "calls.log").read_text() if (state / "calls.log").exists() else True


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize("missing", ["WORK", "SERVE_SH", "STATUS_DIR", "TIME_LIMIT_MIN"])
def test_required_settings(tmp_path, missing):
    proc, state = run_slurm(tmp_path, {missing: None})
    assert proc.returncode != 0
    assert missing in proc.stderr


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_the_end_of_the_job_is_written(tmp_path):
    # the time limit (or scancel) sends SIGTERM: the status says so at once
    p, state = run_slurm(tmp_path, background=True)
    for _ in range(200):
        if (tmp_path / "status" / "42").exists() and status(tmp_path)["state"] == "ready":
            break
        time.sleep(0.05)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=10)
    assert status(tmp_path)["state"] == "ended"


# --- submit_chunks.sh -------------------------------------------------------------------------------

PLAN = {"chunks": [{"epochs": "1-1", "admit_for_hours": 9.5}, {"epochs": "2-2", "admit_for_hours": 9.5},
                   {"epochs": "3-3", "admit_for_hours": 9.25}]}
UV = REC + """case "$*" in
  "run python "*) shift 2; exec "$PYTHON" "$@" ;;
  *"simtest plan "*) cat "$STATE/plan.json"; exit "${PLAN_RC:-0}" ;;
  *"simtest run "*)
    n=$(grep -c 'simtest run ' "$STATE/calls.log"); rc=$(sed -n "${n}p" "$STATE/run-rcs" 2>/dev/null)
    exit "${rc:-0}" ;;
  *"simtest equiv score "*) exit 0 ;;
  *) echo "unexpected uv call: $*" >&2; exit 99 ;;
esac
"""
# The login host. sbatch reads the job script from stdin and answers with a job id (1001, 1002...);
# FAIL_SBATCH=N fails the Nth. The poll prints squeue's state and the job's status file, from
# $STATE/jobs/<id>.squeue and $STATE/jobs/<id>.status (each a sequence: one line per poll, the last
# repeats; the status lines use ';' for newlines).
SSH = REC + """for a; do cmd=$a; done
pop() { f=$1; line=$(head -1 "$f"); if [ "$(wc -l < "$f")" -gt 1 ]; then tail -n +2 "$f" > "$f.t" && mv "$f.t" "$f"; fi; printf '%s' "$line"; }
case $cmd in
  *"sbatch "*)
    cat > "$STATE/job-script-$(grep -c 'sbatch ' "$STATE/calls.log")"
    n=$(grep -c 'sbatch ' "$STATE/calls.log")
    [ "$n" = "${FAIL_SBATCH:-0}" ] && { echo "sbatch: error: Batch job submission failed" >&2; exit 1; }
    echo "$((1000 + n));hopper" ;;
  *"cat > "*"/serve.sh"*) cat > "$STATE/remote-serve.sh" ;;
  *"squeue -h -j "*)
    id=$(echo "$cmd" | sed -E 's/.*squeue -h -j ([0-9]+).*/\\1/')
    echo "squeue=$(pop "$STATE/jobs/$id.squeue")"
    pop "$STATE/jobs/$id.status" | tr ';' '\\n'; echo ;;
  *scancel*) echo "$cmd" >> "$STATE/scancel.log" ;;
  *) echo "$cmd" >> "$STATE/tunnels"; sleep 0.2; exit 255 ;;   # the tunnel (-N): its target
esac
"""
GIT = REC + """case "$*" in
  "rev-parse HEAD") echo "${GIT_HEAD}" ;;
  "status --porcelain") printf '%s' "${GIT_DIRTY:-}" ;;
  *) echo "unexpected git call: $*" >&2; exit 99 ;;
esac
"""
# The servers answer once this chunk's tunnel has started (one more distinct node than the chunks
# before; at most 5 s), as they would through it.
CURL = REC + """case "$*" in *":18000/v1/models"*)
  n=$(grep -c ':18000/v1/models' "$STATE/calls.log")
  for i in $(seq 100); do [ "$(sort -u "$STATE/tunnels" 2>/dev/null | wc -l)" -ge "$n" ] && break; sleep 0.05; done ;;
esac
case "$*" in
  *":18000/v1/models"*) echo '{"data":[{"id": "agent"}]}' ;;
  *":18001/v1/models"*) echo '{"data":[{"id": "simulator"}]}' ;;
  *) exit 7 ;;
esac
"""
LAPTOP = {"uv": UV, "ssh": SSH, "git": GIT, "curl": CURL, "lsof": REC + "exit 1\n"}


def laptop(tmp_path: Path, preflight: str | None = HEAD):
    state, logs = tmp_path / "state", tmp_path / "logs"
    (state / "jobs").mkdir(parents=True)
    logs.mkdir()
    stubs = stub_dir(tmp_path, LAPTOP)
    (state / "plan.json").write_text(json.dumps(PLAN))
    if preflight is not None:
        (logs / "preflight").mkdir()
        (logs / "preflight" / "n2-pilot.ok").write_text(preflight + "\n")
    (tmp_path / "dotenv").write_text("")
    (tmp_path / "durations.json").write_text("{}")
    env = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "STATE": str(state), "LOGS": str(logs),
           "GIT_HEAD": HEAD, "PYTHON": sys.executable, "ENV_FILE": str(tmp_path / "dotenv"),
           "POLL": "0.05", "WAIT_TICK": "0.05"}
    return state, logs, env


def calls_of(state: Path) -> list[str]:
    f = state / "calls.log"
    return f.read_text().splitlines() if f.exists() else []


SUBMIT_ENV = {"LOGIN": "victor@login.hopper", "TIME_LIMIT": "12", "SETUP_HOURS": "0.5",
              "WORK": "/scratch/victor/sim", "SBATCH_ARGS": "--partition=gpu --cpus-per-task=16"}


def submit(tmp_path: Path, env: dict | None = None, preflight: str | None = HEAD):
    state, logs, base = laptop(tmp_path, preflight)
    full = {**base, **SUBMIT_ENV, **(env or {})}
    full = {k: v for k, v in full.items() if v is not None}
    proc = subprocess.run([ZSH, str(SUBMIT), CONFIG, str(tmp_path / "durations.json")], env=full, cwd=REPO,
                          capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
    return proc, state, logs


def sbatches(calls):
    return [c for c in calls if "sbatch " in c]


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_one_job_per_chunk_each_after_the_previous(tmp_path):
    proc, state, logs = submit(tmp_path)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    calls = calls_of(state)
    plan = [c for c in calls if "simtest plan " in c][0]
    assert f"simtest plan {CONFIG} --durations {tmp_path / 'durations.json'} --time-limit 12 --setup-hours 0.5 --json" in plan
    sb = sbatches(calls)
    assert len(sb) == 3
    assert "--dependency" not in sb[0]
    assert "--dependency=afterany:1001" in sb[1] and "--dependency=afterany:1002" in sb[2]
    for c in sb:
        assert "--time=720" in c and "TIME_LIMIT_MIN=720" in c and "WORK=/scratch/victor/sim" in c
        assert "--partition=gpu --cpus-per-task=16" in c and "--parsable" in c
    assert (state / "job-script-1").read_text() == SLURM.read_text()        # the job template, by stdin
    assert "Exploratory (N0 Colab probe" in (state / "remote-serve.sh").read_text()   # serve.sh as it is
    jobs = list((logs / "hopper").glob("n2-pilot-*.jobs"))
    assert len(jobs) == 1
    text = jobs[0].read_text()
    assert f"# commit={HEAD}" in text and "# login=victor@login.hopper" in text and "# time_limit=12" in text
    rows = [x.split() for x in text.splitlines() if not x.startswith("#")]
    assert rows == [["1001", "1-1", "9.5"], ["1002", "2-2", "9.5"], ["1003", "3-3", "9.25"]]
    assert f"run_chunks.sh {jobs[0]}" in proc.stdout


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_a_failed_submission_cancels_the_jobs_already_submitted(tmp_path):
    proc, state, logs = submit(tmp_path, {"FAIL_SBATCH": "3"})
    assert proc.returncode == 1
    cancelled = (state / "scancel.log").read_text()
    assert "1001" in cancelled and "1002" in cancelled


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
@pytest.mark.parametrize("env,preflight,needle", [
    ({"LOGIN": None}, HEAD, "LOGIN"),
    ({"TIME_LIMIT": None}, HEAD, "TIME_LIMIT"),
    ({"SETUP_HOURS": None}, HEAD, "SETUP_HOURS"),
    ({"WORK": None}, HEAD, "WORK"),
    ({"WORK": "scratch/sim"}, HEAD, "absolute"),
    ({}, None, "preflight"),
    ({"GIT_DIRTY": "?? x\n"}, HEAD, "dirty"),
], ids=["login", "time", "setup", "work", "relative-work", "preflight", "dirty"])
def test_submission_refusals(tmp_path, env, preflight, needle):
    proc, state, logs = submit(tmp_path, env, preflight)
    assert proc.returncode == 1
    assert "refused" in proc.stderr and needle in proc.stderr
    assert not [c for c in calls_of(state) if c.startswith(("ssh", "uv"))]


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_submit_dry_run_plans_but_submits_nothing(tmp_path):
    proc, state, logs = submit(tmp_path, {"DRY_RUN": "1"})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not [c for c in calls_of(state) if c.startswith("ssh")]
    assert proc.stdout.count("sbatch --parsable") == 3
    assert "--dependency=afterany:<job 1>" in proc.stdout
    assert not (logs / "hopper").exists()


# --- run_chunks.sh ----------------------------------------------------------------------------------

def jobs_file(tmp_path: Path, rows, commit=HEAD, time_limit="12", setup="0.5") -> Path:
    f = tmp_path / "logs" / "hopper" / "n2-pilot-test.jobs"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("\n".join([f"# config={CONFIG}", f"# commit={commit}", "# login=victor@login.hopper",
                            "# remote_dir=sim-session", f"# time_limit={time_limit}", f"# setup_hours={setup}"]
                           + [" ".join(r) for r in rows]) + "\n")
    return f


def job(state: Path, jid: str, squeue: list[str], statuses: list[str]):
    (state / "jobs" / f"{jid}.squeue").write_text("\n".join(squeue) + "\n")
    (state / "jobs" / f"{jid}.status").write_text("\n".join(statuses) + "\n")


def starting(node: str) -> str:
    now = int(time.time())
    return f"node={node};state=starting;start={now};end={now + 12 * 3600}"


def ready(node: str, hours_left: float) -> str:
    now = int(time.time())
    return f"node={node};state=ready;start={now - 1800};end={now + int(hours_left * 3600)}"


def run_chunks(tmp_path: Path, rows, setup_jobs, env: dict | None = None, commit=HEAD, run_rcs=()):
    state, logs, base = laptop(tmp_path)
    f = jobs_file(tmp_path, rows, commit)
    setup_jobs(state)
    (state / "run-rcs").write_text("".join(f"{rc}\n" for rc in run_rcs))
    full = {**base, "OUTDIR": str(tmp_path / "out"), **(env or {})}
    proc = subprocess.run([ZSH, str(RUN), str(f)], env=full, cwd=REPO, capture_output=True, text=True,
                          timeout=90, stdin=subprocess.DEVNULL)
    return proc, state


def runs(calls):
    return [c for c in calls if "simtest run " in c]


def admit(call: str) -> float:
    return float(call.split("--admit-for ")[1].split()[0])


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_each_chunk_waits_for_its_job_then_runs_on_its_node(tmp_path):
    def jobs(state):
        job(state, "1001", ["PENDING", "RUNNING"], ["", starting("gpu07"), ready("gpu07", 20)])
        job(state, "1002", ["PENDING", "PENDING", "RUNNING"], ["", ready("gpu11", 20)])
    proc, state = run_chunks(tmp_path, [("1001", "1-1", "9.5"), ("1002", "2-2", "9.5")], jobs)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    calls = calls_of(state)
    r = runs(calls)
    assert len(r) == 2
    assert f"simtest run {CONFIG} --epochs 1-1 --admit-for 9.50" in r[0]
    assert f"simtest run {CONFIG} --epochs 2-2 --admit-for 9.50" in r[1]
    tunnels = [c for c in calls if c.startswith("ssh") and " -N " in c]
    assert any("-J victor@login.hopper" in c and c.endswith("victor@gpu07") for c in tunnels)
    assert any(c.endswith("victor@gpu11") for c in tunnels)
    cancelled = (state / "scancel.log").read_text().splitlines()
    first_run = calls.index(r[0])
    first_cancel = next(i for i, c in enumerate(calls) if "scancel 1001" in c)
    assert first_cancel > first_run          # each job is cancelled once its chunk is done: the next starts
    assert any("1002" in c for c in cancelled)


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_a_late_start_shortens_the_admission_window(tmp_path):
    # plan: 12 h jobs, 0.5 h setup, 9.5 h admitted, so 2 h kept for the longest episode. The job has
    # 6 h left when its servers are ready: admit at most 4 h
    def jobs(state):
        job(state, "1001", ["RUNNING"], [ready("gpu07", 6)])
    proc, state = run_chunks(tmp_path, [("1001", "1-1", "9.5")], jobs)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    a = admit(runs(calls_of(state))[0])
    assert 3.95 <= a <= 4.0


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_a_job_without_time_for_an_episode_is_skipped(tmp_path):
    def jobs(state):
        job(state, "1001", ["RUNNING"], [ready("gpu07", 1.5)])     # 1.5 h left, 2 h needed
        job(state, "1002", ["RUNNING"], [ready("gpu11", 20)])
    proc, state = run_chunks(tmp_path, [("1001", "1-1", "9.5"), ("1002", "2-2", "9.5")], jobs)
    assert proc.returncode == 3, proc.stderr + proc.stdout
    r = runs(calls_of(state))
    assert len(r) == 1 and "--epochs 2-2" in r[0]
    assert "1-1: SKIPPED" in proc.stdout
    assert "scancel 1001" in (state / "scancel.log").read_text()


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_a_failed_job_is_skipped_and_the_next_chunk_runs(tmp_path):
    def jobs(state):
        job(state, "1001", ["RUNNING", ""], ["node=gpu07;state=failed;start=1;end=2"])
        job(state, "1002", ["RUNNING"], [ready("gpu11", 20)])
    proc, state = run_chunks(tmp_path, [("1001", "1-1", "9.5"), ("1002", "2-2", "9.5")], jobs)
    assert proc.returncode == 3, proc.stderr + proc.stdout
    assert len(runs(calls_of(state))) == 1
    assert "failed" in proc.stdout


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_an_incomplete_chunk_is_reported_and_the_next_runs(tmp_path):
    def jobs(state):
        job(state, "1001", ["RUNNING"], [ready("gpu07", 20)])
        job(state, "1002", ["RUNNING"], [ready("gpu11", 20)])
    proc, state = run_chunks(tmp_path, [("1001", "1-1", "9.5"), ("1002", "2-2", "9.5")], jobs, run_rcs=(3, 0))
    assert proc.returncode == 3, proc.stderr + proc.stdout
    assert len(runs(calls_of(state))) == 2
    assert "1-1: INCOMPLETE" in proc.stdout


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_a_run_error_stops_and_cancels_every_remaining_job(tmp_path):
    def jobs(state):
        job(state, "1001", ["RUNNING"], [ready("gpu07", 20)])
        job(state, "1002", ["PENDING"], [""])
        job(state, "1003", ["PENDING"], [""])
    proc, state = run_chunks(tmp_path, [("1001", "1-1", "9.5"), ("1002", "2-2", "9.5"), ("1003", "3-3", "9.5")],
                             jobs, run_rcs=(1,))
    assert proc.returncode == 1
    assert len(runs(calls_of(state))) == 1
    cancelled = (state / "scancel.log").read_text()
    assert "1002" in cancelled and "1003" in cancelled


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_run_chunks_refuses_another_commit(tmp_path):
    proc, state = run_chunks(tmp_path, [("1001", "1-1", "9.5")], lambda s: None,
                             commit="fedcba9876543210fedcba9876543210fedcba98")
    assert proc.returncode == 1
    assert "refused" in proc.stderr and "commit" in proc.stderr
    assert not [c for c in calls_of(state) if c.startswith(("ssh", "uv"))]


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_run_chunks_dry_run_touches_nothing(tmp_path):
    proc, state = run_chunks(tmp_path, [("1001", "1-1", "9.5"), ("1002", "2-2", "9.5")], lambda s: None,
                             env={"DRY_RUN": "1", "EQUIV": "1", "EQUIV_CONTEXTS": str(tmp_path / "dotenv")})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not [c for c in calls_of(state) if c.startswith(("ssh", "uv", "curl"))]
    out = proc.stdout
    assert out.count("simtest run ") == 2 and "--admit-for 9.50" in out
    assert out.count("simtest equiv score") == 1          # on the first chunk's servers, before its episodes
    assert out.index("simtest equiv score") < out.index("simtest run ")
    assert "scancel 1001" in out and "tunnel.sh" in out


def test_the_scripts_are_executable():
    for s in (SUBMIT, RUN):
        assert s.read_text().startswith("#!/bin/zsh") and os.access(s, os.X_OK)
    assert SLURM.read_text().startswith("#!/bin/bash")


# --- sol-loop round 1 (02-11-2026) ---------------------------------------------------------------------

def test_the_planners_own_chunks_are_submitted(tmp_path, monkeypatch):
    """S2: simtest plan writes a one-epoch chunk as "3", not "3-3"."""
    from simtest.stats import replay as R

    plan = R.plan_chunks([("C0", "none"), ("C3", "s")], ["a", "b"], 3, 4, {"C0": [60.0], "C3": [120.0]}, 5.0,
                         setup_hours=0.5, draws=20)
    assert [c["epochs"] for c in plan["chunks"]] == ["1-2", "3"]
    monkeypatch.setitem(PLAN, "chunks", plan["chunks"])
    proc, state, logs = submit(tmp_path)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    rows = [x.split() for x in next((logs / "hopper").glob("n2-pilot-*.jobs")).read_text().splitlines()
            if not x.startswith("#")]
    assert rows == [["1001", "1-2", "2.5"], ["1002", "3", "2.5"]]


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_equiv_runs_on_the_first_usable_job_when_the_first_fails(tmp_path):
    """S4: with EQUIV=1 the check runs before any episode, also when the first job failed."""
    def jobs(state):
        job(state, "1001", ["RUNNING", ""], ["node=gpu07;state=failed;start=1;end=2"])
        job(state, "1002", ["RUNNING"], [ready("gpu11", 20)])
        job(state, "1003", ["RUNNING"], [ready("gpu12", 20)])
    ctx = tmp_path / "contexts.jsonl"
    ctx.write_text("{}\n")
    proc, state = run_chunks(tmp_path, [("1001", "1-1", "9.5"), ("1002", "2-2", "9.5"), ("1003", "3-3", "9.5")],
                             jobs, env={"EQUIV": "1", "EQUIV_CONTEXTS": str(ctx)})
    calls = calls_of(state)
    eq = [i for i, c in enumerate(calls) if "simtest equiv score" in c]
    first_run = next(i for i, c in enumerate(calls) if "simtest run " in c)
    assert len(eq) == 1 and eq[0] < first_run, proc.stdout
