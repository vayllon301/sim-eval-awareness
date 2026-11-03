"""explore/colab_probe/session.sh: one Colab VM session runs a config's GPU work (serve.sh once, the
run, the equivalence scores, the GPU readings), guarded by the watchdog, and always ends by stopping
the VM.

Runs the script with stubs first on PATH for everything it could reach: `colab` (new, sessions,
usage, stop), `ssh` (the VM: serve.sh, the readings, the copies back), `uv` (simtest), `curl` (the
servers through the tunnel), `git`, `lsof`, `pgrep`, `caffeinate`. Each records its calls. The dry
run must call none of colab, ssh, uv and curl. Logs go to a temporary LOGS directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "explore" / "colab_probe" / "session.sh"
CONFIG = "configs/coding/pilot.yaml"   # name: n2-pilot
ZSH = shutil.which("zsh")
pytestmark = pytest.mark.skipif(ZSH is None, reason="needs zsh")
HEAD = "0123456789abcdef0123456789abcdef01234567"

REC = 'echo "$(basename "$0") $*" >> "$STATE/calls.log"\n'
COLAB = REC + """case "$*" in
  *" usage") printf 'Current balance: 150.00 compute units\\nUsage rate: 10.05/hr\\nActive assignments: 1\\n' ;;
  *" sessions")
    if [ -e "$STATE/alive" ]; then echo "[probe] https://example.invalid | Hardware: G4 | Shape: STANDARD | Variant: GPU"
    else echo "[colab] No active sessions found on server."; fi ;;
  *" new -s probe --gpu G4") [ -e "$STATE/fail-new" ] && exit 1; touch "$STATE/alive"; echo "[colab] Creating session 'probe'..." ;;
  *" stop -s probe") rm -f "$STATE/alive"; echo "[colab] Session terminated." ;;
  *) echo "unexpected colab call: $*" >&2; exit 99 ;;
esac
"""
# The VM: the last argument is the remote command. FAIL_REMOTE fails the command that contains it.
SSH = REC + """for a; do cmd=$a; done
if [ -n "${FAIL_REMOTE:-}" ]; then case $cmd in *"$FAIL_REMOTE"*) cat >/dev/null; echo "remote failure" >&2; exit 1 ;; esac; fi
case $cmd in
  *"sim-session-step serve"*) cat > "$STATE/serve-stdin"; echo "agent-ready"; echo "simulator-ready" ;;
  *"sim-session-step free-gpu"*) echo "GPU free" ;;
  *"sim-session-step readings"*) echo "$cmd" > "$STATE/readings-cmd"; echo "report"; exit "${READINGS_RC:-0}" ;;
  *"cat > /content/sim.bundle"*) cat > "$STATE/sim.bundle" ;;
  *"tar czf - -C /content probe"*) tar czf - -C "$STATE/vm" probe ;;
  *"tar czf - -C /content interp-smoke"*) tar czf - -C "$STATE/vm" interp-smoke ;;
  *) sleep 0.2; exit 255 ;;                      # the tunnel (-N): drops, and tunnel.sh reconnects
esac
"""
UV = REC + """case "$*" in
  *"simtest run "*) exit "${RUN_RC:-0}" ;;
  *"simtest equiv score "*) exit "${EQUIV_RC:-0}" ;;
  *) echo "unexpected uv call: $*" >&2; exit 99 ;;
esac
"""
CURL = REC + """case "$*" in
  *":18000/v1/models"*) echo '{"object":"list","data":[{"id": "agent","object":"model"}]}' ;;
  *":18001/v1/models"*) echo '{"object":"list","data":[{"id": "simulator","object":"model"}]}' ;;
  *"/metrics"*) printf 'vllm:num_requests_running{engine="0"} 0.0\\nvllm:num_requests_waiting{engine="0"} 0.0\\n' ;;
  *) exit 7 ;;
esac
"""
GIT = REC + """case "$*" in
  "rev-parse HEAD") echo "${GIT_HEAD}" ;;
  "status --porcelain") printf '%s' "${GIT_DIRTY:-}" ;;
  "bundle create "*) echo bundle > "$3" ;;
  *) echo "unexpected git call: $*" >&2; exit 99 ;;
esac
"""
LSOF = REC + 'case "$*" in *"${PORT_BUSY:-none}"*) echo "ssh 4242 victor 7u IPv4 TCP 127.0.0.1:$PORT_BUSY (LISTEN)"; exit 0 ;; esac\nexit 1\n'
PGREP = REC + "exit 1\n"
CAFFEINATE = REC + "exit 0\n"
STUBS = {"colab": COLAB, "ssh": SSH, "uv": UV, "curl": CURL, "git": GIT, "lsof": LSOF, "pgrep": PGREP,
         "caffeinate": CAFFEINATE}


def setup(tmp_path: Path, preflight: str | None = HEAD):
    stubs, state, logs = tmp_path / "stubs", tmp_path / "state", tmp_path / "logs"
    for d in (stubs, state, logs / "equiv", state / "vm" / "probe", state / "vm" / "interp-smoke"):
        d.mkdir(parents=True)
    for name, body in STUBS.items():
        (stubs / name).write_text("#!/bin/sh\n" + body)
        (stubs / name).chmod(0o755)
    (state / "vm" / "probe" / "timings.txt").write_text("start 0s\nsimulator-ready 741s\n")
    (state / "vm" / "interp-smoke" / "report.txt").write_text("vectors PASS\n")
    if preflight is not None:
        (logs / "preflight").mkdir()
        (logs / "preflight" / "n2-pilot.ok").write_text(preflight + "\n")
    (logs / "equiv" / "contexts.jsonl").write_text("{}\n")
    (tmp_path / "dotenv").write_text("")       # an empty stand-in for the keys file
    return stubs, state, logs


def run(tmp_path: Path, env: dict | None = None, *, preflight: str | None = HEAD, args=(CONFIG, "40"),
        before=None, timeout=90):
    stubs, state, logs = setup(tmp_path, preflight)
    if before:
        before(state)
    out = tmp_path / "out"
    full = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "STATE": str(state),
            "LOGS": str(logs), "OUTDIR": str(out), "ENV_FILE": str(tmp_path / "dotenv"),
            "GIT_HEAD": HEAD, "MIX": "c3", "POLL": "0.1", "WAIT_TICK": "0.05", **(env or {})}
    proc = subprocess.run([ZSH, str(SCRIPT), *args], env=full, cwd=REPO, capture_output=True, text=True,
                          timeout=timeout, stdin=subprocess.DEVNULL)
    # the watchdog runs detached and ends once the session is gone: wait for it, so no test leaves it
    wd = out / "watchdog.log"
    for _ in range(100):
        if not wd.exists() or "watchdog ends" in wd.read_text() or (state / "alive").exists():
            break
        time.sleep(0.1)
    calls = (state / "calls.log").read_text().splitlines() if (state / "calls.log").exists() else []
    return proc, calls, out, state


def index(calls, text):
    return next(i for i, c in enumerate(calls) if text in c)


# --- the dry run ------------------------------------------------------------------------------------

def test_dry_run_prints_every_step_and_touches_nothing(tmp_path):
    proc, calls, out, state = run(tmp_path, {"DRY_RUN": "1", "EPOCHS": "1-2", "ADMIT_FOR": "13.5",
                                             "STEPS": "run equiv readings"})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    # read-only checks only: no colab, ssh, uv or curl, and nothing created
    assert not [c for c in calls if c.split()[0] in ("colab", "ssh", "uv", "curl", "caffeinate")]
    assert not out.exists()
    cmds = [x for x in proc.stdout.splitlines() if x.startswith("+ ")]
    text = "\n".join(cmds)
    order = ["colab --auth adc usage", "colab --auth adc new -s probe --gpu G4", "watchdog.sh probe 40",
             "sim-session-step serve", "tunnel.sh probe", "localhost:18000/v1/models", "simtest run",
             "simtest equiv score", "sim-session-step free-gpu", "git bundle create", "sim-session-step readings",
             "tar czf - -C /content interp-smoke", "tar czf - -C /content probe", "colab --auth adc stop -s probe"]
    pos = [next(i for i, c in enumerate(cmds) if o in c) for o in order]
    assert pos == sorted(pos), list(zip(order, pos))
    assert "MIX=c3 bash -s' < explore/colab_probe/serve.sh" in text
    assert f"uv run --env-file {tmp_path / 'dotenv'} simtest run {CONFIG} --epochs 1-2 --admit-for 13.5" in text
    assert "--concurrency 1 12" in text and "--agent-url http://localhost:18000/v1" in text
    assert "START_BALANCE=" in text and "IDLE_MIN=20" in text
    assert "explore/interp/gpu_smoke.sh" in text


def test_dry_run_with_only_the_readings_serves_nothing(tmp_path):
    proc, calls, out, state = run(tmp_path, {"DRY_RUN": "1", "STEPS": "readings"})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "sim-session-step serve" not in proc.stdout and "tunnel.sh" not in proc.stdout
    assert "simtest run" not in proc.stdout and "sim-session-step readings" in proc.stdout
    assert "colab --auth adc stop -s probe" in proc.stdout


def test_dry_run_reports_what_a_real_session_would_refuse(tmp_path):
    proc, calls, out, state = run(tmp_path, {"DRY_RUN": "1"}, preflight=None)
    assert proc.returncode == 1
    assert "refused" in proc.stderr and "preflight" in proc.stderr
    assert "colab --auth adc new" in proc.stdout       # the commands still print
    assert not [c for c in calls if c.split()[0] in ("colab", "ssh", "uv", "curl")]


def test_memory_fractions_go_to_serve_sh_only_when_given(tmp_path):
    proc, calls, out, state = run(tmp_path, {"DRY_RUN": "1", "STEPS": "equiv", "AGENT_MEM": "0.40", "SIM_MEM": "0.45"})
    assert proc.returncode == 0, proc.stderr
    assert "MIX=c3 AGENT_MEM=0.40 SIM_MEM=0.45 bash -s" in proc.stdout
    plain, *_ = run(tmp_path / "plain", {"DRY_RUN": "1"})
    assert "_MEM" not in plain.stdout


def test_keep_skips_the_final_stop_in_the_dry_run(tmp_path):
    proc, calls, out, state = run(tmp_path, {"DRY_RUN": "1", "KEEP": "1"})
    assert proc.returncode == 0, proc.stderr
    assert "+ colab --auth adc stop" not in proc.stdout and "KEEP=1" in proc.stdout


# --- refusals: nothing is created ----------------------------------------------------------------------

@pytest.mark.parametrize("case,env,preflight,needle", [
    ("no-preflight", {}, None, "no preflight"),
    ("old-preflight", {}, "fedcba9876543210fedcba9876543210fedcba98", "not on HEAD"),
    ("dirty", {"GIT_DIRTY": " M src/simtest/run.py\n"}, HEAD, "dirty"),
    ("no-mix", {"MIX": ""}, HEAD, "MIX"),
    ("bad-step", {"STEPS": "run train"}, HEAD, "train"),
    ("port-taken", {"PORT_BUSY": "18001"}, HEAD, "18001"),
    ("no-contexts", {"EQUIV_CONTEXTS": "/nonexistent/contexts.jsonl", "STEPS": "equiv"}, HEAD, "contexts"),
], ids=lambda v: v if isinstance(v, str) and len(v) < 16 else None)
def test_refuses_to_start(tmp_path, case, env, preflight, needle):
    proc, calls, out, state = run(tmp_path, env, preflight=preflight)
    assert proc.returncode == 1, proc.stdout
    assert "refused" in proc.stderr and needle in proc.stderr
    assert not [c for c in calls if c.split()[0] in ("colab", "ssh", "uv")]   # no VM, no balance read


def test_a_bad_cap_is_refused(tmp_path):
    proc, calls, out, state = run(tmp_path, args=(CONFIG, "lots"))
    assert proc.returncode == 1 and "cap" in proc.stderr
    assert not [c for c in calls if c.startswith("colab")]


# --- a session --------------------------------------------------------------------------------------

def test_a_full_session_runs_every_step_and_stops_the_vm(tmp_path):
    proc, calls, out, state = run(tmp_path, {"EPOCHS": "1-2", "ADMIT_FOR": "13.5", "STEPS": "run equiv readings"})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    order = ["colab --auth adc usage", "colab --auth adc new -s probe --gpu G4", "sim-session-step serve",
             "localhost:18000/v1/models", "simtest run", "simtest equiv score", "sim-session-step free-gpu",
             "sim-session-step readings", "tar czf - -C /content interp-smoke", "tar czf - -C /content probe",
             "colab --auth adc stop -s probe"]
    pos = [index(calls, o) for o in order]
    assert pos == sorted(pos), list(zip(order, pos))
    assert "Exploratory (N0 Colab probe" in (state / "serve-stdin").read_text()   # serve.sh, as it is
    assert (out / "probe" / "timings.txt").exists()                               # the VM's logs came back
    assert (out / "interp-smoke" / "report.txt").exists()                         # and the readings
    assert "watchdog: session probe, cap 40 units" in (out / "watchdog.log").read_text()
    assert "start balance 150.00" in (out / "watchdog.log").read_text()          # read before colab new
    run_call = calls[index(calls, "simtest run")]
    assert run_call.endswith(f"simtest run {CONFIG} --epochs 1-2 --admit-for 13.5")
    assert not (state / "alive").exists()


def test_a_failing_run_still_stops_the_vm(tmp_path):
    proc, calls, out, state = run(tmp_path, {"RUN_RC": "1"})
    assert proc.returncode == 1
    assert "run: FAILED" in proc.stdout
    assert index(calls, "simtest run") < index(calls, "colab --auth adc stop -s probe")
    assert not [c for c in calls if "simtest equiv" in c or "sim-session-step readings" in c]
    assert (out / "probe" / "timings.txt").exists()   # serve.sh's logs still came back
    assert not (state / "alive").exists()


def test_a_failing_serve_stops_the_vm_before_any_episode(tmp_path):
    proc, calls, out, state = run(tmp_path, {"FAIL_REMOTE": "sim-session-step serve"})
    assert proc.returncode == 1
    assert not [c for c in calls if c.startswith("uv ")]
    assert any("stop -s probe" in c for c in calls)
    assert not (state / "alive").exists()


def test_a_failing_vm_creation_still_stops_it(tmp_path):
    # colab new can fail after the VM was assigned: the stop runs anyway
    proc, calls, out, state = run(tmp_path, before=lambda st: (st / "fail-new").touch())
    assert proc.returncode == 1
    assert any("stop -s probe" in c for c in calls) and not any("simtest" in c for c in calls)


def test_an_incomplete_run_goes_on_to_the_next_steps(tmp_path):
    proc, calls, out, state = run(tmp_path, {"RUN_RC": "3", "STEPS": "run equiv readings"})
    assert proc.returncode == 3, proc.stderr
    assert "INCOMPLETE" in proc.stdout
    assert index(calls, "simtest run") < index(calls, "simtest equiv score") < index(calls, "sim-session-step readings")
    assert not (state / "alive").exists()


def test_a_failing_equiv_stops_the_vm(tmp_path):
    proc, calls, out, state = run(tmp_path, {"EQUIV_RC": "2"})
    assert proc.returncode == 1
    assert not any("sim-session-step readings" in c for c in calls)
    assert not (state / "alive").exists()


def test_readings_only_skips_the_servers(tmp_path):
    proc, calls, out, state = run(tmp_path, {"STEPS": "readings", "READINGS_STEPS": "agent-lens",
                                             "READINGS_BUDGET_MIN": "30"})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not [c for c in calls if "sim-session-step serve" in c or c.startswith("uv ") or " -N " in c]
    cmd = (state / "readings-cmd").read_text()
    assert "STEPS='agent-lens'" in cmd and "BUDGET_MIN=30" in cmd and "gpu_smoke.sh" in cmd
    assert (state / "sim.bundle").read_text().strip() == "bundle"     # the code at HEAD went over
    assert (out / "interp-smoke" / "report.txt").exists()


def test_a_readings_check_that_fails_is_reported_and_the_outputs_still_come_back(tmp_path):
    proc, calls, out, state = run(tmp_path, {"STEPS": "readings", "READINGS_RC": "1"})
    assert proc.returncode == 1
    assert "readings: exit 1" in proc.stdout
    assert (out / "interp-smoke" / "report.txt").exists()
    assert not (state / "alive").exists()


def test_an_existing_session_is_attached_and_its_old_servers_stopped(tmp_path):
    proc, calls, out, state = run(tmp_path, {"STEPS": "run"}, before=lambda st: (st / "alive").touch())
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert not any(" new " in c for c in calls)
    assert index(calls, "sim-session-step free-gpu") < index(calls, "sim-session-step serve")
    assert "attach" in proc.stdout


def test_keep_leaves_the_vm_up_with_the_watchdog_on_it(tmp_path):
    proc, calls, out, state = run(tmp_path, {"KEEP": "1", "STEPS": "run"})
    try:
        assert proc.returncode == 0, proc.stderr + proc.stdout
        assert not any("stop -s probe" in c for c in calls)
        assert (state / "alive").exists() and "KEEP=1" in proc.stdout
    finally:
        (state / "alive").unlink(missing_ok=True)      # the session goes, and the watchdog with it
        for _ in range(100):
            if "watchdog ends" in (out / "watchdog.log").read_text():
                break
            time.sleep(0.1)
    assert "watchdog ends" in (out / "watchdog.log").read_text()


def test_the_script_is_zsh_and_executable():
    assert SCRIPT.read_text().startswith("#!/bin/zsh")
    assert os.access(SCRIPT, os.X_OK)


# --- defaults (02-11-2026) -----------------------------------------------------------------------------

def test_by_default_no_gpu_time_goes_to_the_smoke_test(tmp_path):
    """The readings step runs gpu_smoke.sh, which reads no episode (about 50 GPU minutes). Until a
    reader for the pilot's episodes exists it runs only when STEPS asks for it."""
    proc, calls, out, state = run(tmp_path, {"DRY_RUN": "1"})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "simtest run" in proc.stdout and "simtest equiv score" in proc.stdout
    assert "gpu_smoke.sh" not in proc.stdout and "free-gpu" not in proc.stdout


def test_missing_contexts_are_built_from_the_runs_own_episodes(tmp_path):
    ctx = tmp_path / "new" / "contexts.jsonl"
    proc, calls, out, state = run(tmp_path, {"DRY_RUN": "1", "EQUIV_CONTEXTS": str(ctx)})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    cmds = [x for x in proc.stdout.splitlines() if x.startswith("+ ")]
    run_i = next(i for i, c in enumerate(cmds) if "simtest run" in c)
    build = next(i for i, c in enumerate(cmds) if "simtest equiv contexts" in c)
    score = next(i for i, c in enumerate(cmds) if "simtest equiv score" in c)
    assert run_i < build < score
    assert f"--out {ctx}" in cmds[build] and "logs/n2-pilot" in cmds[build].replace(str(tmp_path / "logs"), "logs")
    # Without a run in this session there is nothing to build them from: refused, as before.
    proc, *_ = run(tmp_path / "b", {"EQUIV_CONTEXTS": str(ctx), "STEPS": "equiv"})
    assert proc.returncode == 1 and "contexts" in proc.stderr


def test_sim_quant_reaches_serve_sh(tmp_path):
    """S9 (sol-loop round 5): after n2-fp8, SIM_QUANT picks the simulator's FP8 scheme; the session
    passes it on (the default sends nothing, so the G4 lines stay as they are)."""
    proc, *_ = run(tmp_path, {"DRY_RUN": "1", "SIM_QUANT": "fp8_per_channel"})
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "SIM_QUANT=fp8_per_channel bash -s" in proc.stdout
    proc, *_ = run(tmp_path / "d", {"DRY_RUN": "1"})
    assert "SIM_QUANT" not in proc.stdout
