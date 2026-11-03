"""explore/colab_probe/watchdog.sh: stops the Colab VM when the session's units reach the cap, when
nothing has run for IDLE_MIN minutes, or at MAX_MIN; it fails safe (a balance it can't read stops the
VM after BAD_READS polls; a metrics read that fails counts as idle).

The script runs against stubs first on PATH: `colab` (usage, sessions, stop, with the CLI's own
output lines), `curl` (vLLM's /metrics) and `pgrep`. Each stub reads its answers from a sequence
file, one line per call (the last line repeats), and records its calls. POLL is a fraction of a
second and IDLE_MIN / MAX_MIN fractions of a minute, so every test runs in a few seconds.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "explore" / "colab_probe" / "watchdog.sh"
ZSH = shutil.which("zsh")
pytestmark = pytest.mark.skipif(ZSH is None, reason="needs zsh")

# Pops the first line of a sequence file (the last line stays, so it repeats).
POP = """pop() { f="$STATE/$1"; line=$(head -1 "$f"); if [ "$(wc -l < "$f")" -gt 1 ]; then
  tail -n +2 "$f" > "$f.t" && mv "$f.t" "$f"; fi; printf '%s' "$line"; }
"""

COLAB = POP + """echo "colab $*" >> "$STATE/calls.log"
case "$*" in
  *" usage")
    b=$(pop usage)
    case $b in
      FAIL) echo "[colab] Warning: failed to fetch compute-unit info: timeout" >&2; exit 1 ;;
      GARBAGE) echo "something else"; exit 0 ;;
    esac
    printf 'Current balance: %s compute units\\nUsage rate: 10.05/hr\\nActive assignments: 1\\n' "$b" ;;
  *" sessions")
    s=$(pop sessions)
    if [ "$s" = FAIL ]; then echo "Traceback: ConnectionError" >&2; exit 1; fi
    if [ -e "$STATE/alive" ]; then
      echo "[probe] https://example.invalid | Hardware: G4 | Shape: STANDARD | Variant: GPU"
    else
      echo "[colab] No active sessions found on server."
    fi ;;
  *" stop -s probe")
    echo "[colab] Stopping session 'probe'..."
    n=$(cat "$STATE/stop-ignore" 2>/dev/null || echo 0)   # a stop that doesn't take (yet)
    if [ "$n" -gt 0 ]; then echo $((n - 1)) > "$STATE/stop-ignore"; exit 0; fi
    rm -f "$STATE/alive"; echo "[colab] Session terminated." ;;
  *) echo "unexpected colab call: $*" >&2; exit 99 ;;
esac
"""

# vLLM's Prometheus text: HELP/TYPE lines, labels, and a decoy metric that must not count.
CURL = POP + """echo "curl $*" >> "$STATE/calls.log"
case "$*" in *:18000/metrics*) port=18000 ;; *:18001/metrics*) port=18001 ;;
  *) echo "unexpected curl call: $*" >&2; exit 99 ;; esac
m=$(pop "metrics-$port")
[ "$m" = FAIL ] && exit 7
r=${m%,*}; w=${m#*,}
cat <<EOF
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} $r.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="m"} $w.0
vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 9.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.25
EOF
"""

PGREP = POP + """echo "pgrep $*" >> "$STATE/calls.log"
p=$(pop pgrep)
[ "$p" = none ] && exit 1
echo 4242
"""


def run(tmp_path: Path, *, usage="100", sessions="STATE", agent="0,0", sim="0,0", pgrep="none",
        alive=True, cap="10", stop_ignore=0, env=None, timeout=30):
    """Each sequence is a newline-separated string (one answer per call, the last repeats)."""
    stubs, state = tmp_path / "stubs", tmp_path / "state"
    stubs.mkdir()
    state.mkdir()
    for name, body in (("colab", COLAB), ("curl", CURL), ("pgrep", PGREP)):
        (stubs / name).write_text("#!/bin/sh\n" + body)
        (stubs / name).chmod(0o755)
    for name, seq in (("usage", usage), ("sessions", sessions), ("metrics-18000", agent),
                      ("metrics-18001", sim), ("pgrep", pgrep)):
        (state / name).write_text(seq.strip() + "\n")
    if alive:
        (state / "alive").touch()
    (state / "stop-ignore").write_text(f"{stop_ignore}\n")
    full = {"PATH": f"{stubs}:/usr/bin:/bin", "STATE": str(state), "POLL": "0.05",
            "IDLE_MIN": "100", "BAD_READS": "3", **(env or {})}
    args = [ZSH, str(SCRIPT), "probe"] + ([cap] if cap is not None else [])
    proc = subprocess.run(args, env=full, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)
    calls = (state / "calls.log").read_text().splitlines() if (state / "calls.log").exists() else []
    return proc, calls


def stops(calls):
    return [c for c in calls if c.endswith("stop -s probe")]


def test_stops_when_the_units_spent_reach_the_cap(tmp_path):
    proc, calls = run(tmp_path, usage="100\n99\n98.5\n97\n96", cap="3", agent="5,0")
    assert proc.returncode == 0, proc.stderr
    assert len(stops(calls)) == 1
    assert "STOP" in proc.stdout and "cap" in proc.stdout
    assert "3.00 of 3 units" in proc.stdout              # stopped at 97, the first read at the cap
    assert sum(1 for c in calls if c.endswith(" usage")) == 4
    assert "gone" in proc.stdout                          # then it saw the session go, and ended


def test_the_cap_counts_from_the_balance_before_the_vm_was_created(tmp_path):
    # session.sh reads the balance before `colab new`: setup's units count against the cap
    proc, calls = run(tmp_path, usage="100", cap="10", agent="5,0", env={"START_BALANCE": "110.5"})
    assert proc.returncode == 0, proc.stderr
    assert len(stops(calls)) == 1 and "cap" in proc.stdout


def test_under_the_cap_it_keeps_watching(tmp_path):
    # busy and under the cap: only the maximum time stops it
    proc, calls = run(tmp_path, usage="100\n99.9\n99.8", cap="10", agent="5,0",
                      env={"MAX_MIN": "0.01"})
    assert proc.returncode == 0, proc.stderr
    assert "maximum time" in proc.stdout and "cap" not in proc.stdout.split("STOP")[1]


@pytest.mark.parametrize("bad", ["FAIL", "GARBAGE"])
def test_a_balance_it_cannot_read_stops_the_vm_after_n_reads(tmp_path, bad):
    # never "under the cap" by default: three failed reads in a row and it stops
    proc, calls = run(tmp_path, usage=f"100\n{bad}", cap="10", agent="5,0")
    assert proc.returncode == 0, proc.stderr
    assert len(stops(calls)) == 1
    assert "balance" in proc.stdout.split("STOP")[1]
    assert sum(1 for c in calls if c.endswith(" usage")) == 4   # one good read, then 3 failed


def test_no_balance_ever_read_stops_the_vm(tmp_path):
    proc, calls = run(tmp_path, usage="FAIL", cap="10", agent="5,0")
    assert proc.returncode == 0, proc.stderr
    assert len(stops(calls)) == 1 and "balance" in proc.stdout.split("STOP")[1]


def test_a_good_read_resets_the_failed_reads(tmp_path):
    # FAIL, FAIL, ok, FAIL, FAIL, ok...: never 3 in a row, so only the maximum time stops it
    proc, calls = run(tmp_path, usage="\n".join(["FAIL", "FAIL", "100"] * 20), cap="10", agent="5,0",
                      env={"MAX_MIN": "0.02"})
    assert proc.returncode == 0, proc.stderr
    assert "maximum time" in proc.stdout.split("STOP")[1]


def test_stops_when_nothing_runs_for_the_idle_time(tmp_path):
    proc, calls = run(tmp_path, agent="0,0", sim="0,0", pgrep="none", env={"IDLE_MIN": "0.005"})
    assert proc.returncode == 0, proc.stderr
    assert len(stops(calls)) == 1
    assert "idle" in proc.stdout.split("STOP")[1]
    pg = [c for c in calls if c.startswith("pgrep")]
    assert pg and "simtest (run|preflight|equiv)" in pg[0]


def test_a_metrics_read_that_fails_counts_as_idle(tmp_path):
    # the tunnel is down or the servers are gone: that's idle, not busy
    proc, calls = run(tmp_path, agent="FAIL", sim="FAIL", pgrep="none", env={"IDLE_MIN": "0.005"})
    assert proc.returncode == 0, proc.stderr
    assert "idle" in proc.stdout.split("STOP")[1]
    assert "agent=down" in proc.stdout


@pytest.mark.parametrize("agent,sim,pgrep", [
    ("2,0", "0,0", "none"),     # the agent is generating
    ("0,0", "0,3", "none"),     # the simulator has requests waiting
    ("FAIL", "FAIL", "found"),  # a simtest process is alive (between calls, or the servers restarting)
], ids=["running", "waiting", "process"])
def test_anything_running_is_not_idle(tmp_path, agent, sim, pgrep):
    proc, calls = run(tmp_path, agent=agent, sim=sim, pgrep=pgrep,
                      env={"IDLE_MIN": "0.003", "MAX_MIN": "0.015"})
    assert proc.returncode == 0, proc.stderr
    reason = proc.stdout.split("STOP")[1]
    assert "maximum time" in reason and "idle" not in reason


def test_the_idle_clock_restarts_when_something_runs(tmp_path):
    # busy for 12 polls (longer than the idle time), then idle: it must not stop while busy
    busy = "\n".join(["1,0"] * 12 + ["0,0"])
    proc, calls = run(tmp_path, agent=busy, sim="0,0", env={"IDLE_MIN": "0.004"})
    assert proc.returncode == 0, proc.stderr
    assert "idle" in proc.stdout.split("STOP")[1]
    first_stop = next(i for i, c in enumerate(calls) if c.endswith("stop -s probe"))
    agent_reads = [c for c in calls[:first_stop] if ":18000/metrics" in c]
    assert len(agent_reads) >= 13          # every busy reading came before the stop


def test_the_waiting_count_by_reason_is_not_counted(tmp_path):
    # vllm:num_requests_waiting_by_reason is 9 in the stub; only the exact metric names count
    proc, calls = run(tmp_path, agent="0,0", sim="0,0", env={"IDLE_MIN": "0.005"})
    assert "idle" in proc.stdout.split("STOP")[1]


def test_logs_a_line_every_poll(tmp_path):
    # five polls, then the cap: one line each, whatever the machine's speed
    proc, calls = run(tmp_path, agent="1,2", sim="16,8", usage="100\n99.5\n99.5\n99.5\n80", cap="10")
    lines = [x for x in proc.stdout.splitlines() if "balance=" in x]
    assert len(lines) == 5
    assert "agent=1/2" in lines[-2] and "simulator=16/8" in lines[-2] and "spent=0.50" in lines[-2]


def test_exits_when_the_session_is_gone(tmp_path):
    proc, calls = run(tmp_path, alive=False)
    assert proc.returncode == 0, proc.stderr
    assert not stops(calls)
    assert "gone" in proc.stdout


def test_a_failed_session_list_is_not_a_gone_session(tmp_path):
    # `colab sessions` failing must not end the watchdog (the VM would bill unwatched)
    # (a list that fails after the stop is retried with another stop, so there may be more than one)
    proc, calls = run(tmp_path, sessions="FAIL\nFAIL\nFAIL\nSTATE", agent="5,0", env={"MAX_MIN": "0.01"})
    assert proc.returncode == 0, proc.stderr
    assert stops(calls) and "maximum time" in proc.stdout.split("STOP")[1]


def test_after_a_stop_it_retries_until_the_session_is_gone(tmp_path):
    # the first two stop calls return without stopping anything: it stops again, and ends only once
    # the session is gone
    proc, calls = run(tmp_path, usage="FAIL", cap="10", agent="5,0", stop_ignore=2)
    assert proc.returncode == 0, proc.stderr
    assert len(stops(calls)) == 3 and "gone" in proc.stdout


@pytest.mark.parametrize("cap", [None, "abc", "-3", "0"])
def test_a_missing_or_bad_cap_is_refused(tmp_path, cap):
    proc, calls = run(tmp_path, cap=cap)
    assert proc.returncode == 2
    assert not calls                      # refused before calling anything
    assert "cap" in proc.stderr
