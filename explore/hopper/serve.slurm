#!/bin/bash
# Exploratory (Hopper sessions; docs/SESSIONS.md). Not part of the testbed.
#
# A Slurm job that serves both models for one chunk of a run, both at 128K, on the profile of
# explore/colab_probe/serve.sh that PROFILE names: two-gpu (two GPUs of one node, one model per card;
# the default, and the --gres below) or one-gpu-94g (one card of 90-97 GB, both models on it, as on
# Hopper's H100 NVL nodes; submit_chunks.sh then passes --gres=gpu:1; or one-gpu-96g / -141g). The job refuses to serve any
# profile but the one named. submit_chunks.sh, on the laptop, submits it once per chunk; run_chunks.sh, on the laptop,
# waits for it, opens the tunnel and runs the chunk's episodes. Inspect, Docker and the logs stay on
# the laptop; only vLLM runs here.
#
# It runs serve.sh as it is (SERVE_SH: its copy on the cluster) with WORK on scratch, then keeps the
# servers alive until the job ends. It checks both every HEALTH_POLL seconds and ends the job when
# one fails 3 checks in a row, so the GPUs go back to the queue. It writes STATUS_DIR/<job id>:
#   node=<the node>  state=starting|ready|failed|server-died|ended  start=<epoch s>  end=<epoch s>
# where end is the start plus the time limit, and, before it says ready, STATUS_DIR/<job id>.server:
# serve.sh's server.txt (the GPU, both servers' served lines, what vLLM quantized). run_chunks.sh reads
# both through the login host.
#
# The time limit is a parameter, not set here: submit_chunks.sh passes it (sbatch --time, and
# TIME_LIMIT_MIN for the status file). Hopper's gpu partition allows up to 3 days. A job killed at its time limit
# interrupts the episodes still running, and interrupted episodes are re-run later. Long episodes are
# the ones cut most often, so the re-runs favour short ones: each chunk's --admit-for must stop
# admitting episodes early enough to leave room for the longest one.
#
# RELAY_HOST (optional): the login host's name as the node sees it. Set it where SSH from the login
# host into a node is refused (Hopper: even with our job on it, LOG.md 23/24-11-2026). Once both
# servers answer, the job runs `ssh -R` from the node to RELAY_HOST with RELAY_KEY (default
# ~/.ssh/id_ed25519_node, a key the login host accepts for forwarding only), publishing each server
# as a Unix socket in STATUS_DIR/<job id>.sock/ (mode 700): agent.sock and simulator.sock. The servers
# stay on the node's loopback and nothing listens on the cluster network. The status file then says
# sock=<that directory>, and run_chunks.sh forwards the laptop's ports to the sockets. The relay is
# checked with the servers and restarted when it drops; 3 failed restarts in a row end the job
# (state=relay-died).
#
# Required (sbatch --export): WORK (scratch, about 110 GB free; the venv and the models stay there for
# the next jobs), SERVE_SH, STATUS_DIR, TIME_LIMIT_MIN. Optional: MIX (no effect on two GPUs; on one, the memory split), EAGER,
# SIM_QUANT (submit_chunks.sh always names fp8_per_channel), PROFILE (two-gpu; submit_chunks.sh always
# names it), HEALTH_POLL (60).
#SBATCH --job-name=sim-serve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
set -euo pipefail
: "${WORK:?set WORK to a scratch directory with about 110 GB free}"
: "${SERVE_SH:?set SERVE_SH to serve.sh on the cluster}"
: "${STATUS_DIR:?set STATUS_DIR to a directory the login host sees}"
: "${TIME_LIMIT_MIN:?set TIME_LIMIT_MIN to the job time limit in minutes}"
JOB=${SLURM_JOB_ID:?not running under Slurm}
NODE=${SLURMD_NODENAME:-$(hostname -s)}
STATUS=$STATUS_DIR/$JOB
start=$(date +%s)
end=$((start + TIME_LIMIT_MIN * 60))
SOCK=""
[ -z "${RELAY_HOST:-}" ] || SOCK=$STATUS_DIR/$JOB.sock
status() {   # STATE: rewrite the status file at once (a reader never sees half of it)
  { printf 'node=%s\nstate=%s\nstart=%s\nend=%s\nupdated=%s\n' "$NODE" "$1" "$start" "$end" "$(date +%s)"
    [ -z "$SOCK" ] || printf 'sock=%s\n' "$SOCK"; } >"$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
}
RELAY_PID=""
relay_stop() {
  [ -z "$RELAY_PID" ] || kill "$RELAY_PID" 2>/dev/null || true
  RELAY_PID=""
  [ -z "$SOCK" ] || rm -f "$SOCK/agent.sock" "$SOCK/simulator.sock"
}
relay_start() {   # the reverse tunnel; succeeds once both sockets exist on the login host's side
  local i
  relay_stop
  mkdir -p "$SOCK" && chmod 700 "$SOCK" || return 1
  ssh -i "${RELAY_KEY:-$HOME/.ssh/id_ed25519_node}" -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -N \
    -R "$SOCK/agent.sock:127.0.0.1:8000" -R "$SOCK/simulator.sock:127.0.0.1:8001" "$RELAY_HOST" \
    </dev/null >>"$STATUS.relay.log" 2>&1 &
  RELAY_PID=$!
  for i in $(seq 1 "${RELAY_WAIT:-30}"); do
    if [ -S "$SOCK/agent.sock" ] && [ -S "$SOCK/simulator.sock" ]; then return 0; fi
    kill -0 "$RELAY_PID" 2>/dev/null || break
    sleep "${RELAY_TICK:-2}"
  done
  relay_stop   # a failed attempt never stays up to pass the next check
  return 1
}
# The servers outlive serve.sh (setsid), and a job whose process tracking follows the process tree
# (Hopper: proctrack/linuxproc) loses them once serve.sh exits: a cancelled job left both serving,
# outside any job, on its GPU (LOG.md, 23/24-11-2026). So the job stops them itself, by the process
# groups serve.sh records, whenever it exits: TERM first, KILL after STOP_WAIT seconds (20; Slurm's
# KillWait is 30). WORK is shared by jobs (and nodes), so each server records its own group, before
# vLLM starts, in this job's PID_DIR (STATUS_DIR/<job id>.pid.d), and the job stops only those: never
# groups an earlier job left. Cleanup first closes registration (PID_DIR/closed; a server launched
# after that gives up), then reads the groups: a cancel at any moment finds a server or stops it.
export PID_DIR=$STATUS_DIR/$JOB.pid.d
stop_servers() {
  set +e   # cleanup is best-effort: no failed write (a full disk) may stop it before the groups are signalled
  local pg pgs i alive
  { mkdir -p "$PID_DIR" && touch "$PID_DIR/closed"; } \
    || echo "serve.slurm: could not close registration ($PID_DIR/closed); stopping the servers anyway"
  pgs=$(cat "$PID_DIR"/*.pid 2>/dev/null || true)
  [ -n "$pgs" ] || return 0
  for pg in $pgs; do kill -TERM -- "-$pg" 2>/dev/null || true; done
  for i in $(seq 1 "${STOP_WAIT:-20}"); do
    alive=0
    for pg in $pgs; do kill -0 -- "-$pg" 2>/dev/null && alive=1; done
    [ "$alive" = 1 ] || break
    sleep 1
  done
  for pg in $pgs; do kill -KILL -- "-$pg" 2>/dev/null || true; done
  rm -f "$PID_DIR"/*.pid   # 'closed' stays: a server still launching must find it
}
mkdir -p "$STATUS_DIR"
rm -rf "$PID_DIR"
status starting
# The time limit and scancel send SIGTERM (vLLM gets it too).
trap 'set +e; stop_servers; relay_stop' EXIT   # best-effort, the servers first: nothing may keep them up
trap 'status ended; exit 0' TERM

# The profile serve.sh would pick on this node's GPUs (its dry run starts nothing).
if ! dry=$(DRY_RUN=1 bash "$SERVE_SH" 2>&1); then
  echo "$dry"
  status failed
  exit 1
fi
profile=$(echo "$dry" | sed -n 's/^profile: \([^ ]*\).*/\1/p')
if [ "$profile" != "${PROFILE:-two-gpu}" ]; then
  echo "serve.slurm: serve.sh picks the $profile profile here, not ${PROFILE:-two-gpu}: refusing to serve it"
  status failed
  exit 1
fi
echo "$dry"

# A server already answering here is not this job's (one left from an earlier job, or someone else's):
# serving next to it would fail, and its health checks would pass for this job.
for port in 8000 8001; do
  if curl -s -o /dev/null -m 5 "localhost:$port/health"; then
    echo "serve.slurm: something already answers on $NODE:$port: refusing to serve next to it"
    status failed
    exit 1
  fi
done
if ! bash "$SERVE_SH" || ! cp "$WORK/probe/server.txt" "$STATUS.server"; then
  status failed
  exit 1
fi
if [ -n "$SOCK" ] && ! relay_start; then
  echo "serve.slurm: the relay to $RELAY_HOST didn't come up (see $STATUS.relay.log)"
  relay_stop
  status failed
  exit 1
fi
status ready

fails=0 rfails=0
while true; do
  sleep "${HEALTH_POLL:-60}" &
  wait $!
  if curl -sf -m 10 localhost:8000/health >/dev/null && curl -sf -m 10 localhost:8001/health >/dev/null; then
    fails=0
  else
    fails=$((fails + 1))
    echo "$(date -u +%H:%M:%S) a server failed its health check ($fails in a row)"
    if [ "$fails" -ge 3 ]; then
      relay_stop
      status server-died
      exit 1
    fi
  fi
  if [ -n "$SOCK" ] && ! kill -0 "$RELAY_PID" 2>/dev/null; then
    if relay_start; then
      rfails=0
      echo "$(date -u +%H:%M:%S) the relay dropped and is back"
    else
      rfails=$((rfails + 1))
      echo "$(date -u +%H:%M:%S) the relay is down ($rfails failed restarts in a row)"
      if [ "$rfails" -ge 3 ]; then
        relay_stop
        status relay-died
        exit 1
      fi
    fi
  fi
done
