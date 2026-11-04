#!/bin/bash
# Exploratory (Hopper sessions; docs/SESSIONS.md). Not part of the testbed.
#
# A Slurm job that serves both models for one chunk of a run: two GPUs of one node, one model per
# card, both at 128K (explore/colab_probe/serve.sh's two-gpu profile; the job refuses to serve any
# other). submit_chunks.sh, on the laptop, submits it once per chunk; run_chunks.sh, on the laptop,
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
# TIME_LIMIT_MIN for the status file). Hopper's limit is unknown. A job killed at its time limit
# interrupts the episodes still running, and interrupted episodes are re-run later. Long episodes are
# the ones cut most often, so the re-runs favour short ones: each chunk's --admit-for must stop
# admitting episodes early enough to leave room for the longest one.
#
# Required (sbatch --export): WORK (scratch, about 110 GB free; the venv and the models stay there for
# the next jobs), SERVE_SH, STATUS_DIR, TIME_LIMIT_MIN. Optional: MIX (no effect on two GPUs), EAGER,
# SIM_QUANT (submit_chunks.sh always names fp8_per_channel), PROFILE (two-gpu), HEALTH_POLL (60).
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
status() {   # STATE: rewrite the status file at once (a reader never sees half of it)
  printf 'node=%s\nstate=%s\nstart=%s\nend=%s\nupdated=%s\n' "$NODE" "$1" "$start" "$end" "$(date +%s)" >"$STATUS.tmp"
  mv "$STATUS.tmp" "$STATUS"
}
mkdir -p "$STATUS_DIR"
status starting
# The time limit and scancel send SIGTERM (vLLM gets it too).
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

if ! bash "$SERVE_SH" || ! cp "$WORK/probe/server.txt" "$STATUS.server"; then
  status failed
  exit 1
fi
status ready

fails=0
while true; do
  sleep "${HEALTH_POLL:-60}" &
  wait $!
  if curl -sf -m 10 localhost:8000/health >/dev/null && curl -sf -m 10 localhost:8001/health >/dev/null; then
    fails=0
  else
    fails=$((fails + 1))
    echo "$(date -u +%H:%M:%S) a server failed its health check ($fails in a row)"
    if [ "$fails" -ge 3 ]; then
      status server-died
      exit 1
    fi
  fi
done
