#!/bin/bash
# Exploratory (N1 GPU smoke test of the interp tools; docs/INTERP.md, "GPU smoke test").
#
# Runs on the Colab G4 VM (RTX PRO 6000 Blackwell, 96 GB), alone on the GPU: stop
# serve.sh's servers first. Installs vLLM 0.32.0 (as serve.sh) and vllm-lens 1.3.0 into a
# venv of its own: vllm-lens loads into every vLLM process of its environment and forces
# eager mode, so it must not share serve.sh's install. Downloads the two models only if
# serve.sh's copies aren't in /content/models (hf download skips files it already has).
# (explore/interp/gpu_env.sh, shared with read_run.sh.) Then runs explore/interp/gpu_smoke.py's steps in order, one process each, writing
# everything to $OUT, with each step's start and end in $OUT/timings.txt and its output in
# $OUT/<step>.log. A step whose output exists is skipped, so a second run resumes (delete a
# failed step's partial outputs in $OUT/<model>/ first); a failed step is logged and the
# next one runs; the report runs last.
#
# Estimated G4 wall time per step, in minutes (estimates: nothing here has run on a GPU yet):
#   install 3       agent-stock-eager 3       simulator-stock-eager 4
#   download 5*     agent-lens 8              simulator-lens 7
#                   agent-stock-served 5      simulator-stock-served 6
#                   agent-hf 8                report <1
#   * both models, about 100 GB (311 s in N0); about 0 when serve.sh's downloads are there.
# About 50 minutes in all, 8 to 9 compute units. A step is skipped if it would end after
# BUDGET_MIN minutes (default 75) by its estimate; STEPS picks which steps run.
#
# From the laptop (session "probe"; every Colab command uses gcloud ADC):
#   colab --auth adc new -s probe --gpu G4
#   PROXY="colab --auth adc ssh --proxy-mode -s probe"
#   SSH="ssh -o ProxyCommand='$PROXY' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@probe"
#   # the experiment repo at the commit to test, .git included (every output records the commit)
#   tar czf - --exclude .venv --exclude logs -C experiment . | eval "$SSH 'mkdir -p /content/sim && tar xzf - -C /content/sim'"
#   eval "$SSH 'setsid nohup bash /content/sim/explore/interp/gpu_smoke.sh >/content/interp-smoke.log 2>&1 </dev/null &'"
#   eval "$SSH 'cat /content/interp-smoke/timings.txt; tail -3 /content/interp-smoke.log'"   # progress
#   eval "$SSH 'tar czf - -C /content interp-smoke'" > logs/interp/gpu-smoke.tgz             # results, vectors included
#   colab --auth adc stop -s probe                   # always, at the end: the VM bills while it lives
#
# Settings can be overridden with environment variables.
set -euo pipefail
VLLM_VERSION=${VLLM_VERSION:-0.32.0}
LENS_VERSION=${LENS_VERSION:-1.3.0}
AGENT_REPO=${AGENT_REPO:-Qwen/Qwen3.8-27B-FP8}
SIM_REPO=${SIM_REPO:-Qwen/Qwen-AgentWorld-35B-A3B}
CODE=${CODE:-/content/sim}
OUT=${OUT:-/content/interp-smoke}
VENV=${VENV:-/content/interp-venv}
MODELS=${MODELS:-/content/models}
BUDGET_MIN=${BUDGET_MIN:-75}
STEPS=${STEPS:-agent-stock-eager agent-lens simulator-stock-eager simulator-lens agent-stock-served simulator-stock-served agent-hf}
mkdir -p "$OUT" "$MODELS"
# The copied checkout belongs to the laptop's user, so git (here and in the Python steps)
# refuses it as "dubious ownership" and every output recorded no commit (31-10-2026).
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$CODE"
t0=$(date +%s)
stamp() { echo "$1 $(($(date +%s) - t0))s" | tee -a "$OUT/timings.txt"; }

stamp start
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 2000 ]; then echo "the GPU has $used MiB in use: stop serve.sh's servers first"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"
df -h "$MODELS" | tee -a "$OUT/gpu.txt"

source "${BASH_SOURCE[0]%/*}/gpu_env.sh"

smoke() { "$PY" -I "$CODE/explore/interp/gpu_smoke.py" "$@"; }
est() {  # a step's estimated minutes, from the header
  case $1 in
    agent-stock-eager) echo 3 ;; agent-lens) echo 8 ;; agent-stock-served) echo 5 ;; agent-hf) echo 8 ;;
    simulator-stock-eager) echo 4 ;; simulator-lens) echo 7 ;; simulator-stock-served) echo 6 ;;
  esac
}
# step NAME OUTPUT COMMAND...: run unless OUTPUT exists or the step would end after BUDGET_MIN.
step() {
  local name=$1 done=$2 now=$((($(date +%s) - t0) / 60)) min
  min=$(est "$1")
  shift 2
  if [ -e "$done" ]; then stamp "$name skipped (done)"; return; fi
  if [ $((now + min)) -gt "$BUDGET_MIN" ]; then stamp "$name skipped (budget: $now + $min > $BUDGET_MIN min)"; return; fi
  stamp "$name start"
  if "$@" >"$OUT/$name.log" 2>&1; then stamp "$name done"; else stamp "$name FAILED (see $OUT/$name.log)"; fi
}
for s in $STEPS; do
  [ -n "$(est "$s")" ] || { echo "unknown step $s"; exit 1; }
  model=${s%%-*} what=${s#*-}
  case $what in
    stock-*) step "$s" "$OUT/$model/$what.json" env VLLM_LENS_DISABLE=1 "$PY" -I "$CODE/explore/interp/gpu_smoke.py" \
               stock "$MODELS/$model" --name "$model" --mode "${what#stock-}" --out-dir "$OUT" ;;
    *) step "$s" "$OUT/$model/$what.json" smoke "$what" "$MODELS/$model" --name "$model" --out-dir "$OUT" ;;
  esac
done
rc=0
smoke report --out-dir "$OUT" 2>&1 | tee "$OUT/report.txt" || rc=$?
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a "$OUT/gpu.txt"
if [ "$rc" -eq 0 ]; then stamp done; else stamp "done: a check FAILED or is INCOMPLETE (see $OUT/report.txt)"; exit "$rc"; fi
