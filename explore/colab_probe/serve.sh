#!/bin/bash
# Exploratory (N0 Colab probe; LOG.md). Not part of the testbed.
#
# Runs on the GPU machine: a Colab VM, or a node of Lewis's Hopper cluster. Installs a pinned
# vLLM, downloads the agent and the simulator, and serves both, listening on the machine's
# loopback only: the agent (Qwen3.8-27B, official FP8 checkpoint) on :8000 as "agent", the
# simulator (Qwen-AgentWorld-35B-A3B, official bf16 weights quantized to FP8 by vLLM at load) on
# :8001 as "simulator". Writes the GPU, versions, model revisions, the served command lines and
# each step's time to $WORK/probe/, and each server's log to $WORK/probe/<name>.log.
#
# From the laptop (session "probe"; every Colab command uses gcloud ADC):
#   colab --auth adc new -s probe --gpu G4          # or A100, H100 (80 GB: shorter contexts, below)
#   (MIX=c3 for C3-heavy runs, EAGER=1 for steering runs: prefix them to bash -s, e.g. "MIX=c3 bash -s")
#   PROXY="colab --auth adc ssh --proxy-mode -s probe"
#   SSH="ssh -o ProxyCommand='$PROXY' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@probe"
#   eval "$SSH 'bash -s'" < explore/colab_probe/serve.sh                 # until both servers are ready
#   eval "$SSH -N -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001" &      # the tunnel
#   curl -s localhost:18000/v1/models; curl -s localhost:18001/v1/models
#   colab --auth adc stop -s probe                   # always, at the end: the VM bills while it lives
#
# On Hopper (or any Linux GPU machine without Colab's /content), give a work directory with about
# 110 GB free (the downloads are 29 + 67 GB). vLLM goes into a venv there ($WORK/venv, or VENV), and
# Hugging Face's cache into $WORK/hf-home unless HF_HOME is set. On a node behind a login host:
#   ssh <node> 'WORK=/scratch/$USER/sim MIX=c3 bash -s' < explore/colab_probe/serve.sh
#   SSH_TARGET=<node> SSH_JUMP=<login> explore/colab_probe/tunnel.sh >/tmp/tunnel.log 2>&1 &   # the tunnel
#
# Profiles, by the GPUs the machine has (CUDA_VISIBLE_DEVICES, or GPUS, picks them):
#   one-gpu-96g   one 90-135 GB card (Colab G4): both models at 128K, 0.47 / 0.47 (MIX=c3: 0.42 / 0.52)
#   one-gpu-141g  one card of 135 GB or more (H200): as the G4, with more KV cache for each model
#   one-gpu-80g   one 80 GB card (H100, A100): both at 64K. NOT equivalent to the G4 (a loud warning)
#   two-gpu       two cards of 80 GB or more: one model per card, both at 128K, 0.90 each
#
# DRY_RUN=1 prints the profile and the vllm command lines and exits: nothing is installed,
# downloaded, created or started. GPU_MIB and GPU_COUNT stand in for nvidia-smi there. The G4's
# lines are pinned by tests/test_serve_script.py.
#
# SIM_QUANT: how vLLM quantizes the simulator at load. fp8 (the default; every episode so far): in
# vLLM 0.32 the deprecated name of fp8_per_tensor, whose MoE scales its activations over the whole
# batch. fp8_per_channel: per-channel weight scales, per-token activations in every layer, the MoE
# included; it pins the MoE kernel to Triton, the one fp8 uses on the G4 and the H100. Nothing else
# is accepted (docs/BACKENDS.md). SIM_ONLY=1 (re)starts the simulator alone next to a running agent:
# it stops the simulator an earlier run started in this WORK, and keeps its log as simulator.<time>.log.
#
# Settings can be overridden with environment variables (the defaults are the plan's).
set -euo pipefail
VLLM_VERSION=${VLLM_VERSION:-0.32.0}
AGENT_REPO=${AGENT_REPO:-Qwen/Qwen3.8-27B-FP8}
SIM_REPO=${SIM_REPO:-Qwen/Qwen-AgentWorld-35B-A3B}
die() { echo "serve.sh: $*" >&2; exit 1; }
# One backend per experiment (AGENTS.md): every episode so far ran on vLLM 0.32.0.
[ "$VLLM_VERSION" = 0.32.0 ] || die "vLLM is pinned at 0.32.0 (got $VLLM_VERSION): another version is another backend"
SIM_QUANT=${SIM_QUANT:-fp8}
case "$SIM_QUANT" in
  fp8) SIM_QUANT_ARGS=(--quantization fp8) ;;
  fp8_per_channel) SIM_QUANT_ARGS=(--quantization fp8_per_channel --moe-backend triton) ;;
  *) die "SIM_QUANT is fp8 or fp8_per_channel (got $SIM_QUANT): another quantization is another backend" ;;
esac

# The work directory: Colab's /content by default there; anywhere else it must be given.
if [ -z "${WORK:-}" ]; then
  [ -d /content ] || die "set WORK to a directory with about 110 GB free (no /content: this isn't Colab)"
  WORK=/content
fi
# vLLM's install: Colab's system Python (as before); elsewhere a venv in WORK. VENV= (empty) forces
# the system install, VENV=<dir> a venv there.
if [ "$WORK" = /content ]; then VENV=${VENV-}; else VENV=${VENV-$WORK/venv}; fi

# The GPUs. CUDA_VISIBLE_DEVICES (set by a scheduler such as Slurm) or GPUS lists the devices to
# use; otherwise nvidia-smi's. GPU_MIB is the first card's memory.
if [ -n "${GPUS:-}" ]; then :; elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then GPUS=$CUDA_VISIBLE_DEVICES; fi
if [ -z "${GPU_COUNT:-}" ]; then
  if [ -n "${GPUS:-}" ]; then GPU_COUNT=$(echo "$GPUS" | tr ',' '\n' | grep -c .)
  else GPU_COUNT=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | grep -c .); fi
fi
if [ -z "${GPUS:-}" ]; then GPUS=0; i=1; while [ "$i" -lt "$GPU_COUNT" ]; do GPUS+=",$i"; i=$((i + 1)); done; fi
GPU_MIB=${GPU_MIB:-$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)}
AGENT_GPU="" SIM_GPU=""

# Fraction of the GPU's memory for each server (weights, KV cache) and context length, by card.
# AgentWorld's card asks for at least 128K. On a 96 GB G4 both fit at 128K; on an 80 GB A100 or
# H100 the agent (27.4 GiB of weights, about 8 GiB of KV at 128K) does not, so both get 64K.
if [ "$GPU_COUNT" -ge 2 ]; then
  [ "$GPU_MIB" -ge 79000 ] || die "two GPUs of $GPU_MIB MiB are too small: one model per GPU needs about 80 GB each"
  # One model per card: no memory split to choose, so MIX has no effect.
  PROFILE=two-gpu mem=0.90 sim=0.90 len=131072
  AGENT_GPU=$(echo "$GPUS" | cut -d, -f1) SIM_GPU=$(echo "$GPUS" | cut -d, -f2)
elif [ "$GPU_MIB" -ge 90000 ]; then
  if [ "$GPU_MIB" -ge 135000 ]; then PROFILE=one-gpu-141g; else PROFILE=one-gpu-96g; fi
  # On the G4, 0.42 left the agent 7.6 of the 8.3 GiB of KV it needs at 128K (LOG.md, 29-10-2026).
  mem=0.47 sim=0.47 len=131072
  # MIX=c3: runs that are mostly C1-C3 give the simulator more. In session 5 (12-14 C3 episodes at
  # once) the simulator's KV cache sat at 75-99% with 300 preemptions, and the agent's never passed
  # about 30% of its 218K tokens (LOG.md, 01-11-2026). 0.42 leaves the agent about 117K tokens, still
  # about twice what it used. Keep the default for C0-heavy runs: 12 C0 agents need the agent's cache.
  if [ "${MIX:-}" = c3 ]; then mem=0.42 sim=0.52; fi
elif [ "$GPU_MIB" -ge 79000 ]; then
  PROFILE=one-gpu-80g mem=0.44 sim=0.50 len=65536
  cat >&2 <<'EOF'
!!! WARNING: one 80 GB GPU (H100, A100): both models get a 64K context, not the G4's 128K.
!!! Long C3 episodes can pass 64K: session 5's largest simulator call was 69K tokens
!!! (LOG.md, 01-11-2026), so such calls fail with a context-length error.
!!! This backend is NOT equivalent to the G4: an experiment on it runs every condition on it,
!!! never mixed with G4 episodes. Two GPUs, or one H200, serve both at 128K.
EOF
else
  die "one GPU of $GPU_MIB MiB is too small for both models (they need about 80 GB)"
fi
AGENT_MEM=${AGENT_MEM:-$mem}
SIM_MEM=${SIM_MEM:-$sim}
AGENT_LEN=${AGENT_LEN:-$len}
SIM_LEN=${SIM_LEN:-$len}
# Both models are hybrid (linear-attention layers need one cache block per running sequence), and
# vLLM's default of 1024 sequences doesn't fit (the G4 has about 358 blocks for the agent at 0.51
# and 277 for the simulator at 0.43; that is the startup check, not how many long contexts fit).
# 64 is a provisional ceiling above what the client sends; the number of episodes run at once
# (simtest's max_connections) sets the real concurrency (LOG.md, 29-10-2026).
MAX_SEQS=${MAX_SEQS:-64}
OUT=$WORK/probe
MODELS=$WORK/models
# EAGER=1: serve on the engine settings the steering engine uses (vllm-lens forces eager mode and
# vLLM's V1 model runner). The compiled default does not give the eager engine's tokens (session 5,
# LOG.md, 01-11-2026), so steered and unsteered episodes (N4) both run with EAGER=1.
if [ "${EAGER:-}" = 1 ]; then ENGINE=(--enforce-eager); else ENGINE=(); fi

# The command line of each server, built once: the dry run prints it, serve runs it.
AGENT_ARGS=(--enable-auto-tool-choice --tool-call-parser qwen3_coder ${ENGINE[@]+"${ENGINE[@]}"})
SIM_ARGS=("${SIM_QUANT_ARGS[@]}" --trust-remote-code ${ENGINE[@]+"${ENGINE[@]}"})
vllm_args() {   # NAME PORT MODEL_DIR MEM_FRACTION MAX_LEN [extra vllm flags]
  local name=$1 port=$2 model=$3 mem=$4 maxlen=$5
  shift 5
  CMD=(vllm serve "$model" --served-model-name "$name" --host 127.0.0.1 --port "$port"
    --gpu-memory-utilization "$mem" --max-model-len "$maxlen" --max-num-seqs "$MAX_SEQS" --language-model-only
    --reasoning-parser qwen3 "$@")
}
show_cmd() {    # GPU: the command line as run, with the variables that change it
  local env=""
  [ -z "$1" ] || env+="CUDA_VISIBLE_DEVICES=$1 "
  [ "${EAGER:-}" != 1 ] || env+="VLLM_USE_V2_MODEL_RUNNER=0 "
  printf '%s%s' "$env" "${CMD[0]}"
  printf ' %q' "${CMD[@]:1}"
  printf '\n'
}

if [ "${DRY_RUN:-}" = 1 ]; then
  echo "profile: $PROFILE ($GPU_COUNT GPU of $GPU_MIB MiB), vllm $VLLM_VERSION, work $WORK"
  if [ "${SIM_ONLY:-}" != 1 ]; then
    vllm_args agent 8000 "$MODELS/agent" "$AGENT_MEM" "$AGENT_LEN" "${AGENT_ARGS[@]}"
    show_cmd "$AGENT_GPU"
  fi
  vllm_args simulator 8001 "$MODELS/simulator" "$SIM_MEM" "$SIM_LEN" "${SIM_ARGS[@]}"
  show_cmd "$SIM_GPU"
  exit 0
fi

mkdir -p "$OUT" "$MODELS"
t0=$(date +%s)
stamp() { echo "$1 $(($(date +%s) - t0))s" | tee -a "$OUT/timings.txt"; }

stamp start
echo "profile: $PROFILE ($GPU_COUNT GPU of $GPU_MIB MiB)" | tee "$OUT/profile.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"
df -h "$WORK" | tee -a "$OUT/gpu.txt"
if [ -z "$VENV" ]; then
  pip install -q uv
  uv pip install --system -q "vllm==$VLLM_VERSION"
else
  command -v uv >/dev/null || { python3 -m pip install --user -q uv && export PATH="$HOME/.local/bin:$PATH"; }
  [ -x "$VENV/bin/python" ] || uv venv -q --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" -q "vllm==$VLLM_VERSION"
  export PATH="$VENV/bin:$PATH"
  export HF_HOME=${HF_HOME:-$WORK/hf-home}
fi
python -c "import vllm, torch, transformers as t; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', t.__version__)" | tee "$OUT/versions.txt"
stamp vllm-installed

# Both downloads at once; the commit of each model goes in versions.txt.
hf download "$AGENT_REPO" --local-dir "$MODELS/agent" >"$OUT/download-agent.log" 2>&1 & a=$!
hf download "$SIM_REPO" --local-dir "$MODELS/simulator" >"$OUT/download-simulator.log" 2>&1 & s=$!
wait $a && wait $s
python - "$AGENT_REPO" "$SIM_REPO" <<'EOF' | tee -a "$OUT/versions.txt"
import sys
from huggingface_hub import model_info
for repo in sys.argv[1:]:
    print(repo, model_info(repo).sha)
EOF
stamp downloaded

# serve NAME PORT MODEL_DIR MEM_FRACTION MAX_LEN GPU [extra vllm flags]: start one server and wait
# until it answers. The second server starts only after the first holds its memory.
serve() {
  local name=$1 port=$2 model=$3 mem=$4 maxlen=$5 gpu=$6
  shift 6
  echo "$name: memory $mem, context $maxlen" | tee -a "$OUT/versions.txt"
  vllm_args "$name" "$port" "$model" "$mem" "$maxlen" "$@"
  show_cmd "$gpu" | tee -a "$OUT/versions.txt"
  if [ -n "$gpu" ]; then
    CUDA_VISIBLE_DEVICES=$gpu setsid nohup "${CMD[@]}" </dev/null >"$OUT/$name.log" 2>&1 &
  else
    setsid nohup "${CMD[@]}" </dev/null >"$OUT/$name.log" 2>&1 &
  fi
  local pid=$!
  until curl -sf "localhost:$port/health" >/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then tail -50 "$OUT/$name.log"; exit 1; fi
    sleep 5
  done
  stamp "$name-ready"
}
if [ "${EAGER:-}" = 1 ]; then export VLLM_USE_V2_MODEL_RUNNER=0; fi
echo "engine: ${ENGINE[*]:-compiled (vLLM default)}" | tee -a "$OUT/versions.txt"
# SIM_ONLY=1: stop the simulator an earlier run started (setsid made each server its own process
# group, so its engine processes go with it) and wait until it is gone: SIGKILL after 3 minutes,
# and give up after 4.
stop_simulator() {
  local pids pid i
  pids=$(pgrep -f -- "$MODELS/simulator --served-model-name simulator" || true)   # this WORK's only
  for pid in $pids; do kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; done
  for pid in $pids; do
    i=0
    while pgrep -g "$pid" >/dev/null; do
      i=$((i + 1))
      [ "$i" -le 120 ] || die "the running simulator (process group $pid) did not stop"
      if [ "$i" -gt 90 ]; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
      sleep 2
    done
  done
  [ -z "$pids" ] || stamp simulator-stopped
  if [ -f "$OUT/simulator.log" ]; then mv "$OUT/simulator.log" "$OUT/simulator.$(date +%s).log"; fi
}
if [ "${SIM_ONLY:-}" = 1 ]; then
  stop_simulator
else
  serve agent 8000 "$MODELS/agent" "$AGENT_MEM" "$AGENT_LEN" "$AGENT_GPU" "${AGENT_ARGS[@]}"
fi
serve simulator 8001 "$MODELS/simulator" "$SIM_MEM" "$SIM_LEN" "$SIM_GPU" "${SIM_ARGS[@]}"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a "$OUT/gpu.txt"
grep -h -i -E "KV cache|maximum concurrency|prefix cach" "$OUT/agent.log" "$OUT/simulator.log" | tee "$OUT/kv.txt" || true
stamp done
