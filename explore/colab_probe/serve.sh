#!/bin/bash
# Exploratory (N0 Colab probe; LOG.md). Not part of the testbed.
#
# Runs on the Colab VM. Installs a pinned vLLM, downloads the agent and the simulator,
# and serves both on the one GPU, listening on the VM's loopback only: the agent
# (Qwen3.8-27B, official FP8 checkpoint) on :8000 as "agent", the simulator
# (Qwen-AgentWorld-35B-A3B, official bf16 weights quantized to FP8 by vLLM at load) on
# :8001 as "simulator". Writes the GPU, versions, model revisions and each step's time to
# /content/probe/, and each server's log to /content/probe/<name>.log.
#
# From the laptop (session "probe"; every Colab command uses gcloud ADC):
#   colab --auth adc new -s probe --gpu G4          # or A100, H100 (80 GB: shorter contexts, below)
#   PROXY="colab --auth adc ssh --proxy-mode -s probe"
#   SSH="ssh -o ProxyCommand='$PROXY' -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@probe"
#   eval "$SSH 'bash -s'" < explore/colab_probe/serve.sh                 # until both servers are ready
#   eval "$SSH -N -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001" &      # the tunnel
#   curl -s localhost:18000/v1/models; curl -s localhost:18001/v1/models
#   colab --auth adc stop -s probe                   # always, at the end: the VM bills while it lives
#
# Settings can be overridden with environment variables (the defaults are the plan's).
set -euo pipefail
VLLM_VERSION=${VLLM_VERSION:-0.32.0}
AGENT_REPO=${AGENT_REPO:-Qwen/Qwen3.8-27B-FP8}
SIM_REPO=${SIM_REPO:-Qwen/Qwen-AgentWorld-35B-A3B}
# Fraction of the GPU's memory for each server (weights, KV cache) and context length, by card.
# AgentWorld's card asks for at least 128K. On a 96 GB G4 both fit at 128K; on an 80 GB A100 or
# H100 the agent (27.4 GiB of weights, about 8 GiB of KV at 128K) does not, so both get 64K.
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
# On the G4, 0.42 left the agent 7.6 of the 8.3 GiB of KV it needs at 128K (LOG.md, 29-10-2026).
if [ "$GPU_MIB" -ge 90000 ]; then mem=0.47 sim=0.47 len=131072; else mem=0.44 sim=0.50 len=65536; fi
AGENT_MEM=${AGENT_MEM:-$mem}
SIM_MEM=${SIM_MEM:-$sim}
AGENT_LEN=${AGENT_LEN:-$len}
SIM_LEN=${SIM_LEN:-$len}
# Both models are hybrid (linear-attention layers need one cache block per running sequence), and
# vLLM's default of 1024 sequences doesn't fit; the testbed never runs more than a few at once.
MAX_SEQS=${MAX_SEQS:-16}
OUT=/content/probe
mkdir -p "$OUT" /content/models
t0=$(date +%s)
stamp() { echo "$1 $(($(date +%s) - t0))s" | tee -a "$OUT/timings.txt"; }

stamp start
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$OUT/gpu.txt"
df -h /content | tee -a "$OUT/gpu.txt"
pip install -q uv
uv pip install --system -q "vllm==$VLLM_VERSION"
python -c "import vllm, torch, transformers as t; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', t.__version__)" | tee "$OUT/versions.txt"
stamp vllm-installed

# Both downloads at once; the commit of each model goes in versions.txt.
hf download "$AGENT_REPO" --local-dir /content/models/agent >"$OUT/download-agent.log" 2>&1 & a=$!
hf download "$SIM_REPO" --local-dir /content/models/simulator >"$OUT/download-simulator.log" 2>&1 & s=$!
wait $a && wait $s
python - "$AGENT_REPO" "$SIM_REPO" <<'EOF' | tee -a "$OUT/versions.txt"
import sys
from huggingface_hub import model_info
for repo in sys.argv[1:]:
    print(repo, model_info(repo).sha)
EOF
stamp downloaded

# serve NAME PORT MODEL_DIR MEM_FRACTION MAX_LEN [extra vllm flags]: start one server and wait until it
# answers. The second server starts only after the first holds its memory.
serve() {
  local name=$1 port=$2 model=$3 mem=$4 maxlen=$5
  shift 5
  echo "$name: memory $mem, context $maxlen" | tee -a "$OUT/versions.txt"
  setsid nohup vllm serve "$model" --served-model-name "$name" --host 127.0.0.1 --port "$port" \
    --gpu-memory-utilization "$mem" --max-model-len "$maxlen" --max-num-seqs "$MAX_SEQS" --language-model-only \
    --reasoning-parser qwen3 "$@" </dev/null >"$OUT/$name.log" 2>&1 &
  local pid=$!
  until curl -sf "localhost:$port/health" >/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then tail -50 "$OUT/$name.log"; exit 1; fi
    sleep 5
  done
  stamp "$name-ready"
}
serve agent 8000 /content/models/agent "$AGENT_MEM" "$AGENT_LEN" --enable-auto-tool-choice --tool-call-parser qwen3_coder
serve simulator 8001 /content/models/simulator "$SIM_MEM" "$SIM_LEN" --quantization fp8 --trust-remote-code
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a "$OUT/gpu.txt"
grep -h -i -E "KV cache|maximum concurrency|prefix cach" "$OUT/agent.log" "$OUT/simulator.log" | tee "$OUT/kv.txt" || true
stamp done
