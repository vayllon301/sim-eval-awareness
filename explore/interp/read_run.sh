#!/bin/bash
# Exploratory (reading a run from the inside; docs/INTERP.md, "Reading a run"; docs/SESSIONS.md).
#
# Runs on the GPU machine, alone on the GPU (stop serve.sh's servers first): a Colab VM through
# session.sh's read step, or a Hopper node through explore/hopper/read.slurm. Episodes never run here:
# it reads what `simtest interp export` wrote, and writes readings. One engine at a time, one process
# each, in the venv of gpu_env.sh.
#
# Steps, in PARTS' order (default: proj):
#   check     explore/interp/read_check.py, on the first GPU run: the longest and the shortest episode
#             on vllm and on hf, the agreement of the two paths, and the chunk noise. About 15 minutes.
#             Needs VECTORS and PHRASES. Skipped once read-check.json exists. Pass rules: docs/INTERP.md.
#   proj      explore/interp/read_run.py on PROJ_ENGINE (vllm, the episodes' engine and the vectors';
#             hf if vllm-lens fails on long prompts): projections, prompt ends, SAE top-k latents
#             at SAE_LAYERS. Needs VECTORS. Resumable, episode by episode.
#   logit     the logit monitor on hf: one pass per episode plus a short pass per turn and phrase.
#             Needs PHRASES. Resumable, episode by episode.
#   transfer  the Qwen-Scope transfer check on the pilot's transcripts (sae_transfer.py): FVU and
#             spliced loss on hf, FVU alone on vllm, over TRANSFER_PER_CONDITION episodes per condition
#             at TRANSFER_LAYERS. Skipped once its output exists.
#
# Estimated minutes for the pilot's 54 episodes (2.3M tokens), from LOG.md's numbers and the laptop's
# scaling, not measured on a GPU: proj about 15, logit about 25, transfer about 15, plus about 1 per
# engine load. Setup: 3 install, 5 download when the VM has neither.
#
# Colab's SSH allows one session, so a read step can't copy results out while it runs. CHUNK_MIN
# ends proj and logit between episodes, and session.sh calls this again after fetching what is done;
# a VM lost mid-run loses one chunk at most.
#
# Settings (environment): WORK (/content), CODE ($WORK/sim), IN ($WORK/read: episodes/, readings/ on a
# resume), OUT ($IN/readings), VENV ($WORK/interp-venv), MODELS ($WORK/models), PARTS, PROJ_ENGINE,
# VECTORS ($IN/vectors.safetensors, else $WORK/interp-smoke/agent/vectors.safetensors), LAYERS (the
# vector file's chosen layer), PHRASES ($IN/phrases.json), SAE_REPO, SAE_LAYERS (none),
# TRANSFER_LAYERS (32,42,48), TRANSFER_PER_CONDITION (2), CHUNK_MIN (20).
# Exit: 0 when every part is complete, 3 when some are left (run it again), 1 on an error.
set -uo pipefail
VLLM_VERSION=${VLLM_VERSION:-0.32.0}
LENS_VERSION=${LENS_VERSION:-1.3.0}
AGENT_REPO=${AGENT_REPO:-Qwen/Qwen3.8-27B-FP8}
SIM_REPO=${SIM_REPO:-Qwen/Qwen-AgentWorld-35B-A3B}
SAE_REPO=${SAE_REPO:-Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50}
WORK=${WORK:-/content}
CODE=${CODE:-$WORK/sim}
IN=${IN:-$WORK/read}
OUT=${OUT:-$IN/readings}
VENV=${VENV:-$WORK/interp-venv}
MODELS=${MODELS:-$WORK/models}
PARTS=${PARTS:-proj}
PARTS=${PARTS//+/ }     # sbatch --export can't carry spaces
PROJ_ENGINE=${PROJ_ENGINE:-vllm}
PHRASES=${PHRASES:-$IN/phrases.json}
LAYERS=${LAYERS:-}
SAE_LAYERS=${SAE_LAYERS:-}
TRANSFER_LAYERS=${TRANSFER_LAYERS:-32,42,48}
TRANSFER_PER_CONDITION=${TRANSFER_PER_CONDITION:-2}
CHUNK_MIN=${CHUNK_MIN:-20}
if [ -z "${VECTORS:-}" ]; then
  if [ -f "$IN/vectors.safetensors" ]; then VECTORS=$IN/vectors.safetensors
  else VECTORS=$WORK/interp-smoke/agent/vectors.safetensors; fi
fi
mkdir -p "$OUT" "$MODELS"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$CODE"
t0=$(date +%s)
stamp() { echo "$1 $(($(date +%s) - t0))s" | tee -a "$OUT/read-timings.txt"; }

for p in $PARTS; do
  case $p in check|proj|logit|transfer) ;; *) echo "read_run.sh: unknown part $p (PARTS: check proj logit transfer)"; exit 1 ;; esac
done
[ -f "$IN/episodes/manifest.json" ] || { echo "read_run.sh: no episodes in $IN/episodes: run simtest interp export"; exit 1; }
case " $PARTS " in *" proj "*|*" check "*) [ -f "$VECTORS" ] || { echo "read_run.sh: no vector file at $VECTORS"; exit 1; } ;; esac
case " $PARTS " in *" logit "*|*" check "*) [ -f "$PHRASES" ] || { echo "read_run.sh: no phrase file at $PHRASES"; exit 1; } ;; esac
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "$used" -gt 2000 ]; then echo "read_run.sh: the GPU has $used MiB in use: stop serve.sh's servers first"; exit 1; fi

set -e
source "${BASH_SOURCE[0]%/*}/gpu_env.sh"
SAE=$MODELS/sae-agent
sae_layers=$(printf '%s\n' "$SAE_LAYERS" "$TRANSFER_LAYERS" | tr ',' '\n' | grep . | sort -un)
if [ -n "$sae_layers" ] && { [ -n "$SAE_LAYERS" ] || [[ " $PARTS " == *" transfer "* ]]; }; then
  includes=(--include config.json)
  for L in $sae_layers; do includes+=(--include "layer$L.sae.pt"); done
  "$VENV/bin/hf" download "$SAE_REPO" --local-dir "$SAE" "${includes[@]}" >"$OUT/download-sae.log" 2>&1
fi
set +e

rc=0
note() {   # an exit code of read_run.py: 3 (some left) must not hide an error
  case $1 in 0) ;; 3) [ "$rc" = 1 ] || rc=3 ;; *) rc=1 ;; esac
}
reader() { "$PY" -I "$CODE/explore/interp/read_run.py" "$MODELS/agent" --episodes "$IN/episodes" "$@"; }
for p in $PARTS; do
  stamp "$p start"
  case $p in
    check)
      [ ! -e "$OUT/read-check.json" ] || continue
      check() { "$PY" -I "$CODE/explore/interp/read_check.py" "$@" 2>&1 | tee -a "$OUT/check.log"; note "${PIPESTATUS[0]}"; }
      common=(--episodes "$IN/episodes" --vectors "$VECTORS" --phrases "$PHRASES" --out "$OUT")
      check run "$MODELS/agent" --engine vllm "${common[@]}"
      check run "$MODELS/agent" --engine hf "${common[@]}"
      check run "$MODELS/agent" --engine hf --chunk 2048 --label hf-2048 "${common[@]}"
      check compare --out "$OUT" ;;
    proj)
      args=(--engine "$PROJ_ENGINE" --out "$OUT" --parts proj --vectors "$VECTORS" --budget-min "$CHUNK_MIN")
      [ -z "$LAYERS" ] || args+=(--layers "$LAYERS")
      [ -z "$SAE_LAYERS" ] || args+=(--sae-dir "$SAE" --sae-layers "$SAE_LAYERS" --sae-device cuda)
      reader "${args[@]}" 2>&1 | tee -a "$OUT/proj.log"
      note "${PIPESTATUS[0]}" ;;
    logit)
      reader --engine hf --out "$OUT/logit" --parts logit --phrases "$PHRASES" --budget-min "$CHUNK_MIN" 2>&1 | tee -a "$OUT/logit.log"
      note "${PIPESTATUS[0]}" ;;
    transfer)
      for engine in hf vllm; do
        out=$OUT/sae-transfer-$engine.json
        [ ! -e "$out" ] || continue
        extra=()
        [ "$engine" = hf ] || extra=(--fvu-only)
        "$PY" -I "$CODE/explore/interp/sae_transfer.py" "$MODELS/agent" "$SAE" --layers "$TRANSFER_LAYERS" --engine "$engine" \
          --episodes "$IN/episodes" --per-condition "$TRANSFER_PER_CONDITION" --sae-device cuda \
          --sample "$OUT/sae-sample.json" --out "$out" ${extra[@]+"${extra[@]}"} 2>&1 | tee -a "$OUT/transfer.log"
        note "${PIPESTATUS[0]}"
      done ;;
  esac
  stamp "$p done"
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a "$OUT/read-timings.txt"
exit "$rc"
