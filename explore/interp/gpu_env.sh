# Exploratory (GPU work on the interp tools; docs/INTERP.md). Sourced by gpu_smoke.sh and read_run.sh.
#
# The venv and the models. vLLM 0.32.0 (as serve.sh) and vllm-lens 1.3.0 go into a venv of their own:
# vllm-lens loads into every vLLM process of its environment and forces eager mode, so it must not
# share serve.sh's install. The two models come down only if serve.sh's copies aren't in MODELS (hf
# download skips files it already has). Needs VLLM_VERSION, LENS_VERSION, AGENT_REPO, SIM_REPO, CODE,
# OUT, VENV, MODELS and a function `stamp`. Sets PY.
PY="$VENV/bin/python"
# The marker is written only after every install and .pth below succeeded, so a failed install is
# redone from a clean venv on the next run (python appears in the venv before its packages do).
if [ ! -e "$VENV/.setup-complete" ]; then
  command -v uv >/dev/null || { python3 -m pip install --user -q uv && export PATH="$HOME/.local/bin:$PATH"; }
  uv venv -q --clear --python 3.12 "$VENV"
  uv pip install -q --python "$PY" "vllm==$VLLM_VERSION"
  # vllm-lens pins vllm==0.30.0, so it goes on top of 0.32.0 without its dependencies (its patches
  # are checked at startup: gpu_smoke.py sets VLLM_LENS_STRICT_COMPATIBILITY=1). zstandard is the
  # one it needs that vLLM doesn't bring; kernels (the FP8 Triton kernel) and accelerate are for the
  # HF step (transformers 5.19 needs kernels >=0.17,<0.18; an unpinned install failed on 31-10-2026).
  # The constraints keep vLLM's torch and transformers.
  uv pip freeze --python "$PY" | grep -E '^(torch|transformers|triton|vllm)==' > "$VENV/constraints.txt"
  uv pip install -q --python "$PY" --no-deps "vllm-lens==$LENS_VERSION"
  uv pip install -q --python "$PY" -c "$VENV/constraints.txt" zstandard "kernels==0.17.0" accelerate
  # Gated DeltaNet's fast path for the HF step; without it transformers uses its torch path.
  uv pip install -q --python "$PY" -c "$VENV/constraints.txt" flash-linear-attention \
    || echo "flash-linear-attention not installed: the HF step uses transformers' torch path" | tee -a "$OUT/versions.txt"
  SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo "$CODE/src" > "$SITE/simtest-src.pth"  # simtest.interp without installing simtest (it asks for Python 3.14)
  # vllm-lens#52: its pre-hook logs a traceback for every Qwen3.5 layer at every forward pass (we use no
  # pre-hooks). Drop exactly that warning, in this venv only; a post-hook error (a skipped steer) still shows.
  echo 'import logging; logging.getLogger("vllm_lens._worker_ext").addFilter(lambda r: not str(r.msg).startswith("vllm-lens pre-hook error"))' \
    > "$SITE/simtest-quiet-lens-prehook.pth"
  touch "$VENV/.setup-complete"
fi
"$PY" - <<'EOF' | tee -a "$OUT/versions.txt"
import platform
from importlib.metadata import version
print("python", platform.python_version())
for p in ("vllm", "vllm-lens", "torch", "transformers", "tokenizers", "triton", "flashinfer-python", "kernels",
          "accelerate", "flash-linear-attention", "zstandard"):
    try:
        print(p, version(p))
    except Exception:
        print(p, None)
EOF
uv pip freeze --python "$PY" > "$OUT/pip-freeze.txt"
{ git -C "$CODE" rev-parse HEAD 2>/dev/null || echo "none: $CODE is not a git checkout"; } | sed 's/^/experiment commit /' \
  | tee -a "$OUT/versions.txt"
{ git -C "$CODE" status --porcelain -- src/simtest/interp explore/interp 2>/dev/null || true; } | sed 's/^/dirty: /' \
  | tee -a "$OUT/versions.txt"
stamp installed

"$VENV/bin/hf" download "$AGENT_REPO" --local-dir "$MODELS/agent" >"$OUT/download-agent.log" 2>&1 & a=$!
"$VENV/bin/hf" download "$SIM_REPO" --local-dir "$MODELS/simulator" >"$OUT/download-simulator.log" 2>&1 & s=$!
wait $a && wait $s
"$PY" - "$AGENT_REPO" "$SIM_REPO" <<'EOF' | tee -a "$OUT/versions.txt"
import sys
from huggingface_hub import model_info
for repo in sys.argv[1:]:
    print(repo, model_info(repo).sha)
EOF
stamp downloaded
