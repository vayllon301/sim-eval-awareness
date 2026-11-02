"""explore/colab_probe/serve.sh: the G4's vllm command lines stay byte-identical, and the
Hopper profiles (two GPUs, one H200, one 80 GB H100) serve what the plan needs.

The script runs with DRY_RUN=1 (it prints the command lines and touches nothing) and stub
binaries first on PATH that fail loudly if called, so no test can install, download or serve
anything. One test runs the full script against recording stubs, to show the dry run prints
what the script really starts.

The expected G4 lines were captured from the script before Hopper support (experiment
`eedcb55`), with stub binaries recording vllm's argv.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "explore" / "colab_probe" / "serve.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")

G4_MIB = "97887"     # RTX PRO 6000 Blackwell (Colab G4)
H200_MIB = "143771"
H100_MIB = "81559"

AGENT = ("vllm serve /content/models/agent --served-model-name agent --host 127.0.0.1 --port 8000 "
         "--gpu-memory-utilization {mem} --max-model-len {len} --max-num-seqs 64 --language-model-only "
         "--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder")
SIM = ("vllm serve /content/models/simulator --served-model-name simulator --host 127.0.0.1 --port 8001 "
       "--gpu-memory-utilization {mem} --max-model-len {len} --max-num-seqs 64 --language-model-only "
       "--reasoning-parser qwen3 --quantization fp8 --trust-remote-code")
EAGER = "VLLM_USE_V2_MODEL_RUNNER=0 "

# Captured from the pre-Hopper script (see the module docstring), one pair per G4 setting.
G4_LINES = {
    (): [AGENT.format(mem="0.47", len=131072), SIM.format(mem="0.47", len=131072)],
    (("MIX", "c3"),): [AGENT.format(mem="0.42", len=131072), SIM.format(mem="0.52", len=131072)],
    (("EAGER", "1"),): [EAGER + AGENT.format(mem="0.47", len=131072) + " --enforce-eager",
                        EAGER + SIM.format(mem="0.47", len=131072) + " --enforce-eager"],
    (("MIX", "c3"), ("EAGER", "1")): [EAGER + AGENT.format(mem="0.42", len=131072) + " --enforce-eager",
                                      EAGER + SIM.format(mem="0.52", len=131072) + " --enforce-eager"],
}
# The same script on an 80 GB card, before the change (64K, 0.44 / 0.50; MIX had no effect there).
H100_LINES = [AGENT.format(mem="0.44", len=65536), SIM.format(mem="0.50", len=65536)]

FORBIDDEN = ("nvidia-smi", "pip", "uv", "hf", "vllm", "curl", "setsid", "python", "python3")


def _stubs(tmp_path: Path, record: Path | None = None) -> Path:
    """A bin directory that shadows every tool the script could call. Without `record`, each
    stub fails loudly (a dry run must call none of them). With it, the stubs record vllm's
    argv (with the variables that matter) and pretend everything else worked."""
    d = Path(tempfile.mkdtemp(prefix="stubs-", dir=tmp_path))   # one per run: a test may run the script twice
    for name in FORBIDDEN:
        (d / name).write_text(f"#!/bin/sh\necho 'stub {name} called' >&2\nexit 99\n")
    if record is not None:
        (d / "vllm").write_text(
            "#!/bin/bash\n{ e=''; [ -n \"${CUDA_VISIBLE_DEVICES+x}\" ] && e+=\"CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \"; "
            "[ -n \"${VLLM_USE_V2_MODEL_RUNNER+x}\" ] && e+=\"VLLM_USE_V2_MODEL_RUNNER=$VLLM_USE_V2_MODEL_RUNNER \"; "
            f"printf '%s' \"$e\"; printf 'vllm'; printf ' %q' \"$@\"; printf '\\n'; }} >> {record}\n")
        (d / "curl").write_text(   # healthy once the backgrounded stub has recorded its port
            "#!/bin/bash\nport=$(echo \"$*\" | sed -E 's#.*localhost:([0-9]+)/health.*#\\1#')\n"
            f"for i in $(seq 400); do grep -q -- \"--port $port \" {record} 2>/dev/null && exit 0; sleep 0.05; done\nexit 1\n")
        (d / "nvidia-smi").write_text("#!/bin/sh\necho 'name, memory.total [MiB], driver_version'\n")
        (d / "python").write_text("#!/bin/sh\n[ \"$1\" = - ] && cat >/dev/null\necho stub-python\n")
        for name in ("pip", "uv", "hf", "python3"):
            (d / name).write_text("#!/bin/sh\nexit 0\n")
        (d / "setsid").write_text("#!/bin/sh\nexec \"$@\"\n")
    for p in d.iterdir():
        p.chmod(0o755)
    return d


def run(tmp_path: Path, env: dict, dry: bool = True, record: Path | None = None) -> subprocess.CompletedProcess:
    stubs = _stubs(tmp_path, record)
    full = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), **env}
    if dry:
        full["DRY_RUN"] = "1"
    return subprocess.run([BASH, str(SCRIPT)], env=full, capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL)


def lines(proc) -> list[str]:
    assert proc.returncode == 0, proc.stderr
    return [x for x in proc.stdout.splitlines() if x.startswith(("vllm ", "VLLM_", "CUDA_"))]


@pytest.mark.parametrize("extra", list(G4_LINES), ids=lambda e: "+".join(f"{k}={v}" for k, v in e) or "default")
def test_g4_command_lines_are_byte_identical(tmp_path, extra):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1", **dict(extra)})
    assert lines(proc) == G4_LINES[extra]


def test_dry_run_touches_nothing(tmp_path):
    proc = run(tmp_path, {"WORK": str(tmp_path / "work"), "GPU_MIB": G4_MIB, "GPU_COUNT": "1"})
    assert proc.returncode == 0, proc.stderr
    assert "stub" not in proc.stderr          # no tool was called
    assert not (tmp_path / "work").exists()   # and nothing was created


def test_dry_run_prints_the_profile(tmp_path):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1"})
    assert "profile: one-gpu-96g" in proc.stdout


def test_h200_serves_both_models_at_128k(tmp_path):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": H200_MIB, "GPU_COUNT": "1"})
    got = lines(proc)
    assert "profile: one-gpu-141g" in proc.stdout
    assert got == G4_LINES[()]   # the G4's fractions leave each model more KV cache on 141 GB


def test_two_gpus_one_model_per_gpu_at_128k(tmp_path):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": H100_MIB, "GPU_COUNT": "2"})
    agent, sim = lines(proc)
    assert "profile: two-gpu" in proc.stdout
    assert agent == "CUDA_VISIBLE_DEVICES=0 " + AGENT.format(mem="0.90", len=131072)
    assert sim == "CUDA_VISIBLE_DEVICES=1 " + SIM.format(mem="0.90", len=131072)


def test_two_gpus_follow_the_visible_devices(tmp_path):
    """A scheduler (Slurm) may hand out GPUs 3 and 5: each server gets one of them."""
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": H100_MIB, "CUDA_VISIBLE_DEVICES": "3,5"})
    agent, sim = lines(proc)
    assert agent.startswith("CUDA_VISIBLE_DEVICES=3 vllm serve /content/models/agent ")
    assert sim.startswith("CUDA_VISIBLE_DEVICES=5 vllm serve /content/models/simulator ")


def test_two_gpus_with_eager(tmp_path):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": H100_MIB, "GPU_COUNT": "2", "EAGER": "1"})
    agent, sim = lines(proc)
    assert agent == "CUDA_VISIBLE_DEVICES=0 " + EAGER + AGENT.format(mem="0.90", len=131072) + " --enforce-eager"
    assert sim == "CUDA_VISIBLE_DEVICES=1 " + EAGER + SIM.format(mem="0.90", len=131072) + " --enforce-eager"


@pytest.mark.parametrize("mix", ["", "c3"])
def test_one_80gb_gpu_keeps_64k_with_a_loud_warning(tmp_path, mix):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": H100_MIB, "GPU_COUNT": "1", "MIX": mix})
    assert lines(proc) == H100_LINES
    assert "profile: one-gpu-80g" in proc.stdout
    warning = proc.stderr
    assert "WARNING" in warning and "64K" in warning
    assert "NOT equivalent to the G4" in warning
    assert "context" in warning   # long C3 episodes can hit the context limit


def test_a_gpu_too_small_for_both_models_is_refused(tmp_path):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": "49140", "GPU_COUNT": "1"})
    assert proc.returncode != 0
    assert "too small" in proc.stderr


def test_work_dir_replaces_content(tmp_path):
    work = "/scratch/victor/sim"
    proc = run(tmp_path, {"WORK": work, "GPU_MIB": H100_MIB, "GPU_COUNT": "2"})
    out = "\n".join(lines(proc))
    assert "/content" not in proc.stdout
    assert f"{work}/models/agent " in out and f"{work}/models/simulator " in out


def test_outside_colab_the_work_dir_is_required(tmp_path):
    # no WORK and (on this machine) no /content: refuse rather than fill a home directory
    if Path("/content").is_dir():
        pytest.skip("this machine has /content")
    proc = run(tmp_path, {"GPU_MIB": G4_MIB, "GPU_COUNT": "1"})
    assert proc.returncode != 0
    assert "WORK" in proc.stderr


def test_vllm_stays_pinned(tmp_path):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1"})
    assert "vllm 0.32.0" in proc.stdout
    other = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "VLLM_VERSION": "0.33.0"})
    assert other.returncode != 0
    assert "0.32.0" in other.stderr


@pytest.mark.parametrize("env", [
    {"GPU_MIB": G4_MIB, "GPU_COUNT": "1", "MIX": "c3", "VENV": ""},     # system install, as on Colab
    {"GPU_MIB": H100_MIB, "GPU_COUNT": "2", "EAGER": "1"},              # a venv in WORK, as on Hopper
], ids=["one-gpu-system", "two-gpu-venv"])
def test_the_dry_run_prints_what_the_script_starts(tmp_path, env):
    work = tmp_path / "work"
    dry = lines(run(tmp_path, {"WORK": str(work), **env}))
    rec = tmp_path / "rec.txt"
    real = run(tmp_path, {"WORK": str(work), **env}, dry=False, record=rec)
    assert real.returncode == 0, real.stderr + real.stdout
    assert rec.read_text().splitlines() == dry
    versions = (work / "probe" / "versions.txt").read_text()
    assert all(x in versions for x in dry)   # the served command lines are kept with the run's record


def test_serve_script_has_no_colab_only_paths_outside_the_default():
    text = SCRIPT.read_text()
    body = [x for x in text.splitlines() if not x.lstrip().startswith("#")]
    hits = [x for x in body if "/content" in x]
    # /content may appear only where the script picks its default work directory
    assert all("WORK" in x for x in hits), hits
    assert os.access(SCRIPT, os.R_OK)
