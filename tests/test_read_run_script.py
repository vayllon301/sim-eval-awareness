"""explore/interp/read_run.sh, the GPU side of reading a run, and Hopper's job and submit script. Stubs
for python, nvidia-smi, hf, uv and ssh: nothing reads a model and nothing reaches a cluster."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "explore" / "interp" / "read_run.sh"
SLURM = REPO / "explore" / "hopper" / "read.slurm"
SUBMIT = REPO / "explore" / "hopper" / "submit_read.sh"
CONFIG = "configs/coding/pilot.yaml"
BASH, ZSH = shutil.which("bash"), shutil.which("zsh")
REC = 'echo "$(basename "$0") $*" >> "$STATE/calls.log"\n'
# The venv's python: exit codes by which script it runs and with what. Writes what the real one would.
PYTHON = REC + """case "$*" in
  *read_run.py*"--parts proj"*) exit "${PROJ_RC:-0}" ;;
  *read_run.py*"--parts logit"*) exit "${LOGIT_RC:-0}" ;;
  *sae_transfer.py*)
    for a in "$@"; do [ "$prev" = --out ] && touch "$a"; prev=$a; done
    exit "${TRANSFER_RC:-0}" ;;
  *) exit 0 ;;
esac
"""


def setup(tmp_path: Path, used="10"):
    work, stubs, state = tmp_path / "work", tmp_path / "stubs", tmp_path / "state"
    for d in (stubs, state, work / "read" / "episodes", work / "interp-venv" / "bin", work / "sim" / "explore"):
        d.mkdir(parents=True)
    (work / "read" / "episodes" / "manifest.json").write_text("{}")
    (work / "read" / "vectors.safetensors").write_text("v")
    (work / "read" / "phrases.json").write_text("{}")
    (work / "interp-venv" / ".setup-complete").touch()
    shutil.copytree(REPO / "explore" / "interp", work / "sim" / "explore" / "interp", ignore=shutil.ignore_patterns("data", "__pycache__"))
    for name, body in {"python": PYTHON, "hf": REC, "uv": REC}.items():
        p = work / "interp-venv" / "bin" / name
        p.write_text("#!/bin/sh\n" + body)
        p.chmod(0o755)
    gpu = stubs / "nvidia-smi"
    gpu.write_text(f"#!/bin/sh\necho {used}\n")
    gpu.chmod(0o755)
    uv = stubs / "uv"
    uv.write_text("#!/bin/sh\n" + REC)
    uv.chmod(0o755)
    return work, stubs, state


def run(tmp_path, env=None, used="10"):
    work, stubs, state = setup(tmp_path, used)
    full = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "STATE": str(state), "WORK": str(work), **(env or {})}
    proc = subprocess.run([BASH, str(work / "sim" / "explore" / "interp" / "read_run.sh")], env=full, capture_output=True,
                          text=True, timeout=30, stdin=subprocess.DEVNULL)
    calls = (state / "calls.log").read_text().splitlines() if (state / "calls.log").exists() else []
    return proc, calls, work


def test_proj_runs_on_the_episodes_engine_with_the_served_files(tmp_path):
    proc, calls, work = run(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    call = next(c for c in calls if "read_run.py" in c)
    assert "--engine vllm" in call and "--parts proj" in call and f"--vectors {work}/read/vectors.safetensors" in call
    assert f"{work}/models/agent" in call and "--budget-min 20" in call and "--sae" not in call


def test_the_logit_part_runs_on_transformers(tmp_path):
    proc, calls, work = run(tmp_path, {"PARTS": "proj logit"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    call = next(c for c in calls if "--parts logit" in c)
    assert "--engine hf" in call and f"--phrases {work}/read/phrases.json" in call and f"--out {work}/read/readings/logit" in call


def test_latents_are_asked_on_the_gpu(tmp_path):
    proc, calls, work = run(tmp_path, {"SAE_LAYERS": "42"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "--sae-layers 42 --sae-device cuda" in next(c for c in calls if "read_run.py" in c)
    assert any("layer42.sae.pt" in c for c in calls if c.startswith("hf "))


def test_the_transfer_check_splices_on_hf_and_reads_fvu_on_vllm(tmp_path):
    proc, calls, work = run(tmp_path, {"PARTS": "transfer"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    hf, vllm = [c for c in calls if "sae_transfer.py" in c]
    assert "--engine hf" in hf and "--fvu-only" not in hf and "--episodes" in hf and "--per-condition 2" in hf
    assert "--engine vllm" in vllm and "--fvu-only" in vllm
    assert any("layer48.sae.pt" in c for c in calls if c.startswith("hf "))


def test_a_transfer_check_already_done_isnt_redone(tmp_path):
    work, stubs, state = setup(tmp_path)
    (work / "read" / "readings").mkdir()
    (work / "read" / "readings" / "sae-transfer-hf.json").write_text("{}")
    (work / "read" / "readings" / "sae-transfer-vllm.json").write_text("{}")
    full = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "STATE": str(state), "WORK": str(work), "PARTS": "transfer"}
    proc = subprocess.run([BASH, str(work / "sim" / "explore" / "interp" / "read_run.sh")], env=full, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0 and "sae_transfer.py" not in (state / "calls.log").read_text() if (state / "calls.log").exists() else True


@pytest.mark.parametrize("env, rc", [({"PROJ_RC": "3"}, 3), ({"PROJ_RC": "1"}, 1),
                                     ({"PROJ_RC": "3", "LOGIT_RC": "1"}, 1), ({"PROJ_RC": "1", "LOGIT_RC": "3"}, 1)])
def test_some_left_never_hides_an_error(tmp_path, env, rc):
    proc, calls, _ = run(tmp_path, {"PARTS": "proj logit", **env})
    assert proc.returncode == rc


def test_it_refuses_a_busy_gpu(tmp_path):
    proc, calls, _ = run(tmp_path, used="50000")
    assert proc.returncode == 1 and "stop serve.sh" in proc.stdout and not any("read_run.py" in c for c in calls)


def test_it_refuses_missing_inputs(tmp_path):
    work, stubs, state = setup(tmp_path)
    (work / "read" / "episodes" / "manifest.json").unlink()
    full = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "STATE": str(state), "WORK": str(work)}
    proc = subprocess.run([BASH, str(work / "sim" / "explore" / "interp" / "read_run.sh")], env=full, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 1 and "no episodes" in proc.stdout


def test_the_first_gpu_run_checks_both_paths_then_compares(tmp_path):
    proc, calls, work = run(tmp_path, {"PARTS": "check"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    runs = [c for c in calls if "read_check.py" in c]
    assert "run" in runs[0] and "--engine vllm" in runs[0] and "--engine hf" in runs[1] and "--chunk 2048" in runs[2]
    assert " compare " in runs[3]


def test_a_plus_stands_for_a_space_in_parts(tmp_path):
    """sbatch --export can't carry spaces."""
    proc, calls, _ = run(tmp_path, {"PARTS": "proj+logit"})
    assert proc.returncode == 0 and any("--parts logit" in c for c in calls)


def test_the_estimate_is_in_the_header():
    assert "Estimated minutes for the pilot" in SCRIPT.read_text()


# --- Hopper ---------------------------------------------------------------------------------------

def test_the_job_asks_for_one_gpu_and_leaves_the_time_limit_to_sbatch():
    text = SLURM.read_text()
    assert "#SBATCH --gres=gpu:1" in text and "#SBATCH --time" not in text and "read_run.sh" in text


def test_the_job_passes_on_the_readers_exit_code(tmp_path):
    work = tmp_path / "w"
    (work / "sim" / "explore" / "interp").mkdir(parents=True)
    (work / "sim" / "explore" / "interp" / "read_run.sh").write_text('echo "WORK=$WORK"\nexit 3\n')
    proc = subprocess.run([BASH, str(SLURM)], env={"PATH": "/usr/bin:/bin", "WORK": str(work)}, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 3 and f"WORK={work}" in proc.stdout


SSH = REC + """for a; do cmd=$a; done
cat > "$STATE/ssh-stdin-$(grep -c '^ssh' "$STATE/calls.log")" 2>/dev/null
case $cmd in *sbatch*) [ -n "${FAIL_SBATCH:-}" ] && exit 1; echo "1234;hopper" ;; esac
exit 0
"""
GIT = REC + """case "$*" in
  "status --porcelain") printf '%s' "${GIT_DIRTY:-}" ;;
  "bundle create "*) echo bundle > "$3" ;;
esac
"""


def submit(tmp_path, env=None, with_episodes=True):
    stubs, state, logs = tmp_path / "stubs", tmp_path / "state", tmp_path / "logs"
    for d in (stubs, state, logs):
        d.mkdir(exist_ok=True)
    for name, body in {"ssh": SSH, "git": GIT}.items():
        (stubs / name).write_text("#!/bin/sh\n" + body)
        (stubs / name).chmod(0o755)
    if with_episodes:
        d = logs / "read" / "n2-pilot" / "episodes"
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text("{}")
    (tmp_path / "v.safetensors").write_text("v")
    full = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "STATE": str(state), "LOGS": str(logs), "LOGIN": "v@login",
            "TIME_LIMIT": "2", "WORK": "/scratch/v/sim", "VECTORS": str(tmp_path / "v.safetensors"), "SBATCH_ARGS": "--partition=gpu", **(env or {})}
    full = {k: v for k, v in full.items() if v is not None}
    proc = subprocess.run([ZSH, str(SUBMIT), CONFIG], env=full, cwd=REPO, capture_output=True, text=True, timeout=30, stdin=subprocess.DEVNULL)
    calls = (state / "calls.log").read_text().splitlines() if (state / "calls.log").exists() else []
    return proc, calls, state


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_submit_sends_everything_and_submits_one_job(tmp_path):
    proc, calls, state = submit(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    text = "\n".join(calls)
    assert "clone -q /scratch/v/sim/sim.bundle" in text and "tar xzf - -C /scratch/v/sim/read" in text
    assert "cat > /scratch/v/sim/read/vectors.safetensors" in text
    job = text[text.index("sbatch --parsable"):]
    assert "sbatch --parsable --time=120" in job and "--partition=gpu" in job and "CHUNK_MIN=110" in job and "PARTS=proj" in job
    assert "tar czf - -C /scratch/v/sim/read readings" in proc.stdout     # how to bring them back


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
@pytest.mark.parametrize("env, needle", [({"LOGIN": None}, "LOGIN"), ({"TIME_LIMIT": None}, "TIME_LIMIT"), ({"WORK": "rel/path"}, "absolute"),
                                         ({"VECTORS": None}, "need vectors"), ({"PARTS": "logit"}, "need sentences"),
                                         ({"GIT_DIRTY": " M x"}, "dirty")])
def test_submit_refusals(tmp_path, env, needle):
    proc, calls, _ = submit(tmp_path, env)
    assert proc.returncode == 1 and needle in proc.stderr and not any(c.startswith("ssh") for c in calls)


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_submit_refuses_without_episodes(tmp_path):
    proc, calls, _ = submit(tmp_path, with_episodes=False)
    assert proc.returncode == 1 and "no episodes" in proc.stderr


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_submit_dry_run_touches_nothing(tmp_path):
    proc, calls, _ = submit(tmp_path, {"DRY_RUN": "1"})
    assert proc.returncode == 0 and not any(c.startswith("ssh") for c in calls) and "sbatch" in proc.stdout


def test_the_scripts_are_executable():
    for p in (SCRIPT, SLURM, SUBMIT):
        assert os.access(p, os.X_OK), p
