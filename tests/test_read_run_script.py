"""explore/interp/read_run.sh, the GPU side of reading a run, and Hopper's job and submit script. Stubs
for python, nvidia-smi, hf, uv and ssh: nothing reads a model and nothing reaches a cluster."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "explore" / "interp" / "read_run.sh"
SLURM = REPO / "explore" / "hopper" / "read.slurm"
SUBMIT = REPO / "explore" / "hopper" / "submit_read.sh"
CONFIG = "configs/coding/pilot.yaml"
BASH, ZSH = shutil.which("bash"), shutil.which("zsh")
REC = 'echo "$(basename "$0") $*" >> "$STATE/calls.log"\n'
AGENT_REV, SIM_REV = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a", "60d2b0434a53d2e62a7c00a489586815d94ebffb"
SAE_REV = "13d4221569f7ca5d3c1e605e3e3dc95117e4807c"     # the repo's head on 04-11-2026
# The venv's python: exit codes by which script it runs and with what. Writes what the real one would.
PYTHON = REC + """case "$*" in
  *read_run.py*"--parts proj"*) exit "${PROJ_RC:-0}" ;;
  *read_run.py*"--parts logit"*) exit "${LOGIT_RC:-0}" ;;
  *sae_transfer.py*)
    for a in "$@"; do [ "$prev" = --out ] && touch "$a"; prev=$a; done
    exit "${TRANSFER_RC:-0}" ;;
  *gpu_smoke.py*)
    for a in "$@"; do [ "$prev" = --out-dir ] && d=$a/agent; prev=$a; done
    mkdir -p "$d" && touch "$d/vectors.safetensors" "$d/vectors.safetensors.report.json" "$d/lens.json"
    exit "${LENS_RC:-0}" ;;
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


def test_the_saes_download_at_their_pinned_commit_and_the_readings_record_it(tmp_path):
    proc, calls, work = run(tmp_path, {"PARTS": "transfer", "SAE_REVISION": "main"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    hf = next(c for c in calls if "SAE-Res" in c and c.startswith("hf download"))
    assert f"--revision {SAE_REV} " in hf
    assert f"Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50 {SAE_REV}\n" in (work / "read" / "readings" / "versions.txt").read_text()


@pytest.mark.parametrize("parts, minutes", [("proj", [20]), ("logit", [20]), ("logit+transfer", [20]), ("proj+logit", [10, 10]),
                                            ("check+proj+logit+transfer", [10, 10])])
def test_chunk_min_is_split_across_the_parts_that_have_a_budget(tmp_path, parts, minutes):
    proc, calls, _ = run(tmp_path, {"PARTS": parts})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    got = [c.split("--budget-min ")[1].split()[0] for c in calls if "read_run.py" in c]
    assert got == [str(m) for m in minutes]


def run_without_vectors(tmp_path, env):
    work, stubs, state = setup(tmp_path)
    (work / "read" / "vectors.safetensors").unlink()
    full = {"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(tmp_path), "STATE": str(state), "WORK": str(work), **env}
    proc = subprocess.run([BASH, str(work / "sim" / "explore" / "interp" / "read_run.sh")], env=full, capture_output=True, text=True,
                          timeout=30, stdin=subprocess.DEVNULL)
    return proc, (state / "calls.log").read_text().splitlines() if (state / "calls.log").exists() else [], work


def test_the_vectors_part_builds_them_here_and_the_next_part_reads_them(tmp_path):
    proc, calls, work = run_without_vectors(tmp_path, {"PARTS": "vectors+proj"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lens = next(i for i, c in enumerate(calls) if "gpu_smoke.py lens" in c)
    assert f"{work}/models/agent --name agent --out-dir {work}/interp-smoke" in calls[lens]
    proj = next(i for i, c in enumerate(calls) if "read_run.py" in c)
    assert lens < proj and f"--vectors {work}/interp-smoke/agent/vectors.safetensors" in calls[proj]


def test_the_vectors_come_back_with_the_readings(tmp_path):
    proc, calls, work = run_without_vectors(tmp_path, {"PARTS": "vectors"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert {p.name for p in (work / "read" / "readings" / "vectors").iterdir()} == {"vectors.safetensors", "vectors.safetensors.report.json", "lens.json"}


def test_vectors_that_exist_arent_built_again(tmp_path):
    proc, calls, work = run(tmp_path, {"PARTS": "vectors proj"})
    assert proc.returncode == 0 and not any("gpu_smoke.py" in c for c in calls)


def test_a_failed_build_is_an_error(tmp_path):
    proc, calls, work = run_without_vectors(tmp_path, {"PARTS": "vectors", "LENS_RC": "1"})
    assert proc.returncode == 1


def test_the_vectors_part_goes_first(tmp_path):
    proc, calls, work = run_without_vectors(tmp_path, {"PARTS": "proj vectors"})
    assert proc.returncode == 1 and "first" in proc.stdout


def test_the_flag_for_another_cards_vectors_reaches_proj(tmp_path):
    proc, calls, work = run(tmp_path, {"ALLOW_OTHER_BACKEND": "1"})
    assert "--allow-other-backend" in next(c for c in calls if "read_run.py" in c)
    proc, calls, work = run(tmp_path / "x" if (tmp_path / "x").mkdir() is None else None)
    assert "--allow-other-backend" not in next(c for c in calls if "read_run.py" in c)


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


def test_the_reader_needs_the_servers_ids_unless_told_otherwise(tmp_path):
    proc, calls, _ = run(tmp_path, {"PARTS": "proj logit"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not any("--no-server-ids" in c for c in calls)


def test_no_server_ids_reaches_both_parts_of_the_reader(tmp_path):
    proc, calls, _ = run(tmp_path, {"PARTS": "proj logit", "NO_SERVER_IDS": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sum("--no-server-ids" in c for c in calls if "read_run.py" in c) == 2


def test_the_interp_downloads_take_serve_shs_commits(tmp_path):
    """The vectors and readings use the files the servers served."""
    proc, calls, work = run(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    downloads = sorted(c for c in calls if c.startswith("hf download"))
    assert downloads == [f"hf download Qwen/Qwen-AgentWorld-35B-A3B --revision {SIM_REV} --local-dir {work}/models/simulator",
                         f"hf download Qwen/Qwen3.8-27B-FP8 --revision {AGENT_REV} --local-dir {work}/models/agent"]
    versions = (work / "read" / "readings" / "versions.txt").read_text()
    assert f"Qwen/Qwen3.8-27B-FP8 {AGENT_REV}\n" in versions and f"Qwen/Qwen-AgentWorld-35B-A3B {SIM_REV}\n" in versions


def check_file(out, label, uuid):
    (out / f"check-{label}-{uuid}.json").write_text(json.dumps({
        "label": label, "uuid": uuid, "tokens": 3, "turns": [0, 1], "projection": [1.0, 2.0, 3.0], "logprobs": [-1.0, -2.0],
        "monitor": [0.1, 0.2], "seconds": {"pass": 1.0, "phrases": 1.0}}))


def compare(out):
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    return subprocess.run([sys.executable, "-I", str(REPO / "explore" / "interp" / "read_check.py"), "compare", "--out", str(out)],
                          env=env, capture_output=True, text=True, timeout=60)


@pytest.mark.parametrize("gone", ["vllm", "hf", "hf-2048"])
def test_a_check_with_a_path_missing_writes_no_result_so_a_retry_runs_it(tmp_path, gone):
    for label in {"vllm", "hf", "hf-2048"} - {gone}:
        check_file(tmp_path, label, "long")
    proc = compare(tmp_path)
    assert proc.returncode != 0 and gone in proc.stderr and not (tmp_path / "read-check.json").exists()


def test_a_check_with_an_episode_missing_on_one_path_writes_no_result(tmp_path):
    for label in ("vllm", "hf", "hf-2048"):
        check_file(tmp_path, label, "long")
    check_file(tmp_path, "vllm", "short")
    check_file(tmp_path, "hf", "short")
    proc = compare(tmp_path)
    assert proc.returncode != 0 and "hf-2048" in proc.stderr and not (tmp_path / "read-check.json").exists()


def test_a_complete_check_writes_its_result(tmp_path):
    for label in ("vllm", "hf", "hf-2048"):
        check_file(tmp_path, label, "long")
    proc = compare(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert len(json.loads((tmp_path / "read-check.json").read_text())) == 3


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
@pytest.mark.parametrize("parts, minutes", [("proj", 110), ("proj logit", 110), ("proj transfer", 95), ("check proj logit transfer", 80)])
def test_the_job_keeps_room_for_the_parts_that_have_no_budget(tmp_path, parts, minutes):
    """A budgeted part stops on time; check and transfer run to their end whatever is left."""
    (tmp_path / "p.json").write_text("{}")
    proc, calls, _ = submit(tmp_path, {"PARTS": parts, "PHRASES": str(tmp_path / "p.json"), "NO_SERVER_IDS": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"CHUNK_MIN={minutes}," in "\n".join(calls) + "," or f"CHUNK_MIN={minutes} " in "\n".join(calls)


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_a_job_can_build_its_vectors_on_hopper(tmp_path):
    proc, calls, _ = submit(tmp_path, {"PARTS": "vectors proj", "VECTORS": None, "NO_SERVER_IDS": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    job = "\n".join(calls)
    assert "PARTS=vectors+proj" in job and "CHUNK_MIN=100" in job and "cat > /scratch/v/sim/read/vectors.safetensors" not in job


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_a_job_refuses_to_build_vectors_and_send_a_file(tmp_path):
    proc, calls, _ = submit(tmp_path, {"PARTS": "vectors proj", "NO_SERVER_IDS": "1"})
    assert proc.returncode == 1 and "VECTORS" in proc.stderr and not any(c.startswith("ssh") for c in calls)


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_the_flag_for_another_cards_vectors_is_sent_by_name(tmp_path):
    proc, calls, _ = submit(tmp_path, {"ALLOW_OTHER_BACKEND": "1", "NO_SERVER_IDS": "1"})
    assert "ALLOW_OTHER_BACKEND=1" in "\n".join(calls)


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_a_job_too_short_for_its_parts_is_refused(tmp_path):
    (tmp_path / "p.json").write_text("{}")
    proc, calls, _ = submit(tmp_path, {"PARTS": "check proj logit transfer", "PHRASES": str(tmp_path / "p.json"), "NO_SERVER_IDS": "1",
                                       "TIME_LIMIT": "0.5"})
    assert proc.returncode == 1 and "room" in proc.stderr and not any(c.startswith("ssh") for c in calls)


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
@pytest.mark.parametrize("env, needle", [({"LOGIN": None}, "LOGIN"), ({"TIME_LIMIT": None}, "TIME_LIMIT"), ({"WORK": "rel/path"}, "absolute"),
                                         ({"VECTORS": None}, "need vectors"), ({"PARTS": "logit"}, "need sentences"),
                                         ({"GIT_DIRTY": " M x"}, "dirty")])
def test_submit_refusals(tmp_path, env, needle):
    proc, calls, _ = submit(tmp_path, env)
    assert proc.returncode == 1 and needle in proc.stderr and not any(c.startswith("ssh") for c in calls)


def manifest(tmp_path, exported, with_ids):
    d = tmp_path / "logs" / "read" / "n2-pilot" / "episodes"
    d.mkdir(parents=True)
    rows = [{"status": "exported", "uuid": f"u{i}"} for i in range(exported)]
    (d / "manifest.json").write_text(json.dumps({"episodes": rows}, indent=1, sort_keys=True))
    for i in range(with_ids):
        (d / f"u{i}.ids.json").write_text("{}")


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_submit_says_so_before_the_cluster_when_the_servers_ids_are_missing(tmp_path):
    manifest(tmp_path, 2, 1)
    proc, calls, _ = submit(tmp_path, with_episodes=False)
    assert proc.returncode == 1 and "1 of 2 exported episodes have no server IDs" in proc.stderr
    assert not any(c.startswith("ssh") for c in calls)


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_submit_passes_on_a_read_without_a_server_by_name(tmp_path):
    manifest(tmp_path, 2, 0)
    proc, calls, _ = submit(tmp_path, {"NO_SERVER_IDS": "1"}, with_episodes=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NO_SERVER_IDS=1" in next(c for c in calls if "sbatch --parsable" in c)


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_submit_with_every_id_says_nothing_about_them(tmp_path):
    manifest(tmp_path, 2, 2)
    proc, calls, _ = submit(tmp_path, with_episodes=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NO_SERVER_IDS" not in next(c for c in calls if "sbatch --parsable" in c)


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


# --- what crosses sbatch --export ------------------------------------------------------------------

def as_slurm(calls) -> dict:
    """The variables sbatch sets from a call's --export: it splits on commas, so a piece with no = is a
    name to pass on from the environment, whatever came before it."""
    value = next(w for c in calls for w in c.split() if w.startswith("--export=")).removeprefix("--export=")
    return dict(piece.split("=", 1) for piece in value.split(",") if "=" in piece)


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_layer_lists_survive_sbatchs_export_and_reach_the_reader_whole(tmp_path):
    laptop, node = tmp_path / "laptop", tmp_path / "node"
    laptop.mkdir(), node.mkdir()
    proc, calls, _ = submit(laptop, {"LAYERS": "30,42,48", "SAE_LAYERS": "32,42,48", "NO_SERVER_IDS": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    job = as_slurm(calls)
    assert set(job) >= {"WORK", "PARTS", "CHUNK_MIN", "LAYERS", "SAE_LAYERS", "NO_SERVER_IDS"}
    env = {k: job[k] for k in ("PARTS", "CHUNK_MIN", "LAYERS", "SAE_LAYERS", "NO_SERVER_IDS")}
    proc, calls, _ = run(node, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    call = next(c for c in calls if "read_run.py" in c)
    assert "--layers 30,42,48 " in call and "--sae-layers 32,42,48 " in call


def test_a_list_given_with_commas_on_colab_still_works(tmp_path):
    proc, calls, _ = run(tmp_path, {"LAYERS": "30,42,48", "SAE_LAYERS": "32,42,48"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    call = next(c for c in calls if "read_run.py" in c)
    assert "--layers 30,42,48 " in call and "--sae-layers 32,42,48 " in call


@pytest.mark.skipif(ZSH is None, reason="needs zsh")
def test_a_path_with_a_comma_cant_cross_sbatchs_export(tmp_path):
    proc, calls, _ = submit(tmp_path, {"WORK": "/scratch/v,w/sim"})
    assert proc.returncode == 1 and "comma" in proc.stderr and not any(c.startswith("ssh") for c in calls)
