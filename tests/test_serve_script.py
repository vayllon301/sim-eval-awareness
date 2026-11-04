"""explore/colab_probe/serve.sh: the simulator serves fp8_per_channel unless fp8 is named, the G4's
old vllm command lines stay byte-identical under SIM_QUANT=fp8, and the Hopper profiles (two GPUs,
one H200, one 80 GB H100) serve what the plan needs.

The script runs with DRY_RUN=1 (it prints the command lines and touches nothing) and stub
binaries first on PATH that fail loudly if called, so no test can install, download or serve
anything. One test runs the full script against recording stubs, to show the dry run prints
what the script really starts.

The G4 lines of every session so far were captured from the script before Hopper support (experiment
`eedcb55`), with stub binaries recording vllm's argv. They are what SIM_QUANT=fp8 must still print,
so an old session can be reproduced. The default changed on purpose: since LOG.md, 02-11-2026, the
simulator runs fp8_per_channel from the pilot on, so fp8 needs its name.

SIM_ONLY=1 restarts the simulator alone, next to a running agent (docs/BACKENDS.md).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
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

# Captured from the pre-Hopper script (see the module docstring), one pair per G4 setting. Every
# session so far: what SIM_QUANT=fp8 reproduces.
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

# fp8_per_channel, the default: the same line with the per-channel scheme and the MoE kernel pinned
# to Triton. The agent's line never changes.
OLD_SCHEME, FINAL_SCHEME = "--quantization fp8 --trust-remote-code", \
    "--quantization fp8_per_channel --moe-backend triton --trust-remote-code"
SIM_PC = SIM.replace(OLD_SCHEME, FINAL_SCHEME)
assert SIM_PC != SIM
G4_DEFAULT = {k: [a, s.replace(OLD_SCHEME, FINAL_SCHEME)] for k, (a, s) in G4_LINES.items()}
H100_DEFAULT = [H100_LINES[0], H100_LINES[1].replace(OLD_SCHEME, FINAL_SCHEME)]

FORBIDDEN = ("nvidia-smi", "pip", "uv", "hf", "vllm", "curl", "setsid", "python", "python3", "pgrep")


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
            f"printf '%s' \"$e\"; printf 'vllm'; printf ' %q' \"$@\"; printf '\\n'; }} >> {record}\n"
            f"cat {record.parent / 'vllm-says'} 2>/dev/null\n")   # the server's own log lines, when a test gives them
        (d / "curl").write_text(   # healthy once the backgrounded stub has recorded its port
            "#!/bin/bash\nport=$(echo \"$*\" | sed -E 's#.*localhost:([0-9]+)/health.*#\\1#')\n"
            f"for i in $(seq 400); do grep -q -- \"--port $port \" {record} 2>/dev/null && exit 0; sleep 0.05; done\nexit 1\n")
        (d / "nvidia-smi").write_text(
            "#!/bin/sh\ncase \"$*\" in *--query-gpu=name\\ *) printf 'NVIDIA H100 80GB HBM3\\nNVIDIA H100 80GB HBM3\\n' ;;\n"
            "*) echo 'name, memory.total [MiB], driver_version' ;; esac\n")
        (d / "python").write_text("#!/bin/sh\n[ \"$1\" = - ] && cat >/dev/null\necho stub-python\n")
        for name in ("pip", "uv", "hf", "python3"):
            (d / name).write_text("#!/bin/sh\nexit 0\n")
        (d / "setsid").write_text("#!/bin/sh\nexec \"$@\"\n")
        # -f PATTERN: the live pids in RUNNING (a test's stand-in for a simulator already serving),
        # each with the pattern recorded; -g PGID: whether that process group still has a member
        running = record.parent / "running"
        (d / "pgrep").write_text(
            "#!/bin/sh\ncase \"$1\" in\n"
            f"  -f) echo \"$3\" >> {record.parent / 'pgrep.txt'}; st=1; for p in $(cat {running} 2>/dev/null); do "
            "kill -s 0 \"$p\" 2>/dev/null && { echo \"$p\"; st=0; }; done; exit $st ;;\n"
            "  -g) kill -s 0 -- \"-$2\" 2>/dev/null && { echo \"$2\"; exit 0; }; exit 1 ;;\n"
            "esac\nexit 1\n")
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
    """Only the simulator's scheme changes from the pre-Hopper lines."""
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1", **dict(extra)})
    assert lines(proc) == G4_DEFAULT[extra]


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
    assert got == G4_DEFAULT[()]   # the G4's fractions leave each model more KV cache on 141 GB


def test_two_gpus_one_model_per_gpu_at_128k(tmp_path):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": H100_MIB, "GPU_COUNT": "2"})
    agent, sim = lines(proc)
    assert "profile: two-gpu" in proc.stdout
    assert agent == "CUDA_VISIBLE_DEVICES=0 " + AGENT.format(mem="0.90", len=131072)
    assert sim == "CUDA_VISIBLE_DEVICES=1 " + SIM_PC.format(mem="0.90", len=131072)


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
    assert sim == "CUDA_VISIBLE_DEVICES=1 " + EAGER + SIM_PC.format(mem="0.90", len=131072) + " --enforce-eager"


@pytest.mark.parametrize("mix", ["", "c3"])
def test_one_80gb_gpu_keeps_64k_with_a_loud_warning(tmp_path, mix):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": H100_MIB, "GPU_COUNT": "1", "MIX": mix})
    assert lines(proc) == H100_DEFAULT
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


# --- SIM_QUANT and SIM_ONLY (the final scheme, and old sessions; docs/BACKENDS.md) -------------------


@pytest.mark.parametrize("extra", list(G4_LINES), ids=lambda e: "+".join(f"{k}={v}" for k, v in e) or "default")
def test_sim_quant_fp8_reproduces_the_old_sessions_lines(tmp_path, extra):
    """fp8 by name gives every old session's lines, byte for byte, and says it isn't the final scheme."""
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "SIM_QUANT": "fp8", **dict(extra)})
    assert lines(proc) == G4_LINES[extra]
    assert "WARNING" in proc.stderr and "old sessions" in proc.stderr


@pytest.mark.parametrize("extra", list(G4_LINES), ids=lambda e: "+".join(f"{k}={v}" for k, v in e) or "default")
def test_the_default_is_the_final_scheme_and_changes_only_the_simulator(tmp_path, extra):
    default = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1", **dict(extra)})
    named = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "SIM_QUANT": "fp8_per_channel",
                           **dict(extra)})
    agent, sim = lines(named)
    want_agent, want_sim = G4_LINES[extra]
    assert agent == want_agent    # the agent is its own FP8 checkpoint: untouched
    assert sim == want_sim.replace(OLD_SCHEME, FINAL_SCHEME)
    assert lines(default) == lines(named)
    assert "WARNING" not in default.stderr


def test_an_empty_sim_quant_is_the_final_scheme(tmp_path):
    """SIM_QUANT= (empty, as a shell may pass it) is no way to reach fp8."""
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "SIM_QUANT": ""})
    assert lines(proc) == G4_DEFAULT[()]


def test_sim_quant_per_channel_on_every_profile(tmp_path):
    two = run(tmp_path, {"WORK": "/content", "GPU_MIB": H100_MIB, "GPU_COUNT": "2", "SIM_QUANT": "fp8_per_channel"})
    assert lines(two)[1] == "CUDA_VISIBLE_DEVICES=1 " + SIM_PC.format(mem="0.90", len=131072)
    h100 = run(tmp_path, {"WORK": "/content", "GPU_MIB": H100_MIB, "GPU_COUNT": "1", "SIM_QUANT": "fp8_per_channel"})
    assert lines(h100) == [H100_LINES[0], SIM_PC.format(mem="0.50", len=65536)]


@pytest.mark.parametrize("value", ["fp8_per_tensor", "fp8_per_block", "FP8", "nvfp4_per_token", "none"])
def test_sim_quant_refuses_any_other_scheme(tmp_path, value):
    """fp8_per_tensor is fp8's arithmetic under another name: refused too, so one backend has one name."""
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "SIM_QUANT": value})
    assert proc.returncode != 0
    assert "SIM_QUANT" in proc.stderr and "fp8_per_channel" in proc.stderr
    assert "vllm serve" not in proc.stdout


@pytest.mark.parametrize("env", [{"SIM_QUANT": "fp8"}, {"SIM_ONLY": "1"},
                                 {"SIM_ONLY": "1", "SIM_QUANT": "fp8", "MIX": "c3"}],
                         ids=["old-scheme", "sim-only", "both"])
def test_dry_run_with_the_new_options_touches_nothing(tmp_path, env):
    proc = run(tmp_path, {"WORK": str(tmp_path / "work"), "GPU_MIB": G4_MIB, "GPU_COUNT": "1", **env})
    assert proc.returncode == 0, proc.stderr
    assert "stub" not in proc.stderr
    assert not (tmp_path / "work").exists()


def test_sim_only_dry_run_prints_only_the_simulator(tmp_path):
    proc = run(tmp_path, {"WORK": "/content", "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "MIX": "c3", "SIM_ONLY": "1",
                          "SIM_QUANT": "fp8_per_channel"})
    assert lines(proc) == [SIM_PC.format(mem="0.52", len=131072)]   # the G4's MIX=c3 fraction, as before
    assert "profile: one-gpu-96g" in proc.stdout


@pytest.mark.parametrize("env", [
    {"GPU_MIB": G4_MIB, "GPU_COUNT": "1", "MIX": "c3", "VENV": "", "SIM_QUANT": "fp8_per_channel"},
    {"GPU_MIB": G4_MIB, "GPU_COUNT": "1", "MIX": "c3", "VENV": "", "SIM_QUANT": "fp8_per_channel", "SIM_ONLY": "1"},
    {"GPU_MIB": H100_MIB, "GPU_COUNT": "2", "EAGER": "1", "SIM_QUANT": "fp8_per_channel", "SIM_ONLY": "1"},
], ids=["per-channel", "sim-only", "sim-only-two-gpu-eager"])
def test_the_new_options_start_what_the_dry_run_prints(tmp_path, env):
    work = tmp_path / "work"
    dry = lines(run(tmp_path, {"WORK": str(work), **env}))
    rec = tmp_path / "rec.txt"
    real = run(tmp_path, {"WORK": str(work), **env}, dry=False, record=rec)
    assert real.returncode == 0, real.stderr + real.stdout
    assert rec.read_text().splitlines() == dry
    assert all(x in (work / "probe" / "versions.txt").read_text() for x in dry)


def test_sim_only_stops_the_running_simulator_first_and_keeps_its_log(tmp_path):
    """A simulator already serving (here a stand-in in its own process group, as setsid starts
    it) is stopped, and waited for, before the new one starts; the agent is never started, and
    the old server's log is kept beside the new one."""
    work = tmp_path / "work"
    probe = work / "probe"
    probe.mkdir(parents=True)
    (probe / "simulator.log").write_text("the old server's log\n")
    old = subprocess.Popen(["sleep", "300"], start_new_session=True)
    reaper = threading.Thread(target=old.wait, daemon=True)   # reap it at once, so it doesn't linger as a zombie
    reaper.start()
    (tmp_path / "running").write_text(f"{old.pid}\n")
    rec = tmp_path / "rec.txt"
    try:
        proc = run(tmp_path, {"WORK": str(work), "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "MIX": "c3", "VENV": "",
                              "SIM_ONLY": "1", "SIM_QUANT": "fp8_per_channel"}, dry=False, record=rec)
        assert proc.returncode == 0, proc.stderr + proc.stdout
        reaper.join(timeout=10)
        assert old.poll() is not None, "the running simulator was not stopped"
    finally:
        if old.poll() is None:
            old.kill()
    assert rec.read_text().splitlines() == [SIM_PC.format(mem="0.52", len=131072).replace(
        "/content", str(work))]                                     # the simulator only, no agent
    # only this work directory's simulator: another run's (another WORK) is left alone
    pattern = (tmp_path / "pgrep.txt").read_text().splitlines()[0]
    ours = f"/usr/bin/python3 /usr/local/bin/vllm serve {work}/models/simulator --served-model-name simulator --host"
    others = [f"vllm serve /mnt{work}/models/simulator --served-model-name simulator --host",       # a longer WORK
              f"vllm serve {work}/models/simulator2 --served-model-name simulator --host",
              f"vllm serve {work}/models/agent --served-model-name agent --host"]
    match = lambda line: subprocess.run(["grep", "-qE", "--", pattern], input=line, text=True).returncode == 0
    assert match(ours) and not any(match(x) for x in others), pattern
    kept = [p for p in probe.glob("simulator.*.log")]
    assert len(kept) == 1 and kept[0].read_text() == "the old server's log\n"


def test_without_sim_only_nothing_is_stopped(tmp_path):
    """A normal run starts both servers and never looks for a running simulator."""
    rec = tmp_path / "rec.txt"
    proc = run(tmp_path, {"WORK": str(tmp_path / "work"), "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "VENV": ""},
               dry=False, record=rec)
    assert proc.returncode == 0, proc.stderr
    assert not (tmp_path / "pgrep.txt").exists()
    assert len(rec.read_text().splitlines()) == 2


# --- server.txt: what the setup check and the gate read (explore/sim_quant.py) ---------------------

VLLM_SAYS = ("INFO [base_loader.py:151] Quantizing 340 layers of types: mlp.experts: 40 (from moe: fp8_per_channel_static)\n"
             "INFO [fp8.py:525] Using TRITON Fp8 MoE backend out of potential backends: ['TRITON']\n"
             "INFO [__init__.py:797] Selected CutlassFP8ScaledMMLinearKernel for Fp8PtpcOnlineLinearMethod\n"
             "INFO some other line of the server's log\n")


def test_serve_keeps_the_lines_the_setup_check_reads(tmp_path):
    """The served simulator line, what vLLM says it quantized and the GPU, in one file on both Colab
    and Hopper, whichever scheme it served."""
    work, rec = tmp_path / "work", tmp_path / "rec.txt"
    (tmp_path / "vllm-says").write_text(VLLM_SAYS)
    proc = run(tmp_path, {"WORK": str(work), "GPU_MIB": H100_MIB, "GPU_COUNT": "2", "EAGER": "1"}, dry=False, record=rec)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    text = (work / "probe" / "server.txt").read_text().splitlines()
    assert text[0] == "gpu: NVIDIA H100 80GB HBM3"             # both cards the same: one name
    assert sum("--served-model-name simulator" in x and "--quantization fp8_per_channel" in x for x in text) == 1
    assert not any("--served-model-name agent" in x for x in text)
    assert sum("Quantizing 340 layers" in x for x in text) == 1
    assert sum("Using TRITON Fp8 MoE backend" in x for x in text) == 1
    assert sum("Selected CutlassFP8ScaledMMLinearKernel" in x for x in text) == 1
    assert not any("some other line" in x for x in text)


def test_a_sim_only_restart_replaces_the_lines(tmp_path):
    """The file describes the simulator serving now, not the one a restart stopped."""
    work, rec = tmp_path / "work", tmp_path / "rec.txt"
    (work / "probe").mkdir(parents=True)
    (work / "probe" / "server.txt").write_text("the old server's lines\n")
    proc = run(tmp_path, {"WORK": str(work), "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "VENV": "", "SIM_ONLY": "1"},
               dry=False, record=rec)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    text = (work / "probe" / "server.txt").read_text()
    assert "the old server's lines" not in text and "--quantization fp8_per_channel" in text


def test_the_server_lines_survive_a_log_without_them(tmp_path):
    """No quantization line in the log is the setup check's to refuse: serve.sh itself still ends well."""
    work, rec = tmp_path / "work", tmp_path / "rec.txt"
    proc = run(tmp_path, {"WORK": str(work), "GPU_MIB": G4_MIB, "GPU_COUNT": "1", "VENV": ""}, dry=False, record=rec)
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "Quantizing" not in (work / "probe" / "server.txt").read_text()
