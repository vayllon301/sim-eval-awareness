"""explore/colab_probe/tunnel.sh: besides Colab sessions, a plain SSH host (Hopper) gets the same
reconnecting tunnel. Runs the script with a stub ssh that records its arguments and exits, so the
loop reconnects; TUNNEL_TRIES bounds it for the test."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "explore" / "colab_probe" / "tunnel.sh"
ZSH = shutil.which("zsh")
pytestmark = pytest.mark.skipif(ZSH is None, reason="needs zsh")


def run(tmp_path, env):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    rec = tmp_path / "ssh.txt"
    (stubs / "ssh").write_text(f"#!/bin/sh\necho \"$*\" >> {rec}\nexit 255\n")
    (stubs / "colab").write_text("#!/bin/sh\necho 'colab stub called' >&2\nexit 99\n")
    (stubs / "sleep").write_text("#!/bin/sh\nexit 0\n")
    for p in stubs.iterdir():
        p.chmod(0o755)
    proc = subprocess.run([ZSH, str(SCRIPT)], env={"PATH": f"{stubs}:/usr/bin:/bin", **env},
                          capture_output=True, text=True, timeout=30)
    return proc, (rec.read_text().splitlines() if rec.exists() else [])


def test_a_plain_ssh_host_gets_a_reconnecting_tunnel(tmp_path):
    proc, calls = run(tmp_path, {"SSH_TARGET": "gpu-node-7", "SSH_JUMP": "victor@login", "TUNNEL_TRIES": "3"})
    assert proc.returncode == 0, proc.stderr
    assert len(calls) == 3                       # reconnected after each drop
    for c in calls:
        assert c.startswith("-J victor@login ")
        assert "-N -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001 gpu-node-7" in c
        assert "ServerAliveInterval=15" in c and "ExitOnForwardFailure=yes" in c
    assert "colab stub called" not in proc.stderr


def test_no_jump_host_means_no_jump(tmp_path):
    proc, calls = run(tmp_path, {"SSH_TARGET": "victor@hopper", "TUNNEL_TRIES": "1"})
    assert proc.returncode == 0, proc.stderr
    assert calls and not calls[0].startswith("-J")
    assert calls[0].endswith("victor@hopper")
