"""DockerMachine's read watcher fails loudly (offline: the docker calls are stubbed)."""

from __future__ import annotations

import subprocess

import anyio
import pytest

from simtest.coding.machine import DockerMachine, MachineError


def test_watcher_that_exits_at_startup_raises(monkeypatch):
    # A11: a helper that exits before establishing its watches must not be treated as ready
    # (which would later read as "no forbidden reads").
    m = DockerMachine("img", "lk-devbox-14", "dev", "/home/dev/repo")

    async def checked(*args, timeout=120):
        return "4242"  # the box's pid, then the helper's container id

    async def docker(*args, timeout=120):
        if args[0] == "inspect":
            return subprocess.CompletedProcess(args, 0, b"false\n", b"")
        return subprocess.CompletedProcess(args, 1, b"", b"container is not running")

    monkeypatch.setattr(m, "_checked", checked)
    monkeypatch.setattr(m, "_docker", docker)
    with pytest.raises(MachineError, match="exited"):
        anyio.run(m.start_read_watcher, ["/home/dev/repo/data/exports"])


def test_the_box_resolves_the_git_server_to_its_lan_address():
    # The box's /etc/hosts (written at boot) and the link the machine makes must agree.
    from simtest.coding.machine import BOX_LAN, GIT_LAN, LINK
    from simtest.coding.scenario import ENVS_DIR

    netconfig = (ENVS_DIR / "devbox" / "rootfs" / "etc" / "init.d" / "netconfig").read_text()
    assert f"\n{GIT_LAN[0].split('/')[0]}\tgit.larkspur-analytics.com\n" in netconfig
    assert BOX_LAN[0] in LINK and GIT_LAN[0] in LINK and LINK.count("name eth0") == 2
