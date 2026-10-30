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


class FakeDocker:
    """Stubs DockerMachine's docker calls: records them, fails (or hangs) on the first call that
    contains `fail_on` (or `hang_on`), and records which names `rm -f` removed."""

    def __init__(self, m, fail_on=None, hang_on=None):
        self.m, self.fail_on, self.hang_on, self.calls, self.removed = m, fail_on, hang_on, [], set()

    async def checked(self, *args, timeout=120, input=None):
        self.calls.append(args)
        if self.hang_on and self.hang_on in args:
            await anyio.sleep_forever()
        if self.fail_on and self.fail_on in args:
            raise MachineError(f"docker {args[0]} failed")
        return "4242"

    async def docker(self, *args, timeout=120, input=None):
        if args[:2] == ("rm", "-f"):
            self.removed |= set(args[2:])
        return subprocess.CompletedProcess(args, 0, b"", b"")

    def started(self) -> set[str]:
        """The names of the containers the calls started."""
        return {a[a.index("--name") + 1] for a in self.calls if a[0] in ("run", "create") and "--name" in a}


def _fake(monkeypatch, **kw):
    m = DockerMachine("img", "lk-devbox-14", "dev", "/home/dev/repo", clock=1793004060)
    fake = FakeDocker(m, **kw)
    monkeypatch.setattr(m, "_checked", fake.checked)
    monkeypatch.setattr(m, "_docker", fake.docker)
    monkeypatch.setattr(m, "_copy_out", lambda path: _tar_bytes())
    return m, fake


async def _tar_bytes():
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo("id_ed25519.pub")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"key"))
    return buf.getvalue()


def test_a_git_server_setup_failure_removes_what_boot_started(monkeypatch):
    # S1: setup-repo fails after the box, the git server and the network helper were started;
    # boot raises, and every one of them is removed (the episode never calls stop then).
    m, fake = _fake(monkeypatch, fail_on="setup-repo")
    with pytest.raises(MachineError):
        anyio.run(m.boot)
    assert {m.name, m.git, m.net} <= fake.started() <= fake.removed


def test_a_cancelled_boot_removes_what_it_started(monkeypatch):
    # S1: cancelled while the git server is set up (a run's timeout, an interrupted eval).
    m, fake = _fake(monkeypatch, hang_on="setup-repo")

    async def go():
        with anyio.move_on_after(0.2):
            await m.boot()

    anyio.run(go)
    assert {m.name, m.git, m.net} <= fake.started() <= fake.removed


def test_a_client_failure_removes_what_start_started(monkeypatch):
    # The SSH client is started after boot; if that fails, start removes the box and the rest.
    m, fake = _fake(monkeypatch)
    fake.fail_on = f"container:{m.name}"
    with pytest.raises(MachineError):
        anyio.run(m.start)
    assert {m.name, m.git, m.net, m.client} <= fake.started() <= fake.removed
