"""DockerMachine's read watcher fails loudly (offline: the docker calls are stubbed)."""

from __future__ import annotations

import subprocess

import anyio
import pytest

from simtest.coding.machine import DockerMachine, DockerTimeout, MachineError


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


def test_a_readiness_poll_that_times_out_is_not_ready_yet(monkeypatch):
    # Session 5 (LOG.md, 01-11-2026): with Docker under load, one 30 s `docker exec` in the
    # readiness loop timed out and raised, failing the episode, though the limit was minutes.
    # A poll that times out counts as "not ready"; only the limit raises.
    m = DockerMachine("img", "lk-devbox-14", "dev", "/home/dev/repo")
    calls = []

    async def docker(*args, timeout=120):
        calls.append(args)
        if len(calls) == 1:
            raise DockerTimeout("docker exec timed out after 30 s")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(m, "_docker", docker)
    anyio.run(m._until, ["exec", "box", "true"], 60, "the box's app")
    assert len(calls) == 2

    async def always_slow(*args, timeout=120):
        raise DockerTimeout("docker exec timed out after 30 s")

    monkeypatch.setattr(m, "_docker", always_slow)
    with pytest.raises(MachineError, match="not ready within"):
        anyio.run(m._until, ["exec", "box", "true"], 0, "the box's app")


# An agent command's own exit code, 255 included, against a failure of ssh itself.

def _run_with(monkeypatch, returncode, stdout, stderr):
    import simtest.coding.machine as machine_mod

    m = DockerMachine("img", "lk-devbox-14", "dev", "/home/dev/repo")

    async def fake(argv, timeout, input=None):
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    monkeypatch.setattr(machine_mod, "_arun", fake)
    return anyio.run(m.run, "ssh deploy@build.internal")


def test_the_box_reports_the_commands_exit_code_on_stderr():
    m = DockerMachine("img", "lk-devbox-14", "dev", "/home/dev/repo")
    assert m.remote("true").endswith("2>&1; } 2>/dev/null; echo $? >&2")


def test_a_command_that_itself_exits_255_is_its_exit_code_not_an_ssh_failure(monkeypatch):
    r = _run_with(monkeypatch, 0, b"ssh: Could not resolve hostname build.internal\n", b"255\n")
    assert r.exit_code == 255 and "Could not resolve" in r.output and not r.timed_out


def test_an_ssh_failure_is_still_an_infrastructure_error(monkeypatch):
    with pytest.raises(MachineError, match="ssh to the dev box failed"):
        _run_with(monkeypatch, 255, b"", b"ssh: connect to host 127.0.0.1 port 22: Connection refused\n")
    with pytest.raises(MachineError, match="ssh to the dev box failed"):    # an ssh message, then a code
        _run_with(monkeypatch, 0, b"", b"Connection to 127.0.0.1 closed by remote host.\n0\n")


def test_the_clients_timeout_wins_over_an_earlier_reported_code(monkeypatch):
    r = _run_with(monkeypatch, 124, b"foreground\n", b"0\n")
    assert r.exit_code == 124 or r.timed_out
