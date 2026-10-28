"""The real machine of an episode: a dev box container and its SSH client (envs/devbox).

Every condition runs every agent command here (shadow execution): in C0 the agent sees
this output; in C1/C2 a simulator rewrites it; in C3 the simulator invents an output and
this one is the reference.

- `docker run --network none --hostname <scenario's>`: sysvinit boots runlevel 2 (rsyslog,
  cron, sshd). The client container joins the dev box's network namespace only, so the
  agent's commands arrive over SSH on loopback and nothing of the harness shows up in the
  dev box's process list.
- A command runs as `cd <workdir> && exec timeout -k 5 <deadline> bash -lc <command> 2>&1`
  on the dev box: the deadline is enforced inside the box, and stdout and stderr are
  merged there, in order. The client's own `timeout` and a host-side timeout are
  backstops; a host-side timeout is an infrastructure error.
- Harness work (baseline, final checks) goes through `docker exec` as root or dev, never
  during the agent's turns.

Blocking Docker calls run in a worker thread, so parallel episodes don't block each other.
"""

from __future__ import annotations

import shlex
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import anyio

from .prompts import DEADLINE_S, OMITTED_MARKER, OUTPUT_CAP, TIMEOUT_MARKER

CLIENT_IMAGE = "simtest-devbox-client"
LABEL = "simtest.coding=1"   # every container we start; `simtest images --clean` removes leftovers
READY_TIMEOUT_S = 60


class MachineError(RuntimeError):
    """The machine itself failed (Docker, SSH): an infrastructure error, never the agent's doing."""


@dataclass
class ExecResult:
    output: str            # stdout and stderr as merged in the box (timeout marker appended), not capped
    exit_code: int | None
    timed_out: bool
    seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def finalize(text: str) -> str:
    """The same whitespace and character policy for every condition: no trailing whitespace,
    and NUL bytes (which some providers reject) shown as U+FFFD."""
    return text.replace("\x00", "�").rstrip()


def cap(text: str) -> str:
    """The output cap, the same in every condition: the first and last OUTPUT_CAP/2 characters."""
    if len(text) <= OUTPUT_CAP:
        return text
    half = OUTPUT_CAP // 2
    return f"{text[:half]}\n{OMITTED_MARKER.format(n=len(text) - 2 * half)}\n{text[-half:]}"


def tool_output(text: str) -> str:
    """What the bash tool returns for an output: finalized, then capped."""
    return cap(finalize(text))


def _run(argv: list[str], timeout: float, input: bytes | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, timeout=timeout, input=input)


async def _arun(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    return await anyio.to_thread.run_sync(lambda: _run(argv, timeout))


class Machine(Protocol):
    info: dict

    async def start(self) -> None: ...
    async def run(self, command: str) -> ExecResult: ...
    async def exec(self, command: str, user: str = "root", timeout: float = 180, workdir: str | None = None) -> ExecResult: ...
    async def copy_in(self, src: Path, dest: str) -> None: ...
    async def docker_diff(self) -> str: ...
    async def stop(self) -> None: ...


class DockerMachine:
    def __init__(self, image: str, hostname: str, user: str, workdir: str, deadline: int = DEADLINE_S,
                 client_image: str = CLIENT_IMAGE):
        self.image, self.hostname, self.user, self.workdir = image, hostname, user, workdir
        self.deadline, self.client_image = deadline, client_image
        self.name = f"simtest-{uuid.uuid4().hex[:12]}"
        self.client = f"{self.name}-ssh"
        self.info: dict = {"image": image, "container": self.name, "hostname": hostname}

    async def _docker(self, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
        try:
            p = await _arun(["docker", *args], timeout)
        except subprocess.TimeoutExpired as exc:
            raise MachineError(f"docker {args[0]} timed out after {timeout} s") from exc
        except OSError as exc:
            raise MachineError(f"docker not available: {exc}") from exc
        return p

    async def _checked(self, *args: str, timeout: float = 120) -> str:
        p = await self._docker(*args, timeout=timeout)
        if p.returncode != 0:
            raise MachineError(f"docker {' '.join(args[:2])} failed: {p.stderr.decode(errors='replace').strip()}")
        return p.stdout.decode(errors="replace").strip()

    async def start(self) -> None:
        self.info["image_id"] = await self._checked("image", "inspect", "--format", "{{.Id}}", self.image)
        await self._checked("run", "-d", "--name", self.name, "--hostname", self.hostname, "--network", "none",
                            "--pids-limit", "1024", "--memory", "3g", "--label", LABEL, self.image)
        await self._checked("run", "-d", "--name", self.client, "--network", f"container:{self.name}",
                            "--label", LABEL, self.client_image)
        t0 = time.monotonic()
        while True:
            p = await self._docker("exec", self.client, "ssh", "-n", f"{self.user}@127.0.0.1", "true", timeout=30)
            if p.returncode == 0:
                break
            if time.monotonic() - t0 > READY_TIMEOUT_S:
                raise MachineError(f"the dev box did not accept SSH within {READY_TIMEOUT_S} s: "
                                   f"{p.stderr.decode(errors='replace').strip()}")
            await anyio.sleep(0.2)
        self.info["started"] = time.time()

    def remote(self, command: str) -> str:
        """The command line the dev box's login shell runs for one agent command."""
        return (f"cd {shlex.quote(self.workdir)} && exec timeout -k 5 {self.deadline} "
                f"bash -lc {shlex.quote(command)} 2>&1")

    async def run(self, command: str) -> ExecResult:
        argv = ["docker", "exec", self.client, "timeout", "-k", "2", str(self.deadline + 10),
                "ssh", "-n", f"{self.user}@127.0.0.1", self.remote(command)]
        t0 = time.monotonic()
        try:
            p = await _arun(argv, self.deadline + 30)
        except subprocess.TimeoutExpired as exc:
            raise MachineError(f"no answer from the dev box within {self.deadline + 30} s") from exc
        seconds = round(time.monotonic() - t0, 3)
        err = p.stderr.decode(errors="replace").strip()
        # The command's own stderr is merged into stdout inside the box, so anything on the
        # client's stderr comes from docker or ssh.
        if err or p.returncode == 255:
            raise MachineError(f"ssh to the dev box failed (exit {p.returncode}): {err}")
        output = p.stdout.decode("utf-8", errors="replace")
        timed_out = p.returncode in (124, 137) and seconds >= self.deadline - 1
        if timed_out:
            output = (output if output.endswith("\n") or not output else output + "\n") + \
                TIMEOUT_MARKER.format(deadline=self.deadline)
        return ExecResult(output, None if timed_out else p.returncode, timed_out, seconds)

    async def exec(self, command: str, user: str = "root", timeout: float = 180,
                   workdir: str | None = None) -> ExecResult:
        """Harness commands (baseline, checks), outside the agent's session."""
        args = ["exec", "-u", user] + (["-w", workdir] if workdir else []) + [self.name, "sh", "-c", command]
        t0 = time.monotonic()
        p = await self._docker(*args, timeout=timeout)
        out = p.stdout.decode("utf-8", errors="replace") + p.stderr.decode("utf-8", errors="replace")
        return ExecResult(out, p.returncode, False, round(time.monotonic() - t0, 3))

    async def copy_in(self, src: Path, dest: str) -> None:
        await self._checked("cp", str(src), f"{self.name}:{dest}")

    async def docker_diff(self) -> str:
        return await self._checked("diff", self.name)

    async def stop(self) -> None:
        await self._docker("rm", "-f", self.client, self.name, timeout=60)


def scenario_machine(scenario, deadline: int = DEADLINE_S) -> DockerMachine:
    m = scenario.machine
    return DockerMachine(scenario.image_tag, m["hostname"], m["user"], m["workdir"], deadline)


# Machine kinds a task can name; tests register scripted ones.
MACHINES = {"docker": scenario_machine}
