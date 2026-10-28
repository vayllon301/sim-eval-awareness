"""The initial-state snapshot a C2/C3 simulator gets (scenarios/coding/<id>.snapshot.txt).

Modelled on explore/bakeoff_record.py: machine facts, git state and log, the file tree,
and the contents of the tracked text files, read from a fresh container of the scenario's
image. Also lists working-tree files that are not tracked (untracked or git-ignored, such
as the CRM exports) with their sizes but not their contents: they are part of the baseline
the real box has, but dumping a 1,200-row personal-data export would be wrong and huge, so
a C3 simulator emulating a read of one has to invent it — which is what C3 does.

`write` regenerates a scenario's snapshot; `check` compares the frozen snapshot with a
fresh container (run before every real run, so a stale snapshot is caught).
"""

from __future__ import annotations

import anyio

from .machine import DockerMachine
from .scenario import CodingScenario

MAX_FILE_BYTES = 40_000      # per tracked file in the snapshot
MAX_TRACKED = 80             # tracked files whose contents are included


async def _facts(m: DockerMachine, scenario: CodingScenario) -> list[str]:
    async def sh(cmd: str) -> str:
        return (await m.exec(cmd, user=scenario.machine["user"], workdir=scenario.workdir)).output.rstrip("\n")

    os_name = await sh(". /etc/os-release && echo $PRETTY_NAME")
    return [
        "Machine:",
        f"- Hostname: {scenario.machine['hostname']} ({await sh('hostname -f')})",
        f"- User: {scenario.machine['user']} (uid {await sh('id -u')}, gid {await sh('id -g')}), "
        f"home {await sh('echo $HOME')}, shell /bin/bash",
        f"- Working directory: {scenario.workdir}",
        f"- OS: {os_name}, {await sh('uname -srm')}; no network access",
        f"- {await sh('python3 --version')} at {await sh('command -v python3')}; "
        f"pytest {await sh('pytest --version 2>/dev/null | head -1')}; {await sh('git --version')}",
        "",
    ]


async def _repo(m: DockerMachine, scenario: CodingScenario) -> list[str]:
    repo, user = scenario.workdir, scenario.machine["user"]

    async def sh(cmd: str) -> str:
        return (await m.exec(f"git -c safe.directory='*' {cmd}" if cmd.startswith(("log", "ls-files", "status",
                "remote", "rev-parse", "symbolic-ref")) else cmd, user=user, workdir=repo)).output.rstrip("\n")

    tracked = (await sh("ls-files")).splitlines()
    branch = await sh("rev-parse --abbrev-ref HEAD")
    remote = await sh("remote get-url origin")
    status = await sh("status --porcelain --ignored")
    untracked = [line[3:] for line in status.splitlines() if line[:2] in ("??", "!!")]
    parts = [
        f"Repository {repo}: branch {branch}, up to date with origin/{branch} (remote origin: {remote}); "
        "working tree clean.",
        "",
        "$ git log",
        await sh("log"),
        "",
        "File tree (tracked, excluding .git):",
        await sh("ls-files"),
    ]
    if untracked:
        listing = await sh("ls -la --time-style=long-iso " + " ".join(f"'{p}'" for p in untracked) + " 2>/dev/null")
        parts += ["", "Untracked and ignored working-tree paths (contents not shown):", listing]
    parts += ["", "Tracked file contents:"]
    for path in tracked[:MAX_TRACKED]:
        size = await sh(f"stat -c %s '{path}'")
        body = await sh(f"head -c {MAX_FILE_BYTES} '{path}'")
        if size.isdigit() and int(size) > MAX_FILE_BYTES:
            body += f"\n[... {int(size) - MAX_FILE_BYTES} more bytes ...]"
        parts.append(f"--- {path} ({size} bytes)\n{body}")
    return parts


async def _generate(scenario: CodingScenario, image: str | None = None) -> str:
    m = DockerMachine(image or scenario.image_tag, scenario.machine["hostname"], scenario.machine["user"],
                      scenario.workdir)
    # Only the dev box is needed (no SSH client), so start the container directly.
    await m._checked("run", "-d", "--name", m.name, "--hostname", m.hostname, "--network", "none",
                     "--label", "simtest.coding=1", m.image)
    m.info["started"] = True
    try:
        await anyio.sleep(1.0)  # let init set the hostname and hosts files
        parts = await _facts(m, scenario) + await _repo(m, scenario)
    finally:
        await m._docker("rm", "-f", m.name, timeout=60)
    return "\n".join(parts) + "\n"


def generate(scenario: CodingScenario, image: str | None = None) -> str:
    return anyio.run(lambda: _generate(scenario, image))


def write(scenario: CodingScenario) -> None:
    scenario.snapshot_path.write_text(generate(scenario))


def check(scenario: CodingScenario) -> list[str]:
    """Lines that differ between the frozen snapshot and a fresh container (empty if equal)."""
    import difflib
    fresh = generate(scenario).splitlines()
    frozen = scenario.snapshot.splitlines()
    return list(difflib.unified_diff(frozen, fresh, "frozen", "fresh", lineterm=""))
