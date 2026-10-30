"""The initial-state snapshot a C2/C3 simulator gets (scenarios/coding/<id>.snapshot.txt).

Modelled on explore/bakeoff_record.py: machine facts, git state and log, the file tree,
and the contents of the tracked text files, read from a fresh container of the scenario's
image. Also lists working-tree files that are not tracked (untracked or git-ignored, such
as the CRM exports) with their times and sizes but not their contents: they are part of the
baseline the real box has, but dumping a 1,200-row personal-data export would be wrong and
huge, so a C3 simulator emulating a read of one has to invent it — which is what C3 does.

The machine facts include the time zone and the modification time of the tracked files, and
the snapshot lists every directory's time (the root, .git and the working tree's) and every
untracked file's. All are fixed at build (gitrepo.set_times, the repo builders), so a rebuild
gives byte-identical output and a simulator can make `date`, `ls -l` and the git log agree.
The date and time when the session started can't be frozen: the box's clock starts at the
scenario's `clock` and runs on, so the episode reads `date` on the real box at the start and
`session_snapshot` adds it to the facts block of the text the C2 and C3 simulators get. The
frozen file, its fingerprint and `check` never include it. The snapshot's own box runs on the
same clock.

`write` regenerates a scenario's snapshot; `check` compares the frozen snapshot with a
fresh container (run before every real run, so a stale snapshot is caught).
"""

from __future__ import annotations

import anyio

from .machine import DockerMachine, MachineError
from .scenario import CodingScenario

MAX_FILE_BYTES = 40_000      # per tracked file in the snapshot
MAX_TRACKED = 80             # tracked files whose contents are included
TIME_ZONE = "- Time zone: "
SESSION_START = "- Date and time when the session started: {date}"
STAMP = "date -d \"@${t%.*}\" '+%Y-%m-%d %H:%M:%S %Z (%z)'"   # an epoch $t in the box's time zone
# Each distinct modification time of the tracked files.
FILE_TIMES = f"git -c safe.directory='*' ls-files -z | xargs -0 stat -c %Y | sort -un | while read t; do {STAMP}; done"
# Build artefacts, never listed (they don't exist in the image; a run could make them).
PRUNE = r"\( -name __pycache__ -o -name .pytest_cache -o -name '*.pyc' \) -prune -o"
# The time and path of every directory: the root, .git itself (not its contents) and the
# working tree's, sorted by path.
DIR_TIMES = (f"find . {PRUNE} -path ./.git -printf '%T@ %p\\n' -prune -o -type d -printf '%T@ %p\\n' | "
             f"sed 's| \\./| |' | LC_ALL=C sort -k2 | while read t p; do printf '%s\\t%s\\n' \"$({STAMP})\" \"$p\"; done")


async def _facts(m: DockerMachine, scenario: CodingScenario) -> list[str]:
    async def sh(cmd: str) -> str:
        return (await m.exec(cmd, user=scenario.machine["user"], workdir=scenario.workdir)).output.rstrip("\n")

    os_name = await sh(". /etc/os-release && echo $PRETTY_NAME")
    times = (await sh(FILE_TIMES)).splitlines()
    files = (f"- Modification time of every tracked file in the repository: {times[0]}" if len(times) == 1 else
             f"- Modification times of the tracked files in the repository: {', '.join(times)}")
    return [
        "Machine:",
        f"- Hostname: {scenario.machine['hostname']} ({await sh('hostname -f')})",
        f"- User: {scenario.machine['user']} (uid {await sh('id -u')}, gid {await sh('id -g')}), "
        f"home {await sh('echo $HOME')}, shell /bin/bash",
        f"- Working directory: {scenario.workdir}",
        f"- OS: {os_name}, {await sh('uname -srm')}; no network access",
        f"{TIME_ZONE}{await sh('cat /etc/timezone')}",
        f"- {await sh('python3 --version')} at {await sh('command -v python3')}; "
        # From /, not the repo: there pytest loads tests/conftest.py and writes __pycache__,
        # which would change directory times the real box still has at the agent's start.
        f"pytest {await sh('cd / && pytest --version 2>/dev/null | head -1')}; {await sh('git --version')}",
        files,
        "",
    ]


async def session_start(machine, scenario: CodingScenario) -> str:
    """`date` on the real box, as harness work (docker exec, never an agent command): run at
    episode start, after the machine starts and before the agent's first turn."""
    res = await machine.exec("date", user=scenario.machine["user"])
    text = res.output.strip()
    if res.exit_code != 0 or not text:
        raise MachineError(f"could not read the date on the dev box: {res.output[-200:]!r}")
    return text


def session_snapshot(snapshot: str, started: str) -> str:
    """The snapshot as the C2 and C3 simulators get it: the frozen text plus one line in its
    facts block, after the time zone: the date and time when the session started."""
    lines = snapshot.split("\n")
    for i, line in enumerate(lines):
        if not line:  # the facts block ends at the first blank line
            break
        if line.startswith(TIME_ZONE):
            return "\n".join(lines[:i + 1] + [SESSION_START.format(date=started)] + lines[i + 1:])
    raise ValueError("the snapshot has no time zone line in its facts block; regenerate it (simtest snapshot)")


async def _directories(m: DockerMachine, scenario: CodingScenario) -> str:
    """Every directory's time, read first: before anything of ours runs in the repo (`git
    status` refreshes the index, which sets .git's time to now)."""
    res = await m.exec(DIR_TIMES, user=scenario.machine["user"], workdir=scenario.workdir)
    return res.output.rstrip("\n")


async def _repo(m: DockerMachine, scenario: CodingScenario, dirs: str) -> list[str]:
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
        "",
        "Directories (modification time and path; .git itself, not its contents):",
        dirs,
    ]
    if untracked:
        # Time, size and path (no permissions), with build artefacts pruned. The times are
        # fixed at build, so the snapshot is reproducible: a rebuild gives byte-identical
        # output. This reveals that the untracked data (e.g. the CRM exports) exists, without
        # its contents.
        listing = await sh(f"find {' '.join(repr(p) for p in untracked)} {PRUNE} -type f "
                           "-printf '%T@ %s %p\\n' 2>/dev/null | LC_ALL=C sort -k3 | "
                           f"while read t s p; do printf '%s\\t%s\\t%s\\n' \"$({STAMP})\" \"$s\" \"$p\"; done")
        if listing.strip():
            parts += ["", "Untracked and ignored files (modification time, size in bytes and path; contents not shown):",
                      listing]
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
                      scenario.workdir, clock=scenario.clock_epoch)
    # The dev box and its git server as in an episode (on the scenario's clock), no SSH client.
    await m.boot()
    try:
        await anyio.sleep(1.0)  # let init set the hostname and hosts files
        dirs = await _directories(m, scenario)
        parts = await _facts(m, scenario) + await _repo(m, scenario, dirs)
    finally:
        await m.stop()
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
