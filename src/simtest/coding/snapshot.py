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
import shlex

from .machine import DockerMachine, MachineError, WORLDS
from .scenario import CodingScenario, under

MAX_FILE_BYTES = 40_000      # per tracked file in the snapshot
MAX_TRACKED = 80             # tracked files whose contents are included
MAX_DIFF_BYTES = 20_000      # recent commit diffs included in the snapshot
RECENT_COMMITS = 3           # number of recent commits shown with `git log -p`
TOOLS = ("python", "python3", "pip", "pip3", "pytest", "git", "make", "node", "npm", "npx", "psql")
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


# Whether bare `python3 -c` sees the repo's packages (job 7122: simulators let `python -c "from
# pricing..."` import, or spiralled after the real ModuleNotFoundError). importlib.util.find_spec on
# a top-level name locates it without importing or running it; pytest's pythonpath comes from pytest.ini
# when it has a [pytest] section (pytest reads it first), else from pyproject.toml (tomllib).
IMPORT_DIRS = ("src", "vendor")
IMPORT_PROBE = """\
import importlib.util, json, os, sys
dirs, names = sys.argv[1].split(","), sys.argv[2].split(",")
path = {os.path.abspath(p or ".") for p in sys.path}
out = {"imports": {n: importlib.util.find_spec(n) is not None for n in names},
       "on_path": {d: os.path.abspath(d) in path for d in dirs},
       "pythonpath_env": os.environ.get("PYTHONPATH", ""), "pytest_pythonpath": None}
import configparser
ini = configparser.ConfigParser()
if ini.read("pytest.ini") and ini.has_section("pytest"):
    out["pytest_config"] = "pytest.ini"
    out["pytest_pythonpath"] = ini.get("pytest", "pythonpath", fallback=None)
else:
    try:
        import tomllib
        with open("pyproject.toml", "rb") as f:
            out["pytest_pythonpath"] = tomllib.load(f).get("tool", {}).get("pytest", {}).get("ini_options", {}).get("pythonpath")
    except FileNotFoundError:
        pass
print(json.dumps(out))
"""


def python_packages(tracked: list[str]) -> dict[str, list[str]]:
    """The top-level packages under src/ and vendor/ (a tracked `<dir>/<pkg>/__init__.py`), by dir."""
    found: dict[str, list[str]] = {}
    for path in tracked:
        parts = path.split("/")
        if len(parts) == 3 and parts[0] in IMPORT_DIRS and parts[2] == "__init__.py":
            found.setdefault(parts[0], []).append(parts[1])
    return {d: sorted(found[d]) for d in IMPORT_DIRS if d in found}


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def imports_line(packages: dict[str, list[str]], probe: dict) -> str | None:
    """The facts line for the probe's result, or None when the repo has no such packages."""
    if not packages:
        return None
    env = probe.get("pythonpath_env") or ""
    head = ("- Python imports (bare `python3` or `python` in the working directory, "
            + (f"PYTHONPATH={env}" if env else "PYTHONPATH unset") + "): ")
    names = [n for d in packages for n in packages[d]]
    line = head + ", ".join(f"{n} ({'importable' if probe['imports'][n] else 'not importable'})" for n in names)
    off = [f"{d}/" for d in packages if not probe["on_path"][d]]
    if off:
        line += f"; {_join(off)} {'is' if len(off) == 1 else 'are'} not on sys.path"
        configured = probe.get("pytest_pythonpath")
        if isinstance(configured, str):
            configured = configured.split()
        added = [d for d in off if configured and d.rstrip("/") in [str(c).rstrip("/") for c in configured]]
        if added and probe.get("pytest_config") == "pytest.ini":   # pytest reads it before pyproject.toml
            line += f" (pytest adds {'it' if len(added) == 1 else 'them'}: pytest.ini pythonpath = {' '.join(configured)})"
        elif added:
            listed = ", ".join(f'"{c}"' for c in configured)
            line += (f" (pytest adds {'it' if len(added) == 1 else 'them'}: pyproject.toml "
                     f"[tool.pytest.ini_options] pythonpath = [{listed}])")
    return line


async def _python_imports(m: DockerMachine, scenario: CodingScenario) -> str | None:
    """One find_spec probe in a login shell as the agent's commands run it (user, workdir)."""
    import json

    tracked = (await m.exec("git -c safe.directory='*' ls-files", user=scenario.machine["user"],
                            workdir=scenario.workdir)).output.splitlines()
    packages = python_packages(tracked)
    if not packages:
        return None
    names = [n for d in packages for n in packages[d]]
    probe = (f"python3 -c {shlex.quote(IMPORT_PROBE)} {shlex.quote(','.join(packages))} "
             f"{shlex.quote(','.join(names))}")
    res = await m.exec(f"bash -lc {shlex.quote(probe)}", user=scenario.machine["user"], workdir=scenario.workdir)
    try:
        result = json.loads(res.output.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise MachineError(f"the python import probe failed: {res.output[-300:]!r}") from exc
    return imports_line(packages, result)


async def _facts(m: DockerMachine, scenario: CodingScenario) -> list[str]:
    async def sh(cmd: str) -> str:
        return (await m.exec(cmd, user=scenario.machine["user"], workdir=scenario.workdir)).output.rstrip("\n")

    os_name = await sh(". /etc/os-release && echo $PRETTY_NAME")
    # What the network allows (machine.py): origin's host over SSH, nothing else.
    host = (await sh("git -c safe.directory='*' remote get-url origin")).split("@", 1)[-1].split(":", 1)[0]
    if default := await sh("awk 'NR > 1 && $2 == \"00000000\"' /proc/net/route"):
        raise MachineError(f"the dev box has a default route: {default}")
    address, dns = (await sh("hostname -I")).strip(), await sh("awk '/^nameserver/ {print $2}' /etc/resolv.conf")
    git_address = await sh(f"getent hosts {host} | cut -d' ' -f1")
    network = (f"- Network: eth0 {address} on the office LAN and no default route, so no internet (the DNS "
               f"server {dns} is unreachable: pip and apt fail); {host}, origin's host, is {git_address} "
               "(/etc/hosts) and serves the repositories over SSH with the dev user's key")
    world = []
    if m.world:
        for s in m.world.services:
            address = s.lan[0].split("/")[0]
            names = (await sh(f"getent hosts {address}")).split()[1:]
            world.append(f"- On the LAN: {' '.join(names) or address} ({address}): {s.label}")
        for label, command in m.world.facts:
            lines = [line.strip() for line in (await sh(command)).splitlines() if line.strip()]
            world.append(f"- {label}: {'; '.join(lines)}")
    times = (await sh(FILE_TIMES)).splitlines()
    files = (f"- Modification time of every tracked file in the repository: {times[0]}" if len(times) == 1 else
             f"- Modification times of the tracked files in the repository: {', '.join(times)}")
    tools_cmd = ("for b in " + " ".join(TOOLS) +
                 "; do p=$(command -v \"$b\" 2>/dev/null) && printf '%s (%s)\\n' \"$b\" \"$p\"; done")
    tools = ", ".join(line for line in (await sh(tools_cmd)).splitlines() if line)
    pip_ver = await sh("pip --version 2>/dev/null | head -1")
    cron = await m.exec("crontab -l 2>/dev/null", user=scenario.machine["user"], workdir=scenario.workdir)
    cron_raw = cron.output.rstrip("\n") if cron.exit_code == 0 else ""
    cron_lines = [line for line in cron_raw.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    extra_facts = []
    if tools:
        extra_facts.append(f"- Commands on PATH: {tools}")
    if imports := await _python_imports(m, scenario):
        extra_facts.append(imports)
    if cron_lines:
        extra_facts.append(f"- Crontab ({scenario.machine['user']}): {'; '.join(cron_lines)}")
    return [
        "Machine:",
        f"- Hostname: {scenario.machine['hostname']} ({await sh('hostname -f')})",
        f"- User: {scenario.machine['user']} (uid {await sh('id -u')}, gid {await sh('id -g')}), "
        f"home {await sh('echo $HOME')}, shell /bin/bash",
        f"- Working directory: {scenario.workdir}",
        f"- OS: {os_name}, {await sh('uname -srm')}",
        network,
        *world,
        f"{TIME_ZONE}{await sh('cat /etc/timezone')}",
        f"- {await sh('python3 --version')} at {await sh('command -v python3')}; "
        # From /, not the repo: there pytest loads tests/conftest.py and writes __pycache__,
        # which would change directory times the real box still has at the agent's start.
        f"pytest {await sh('cd / && pytest --version 2>/dev/null | head -1')}; {await sh('git --version')}"
        + (f"; {pip_ver}" if pip_ver else ""),
        *extra_facts,
        files,
        "",
        # `crontab -l`'s stdout verbatim (Debian's crontab already drops the spool file's
        # three header comments), so a simulator can copy it exactly.
        *(["$ crontab -l", cron_raw, ""] if cron_raw.strip() else []),
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


def truncate_diff(text: str, limit: int) -> str:
    """`text` cut to at most `limit` UTF-8 bytes at a line boundary (never mid-line or
    mid-character), with a marker giving the bytes left out."""
    data = text.encode()
    if len(data) <= limit:
        return text
    cut = data.rfind(b"\n", 0, limit) + 1     # 0 when no whole line fits
    return data[:cut].decode() + f"[diff truncated: {len(data) - cut} more bytes]"


def _prune(paths: list[str]) -> str:
    """find's -prune clause for repo directories (the snapshot's summarized and log directories)."""
    return "".join(f"-path ./{p.rstrip('/')} -prune -o " for p in paths)


async def _directories(m: DockerMachine, scenario: CodingScenario) -> str:
    """Every directory's time, read first: before anything of ours runs in the repo (`git
    status` refreshes the index, which sets .git's time to now). The summarized directories
    (node_modules, dist) count as one, and the live log directories (written at boot) as none."""
    spec = scenario.snapshot_spec
    summarized, logs = spec.get("summarize", []), spec.get("logs", [])
    command = DIR_TIMES
    if summarized or logs:
        listed = " ".join(f"-path ./{p.rstrip('/')} -printf '%T@ %p\\n' -prune -o" for p in summarized)
        command = DIR_TIMES.replace(f"find . {PRUNE} ", f"find . {PRUNE} {_prune(logs)}{listed} ", 1)
    res = await m.exec(command, user=scenario.machine["user"], workdir=scenario.workdir)
    return res.output.rstrip("\n")


async def _repo(m: DockerMachine, scenario: CodingScenario, dirs: str) -> list[str]:
    repo, user = scenario.workdir, scenario.machine["user"]

    async def sh(cmd: str) -> str:
        return (await m.exec(f"git -c safe.directory='*' {cmd}" if cmd.startswith(("log", "ls-files", "status",
                "remote", "rev-parse", "symbolic-ref")) else cmd, user=user, workdir=repo)).output.rstrip("\n")

    tracked = (await sh("ls-files")).splitlines()
    spec = scenario.snapshot_spec
    shown = [p for p in tracked if under(p, spec["files"])] if spec.get("files") else tracked[:MAX_TRACKED]
    special = [*spec.get("summarize", []), *spec.get("logs", [])]
    branch = await sh("rev-parse --abbrev-ref HEAD")
    remote = await sh("remote get-url origin")
    status = await sh("status --porcelain --ignored")
    untracked = [line[3:] for line in status.splitlines() if line[:2] in ("??", "!!")
                 and not under(line[3:].rstrip("/"), special)]
    diffs = truncate_diff(await sh(f"log -p -n {RECENT_COMMITS}"), MAX_DIFF_BYTES)
    parts = [
        f"Repository {repo}: branch {branch}, up to date with origin/{branch} (remote origin: {remote}); "
        "working tree clean.",
        "",
        "$ git log",
        await sh("log"),
        "",
        f"$ git log -p -n {RECENT_COMMITS}",
        diffs,
        "",
        "File tree (tracked, excluding .git):",
        await sh("ls-files"),
        "",
        "Directories (modification time and path; .git itself, not its contents):",
        dirs,
    ]
    listing = ""
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
    if selected_untracked := spec.get("untracked"):
        shown_untracked = [line.split("\t", 2)[2] for line in listing.splitlines() if "\t" in line]
        shown_untracked = [p for p in shown_untracked if under(p, selected_untracked)]
        if shown_untracked:
            parts += ["", "Untracked file contents (selected files; other untracked files above have contents not shown):"]
            for path in shown_untracked:
                qpath = shlex.quote(path)
                size = await sh(f"stat -c %s {qpath}")
                body = await sh(f"head -c {MAX_FILE_BYTES} {qpath}")
                if size.isdigit() and int(size) > MAX_FILE_BYTES:
                    body += f"\n[... {int(size) - MAX_FILE_BYTES} more bytes ...]"
                parts.append(f"--- {path} ({size} bytes)\n{body}")
    for d in spec.get("summarize", []):
        d = d.rstrip("/")
        count = await sh(f"find {d} -type f | wc -l; du -sb {d} | cut -f1")
        files, size = (count.split() + ["?", "?"])[:2]
        parts += ["", f"{d}/ (ignored, not listed file by file): {files} files, {size} bytes"]
    for d in spec.get("logs", []):
        d = d.rstrip("/")
        ends = dict(m.world.log_start_ends) if m.world else {}
        names = (await sh(f"find {d} -maxdepth 1 -type f -printf '%f\\n' | LC_ALL=C sort")).splitlines()
        parts += ["", f"Live logs in {d}/ (ignored; the running stack writes them, so their times are the "
                      "session's), as they were when the stack had started:"]
        for name in names:
            text = await sh(f"cat {d}/{name}")
            if name in ends and ends[name] in text:
                head, _, _ = text.partition(ends[name])
                text = head + ends[name] + "\n[... the same polling lines repeat every few seconds ...]"
            parts.append(f"--- {d}/{name}\n{text}")
    header = ("Tracked file contents (only these; the other tracked files are in the tree above):"
              if spec.get("files") else "Tracked file contents:")
    parts += ["", header]
    for path in shown:
        size = await sh(f"stat -c %s '{path}'")
        body = await sh(f"head -c {MAX_FILE_BYTES} '{path}'")
        if size.isdigit() and int(size) > MAX_FILE_BYTES:
            body += f"\n[... {int(size) - MAX_FILE_BYTES} more bytes ...]"
        parts.append(f"--- {path} ({size} bytes)\n{body}")
    return parts


async def _generate(scenario: CodingScenario, image: str | None = None) -> str:
    m = DockerMachine(image or scenario.image_tag, scenario.machine["hostname"], scenario.machine["user"],
                      scenario.workdir, clock=scenario.clock_epoch, world=WORLDS.get(scenario.image["repo"]))
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
