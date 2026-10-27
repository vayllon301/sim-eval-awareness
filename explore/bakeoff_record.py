"""Exploratory: record the real outputs of the bake-off command sequences in Docker.

Not part of the testbed, and not evidence for any comparison. For the simulator bake-off
(LOG.md, 26-10-2026). Each sequence in envs/pyrepo/sequences.yaml runs in a fresh
`pyrepo` container (no network, hostname lk-devbox-14). Each command runs as `dev`
through `docker exec ... bash -lc <command>` in the repo, with stdout and stderr merged
into one pipe on the host side (so `ps` shows only `bash -lc <command>`) and a 30 s
timeout. Everything is recorded twice, in fresh containers; a step whose output or exit
code differs between the two recordings is marked non-deterministic.

Also writes snapshot.txt: the initial state every simulator gets (machine facts, git
state and log, file tree, full contents of the tracked files), read from a fresh
container.

Run (free, local; Docker running):
    docker build -t pyrepo envs/pyrepo
    uv run python -I explore/bakeoff_record.py logs/bakeoff
"""
import datetime, json, os, subprocess, sys, time

import yaml

OUT = sys.argv[1]
IMAGE, HOST, REPO, TIMEOUT = "pyrepo", "lk-devbox-14", "/home/dev/pricing-service", 30
SEQS = yaml.safe_load(open("envs/pyrepo/sequences.yaml"))["sequences"]


def start(name):
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--rm", "--network", "none", "--hostname", HOST,
                    "--name", name, IMAGE], check=True, capture_output=True)


def run(name, command):
    t = time.time()
    try:
        p = subprocess.run(["docker", "exec", "-u", "dev", "-w", REPO, name, "bash", "-lc", command],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=TIMEOUT)
        out, code = p.stdout, p.returncode
    except subprocess.TimeoutExpired as e:
        subprocess.run(["docker", "exec", "-u", "root", name, "pkill", "-KILL", "-f", "bash -lc"], capture_output=True)
        out, code = e.output or b"", 124
    return out.decode("utf-8", "replace"), code, round(time.time() - t, 3)


def record(rep):
    rows = []
    for seq in SEQS:
        name = f"bakeoff-{seq['id']}-{rep}"
        start(name)
        for i, cmd in enumerate(seq["commands"]):
            out, code, secs = run(name, cmd)
            rows.append({"seq": seq["id"], "step": i, "command": cmd, "output": out, "exit_code": code, "seconds": secs})
            print(rep, seq["id"], i, code, secs, cmd.splitlines()[0][:70], flush=True)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return rows


def snapshot(started):
    name = "bakeoff-snapshot"
    start(name)
    sh = lambda c: run(name, c)[0].rstrip("\n")
    files = sh("git ls-files").splitlines()
    os_name = sh(". /etc/os-release && echo $PRETTY_NAME")
    parts = [
        "Machine:",
        f"- Hostname: {HOST}",
        "- User: dev (uid 1000, gid 1000), home /home/dev, shell /bin/bash",
        f"- Working directory: {REPO}",
        f"- OS: {os_name}, {sh('uname -m')}; container image python:3.12-slim; no network access",
        f"- {sh('python --version')} at /usr/local/bin/python (also python3); "
        f"{sh('pip --version').split(' from ')[0]}; pytest {sh('pytest --version').split()[-1]} installed system-wide",
        f"- {sh('git --version')}",
        f"- Session started: {started}",
        "",
        f"Repository {REPO}: branch main, up to date with origin/main "
        f"(remote origin: {sh('git remote get-url origin')}); working tree clean.",
        "",
        "$ git log",
        sh("git log"),
        "",
        "File tree (excluding .git):",
        sh("find . -path ./.git -prune -o -print | sort"),
        "",
        "File contents:",
    ]
    for path in files:
        body = sh(f"cat {path}")
        parts.append(f"--- {path} ({sh(f'stat -c %s {path}')} bytes)\n{body}")
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    return "\n".join(parts) + "\n"


os.makedirs(OUT, exist_ok=True)
started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
image = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
                       capture_output=True, text=True, check=True).stdout.strip()
with open(f"{OUT}/snapshot.txt", "w") as f:
    f.write(snapshot(started))
a, b = record("a"), record("b")
with open(f"{OUT}/real.jsonl", "w") as f:
    for ra, rb in zip(a, b):
        ra.update(deterministic=(ra["output"], ra["exit_code"]) == (rb["output"], rb["exit_code"]),
                  output_b=rb["output"], exit_code_b=rb["exit_code"], image=image)
        f.write(json.dumps(ra) + "\n")
print("non-deterministic:", [(r["seq"], r["step"], r["command"][:40]) for r in a if not r["deterministic"]])
