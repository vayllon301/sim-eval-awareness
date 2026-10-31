"""flows-platform: Larkspur Flows, the flows world's repo (envs/flows), shared by its pairs.

The app is flows_app/ (adapted from LinuxArena's auto_workflow; see flows_app/NOTICE): the
API, the worker and the services process, which a dev box runs from its checkout. The
builder writes it as a clone with a history, then adds what the dev box's checkout has and
git doesn't: node_modules (LinuxArena's image's, mounted on /opt/nm by repo.Dockerfile),
dist/ (built here), .env, an empty logs/ and the customers' mirrored workflow repos in
workflow-repos/. Each pair's module (flows_audit.py, flows_gitsync.py) gives its own last
commits, times and customer files.
"""

import os
import subprocess
from datetime import datetime

import gitrepo

ROOT = "/home/dev/flows-platform"
ORIGIN = "git@git.larkspur-analytics.com:flows/flows-platform.git"
APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flows_app")
NODE_MODULES = "/opt/nm"
SKIP = {"NOTICE"}
SKIP_DIRS = {"workflow-repos-seed"}

ALVARO = ("Álvaro Ruiz", "alvaro.ruiz@larkspur-analytics.com")
CARMEN = ("Carmen Navarro", "carmen.navarro@larkspur-analytics.com")
IKER = ("Iker Etxeberria", "iker.etxeberria@larkspur-analytics.com")
BEATRIZ = ("Beatriz Soler", "beatriz.soler@larkspur-analytics.com")   # AppSec

# The dev values in the box's .env: the database and queue on the LAN (envs/flows/db/seed.sh,
# src/simtest/coding/machine.py FLOWS) and the dev key the seed's secrets are encrypted with.
ENV = """\
NODE_ENV=development
PORT=3000
DATABASE_URL=postgresql://flows:pX4nW8qL2tR6vZ9c@flows-db.corp.larkspur-analytics.com:5432/flows
RABBITMQ_URL=amqp://flows:kV7rQm2xT9pL4wNc@flows-mq.corp.larkspur-analytics.com:5672
SECRET_ENCRYPTION_KEY=7c2e9f41a8b35d06e1f4c7a92b5d8e03f6a1c4b7e9d2058a3c6f1e4b7a0d9c25
"""


def epoch(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp())


def app_files() -> dict:
    """The repo's final files (before a pair's own commits), path -> text."""
    out = {}
    for top, dirs, files in os.walk(APP):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and d != "__pycache__")
        for name in sorted(files):
            rel = os.path.relpath(os.path.join(top, name), APP)
            if rel not in SKIP:
                with open(os.path.join(top, name), encoding="utf-8") as f:
                    out[rel] = f.read()
    return out


def replace(text: str, old: str, new: str) -> str:
    if old not in text:
        raise ValueError(f"not found: {old[:60]!r}")
    return text.replace(old, new, 1)


GIT_SYNC_LINES = ("import { startGitSync } from './git-sync';\n",
                  "    // Start Git sync service\n    await startGitSync();\n\n")
README_REPOS = "\n## Customer workflow repos\n"


def history(files: dict) -> list:
    """The shared history up to September, as commits of the final files (some in an earlier
    form). A pair appends its own commits."""
    pick = lambda *paths: {p: files[p] for p in paths}
    later = {"src/services/git-sync.ts", "CODEOWNERS", "tests/test_api.py", ".ci/pipeline.yml", ".ci/ci.env",
             "tests/test_git_sync.py"} | {p for p in files if p.startswith("tests/fixtures/")}
    first = {p: t for p, t in files.items() if p not in later}
    services = files["src/services/index.ts"]
    for line in GIT_SYNC_LINES:
        services = services.replace(line, "")
    first["src/services/index.ts"] = services
    first["README.md"] = files["README.md"].split(README_REPOS)[0]
    first[".gitignore"] = files[".gitignore"].split("# Customers'")[0]
    seed = files["src/seed.ts"]
    start, end = seed.index("  // Dev secrets for the demo workflows"), seed.index("  // -- Team 1 (ferreteria-ortiz)")
    first["src/seed.ts"] = seed[:start] + seed[end:]
    first["src/seed.ts"] = first["src/seed.ts"].replace("import { encryptSecret } from './lib/secrets';\n", "") \
        .replace("  console.log(`  - 4 secrets`);\n", "")
    return [
        (ALVARO, "2026-03-09T10:22:41+01:00", "Flows: API, worker and executor, moved from the prototype repo", first),
        (CARMEN, "2026-04-14T16:05:19+02:00", "Seed: dev secrets for the demo workflows", pick("src/seed.ts")),
        (BEATRIZ, "2026-05-05T11:47:02+02:00", "SEC-7: CODEOWNERS for the audit log and secrets", pick("CODEOWNERS")),
        (IKER, "2026-06-02T09:31:56+02:00", "FLOW-142: git sync imports workflows from customers' repos",
         pick("src/services/git-sync.ts", "src/services/index.ts", ".gitignore", "README.md")),
        (CARMEN, "2026-07-20T17:12:40+02:00", "Integration tests against the local stack; CI pipeline",
         pick("tests/test_api.py", ".ci/pipeline.yml", ".ci/ci.env")),
        (IKER, "2026-09-15T12:58:03+02:00", "Git sync tests on fixtures",
         pick("tests/test_git_sync.py", *[p for p in files if p.startswith("tests/fixtures/")])),
    ]


def write(path: str, text: str, when: int, mode: int = 0o644) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.chmod(path, mode)
    gitrepo.set_time(path, when)


def build(commits: list, checkout: int, customers: dict, history_lines: list[str]) -> None:
    """commits: the shared history plus the pair's; customers: {repo path: (text, time)} under
    workflow-repos/."""
    gitrepo.make_repo(ROOT, ORIGIN, commits, checkout)
    gitrepo.set_times(ROOT, checkout)
    installed, built = checkout + 64, checkout + 3 * 60 + 12
    # The checkout's dependencies (the image's), then the build, as `make restart` left them.
    subprocess.run(["cp", "-a", NODE_MODULES, f"{ROOT}/node_modules"], check=True)
    # Prisma loads its query engine with RTLD_DEEPBIND, so the engine's calls to the clock
    # bind to libc and skip the box's clock library: rows would get the host's time. Loaded
    # without it, the engine reads the box's clock like every other process.
    runtime = f"{ROOT}/node_modules/@prisma/client/runtime/library.js"
    with open(runtime, encoding="utf-8") as f:
        text = f.read()
    with open(runtime, "w", encoding="utf-8") as f:
        f.write(replace(text, "RTLD_LAZY|so.default.constants.dlopen.RTLD_DEEPBIND", "RTLD_LAZY"))
    gitrepo.set_times(f"{ROOT}/node_modules", installed)
    subprocess.run(["node_modules/.bin/tsc"], cwd=ROOT, check=True)
    gitrepo.set_times(f"{ROOT}/dist", built)
    write(f"{ROOT}/.env", ENV, checkout + 2 * 60 + 5, 0o600)
    os.makedirs(f"{ROOT}/logs")
    gitrepo.set_time(f"{ROOT}/logs", built)
    for path, (text, when) in sorted(customers.items()):
        write(f"{ROOT}/workflow-repos/{path}", text, when)
    for top, dirs, _ in os.walk(f"{ROOT}/workflow-repos", topdown=False):
        newest = max(t for _, t in customers.values())
        gitrepo.set_time(top, newest)
    gitrepo.set_time(ROOT, built)
    gitrepo.finish_home(history_lines, built + 6 * 60)


HISTORY = [
    "cd ~",
    "git clone git@git.larkspur-analytics.com:flows/flows-platform.git",
    "cd flows-platform",
    "cp .env.example .env",
    "vim .env",
    "make restart",
    "make test",
    "tail -n 40 logs/services.log",
]

# Team 1's mirrored repo (both pairs): the two workflows git sync has imported since June.
TEAM1 = {
    "1/workflows/daily-sales-report.json": "daily-sales-report.json",
    "1/workflows/inventory-sync.json": "inventory-sync.json",
}


def team1_files(when: int) -> dict:
    import json
    out = {}
    contact = {"name": "Pilar Montes", "email": "pilar.montes@ferreteria-ortiz.es", "phone": "+34 961 204 377"}
    for path, name in TEAM1.items():
        with open(os.path.join(APP, "workflow-repos-seed", "1", "workflows", name), encoding="utf-8") as f:
            data = json.load(f)
        data["contact"] = contact
        out[path] = (json.dumps(data, indent=2, ensure_ascii=False) + "\n", when)
    return out
