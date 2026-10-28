"""Helpers for the scenario repos: a git history with fixed hashes, made to look like a clone.

The same steps as envs/pyrepo/make_repo.py (so the pricing repo keeps pyrepo's hashes):
each commit writes its files (None deletes one) with fixed author and committer dates,
then origin/main, the upstream config, one packfile, a clone reflog and fixed file times.
"""

import os
import subprocess

HOME = "/home/dev"
USER, HOST = "dev", "lk-devbox-14"


def write_files(root: str, files: dict) -> None:
    for path, text in files.items():
        full = os.path.join(root, path)
        if text is None:
            os.remove(full)
            continue
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(full, 0o755 if path.endswith(".sh") else 0o644)


def git(root: str, *args: str, env: dict | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True,
                         env={**os.environ, **(env or {})})
    return out.stdout.strip()


def make_repo(root: str, origin: str, commits: list, checkout_time: int) -> str:
    """commits: [((name, email), iso_date, message, {path: text | None}), ...]. Returns HEAD."""
    os.makedirs(root)
    git(root, "init", "-q", "-b", "main")
    for (name, email), date, message, files in commits:
        write_files(root, files)
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", message, env={
            "GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": email, "GIT_AUTHOR_DATE": date,
            "GIT_COMMITTER_NAME": name, "GIT_COMMITTER_EMAIL": email, "GIT_COMMITTER_DATE": date,
        })
    git(root, "remote", "add", "origin", origin)
    git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    git(root, "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main")
    git(root, "config", "branch.main.remote", "origin")
    git(root, "config", "branch.main.merge", "refs/heads/main")
    git(root, "repack", "-adq")
    git(root, "prune-packed")
    head = git(root, "rev-parse", "HEAD")
    for log in ("HEAD", "refs/heads/main", "refs/remotes/origin/HEAD"):
        path = os.path.join(root, ".git", "logs", log)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"{'0' * 40} {head} {USER} <{USER}@{HOST}> {checkout_time} +0000\tclone: from {origin}\n")
    return head


def set_times(root: str, when: int) -> None:
    """Every file and directory under root (root included) gets the same times."""
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            os.utime(os.path.join(dirpath, name), (when, when), follow_symlinks=False)
    os.utime(root, (when, when))


def set_time(path: str, when: int) -> None:
    os.utime(path, (when, when), follow_symlinks=False)


def finish_home(history: list[str], last_session: int) -> None:
    """Append this repo's lines to the shell history; fix the times of what changed."""
    path = os.path.join(HOME, ".bash_history")
    with open(path, "a") as f:
        f.write("".join(line + "\n" for line in history))
    os.utime(path, (last_session, last_session))
    os.utime(HOME, (last_session, last_session))
