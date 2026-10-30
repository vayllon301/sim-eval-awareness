"""Build the Docker images the coding scenarios need, and clean up stray containers.

Four images (envs/): the SSH client (simtest-devbox-client), the company's git server
(simtest-devbox-git), the dev box base (simtest-devbox) and one image per repo and variant
(simtest-devbox-<repo>:<variant>). The base build is given the client's public key, so only
that client can log in, and the git server's host key, which the box's known_hosts records.
An image is rebuilt when one it depends on was. Images are built for the host architecture;
they pin debian:trixie-slim by digest.
"""

from __future__ import annotations

import subprocess

from .machine import CLIENT_IMAGE, GIT_IMAGE, LABEL
from .scenario import ENVS_DIR, list_scenarios, load_scenario

BASE_IMAGE = "simtest-devbox"
DEVBOX_DIR = ENVS_DIR / "devbox"
CODING_DIR = ENVS_DIR / "coding"


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, check=True, **kw)


def _exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def _build(tag: str, path, rebuild: bool, *args: str) -> bool:
    """Build an image if asked or missing; whether it was built."""
    if not (rebuild or not _exists(tag)):
        return False
    print(f"building {tag}")
    _run(["docker", "build", *args, "-t", tag, str(path)])
    return True


def build_client(rebuild: bool = False) -> bool:
    return _build(CLIENT_IMAGE, DEVBOX_DIR / "client", rebuild)


def build_git(rebuild: bool = False) -> bool:
    return _build(GIT_IMAGE, DEVBOX_DIR / "git", rebuild)


def _cat(image: str, path: str) -> str:
    out = _run(["docker", "run", "--rm", "--entrypoint", "cat", image, path], capture_output=True, text=True)
    return out.stdout.strip()


def client_public_key() -> str:
    return _cat(CLIENT_IMAGE, "/root/.ssh/id_ed25519.pub")


def git_host_key() -> str:
    return _cat(GIT_IMAGE, "/etc/ssh/ssh_host_ed25519_key.pub")


def build_base(rebuild: bool = False) -> bool:
    """The base, rebuilt when the client or the git server was (their keys are in it)."""
    keys = build_client(rebuild) | build_git(rebuild)
    if not (rebuild or keys or not _exists(BASE_IMAGE)):
        return False
    return _build(BASE_IMAGE, DEVBOX_DIR, True, "--build-arg", f"CLIENT_KEY={client_public_key()}",
                  "--build-arg", f"GIT_HOST_KEY={git_host_key()}")


def scenario_image(repo: str, variant: str, rebuild: bool = False) -> str:
    tag = f"{BASE_IMAGE}-{repo}:{variant}"
    if rebuild or not _exists(tag):
        print(f"building {tag}")
        _run(["docker", "build", "--build-arg", f"BASE={BASE_IMAGE}", "--build-arg", f"REPO={repo}",
              "--build-arg", f"VARIANT={variant}", "-t", tag, "-f", str(CODING_DIR / "Dockerfile"),
              str(CODING_DIR)])
    return tag


def scenario_images() -> list[tuple[str, str]]:
    """Every (repo, variant) a scenario uses."""
    seen = {}
    for sid in list_scenarios():
        img = load_scenario(sid).image
        seen[(img["repo"], img["variant"])] = None
    return list(seen)


def build_all(rebuild: bool = False) -> list[str]:
    base = build_base(rebuild)
    return [scenario_image(repo, variant, rebuild or base) for repo, variant in scenario_images()]


def clean() -> int:
    """Remove any leftover simtest containers (a crashed run). Returns how many."""
    out = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={LABEL}"], capture_output=True, text=True)
    ids = out.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    return len(ids)
