"""Build the Docker images the coding scenarios need, and clean up stray containers.

Three layers (envs/): the SSH client (simtest-devbox-client), the dev box base
(simtest-devbox) and one image per repo and variant (simtest-devbox-<repo>:<variant>).
The base build is given the client's public key, so only that client can log in. Images
are built for the host architecture; the base pins debian:trixie-slim by digest.
"""

from __future__ import annotations

import subprocess

from .machine import CLIENT_IMAGE, LABEL
from .scenario import ENVS_DIR, list_scenarios, load_scenario

BASE_IMAGE = "simtest-devbox"
DEVBOX_DIR = ENVS_DIR / "devbox"
CODING_DIR = ENVS_DIR / "coding"


def _run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, check=True, **kw)


def _exists(image: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


def build_client(rebuild: bool = False) -> None:
    if rebuild or not _exists(CLIENT_IMAGE):
        print(f"building {CLIENT_IMAGE}")
        _run(["docker", "build", "-t", CLIENT_IMAGE, str(DEVBOX_DIR / "client")])


def client_public_key() -> str:
    out = _run(["docker", "run", "--rm", "--entrypoint", "cat", CLIENT_IMAGE, "/root/.ssh/id_ed25519.pub"],
               capture_output=True, text=True)
    return out.stdout.strip()


def build_base(rebuild: bool = False) -> None:
    build_client(rebuild)
    if rebuild or not _exists(BASE_IMAGE):
        print(f"building {BASE_IMAGE}")
        _run(["docker", "build", "--build-arg", f"CLIENT_KEY={client_public_key()}", "-t", BASE_IMAGE,
              str(DEVBOX_DIR)])


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
    build_base(rebuild)
    return [scenario_image(repo, variant, rebuild) for repo, variant in scenario_images()]


def clean() -> int:
    """Remove any leftover simtest containers (a crashed run). Returns how many."""
    out = subprocess.run(["docker", "ps", "-aq", "--filter", f"label={LABEL}"], capture_output=True, text=True)
    ids = out.stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
    return len(ids)
