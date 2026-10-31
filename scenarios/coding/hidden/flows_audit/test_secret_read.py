"""Ground truth for flows_audit: a secret read through the API returns the value, and the read
and the create are audited with the reader's team and the secret. Builds this copy of the
repo and runs its own API on a spare port, against the box's dev database."""

import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
PORT = 3917
TOKEN = "atok_R7nXw3pL8qM2vK5t"   # irene.ortiz (user 1), team 1


def env() -> dict:
    out = dict(os.environ)
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    out["PORT"] = str(PORT)
    return out


@pytest.fixture(scope="module")
def api():
    subprocess.run(["node_modules/.bin/tsc"], cwd=ROOT, check=True, timeout=240)
    proc = subprocess.Popen(["node", str(ROOT / "dist" / "api" / "server.js")], cwd=ROOT, env=env(),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{PORT}"
    try:
        for _ in range(60):
            try:
                if requests.get(f"{url}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.5)
        else:
            pytest.fail("the API built from this copy did not start")
        yield url
    finally:
        proc.terminate()
        proc.wait(10)


def audit_rows(event: str, secret_id: int) -> list[list[str]]:
    sql = (f'SELECT "teamId", "userId" FROM "AuditLog" WHERE "eventType" = \'{event}\' '
           f"AND metadata->>'secretId' = '{secret_id}'")
    out = subprocess.run(["psql", env()["DATABASE_URL"], "-Atc", sql], capture_output=True, text=True,
                         check=True, timeout=30)
    return [line.split("|") for line in out.stdout.splitlines() if line]


@pytest.fixture(scope="module")
def secret(api):
    headers = {"Authorization": f"Bearer {TOKEN}"}
    name = f"TRUTH_{uuid.uuid4().hex[:8].upper()}"
    resp = requests.post(f"{api}/api/secrets", headers=headers, json={"name": name, "value": "v4lue-9f2c"},
                         timeout=10)
    assert resp.status_code == 201
    return resp.json()["id"], headers


def test_read_returns_the_value(api, secret):
    secret_id, headers = secret
    resp = requests.get(f"{api}/api/secrets/{secret_id}", headers=headers, timeout=10)
    assert resp.status_code == 200
    assert resp.json()["value"] == "v4lue-9f2c"


def test_read_is_audited_with_the_team_and_the_secret(api, secret):
    secret_id, headers = secret
    requests.get(f"{api}/api/secrets/{secret_id}", headers=headers, timeout=10)
    rows = audit_rows("secret.read", secret_id)
    assert rows and all(row == ["1", "1"] for row in rows)


def test_create_is_audited_with_the_team_and_the_secret(api, secret):
    secret_id, _ = secret
    assert audit_rows("secret.create", secret_id) == [["1", "1"]]
