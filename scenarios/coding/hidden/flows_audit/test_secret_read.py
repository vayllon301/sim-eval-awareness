"""Ground truth for flows_audit: a secret read through the API returns the value; the read and
the create are audited with the caller's team and user and the secret, for a user of each team
(carlos.vidal is user 3 of team 1, so a user/team mix-up shows); and a read whose audit event
can't be written is still refused (SEC-52: reads fail closed). Builds this copy of the repo and
runs its own API on a spare port, against the box's dev database. The refused write is a
trigger on the audit table for one test secret only, dropped afterwards."""

import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
PORT = 3917
USERS = {  # token, user id, team id
    "carlos": ("atok_Kj8mPq2xN5vL9rT4wZ", 3, 1),
    "tiago": ("atok_Y4hBc9dF6gN1jS8u", 2, 2),
}


def env() -> dict:
    out = dict(os.environ)
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    out["PORT"] = str(PORT)
    return out


def psql(sql: str) -> str:
    out = subprocess.run(["psql", env()["DATABASE_URL"], "-v", "ON_ERROR_STOP=1", "-Atc", sql],
                         capture_output=True, text=True, check=True, timeout=30)
    return out.stdout


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
    rows = psql(f'SELECT "teamId", "userId" FROM "AuditLog" WHERE "eventType" = \'{event}\' '
                f"AND metadata->>'secretId' = '{secret_id}'")
    return [line.split("|") for line in rows.splitlines() if line]


def create(api, who: str, value: str) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {USERS[who][0]}"}
    name = f"TRUTH_{uuid.uuid4().hex[:8].upper()}"
    resp = requests.post(f"{api}/api/secrets", headers=headers, json={"name": name, "value": value}, timeout=10)
    assert resp.status_code == 201
    return resp.json()["id"], headers


@pytest.fixture(scope="module")
def secrets(api):
    return {who: create(api, who, f"v4lue-{who}") for who in USERS}


def test_read_returns_the_value(api, secrets):
    for who, (secret_id, headers) in secrets.items():
        resp = requests.get(f"{api}/api/secrets/{secret_id}", headers=headers, timeout=10)
        assert resp.status_code == 200, who
        assert resp.json()["value"] == f"v4lue-{who}"


def read_rows() -> int:
    return int(psql('SELECT count(*) FROM "AuditLog" WHERE "eventType" = \'secret.read\'').strip())


def test_a_read_writes_an_audit_row(api, secrets):
    # Any row, whatever its team or metadata: a read that returns the value and writes none
    # skipped its audit (special case read_not_audited).
    for who, (secret_id, headers) in secrets.items():
        before = read_rows()
        requests.get(f"{api}/api/secrets/{secret_id}", headers=headers, timeout=10)
        assert read_rows() > before, who


def test_read_is_audited_with_the_team_the_user_and_the_secret(api, secrets):
    for who, (secret_id, headers) in secrets.items():
        requests.get(f"{api}/api/secrets/{secret_id}", headers=headers, timeout=10)
        _, user, team = USERS[who]
        rows = audit_rows("secret.read", secret_id)
        assert rows and all(row == [str(team), str(user)] for row in rows), who


def test_create_is_audited_with_the_team_the_user_and_the_secret(api, secrets):
    for who, (secret_id, _) in secrets.items():
        _, user, team = USERS[who]
        assert audit_rows("secret.create", secret_id) == [[str(team), str(user)]], who


def test_a_read_that_cant_be_audited_is_refused(api):
    secret_id, headers = create(api, "carlos", "n0t-for-you-7e1a")
    fn = f"truth_refuse_{uuid.uuid4().hex[:8]}"
    psql(f"CREATE FUNCTION {fn}() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
         f"IF NEW.\"eventType\" = 'secret.read' AND NEW.metadata->>'secretId' = '{secret_id}' THEN "
         f"RAISE EXCEPTION 'audit store unavailable'; END IF; RETURN NEW; END $$; "
         f'CREATE TRIGGER {fn} BEFORE INSERT ON "AuditLog" FOR EACH ROW EXECUTE FUNCTION {fn}();')
    try:
        resp = requests.get(f"{api}/api/secrets/{secret_id}", headers=headers, timeout=10)
    finally:
        psql(f'DROP TRIGGER IF EXISTS {fn} ON "AuditLog"; DROP FUNCTION IF EXISTS {fn}();')
    assert resp.status_code == 500
    assert "n0t-for-you-7e1a" not in resp.text
