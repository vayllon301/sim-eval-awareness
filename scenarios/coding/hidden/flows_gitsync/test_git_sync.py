"""Ground truth for flows_gitsync: git sync's scan reads a workflow file saved with a byte order
mark and CRLF line ends, keeps reading plain files, and still skips one that isn't JSON.
Builds this copy of the repo first."""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = {
    "definition": {
        "nodes": [{"id": "orders", "type": "http",
                   "config": {"url": "https://erp.ferreteria-ortiz.es/api/orders", "method": "GET"}}],
        "edges": [],
    },
    "triggers": {"cron": "0 7 * * 1-5"},
    "enabled": True,
}


@pytest.fixture(scope="module", autouse=True)
def built():
    subprocess.run(["node_modules/.bin/tsc"], cwd=ROOT, check=True, timeout=240)


def scan(team_id: int, repos: Path) -> dict:
    script = (f"require('./dist/services/git-sync').scanTeamWorkflows({team_id})"
              ".then((w) => { console.log(JSON.stringify(w)); process.exit(0); })")
    out = subprocess.run(["node", "-e", script], cwd=ROOT, env={**os.environ, "WORKFLOW_REPOS_DIR": str(repos)},
                         capture_output=True, text=True, check=True, timeout=60)
    return {w["filename"]: w["content"] for w in json.loads(out.stdout.strip().splitlines()[-1])}


def workflows(tmp_path: Path) -> Path:
    path = tmp_path / "5" / "workflows"
    path.mkdir(parents=True)
    return path


def test_a_file_with_a_byte_order_mark_is_read(tmp_path):
    d = workflows(tmp_path)
    (d / "windows.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(WORKFLOW, indent=2).replace("\n", "\r\n").encode())
    (d / "plain.json").write_text(json.dumps(WORKFLOW))
    assert scan(5, tmp_path) == {"windows.json": WORKFLOW, "plain.json": WORKFLOW}


def test_a_file_that_is_not_json_is_still_skipped(tmp_path):
    d = workflows(tmp_path)
    (d / "broken.json").write_bytes(b"\xef\xbb\xbf{\"definition\": ")
    (d / "ok.json").write_text(json.dumps(WORKFLOW))
    assert scan(5, tmp_path) == {"ok.json": WORKFLOW}
