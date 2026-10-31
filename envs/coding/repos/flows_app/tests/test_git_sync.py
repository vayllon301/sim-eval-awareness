"""Git sync's scan on the fixtures (tests/fixtures/workflow-repos), through dist/ (`make build` first)."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "workflow-repos"


def scan(team_id: int, repos: Path = FIXTURES) -> list[dict]:
    script = (f"require('./dist/services/git-sync').scanTeamWorkflows({team_id})"
              ".then((w) => { console.log(JSON.stringify(w)); process.exit(0); })")
    out = subprocess.run(["node", "-e", script], cwd=ROOT, env={**os.environ, "WORKFLOW_REPOS_DIR": str(repos)},
                         capture_output=True, text=True, check=True, timeout=60)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_scan_reads_each_workflow_file():
    workflows = scan(1)
    assert sorted(w["filename"] for w in workflows) == ["low-stock-alert.json", "weekly-orders-export.json"]
    assert all(w["teamId"] == 1 for w in workflows)


def test_scan_keeps_the_definition_and_triggers():
    by_name = {w["filename"]: w["content"] for w in scan(1)}
    content = by_name["low-stock-alert.json"]
    assert [n["id"] for n in content["definition"]["nodes"]] == ["fetch-stock", "any-low", "notify"]
    assert content["triggers"] == {"cron": "0 8 * * 1-5"}


def test_scan_skips_a_file_that_is_not_json(tmp_path):
    workflows = tmp_path / "1" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "broken.json").write_text('{"definition": ')
    (workflows / "ok.json").write_text(json.dumps({"definition": {"nodes": [], "edges": []}}))
    (workflows / "notes.txt").write_text("not a workflow")
    assert [w["filename"] for w in scan(1, tmp_path)] == ["ok.json"]


def test_a_team_without_a_repo_has_no_workflows(tmp_path):
    assert scan(7, tmp_path) == []
