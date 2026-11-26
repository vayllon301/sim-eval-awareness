"""Offline boundary checks for the development-only C2 replay."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def checkout_import_path(monkeypatch):
    # The pytest console entrypoint, used by CI, does not add the checkout root.
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "explore" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("command", [
    "python3 -c 'from pathlib import Path; Path(\"x\").write_text(\"fixed\")'",
    "perl -pi -e 's/old/new/' src/app.ts",
    "make test",
    "find src -name '*.ts' -exec sed -i 's/old/new/g' {} +",
    "sort -o src/app.ts src/app.ts",
])
def test_reference_stops_at_writes_and_unproven_commands(command):
    builder = script("build_reference")
    steps = [{"command": c, "output": "output"} for c in ("cat src/app.ts", command, "git log -1")]
    assert builder.prefix(steps, {"cat src/app.ts", "git log -1"}) == steps[:1]


def test_reference_keeps_known_reads():
    builder = script("build_reference")
    steps = [{"command": c, "output": "output"} for c in ("git status", "cat src/app.ts", "ls -la")]
    assert builder.prefix(steps, {s["command"] for s in steps}) == steps


def test_unknown_read_form_abstains_instead_of_extending_allowlist():
    builder = script("build_reference")
    steps = [{"command": "cat src/app.ts", "output": "old"},
             {"command": "cat src/app.ts; sort -o src/app.ts src/app.ts", "output": "new"}]
    assert builder.prefix(steps, {"cat src/app.ts"}) == steps[:1]


def job(condition="C2"):
    return {"key": f"flows_audit:0:{condition}", "id": "flows_audit", "index": 0,
            "condition": condition, "command": "cat missing.txt", "real": "unique current answer",
            "messages": [{"role": "system", "content": "An initial snapshot"},
                         {"role": "user", "content": "cat missing.txt"}], "history": []}


def reply(text, stop="stop"):
    return {"model": "fake-local", "choices": [{"message": {"content": text}, "finish_reason": stop}]}


def test_truncated_exact_output_never_passes():
    replay = script("c2_controlled_replay")
    result = replay.evaluate(job(), reply(job()["real"], "length"))
    assert result["grade"]["grade"] == "exact"
    assert not result["faithful"] and not result["complete"]


def test_current_output_is_not_assumed_given_to_reference_c2():
    replay = script("c2_controlled_replay")
    result = replay.evaluate(job(), reply("a different answer"))
    assert result["grade"]["grade"] == "needs_judgement"
    assert not result["faithful"]


def test_unclosed_thought_never_passes():
    replay = script("c2_controlled_replay")
    result = replay.evaluate(job(), reply("<think>unfinished"))
    assert not result["complete"] and not result["faithful"]


def manifest(replay):
    return {"config": {"scenarios": ["flows_audit"], "minimum_gain": 0.15},
            "jobs": [job("C2"), job("C3")]}


def test_missing_pair_and_failed_request_hold_the_screen():
    replay = script("c2_controlled_replay")
    data = manifest(replay)
    rows = [{"key": data["jobs"][0]["key"], "manifest_sha256": replay.digest(data),
             "complete": True, "faithful": True, "grade": {"grade": "exact", "tells": []}}]
    assert replay.report(data, rows)["status"] == "INCOMPLETE"
    rows.append({"key": data["jobs"][1]["key"], "manifest_sha256": replay.digest(data), "error": "offline"})
    report = replay.report(data, rows)
    assert report["status"] == "INCOMPLETE"
    assert report["counts"]["C3"]["errors"] == 1


def test_report_rejects_duplicate_or_stale_results():
    replay = script("c2_controlled_replay")
    data = manifest(replay)
    row = {"key": data["jobs"][0]["key"], "manifest_sha256": replay.digest(data)}
    with pytest.raises(ValueError):
        replay.report(data, [row, row])
    with pytest.raises(ValueError):
        replay.report(data, [dict(row, manifest_sha256="old configuration")])


def test_run_saves_full_reply_and_refuses_to_overwrite(tmp_path, monkeypatch):
    import io
    import json

    replay = script("c2_controlled_replay")
    data = manifest(replay)
    data["code_commit"] = "seeded"
    data["config"].update(model="fake-local", generation={}, wall_seconds=60, request_timeout_seconds=1)
    content = "a long answer\n" * 1000
    monkeypatch.setattr(replay, "head", lambda: "seeded")
    monkeypatch.setattr(replay.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(json.dumps(reply(content)).encode()))
    replay.run(tmp_path, data)
    rows = [json.loads(line) for line in (tmp_path / "responses.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["reply"]["choices"][0]["message"]["content"] == content
    assert rows[0]["parsed_output"] == content
    with pytest.raises(FileExistsError):
        replay.run(tmp_path, data)


def test_fixed_kind_selection_ignores_outcomes():
    replay = script("c2_controlled_replay")
    steps = [{"command": c} for c in ("git status", "cat a", "cat b", "pytest -q", "pytest -v")]
    assert replay.select_steps(steps, ["read", "test"]) == [1, 3]


def test_c0_requires_executed_matching_ground_truth():
    replay = script("c2_controlled_replay")
    sample = {"scores": {"coding_scorer": {"metadata": {"end_reason": "no_tool_call", "flagged": False}}},
              "store": {"CodingEpisode:steps": [{"tool": "bash", "executed": True,
                        "result": {"output": "real"}, "observation": "invented"}]}}
    with pytest.raises(ValueError, match="differs"):
        replay.bash_steps(sample)
    sample["store"]["CodingEpisode:steps"][0]["observation"] = "real"
    assert len(replay.bash_steps(sample)) == 1
    sample["store"]["CodingEpisode:steps"][0]["executed"] = False
    with pytest.raises(ValueError, match="ground truth"):
        replay.bash_steps(sample)


def _c0_log(path, samples, condition="C0"):
    import json
    import zipfile

    with zipfile.ZipFile(path, "w") as z:
        z.writestr("_journal/start.json", json.dumps({"eval": {"task_args": {"condition": condition}}}))
        for sid, commands in samples.items():
            z.writestr(f"samples/{sid}_epoch_1.json", json.dumps({
                "id": sid, "scores": {"coding_scorer": {"metadata": {"end_reason": "no_tool_call", "flagged": False}}},
                "store": {"CodingEpisode:steps": [{"tool": "bash", "executed": True, "command": c, "observation": f"out {c}"}
                                                  for c in commands]}}))


def test_build_reference_writes_only_the_named_scenarios(tmp_path, monkeypatch):
    import json

    builder = script("build_reference")
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"a": ["ls", "cat x"], "b": ["ls"]}))
    monkeypatch.setattr(builder, "ALLOWLIST", allow)
    logs = tmp_path / "logs"
    logs.mkdir()
    # "c" has no audited allowlist and "a" a mutating step: only "a" is asked for
    _c0_log(logs / "c0.eval", {"a": ["ls", "cat x", "sed -i s/1/2/ x", "cat x"], "b": ["ls"], "c": ["ls"]})
    _c0_log(logs / "c3.eval", {"a": ["ls"]}, condition="C3")
    out = tmp_path / "out"
    out.mkdir()
    builder.main([str(logs), "--scenarios", "a", "--out", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["a.reference.json"]
    assert json.loads((out / "a.reference.json").read_text()) == [{"command": "ls", "output": "out ls"},
                                                                  {"command": "cat x", "output": "out cat x"}]
    with pytest.raises(SystemExit, match="d"):              # a named scenario missing from the log
        builder.main([str(logs), "--scenarios", "a", "d", "--out", str(out)])


def test_build_reference_keeps_its_one_argument_form(tmp_path, monkeypatch):
    import json

    builder = script("build_reference")
    allow = tmp_path / "allow.json"
    allow.write_text(json.dumps({"a": ["ls"], "b": ["ls"]}))
    monkeypatch.setattr(builder, "ALLOWLIST", allow)
    logs = tmp_path / "logs"
    logs.mkdir()
    _c0_log(logs / "c0.eval", {"a": ["ls"], "b": ["ls", "make test"]})
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scenarios/coding").mkdir(parents=True)
    builder.main([str(logs)])
    assert sorted(p.name for p in (tmp_path / "scenarios/coding").iterdir()) == ["a.reference.json", "b.reference.json"]


def test_the_python_references_are_audited_read_only_prefixes():
    import json
    import re

    allow = json.loads((Path(__file__).parents[1] / "configs/coding/c2_reference_readonly.json").read_text())
    for sid in ("pricing_cap", "pricing_cap_benign", "billing_rounding", "billing_rounding_benign",
                "orders_schema", "orders_schema_benign"):
        ref = json.loads((Path(__file__).parents[1] / f"scenarios/coding/{sid}.reference.json").read_text())
        assert ref and [x["command"] for x in ref] == allow[sid], sid
        for x in ref:   # no test run, write, or code execution
            assert not re.search(r"\b(?:pytest|make|python3?|pip3?|node|npm|tee|commit|revert|checkout)(?:\s|$)|"
                                 r"sed -i| > |>>", x["command"]), x["command"]
