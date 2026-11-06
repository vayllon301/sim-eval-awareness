"""Offline checks that a continuation never replaces recorded samples."""

import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    from explore import c2_controlled_resume as resume
    original = tmp_path / "original"
    original.mkdir()
    model = {k: 1 for k in resume.MODEL_FIELDS}
    model["identifier"] = "local"
    manifest = {"code_commit": resume.BASE,
                "config": {"model": "local", "generation": {"max_tokens": 65536},
                           "scenarios": ["seed"], "minimum_gain": .15,
                           "wall_seconds": 5400, "request_timeout_seconds": 600}, "jobs": []}
    for i in range(24):
        manifest["jobs"].append({"key": str(i), "id": "seed", "index": i,
                                 "condition": "C2" if i % 2 else "C3",
                                 "command": "cat a", "real": "content",
                                 "messages": [{"role": "system", "content": "snapshot"},
                                              {"role": "user", "content": str(i)}], "history": []})
    reply = {"model": "local", "choices": [{"message": {"content": "content"}, "finish_reason": "stop"}]}
    rows = [{"key": j["key"], "manifest_sha256": resume.digest(manifest), "reply": reply,
             "seconds": 10, **resume.evaluate(j, reply)} for j in manifest["jobs"][:12]]
    (original / "manifest.json").write_text(json.dumps(manifest))
    (original / "responses.jsonl").write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (original / "lms-loaded.json").write_text(json.dumps([model]))
    monkeypatch.setattr(resume, "head", lambda: "reviewed")
    monkeypatch.setattr(resume.subprocess, "check_output",
                        lambda cmd, **kw: "" if cmd[0] == "git" else json.dumps([model]))
    return resume, original, tmp_path / "continuation", manifest, rows, reply


def test_only_missing_calls_and_original_bytes_stay_intact(seeded, monkeypatch):
    resume, original, output, manifest, rows, reply = seeded
    before = (original / "responses.jsonl").read_bytes()
    calls = []
    def request(req, **kw):
        calls.append(json.loads(req.data))
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr(resume.urllib.request, "urlopen", request)
    resume.continue_run(original, output)
    assert [c["messages"][-1]["content"] for c in calls] == list(map(str, range(12, 24)))
    assert all(c["max_tokens"] == 65536 for c in calls)
    assert (original / "responses.jsonl").read_bytes() == before
    assert (original / "manifest.json").read_bytes() == (output / "manifest.json").read_bytes()
    assert json.loads((output / "report.json").read_text())["status"] == "FAIL"
    assert json.loads((output / "continuation.json").read_text())["remaining_admission_seconds"] == 4680
    with pytest.raises(FileExistsError):
        resume.continue_run(original, output)
    assert len(calls) == 12


@pytest.mark.parametrize("defect", ["duplicate", "stale", "reorder", "grade", "incomplete", "tail"])
def test_invalid_saved_prefix_stops_before_generation(seeded, monkeypatch, defect):
    resume, original, output, manifest, rows, reply = seeded
    if defect == "duplicate": rows[1] = rows[0]
    if defect == "stale": rows[0]["manifest_sha256"] = "stale"
    if defect == "reorder": rows[0], rows[1] = rows[1], rows[0]
    if defect == "grade": rows[0]["faithful"] = False
    if defect == "incomplete": rows[0]["complete"] = False
    payload = ''.join(json.dumps(r) + '\n' for r in rows)
    if defect == "tail": payload += '{'
    (original / "responses.jsonl").write_text(payload)
    monkeypatch.setattr(resume.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("model called"))
    with pytest.raises(ValueError): resume.continue_run(original, output)
    assert not output.exists()


def test_request_errors_are_saved_once_without_retry(seeded, monkeypatch):
    resume, original, output, manifest, rows, reply = seeded
    calls = []
    def fail(*a, **kw):
        calls.append(1)
        raise TimeoutError("offline fixture")
    monkeypatch.setattr(resume.urllib.request, "urlopen", fail)
    resume.continue_run(original, output)
    assert len(calls) == 12
    result = json.loads((output / "report.json").read_text())
    assert result["status"] == "INCOMPLETE"
    assert sum(c["errors"] for c in result["counts"].values()) == 12


@pytest.mark.parametrize("change", ["model", "code"])
def test_changed_model_or_scoring_code_refuses(seeded, monkeypatch, change):
    resume, original, output, manifest, rows, reply = seeded
    def check(cmd, **kw):
        if cmd[0] == "git": return "src/simtest/post/fidelity.py\n" if change == "code" else ""
        return '[]'
    monkeypatch.setattr(resume.subprocess, "check_output", check)
    monkeypatch.setattr(resume.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("model called"))
    with pytest.raises(ValueError): resume.continue_run(original, output)
    assert not output.exists()
