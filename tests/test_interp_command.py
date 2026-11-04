"""`simtest interp`: where its files go, and what it refuses. The pieces it calls have their own tests."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from pilot_helpers import DESIGN, full, verdict
from simtest.cli import main
from simtest.interp import command as C
from test_interp_validate import KAPPA_OK, logit, make, summary


def test_reader_files_live_beside_the_logs_not_in_them():
    """Inspect lists every JSON under a log directory as a log."""
    assert C.default_dir(Path("logs/n2-pilot"), "episodes") == Path("logs/read/n2-pilot/episodes")


def test_the_command_is_on_the_cli():
    with pytest.raises(SystemExit) as e:
        main(["interp", "--help"])
    assert e.value.code == 0


def report_args(tmp_path, **over):
    d = {"interp_cmd": "report", "log_dir": tmp_path / "n2-pilot", "readings": tmp_path / "r", "judge": None, "kappa": None,
         "phrases": None, "layer": None, "json": False}
    return Namespace(**{**d, **over})


def test_the_report_prints_from_its_files(tmp_path, monkeypatch, capsys):
    eps, res, read = make(signal=1.5)
    monkeypatch.setattr(C, "_run", lambda d: (eps, DESIGN))
    monkeypatch.setattr(C, "_judge", lambda j, k: (res, KAPPA_OK))
    (tmp_path / "r").mkdir()
    for u, r in read.items():
        (tmp_path / "r" / f"{u}.summary.json").write_text(json.dumps(r["summary"]))
        (tmp_path / "r" / f"{u}.logit.json").write_text(json.dumps(r["logit"]))
    C.main(report_args(tmp_path, judge=tmp_path / "j", kappa=tmp_path / "k"))
    out = capsys.readouterr().out
    assert "layer 42" in out and "logit:eval" in out and "DRAFT rule" in out and "chosen:" in out
    C.main(report_args(tmp_path, judge=tmp_path / "j", kappa=tmp_path / "k", json=True))
    assert json.loads(capsys.readouterr().out)["rule"]["status"] == "chosen"


def test_several_layers_ask_for_one(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_run", lambda d: ([], DESIGN))
    (tmp_path / "r").mkdir()
    s = summary("u", {"eval_awareness": {0: (1.0, 2.0)}})
    s["by_turn"]["eval_awareness@36"] = s["by_turn"]["eval_awareness@42"]
    (tmp_path / "r" / "u.summary.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="give --layer"):
        C.main(report_args(tmp_path))


def test_sentences_come_only_from_a_judge_that_passed(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "_run", lambda d: (full(), DESIGN))
    monkeypatch.setattr(C, "_judge", lambda j, k: ([], {"valid": False, "reason": "no pass recorded"}))
    args = Namespace(interp_cmd="phrases", log_dir=tmp_path / "n2-pilot", judge=tmp_path / "j", kappa=tmp_path / "k", episodes=None,
                     out=tmp_path / "p.json", draft=Path("explore/interp/data/logit_monitor_phrases.json"))
    with pytest.raises(SystemExit, match="NOT VALID"):
        C.main(args)
    assert not (tmp_path / "p.json").exists()


def test_phrases_write_a_file_the_reader_can_take(tmp_path, monkeypatch):
    eps = full()
    res = [verdict(e["judge_key"], positive=False) for e in eps] + [verdict(e["judge_key"], positive=False, label="training") for e in eps]
    monkeypatch.setattr(C, "_run", lambda d: (eps, DESIGN))
    monkeypatch.setattr(C, "_judge", lambda j, k: (res, KAPPA_OK))
    args = Namespace(interp_cmd="phrases", log_dir=tmp_path / "n2-pilot", judge=tmp_path / "j", kappa=tmp_path / "k", episodes=tmp_path,
                     out=tmp_path / "p.json", draft=Path("explore/interp/data/logit_monitor_phrases.json"))
    C.main(args)
    doc = json.loads((tmp_path / "p.json").read_text())
    assert set(doc["groups"]) == {"eval", "training", "control"} and doc["provenance"]


def test_logit_helper_is_in_the_shape_of_the_readers_file():
    assert set(logit("u", {0: 0.5})["logit_monitor"]) >= {"status", "path", "phrases", "by_turn"}
