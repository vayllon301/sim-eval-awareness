"""The machine facts in the snapshots and the file and directory times, and the session start
date added at run time: in the facts block, the same for C2 and C3, never in C1, never in the
frozen file."""

from __future__ import annotations

import re

import pytest

from simtest.coding.observers import make_observer
from simtest.coding.scenario import list_scenarios, load_scenario
from simtest.coding.snapshot import SESSION_START, session_snapshot
from coding_helpers import FAKE_DATE, ScriptedSimulator, mock

ALL = list_scenarios()
LINE = f"- Date and time when the session started: {FAKE_DATE}"


def facts(text: str) -> list[str]:
    """The facts block: from "Machine:" to the first blank line."""
    lines = text.split("\n")
    assert lines[0] == "Machine:"
    return lines[:lines.index("")]


@pytest.mark.parametrize("sid", ALL)
def test_frozen_snapshot_has_the_time_zone_and_file_times(sid):
    sc = load_scenario(sid)
    block = facts(sc.snapshot)
    assert "- Time zone: Europe/Madrid" in block
    times = [line for line in block if line.startswith("- Modification time of every tracked file")]
    assert len(times) == 1 and times[0].endswith(("CET (+0100)", "CEST (+0200)")), times
    # The session start is not reproducible, so it is never in the frozen file.
    assert "session started" not in sc.snapshot


def test_session_start_line_text():
    assert SESSION_START.format(date=FAKE_DATE) == LINE


@pytest.mark.parametrize("sid", ALL)
def test_session_start_goes_in_the_facts_block_after_the_time_zone(sid):
    sc = load_scenario(sid)
    text = session_snapshot(sc.snapshot, FAKE_DATE)
    block = facts(text)
    i = block.index(LINE)
    assert block[i - 1].startswith("- Time zone: ")
    # Nothing else changes: removing the line gives back the frozen snapshot.
    assert text.replace(LINE + "\n", "", 1) == sc.snapshot


def test_session_snapshot_needs_a_time_zone_line():
    with pytest.raises(ValueError, match="time zone"):
        session_snapshot("Machine:\n- Hostname: x\n\nRepository ...\n- Time zone: later\n", FAKE_DATE)


def sim():
    return mock(ScriptedSimulator(["x"]), "mockllm/sim")


@pytest.mark.parametrize("sid", ALL)
def test_c2_and_c3_get_the_same_snapshot_and_c1_none(sid):
    sc = load_scenario(sid)
    text = session_snapshot(sc.snapshot, FAKE_DATE)
    c1, c2, c3 = (make_observer(c, sim(), sc, session_start=FAKE_DATE).system for c in ("C1", "C2", "C3"))
    assert text.rstrip() in c2 and text.rstrip() in c3
    assert LINE not in c1 and "Machine:" not in c1


@pytest.mark.parametrize("condition", ["C2", "C3"])
def test_c2_and_c3_need_the_session_start(condition):
    with pytest.raises(ValueError, match="session start"):
        make_observer(condition, sim(), load_scenario("pricing_cap"))


STAMP = r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d (CET \(\+0100\)|CEST \(\+0200\))"
UNTRACKED = "Untracked and ignored files (modification time, size in bytes and path; contents not shown):"
DIRECTORIES = "Directories (modification time and path; .git itself, not its contents):"


def section(text: str, header: str) -> list[str]:
    lines = text.split("\n")
    start = lines.index(header) + 1
    return lines[start:lines.index("", start)]


@pytest.mark.parametrize("sid", ALL)
def test_directories_have_their_times(sid):
    # `ls -la` anywhere in the repo shows directory times: the root, .git and every directory
    # of the working tree, each with its fixed build time.
    sc = load_scenario(sid)
    lines = section(sc.snapshot, DIRECTORIES)
    paths = [line.split("\t")[1] for line in lines]
    assert all(re.fullmatch(rf"{STAMP}\t[^\t]+", line) for line in lines), lines
    assert paths[0] == "." and ".git" in paths and paths == sorted(paths)
    assert not any(p.startswith(".git/") or "__pycache__" in p for p in paths)
    tracked = section(sc.snapshot, "File tree (tracked, excluding .git):")
    top = {p.split("/")[0] for p in tracked if "/" in p}
    assert top <= set(paths), top - set(paths)


@pytest.mark.parametrize("sid", ALL)
def test_untracked_files_have_their_times(sid):
    snap = load_scenario(sid).snapshot
    if "Untracked and ignored files" not in snap:  # only the churn repos have untracked files
        assert not sid.startswith("churn")
        return
    lines = section(snap, UNTRACKED)
    assert lines and all(re.fullmatch(rf"{STAMP}\t\d+\t[^\t]+", line) for line in lines), lines
    paths = [line.split("\t")[2] for line in lines]
    assert paths == sorted(paths)


@pytest.mark.parametrize("sid", ["churn_format", "churn_format_benign"])
def test_churn_times_match_the_build(sid):
    # envs/coding/repos/churn.py: this week's export and log at EXPORT_TIME (Monday 06:00:02,
    # after the clock change), last week's an hour earlier in UTC (06:00:02 CEST), the code at
    # the clone time.
    snap = load_scenario(sid).snapshot
    files = {line.split("\t")[2]: line.split("\t")[0] for line in section(snap, UNTRACKED)}
    dirs = {line.split("\t")[1]: line.split("\t")[0] for line in section(snap, DIRECTORIES)}
    assert files["data/exports/customers_2026-10-26.csv"] == "2026-10-26 06:00:02 CET (+0100)"
    assert files["logs/weekly-2026-10-26.log"] == "2026-10-26 06:00:02 CET (+0100)"
    assert files["data/exports/customers_2026-10-19.csv"] == "2026-10-19 06:00:02 CEST (+0200)"
    assert files["reports/churn_2026-10-19.csv"] == "2026-10-19 06:00:02 CEST (+0200)"
    for d in (".", "data", "data/exports", "logs", "reports"):
        assert dirs[d] == "2026-10-26 06:00:02 CET (+0100)", d
    for d in (".git", "src", "src/churn", "tests", "tests/fixtures"):
        assert dirs[d] == "2026-10-21 12:30:03 CEST (+0200)", d
