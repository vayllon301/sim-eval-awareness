"""The machine facts in the snapshots, and the session start date added at run time: in the
facts block, the same for C2 and C3, never in C1, never in the frozen file."""

from __future__ import annotations

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
