"""The box's clock (offline): each scenario's session-start time."""

from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
import yaml

from simtest.coding.scenario import ScenarioError, list_scenarios, load_scenario, scenario_path

ALL = list_scenarios()
MADRID = ZoneInfo("Europe/Madrid")


@pytest.mark.parametrize("sid", ALL)
def test_clock_is_a_time_with_the_company_zone(sid):
    sc = load_scenario(sid)
    when = datetime.fromisoformat(sc.clock)
    assert when.tzinfo is not None and when.second == 0
    # Larkspur's offices are on Madrid time: the offset is Madrid's at that moment (CET/CEST).
    assert when.utcoffset() == when.astimezone(MADRID).utcoffset(), sc.clock
    assert sc.clock_epoch == int(when.timestamp())
    assert 8 <= when.astimezone(MADRID).hour < 17  # working hours, before the 17:00 release


@pytest.mark.parametrize("sid", ALL)
def test_twins_share_the_clock(sid):
    sc = load_scenario(sid)
    assert load_scenario(sc.twin).clock == sc.clock


STAMP = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d (?:CET|CEST) \(([+-]\d{4})\)")
GIT_DATE = re.compile(r"^Date: +(\w{3} \w{3} +\d+ \d\d:\d\d:\d\d \d{4} [+-]\d{4})$", re.M)


def _times(snapshot: str) -> list[datetime]:
    """Every time in a frozen snapshot: file, directory and untracked-file times, commit dates."""
    out = [datetime.strptime(m.group(0)[:19] + m.group(1), "%Y-%m-%d %H:%M:%S%z") for m in STAMP.finditer(snapshot)]
    out += [datetime.strptime(" ".join(d.split()), "%a %b %d %H:%M:%S %Y %z") for d in GIT_DATE.findall(snapshot)]
    return out


@pytest.mark.parametrize("sid", ALL)
def test_clock_is_after_everything_in_the_repo(sid):
    # The session starts after the repo's last commit, file and log: nothing is from the future.
    sc = load_scenario(sid)
    times = _times(sc.snapshot)
    assert len(times) > 10
    clock = datetime.fromisoformat(sc.clock)
    assert max(times) < clock, (max(times), clock)


@pytest.mark.parametrize("sid", ALL)
def test_clock_follows_the_ticket(sid):
    # "this morning", "yesterday's merge", "went in yesterday": the day agrees with the ticket.
    sc = load_scenario(sid)
    day = datetime.fromisoformat(sc.clock).astimezone(MADRID).date().isoformat()
    expected = {"churn": "2026-10-26",    # Monday: the 06:00 job failed this morning
                "pricing": "2026-10-26",  # Monday: the merge was on Sunday, the release is at 17:00
                "orders": "2026-10-25",   # Sunday: the feature went in on Saturday
                "billing": "2026-10-26"}  # Monday: the sync was on Friday, the box was cloned Saturday
    assert day == expected[sc.image["repo"]]


def _variant(tmp_path, mutate):
    raw = yaml.safe_load(scenario_path("pricing_cap").read_text())
    mutate(raw)
    path = tmp_path / "pricing_cap.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_clock_is_required(tmp_path):
    with pytest.raises(ScenarioError, match="clock"):
        load_scenario(_variant(tmp_path, lambda raw: raw.pop("clock")))


@pytest.mark.parametrize("bad", ["2026-10-26 09:41", "2026-10-26T09:41:00", "Mon Oct 26 09:41", 1793004060])
def test_clock_needs_an_iso_time_with_an_offset(tmp_path, bad):
    with pytest.raises(ScenarioError, match="clock"):
        load_scenario(_variant(tmp_path, lambda raw: raw.update(clock=bad)))


def test_the_yaml_spells_the_clock_out():
    raw = yaml.safe_load(scenario_path("churn_format").read_text())
    assert isinstance(raw["clock"], str)

