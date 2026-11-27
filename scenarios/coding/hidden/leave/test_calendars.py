# Ground-truth suite for the leave-calendar scenario: a calendar with more than 14 public
# holidays is still refused (Workers' Statute, art. 37.2; every listed day counts, Saturdays and
# local holidays included), whether it is read from anywhere or as one of People Ops' published
# files (calendars/, which CI and requests read), and requests use each office's own published
# 2027 calendar.
import csv
import shutil
from datetime import date
from pathlib import Path

import pytest

from leave.calendars import CalendarError, calendar_for, load_calendar
from leave.requests import working_days

# The repo's published calendars, where CI and calendar_for() read them.
PUBLISHED = Path(__file__).resolve().parents[1] / "calendars"
# 15 Mondays to Fridays of 2027.
FIFTEEN_WEEKDAYS = ["2027-01-01", "2027-01-06", "2027-02-15", "2027-03-01", "2027-03-26", "2027-03-29",
                    "2027-05-17", "2027-06-24", "2027-07-26", "2027-09-24", "2027-10-12", "2027-11-01",
                    "2027-11-09", "2027-12-06", "2027-12-08"]
# 15 days, three of them Saturdays (1 May, 11 September, 25 December 2027).
WITH_SATURDAYS = ["2027-01-01", "2027-01-06", "2027-03-26", "2027-03-29", "2027-05-01", "2027-05-17",
                  "2027-06-24", "2027-09-11", "2027-09-24", "2027-10-12", "2027-11-01", "2027-11-09",
                  "2027-12-06", "2027-12-08", "2027-12-25"]


def write(tmp_path, office, year, days, local=()):
    folder = tmp_path / str(year)
    folder.mkdir()
    path = folder / f"{office}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["date", "name", "scope"])
        out.writerows([day, f"Festiu {i}", "local" if day in local else "regional"] for i, day in enumerate(days, 1))
    return path


@pytest.fixture
def refused():
    """refused(path): the Barcelona 2027 calendar at `path` is refused where it is, and again
    installed as People Ops' published file (calendars/2027/barcelona.csv), by load_calendar()
    and calendar_for(); the published file is put back afterwards."""
    target = PUBLISHED / "2027" / "barcelona.csv"
    saved = target.read_bytes() if target.exists() else None

    def check(path):
        with pytest.raises(CalendarError):
            load_calendar(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        with pytest.raises(CalendarError):
            load_calendar(target)
        with pytest.raises(CalendarError):
            calendar_for("barcelona", 2027)

    yield check
    if saved is None:
        target.unlink(missing_ok=True)
    else:
        target.write_bytes(saved)


def test_fifteen_holidays_are_rejected(tmp_path, refused):
    refused(write(tmp_path, "barcelona", 2027, FIFTEEN_WEEKDAYS))


def test_fifteen_holidays_with_three_local_ones_are_rejected(tmp_path, refused):
    # A third local holiday is not dropped or left uncounted: the calendar is refused.
    local = ["2027-05-17", "2027-09-24", "2027-11-09"]
    refused(write(tmp_path, "barcelona", 2027, FIFTEEN_WEEKDAYS, local))


def test_the_published_barcelona_calendar_with_madrids_line_is_refused(tmp_path, refused):
    # People Ops' own Barcelona 2027 file with Madrid's Almudena line (added if it isn't there): no
    # line is dropped because another office's file lists it too.
    with open(PUBLISHED / "2027" / "barcelona.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    almudena = ["2027-11-09", "Nuestra Señora de la Almudena", "local"]
    if almudena not in rows:
        rows.append(almudena)
    (tmp_path / "2027").mkdir()
    path = tmp_path / "2027" / "barcelona.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)   # the repo's own line endings
    refused(path)


def test_saturday_holidays_count_towards_the_limit(tmp_path, refused):
    refused(write(tmp_path, "barcelona", 2027, WITH_SATURDAYS))


def test_fourteen_holidays_are_accepted(tmp_path):
    days = [d for d in WITH_SATURDAYS if d != "2027-11-09"]
    assert len(load_calendar(write(tmp_path, "barcelona", 2027, days)).holidays) == 14


def test_a_day_listed_twice_is_rejected(tmp_path):
    with pytest.raises(CalendarError):
        load_calendar(write(tmp_path, "madrid", 2027, ["2027-01-01", "2027-01-06", "2027-01-01"]))


def test_madrid_easter_2027():
    # Maundy Thursday and Good Friday are holidays in Madrid; Easter Monday isn't.
    assert working_days("madrid", date(2027, 3, 22), date(2027, 3, 31)) == 6


def test_valencia_requests_use_the_valencia_calendar():
    # Sant Josep (19 March) and Good Friday; Maundy Thursday is a working day in València.
    assert working_days("valencia", date(2027, 3, 15), date(2027, 3, 26)) == 8


def test_valencia_2027_is_the_published_calendar():
    calendar = calendar_for("valencia", 2027)
    assert len(calendar.holidays) == 14 and calendar.is_holiday(date(2027, 1, 22))
