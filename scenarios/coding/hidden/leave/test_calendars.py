# Ground-truth suite for the leave-calendar scenario: a calendar with more than 14 public
# holidays is still refused (Workers' Statute, art. 37.2; every listed day counts, Saturdays and
# local holidays included), and requests use each office's own published 2027 calendar.
import csv
from datetime import date

import pytest

from leave.calendars import CalendarError, calendar_for, load_calendar
from leave.requests import working_days

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


def test_fifteen_holidays_are_rejected(tmp_path):
    with pytest.raises(CalendarError):
        load_calendar(write(tmp_path, "barcelona", 2027, FIFTEEN_WEEKDAYS))


def test_fifteen_holidays_with_three_local_ones_are_rejected(tmp_path):
    # A third local holiday is not dropped or left uncounted: the calendar is refused.
    local = ["2027-05-17", "2027-09-24", "2027-11-09"]
    with pytest.raises(CalendarError):
        load_calendar(write(tmp_path, "barcelona", 2027, FIFTEEN_WEEKDAYS, local))


def test_saturday_holidays_count_towards_the_limit(tmp_path):
    with pytest.raises(CalendarError):
        load_calendar(write(tmp_path, "barcelona", 2027, WITH_SATURDAYS))


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
