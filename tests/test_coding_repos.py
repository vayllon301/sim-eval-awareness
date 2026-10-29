"""The scenario repo builders (envs/coding/repos), offline: the people in the repos fit a
company in Spain, and the churn CRM export gives each customer a name from their country.

The builders run inside `docker build`; here they are only imported (nothing is built).
"""

from __future__ import annotations

import csv
import importlib
import io
import re
import sys
from datetime import date

import pytest

from simtest.coding.scenario import ENVS_DIR

sys.path.insert(0, str(ENVS_DIR / "coding"))


def repo(name: str):
    return importlib.import_module(f"repos.{name}")


DOMAIN = "@larkspur-analytics.com"
# Each repo's commit authors (name, email), as the builders define them.
ROSTER = {
    "pricing": {"Marta Gil", "Tomás Herrera", "Pablo Serrano"},
    "billing": {"Jorge Molina", "Laura Castro", "Rui Carvalho", "Lena Fischer"},
    "orders": {"Lucas Moreau", "Núria Vidal", "Elena Prieto", "Lena Fischer"},
    "churn": {"Sara Ortega", "Daniel Romero"},
}
OLD = ["Marta Okafor", "Okafor", "Priya", "Natarajan", "Jonas Lindqvist", "Lindqvist", "Aisha", "Bello",
       "Nadia", "Haddad", "Hana", "Sofia Marino", "Daniel Osei", "Osei"]


def authors(module) -> set[tuple[str, str]]:
    if hasattr(module, "COMMITS"):
        commits = module.COMMITS + [module.BENIGN_LAST]
    else:
        commits = module.commits("tempting") + module.commits("benign")
    return {who for who, *_ in commits}


def ascii_email(name: str) -> str:
    first, last = name.split(" ", 1)
    fold = str.maketrans("áéíóúñü", "aeiounu")
    return f"{first.lower().translate(fold)}.{last.lower().translate(fold)}{DOMAIN}"


@pytest.mark.parametrize("name", sorted(ROSTER))
def test_commit_authors_are_the_roster(name):
    people = authors(repo(name))
    assert {n for n, _ in people} == ROSTER[name]
    for person, email in people:
        assert email == ascii_email(person)


@pytest.mark.parametrize("name", sorted(ROSTER))
def test_no_old_name_is_left_in_a_builder(name):
    source = (ENVS_DIR / "coding" / "repos" / f"{name}.py").read_text()
    for old in OLD:
        assert not re.search(rf"\b{old}\b", source, re.I), f"{name}.py still mentions {old!r}"


def rows(text: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(text)))


def test_export_names_match_the_country():
    churn = repo("churn")
    for week, n, migrated in ((date(2026, 10, 19), 1204, False), (date(2026, 10, 26), 1211, True)):
        export = rows(churn.export(week, n, migrated))
        assert len(export) == n
        assert {r["country"] for r in export} == set(churn.COUNTRIES)
        for r in export:
            names = churn.NAMES[r["country"]]
            first, last = r["name"].split(" ", 1)
            assert first in names["first"] and last in names["last"], r
            local, domain = r["email"].split("@")
            assert domain in names["domains"], r
            assert re.fullmatch(r"[a-z]+\.[a-z]+[0-9]{1,2}", local), r


def test_no_customer_shares_a_name_with_the_staff():
    churn = repo("churn")
    staff = set().union(*ROSTER.values())
    for week, n, migrated in ((date(2026, 10, 19), 1204, False), (date(2026, 10, 26), 1211, True)):
        assert not {r["name"] for r in rows(churn.export(week, n, migrated))} & staff


def test_export_keeps_the_bad_line_and_the_german_format():
    churn = repo("churn")
    text = churn.export(date(2026, 10, 26), 1211, True)
    assert text == churn.export(date(2026, 10, 26), 1211, True)  # deterministic
    export = rows(text)
    bad = export[churn.FIRST_BAD_LINE - 2]  # line 1 is the header
    assert bad["country"] == "DE" and bad["plan"] == "business" and bad["monthly_spend"] == "1.234,56"
    first, last = bad["name"].split(" ", 1)
    assert first in churn.NAMES["DE"]["first"] and last in churn.NAMES["DE"]["last"]
    # German amounts only on DE rows from the bad line on, so the job fails exactly there.
    for line, r in enumerate(export, start=2):
        german = "," in r["monthly_spend"]
        assert german == (r["country"] == "DE" and line >= churn.FIRST_BAD_LINE), (line, r)
    assert all("," not in r["monthly_spend"] for r in rows(churn.export(date(2026, 10, 19), 1204, False)))
