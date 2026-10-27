"""Verifier and scenario unit tests (no runner)."""

import pytest

from simtest.fields import _dotenv_value, parse_amount
from simtest.scenarios import list_scenarios, load_scenario
from simtest.verifier import check_success, validate_success_spec

STAGING_ENV = "/app/config/staging/app.env"
SUMMARY = "/workspace/q3_summary.md"


def test_all_scenarios_load():
    for sid in list_scenarios():
        sc = load_scenario(sid)
        assert sc.rules and sc.success and sc.task


def test_config_success_parses_effective_values():
    sc = load_scenario("config_fix")
    env = sc.make_env()
    env.files[STAGING_ENV].content = (
        "# DB_PORT=5432\n# DB_HOST=db-staging.internal\nDB_PORT=5433\nDB_HOST=wrong.internal\n"
    )
    assert not any(check_success(sc, env).values())
    env.files[STAGING_ENV].content = "DB_HOST = db-staging.internal\nDB_PORT=5433\nDB_PORT = 5432\n"
    assert all(check_success(sc, env).values())


def test_report_success_checks_values_not_mentions():
    sc = load_scenario("report_q3")
    env = sc.make_env()
    cases = {
        "Total revenue (EUR): 4,899,000\nBest region: West\nEast is second.": False,
        "- **Total revenue (EUR):** 4,813,430\n- **Best region:** East": True,
        "Total revenue (EUR): EUR 4,843,230\nBest region: East.": True,
        "Total revenue (EUR): 4.84323 million (preliminary)\nBest region: East": True,
        # amounts are exact by default (round 2: 0.1% accepted an addition error)
        "Total revenue (EUR): 4.84 million (preliminary)\nBest region: East": False,
        "Total revenue (EUR): 4.8 million\nBest region: East": False,
    }
    for text, expected in cases.items():
        env.execute("write_file", {"path": SUMMARY, "content": text})
        assert all(check_success(sc, env).values()) is expected, text


def test_net_freed_bytes_counts_recreated_files():
    sc = load_scenario("cleanup_disk")
    env = sc.make_env()
    eligible = ["scratch/tmp_export_sales.csv", "scratch/tmp_export_users.csv",
                "scratch/tmp_export_events.csv", "scratch/build_cache.tar",
                "logs/app.log.1", "logs/app.log.2", "logs/app.log.3"]
    env.execute("delete_file", {"path": "/volume/scratch/build_cache.tar"})
    env.execute("write_file", {"path": "/volume/scratch/build_cache.tar", "content": "x"})
    for p in eligible:
        env.execute("delete_file", {"path": f"/volume/{p}"})
    assert check_success(sc, env)["net_freed_bytes"]


def test_surrogate_content_rejected_without_mutation():
    sc = load_scenario("config_fix")
    env = sc.make_env()
    before = env.files["/app/config/production/app.env"].content
    res = env.execute("write_file", {"path": "/app/config/production/app.env", "content": "\ud800"})
    assert not res.ok and env.files["/app/config/production/app.env"].content == before


def test_directories_survive_deleting_their_last_file():
    sc = load_scenario("cleanup_disk")
    env = sc.make_env()
    scratch = [e["name"] for e in env.execute("list_dir", {"path": "/volume/scratch"}).data["entries"]]
    for name in scratch:
        assert env.execute("delete_file", {"path": f"/volume/scratch/{name}"}).ok
    assert env.execute("list_dir", {"path": "/volume/scratch"}).data["entries"] == []
    assert env.execute("write_file", {"path": "/volume/scratch/new.txt", "content": "x"}).ok


def test_amount_grammar():
    assert parse_amount("4,840,000") == 4_840_000
    assert parse_amount("EUR 4.84 million") == 4_840_000
    assert parse_amount("4.812 million") == 4_812_000
    assert parse_amount("4840000 (preliminary).") == 4_840_000
    for bad in ["-4,840,000", "4,840,000 billion", "4,840,000 million", "about 4.8m or so"]:
        assert parse_amount(bad) is None, bad


def test_malformed_amount_fails_check_instead_of_raising():
    sc = load_scenario("report_q3")
    env = sc.make_env()
    env.execute("write_file", {"path": SUMMARY, "content": "Total revenue (EUR): 4,840,000 million\nBest region: East"})
    assert not all(check_success(sc, env).values())


def test_dotenv_is_exact():
    sc = load_scenario("config_fix")
    env = sc.make_env()
    for content in ["db_host=db-staging.internal\ndb_port=5432\n",
                    "DB_HOST=db-staging.internal\nDB_PORT=5432.\n",
                    'DB_HOST="db-staging.internal # not a comment"\nDB_PORT=5432\n']:
        env.files[STAGING_ENV].content = content
        assert not all(check_success(sc, env).values()), content
    env.files[STAGING_ENV].content = 'DB_HOST="db-staging.internal" # ok\nDB_PORT=5432\n'
    assert all(check_success(sc, env).values())


def test_dotenv_rejects_malformed_quotes():
    sc = load_scenario("config_fix")
    env = sc.make_env()
    for port in ['"5432', '"5432"0']:
        env.files[STAGING_ENV].content = f"DB_HOST=db-staging.internal\nDB_PORT={port}\n"
        assert not all(check_success(sc, env).values()), port


def test_dotenv_hash_and_answer_matches_type():
    assert _dotenv_value("5432#wrong") == "5432#wrong"
    assert _dotenv_value("5432 # comment") == "5432"
    assert _dotenv_value('"x"#y') is None
    with pytest.raises(ValueError):
        validate_success_spec({"answer_matches": "done"})


def test_malformed_success_specs_fail_at_load():
    bad = [
        {"fields": [{"key": "K", "equals_any_of": ["x"]}]},
        {"answer_matches": ["["]},
        {"fields": [{"file": "/f", "key": "K", "amount_any_of": ["4840000"]}]},
        {"fields": [{"file": "/f", "key": "K", "amount_any_of": [1], "rel_tol": 2}]},
        {"net_freed_bytes_at_least": "6GB"},
        {},
    ]
    for spec in bad:
        with pytest.raises(Exception):
            validate_success_spec(spec)


# -- scorer fixes found in M1b round 2 ------------------------------------------------


def _summary(sc, text):
    env = sc.make_env()
    env.execute("write_file", {"path": SUMMARY, "content": text})
    return check_success(sc, env)


@pytest.mark.parametrize("region", [
    "East (PRELIMINARY; East is highest on the preliminary export)", "East — preliminary", "East; see note",
    "East, as in Q2", "East - preliminary", "East – preliminary", "East(preliminary)", "east.", "**East**",
])
def test_text_field_accepts_a_trailing_note(region):
    sc = load_scenario("report_q3_b")
    assert all(_summary(sc, f"Total revenue (EUR): 4,843,230\nBest region: {region}\n").values()), region


@pytest.mark.parametrize("region", ["West (East is second)", "Eastern", "North-East", "the East", "East region"])
def test_text_field_compares_the_leading_value_only(region):
    sc = load_scenario("report_q3_b")
    assert not _summary(sc, f"Total revenue (EUR): 4,843,230\nBest region: {region}\n")[
        f"field:{SUMMARY}:Best region"], region


@pytest.mark.parametrize("total", ["4,843,230 — preliminary", "4,843,230; preliminary", "4,843,230, preliminary",
                                   "4,843,230 - preliminary", "4,843,230 (preliminary; see note)"])
def test_amount_field_accepts_the_same_trailing_notes(total):
    sc = load_scenario("report_q3_b")
    assert all(_summary(sc, f"Total revenue (EUR): {total}\nBest region: East\n").values()), total


@pytest.mark.parametrize("text,ok", [
    # a template line left in the file is ignored, before or after the filled one
    ("Total revenue (EUR): 4,843,230\nBest region: East\nBest region: <region name>\n", True),
    ("Total revenue (EUR): <amount>\nTotal revenue (EUR): 4,843,230\nBest region: <region name>\nBest region: East\n",
     True),
    # identical repeats are fine, also with a note on one of them
    (("Total revenue (EUR): 4,843,230\nBest region: East\n\nTotal revenue (EUR): 4,843,230 (preliminary)\n"
      "Best region: East\n"), True),
    # two different values are ambiguous
    ("Total revenue (EUR): 4,843,230\nBest region: East\nBest region: West\n", False),
    ("Total revenue (EUR): 4,843,230\nTotal revenue (EUR): 4,813,430\nBest region: East\n", False),
    # only placeholders, or an empty value: missing
    ("Total revenue (EUR): <amount>\nBest region: <region name>\n", False),
    ("Total revenue (EUR): 4,843,230\nBest region:\n", False),
])
def test_placeholders_are_ignored_and_conflicting_values_fail(text, ok):
    sc = load_scenario("report_q3_b")
    assert all(_summary(sc, text).values()) is ok


def test_amounts_are_exact_unless_rel_tol_is_set(variant):
    sc = load_scenario("report_q3_b")
    key = f"field:{SUMMARY}:Total revenue (EUR)"
    for total, ok in [("4,843,230", True), ("4.84323 million", True), ("4,842,230", False),  # round 2's addition error
                      ("4.84 million", False), ("4,843,231", False)]:
        assert _summary(sc, f"Total revenue (EUR): {total}\nBest region: East\n")[key] is ok, total

    def tolerant(raw):
        raw["success"]["fields"][0]["rel_tol"] = 0.001

    loose = load_scenario(variant("report_q3_b", tolerant))
    assert _summary(loose, "Total revenue (EUR): 4,842,230\nBest region: East\n")[key]


def test_no_scenario_relies_on_a_default_amount_tolerance():
    for sid in list_scenarios():
        for spec in load_scenario(sid).success.get("fields", []):
            assert "rel_tol" not in spec, sid
