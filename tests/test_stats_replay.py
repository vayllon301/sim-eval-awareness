"""`simtest plan`: replaying measured durations through the run's slots (src/simtest/stats/replay.py)."""

from __future__ import annotations

import json
import random

import pytest

from simtest.stats import replay as R

ARMS = [("C0", "none"), ("C3", "s")]


def test_queue_mode_packs_every_condition_into_the_same_slots():
    # 2 slots, 2 scenarios, 1 epoch: C0 episodes take 1 min, C3 10 min. One queue: the two C3
    # run side by side, and the C0 ones fill the other slot. 10 + at most 1 minute.
    minutes = {"C0": [1.0], "C3": [10.0]}
    end = R.replay_once(ARMS, ["a", "b"], 1, 2, minutes, random.Random(0), mode="queue")
    assert 10.0 <= end <= 12.0
    # One condition after another: C0 (1 min) then C3 (10 min).
    assert R.replay_once(ARMS, ["a", "b"], 1, 2, minutes, random.Random(0), mode="sequential") == 11.0


def test_sequential_reruns_run_alone_after_every_condition(monkeypatch):
    minutes = {"C0": [10.0], "C3": [10.0]}

    def draws():
        calls = [0]

        def draw(rng, mins, fail_rate):   # the very first attempt fails at half its duration
            calls[0] += 1
            return (5.0, True) if calls[0] == 1 else (10.0, False)
        return draw

    monkeypatch.setattr(R, "_draw", draws())
    seq = R.replay_once(ARMS, ["a", "b"], 1, 2, minutes, random.Random(0), mode="sequential", fail_rate=0.5, retries=1)
    assert seq == 30.0   # C0 (10), C3 (10), then C0's re-run alone (10)
    monkeypatch.setattr(R, "_draw", draws())
    q = R.replay_once(ARMS, ["a", "b"], 1, 2, minutes, random.Random(0), mode="queue", fail_rate=0.5, retries=1)
    assert q == 25.0     # 0-5 fails, 0-10, 5-15, 10-20; the re-run, behind the others, 15-25


def test_replay_summary_and_units():
    r = R.replay(ARMS, ["a"], 2, 2, {"C0": [60.0], "C3": [60.0]}, draws=50, setup_hours=0.5, units_per_hour=10)
    assert r["episodes"] == 4 and r["mean_hours"] == pytest.approx(2.5) and r["mean_units"] == pytest.approx(25)
    assert r["slot_use"] == pytest.approx(1.0)


def test_unmeasured_conditions_borrow_and_scale():
    data = {"minutes": {"C0": [10.0], "C3": [100.0]}, "scale": {"C0": 1.5}, "assume": {"C2": "C3"}}
    assert R.condition_minutes(data, ["C0", "C2"]) == {"C0": [15.0], "C2": [100.0]}
    with pytest.raises(ValueError):
        R.condition_minutes(data, ["C1"])


def test_chunks_fit_the_limit_and_leave_room_for_the_longest_episode():
    minutes = {"C0": [60.0], "C3": [120.0]}
    plan = R.plan_chunks(ARMS, ["a", "b"], 4, 4, minutes, 5.0, setup_hours=0.5, draws=50)
    # One epoch is 4 episodes on 4 slots: 2 h. Two epochs: 4 h + 0.5 setup fits 5 h; three don't.
    assert plan["epochs_per_job"] == 2 and [c["epochs"] for c in plan["chunks"]] == ["1-2", "3-4"]
    assert plan["chunks"][0]["admit_for_hours"] == 2.5   # 5 - 0.5 setup - 2 h longest episode
    with pytest.raises(ValueError):
        R.plan_chunks(ARMS, ["a"], 2, 4, minutes, 2.4, setup_hours=0.5, draws=10)


def test_cli_plan_for_the_pilot(capsys):
    from simtest.cli import main

    main(["plan", "configs/coding/pilot.yaml", "--durations", "explore/colab_probe/durations-g4.json",
          "--draws", "200", "--compare", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert out["queue"]["episodes"] == 54
    assert out["queue"]["mean_hours"] < out["sequential"]["mean_hours"]
    main(["plan", "configs/coding/pilot.yaml", "--durations", "explore/colab_probe/durations-g4.json",
          "--draws", "100", "--time-limit", "24"])
    text = capsys.readouterr().out
    assert "Estimate only" in text and "--admit-for" in text


def test_a_rerun_starts_only_after_its_failed_attempt_ends(monkeypatch):
    """S3 (sol-loop round 1): two slots, an attempt failing at minute 100, another done at 1, a
    10-minute re-run. The re-run can't start before 100, so the run ends at 110, not 100."""
    seq = iter([(100.0, True), (1.0, False), (10.0, False)])
    monkeypatch.setattr(R, "_draw", lambda rng, mins, fail_rate: next(seq))
    end, left = R._schedule([("C3", "s", "a", 1), ("C0", "none", "a", 1)], 2, {"C0": [1.0], "C3": [1.0]},
                            random.Random(0), 0.5, 2)
    assert end == 110.0 and left == []
