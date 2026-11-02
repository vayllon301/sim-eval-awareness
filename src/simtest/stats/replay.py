"""How long a run takes on the GPU, by replaying measured episode durations through the run's
slots (`simtest plan`). Nothing runs: it only schedules.

One replay:
- every episode of the config (arms × scenarios × epochs) gets a duration drawn with
  replacement from its condition's measured minutes (times its scale, for example C0 measured at
  6 at once and run at 12). A condition never measured borrows another's (`assume`, C2 as C3);
- with probability `fail_rate`, an attempt fails part way (a uniform share of its duration) and
  is re-run from the start, up to `retries` times, with a fresh duration;
- `queue` mode is `simtest run` today (src/simtest/runqueue.py): one seeded, blocked order for
  every condition, `in_flight` slots in total, a re-run at the back of the same queue.
  `sequential` mode is the old one: one condition after another, and a condition's re-runs after
  every condition, alone (LOG.md, 02-11-2026).
- The session lasts the setup plus the time until the last episode ends. Episodes are assumed no
  faster when fewer run at once, so the tail is if anything overstated.

Many replays give the distribution of hours and units. `plan_chunks` splits a run into chunks of
epochs that fit a job's time limit (Hopper), with an admission deadline that leaves room for the
longest measured episode, so no running episode is cut at the wall. Cutting long episodes and
re-running them would favour short ones.
"""

from __future__ import annotations

import heapq
import json
import math
import random
from pathlib import Path

from ..runqueue import episode_order


def load_durations(path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data.get("minutes"), dict) or not data["minutes"]:
        raise ValueError(f"{path}: needs a 'minutes' mapping condition -> list of minutes")
    return data


def condition_minutes(data: dict, conditions) -> dict[str, list[float]]:
    out = {}
    assume, scale = data.get("assume") or {}, data.get("scale") or {}
    for c in conditions:
        src = c if c in data["minutes"] else assume.get(c)
        if src not in data["minutes"]:
            raise ValueError(f"no durations for {c} (and no 'assume' entry pointing to measured ones)")
        out[c] = [float(m) * float(scale.get(src, 1.0)) for m in data["minutes"][src]]
    return out


def _draw(rng, minutes, fail_rate):
    d = rng.choice(minutes)
    if fail_rate and rng.random() < fail_rate:
        return d * rng.random(), True
    return d, False


def _schedule(queue, slots, minutes, rng, fail_rate, retries, start=0.0, requeue=True):
    """List scheduling of `queue` (keys) on `slots`; returns (end time, failed keys left over).
    With `requeue`, a failed attempt goes to the back of the same queue while it has retries."""
    queue = list(queue)
    tries = {}
    free = [start] * slots
    heapq.heapify(free)
    end, left = start, []
    i = 0
    while i < len(queue):
        key = queue[i]
        i += 1
        t = heapq.heappop(free)
        d, failed = _draw(rng, minutes[key[0]], fail_rate)
        heapq.heappush(free, t + d)
        end = max(end, t + d)
        if failed:
            tries[key] = tries.get(key, 0) + 1
            if requeue and tries[key] <= retries:
                queue.append(key)
            else:
                left.append(key)
    return end, left


def replay_once(arms, scenarios, epochs, in_flight, minutes, rng, mode="queue", fail_rate=0.0, retries=2,
                order_seed=0) -> float:
    """Minutes from the first episode's start to the last one's end."""
    if mode == "queue":
        order = episode_order(arms, scenarios, epochs, order_seed)
        end, _ = _schedule(order, in_flight, minutes, rng, fail_rate, retries)
        return end
    if mode != "sequential":
        raise ValueError(f"mode must be queue or sequential, got {mode!r}")
    t, failed = 0.0, []
    for arm in arms:
        keys = [(arm[0], arm[1], sc, e) for e in range(1, epochs + 1) for sc in scenarios]
        t, left = _schedule(keys, in_flight, minutes, rng, fail_rate, retries, start=t, requeue=False)
        failed.append(left)
    for left in failed:   # each condition's re-runs, after every condition, alone
        for _ in range(retries):
            if not left:
                break
            t, left = _schedule(left, in_flight, minutes, rng, fail_rate, retries, start=t, requeue=False)
    return t


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))]


def replay(arms, scenarios, epochs, in_flight, minutes, *, draws=5000, seed=0, setup_hours=0.25,
           units_per_hour=10.05, **kw) -> dict:
    rng = random.Random(seed)
    hours = [setup_hours + replay_once(arms, scenarios, epochs, in_flight, minutes, rng,
                                       order_seed=rng.randrange(2**31), **kw) / 60 for _ in range(draws)]
    mean = sum(hours) / len(hours)
    work = sum(sum(minutes[a[0]]) / len(minutes[a[0]]) for a in arms) * len(scenarios) * epochs / 60
    return {"episodes": len(arms) * len(scenarios) * epochs, "draws": draws, "mean_hours": mean,
            "p10_hours": _pct(hours, 0.10), "p50_hours": _pct(hours, 0.50), "p90_hours": _pct(hours, 0.90),
            "p95_hours": _pct(hours, 0.95), "mean_units": mean * units_per_hour,
            "p10_units": _pct(hours, 0.10) * units_per_hour, "p90_units": _pct(hours, 0.90) * units_per_hour,
            "slot_use": work / ((mean - setup_hours) * in_flight) if mean > setup_hours else None,
            "setup_hours": setup_hours, "units_per_hour": units_per_hour}


def plan_chunks(arms, scenarios, epochs, in_flight, minutes, time_limit_hours, *, setup_hours=0.25, draws=2000,
                seed=0, **kw) -> dict:
    """The most epochs per job whose p95 session fits the limit, and each job's admission deadline:
    the limit minus the setup and the longest measured episode."""
    longest = max(max(minutes[a[0]]) for a in arms) / 60
    admit_for = time_limit_hours - setup_hours - longest
    if admit_for <= 0:
        raise ValueError(f"a job of {time_limit_hours} h can't hold the setup ({setup_hours} h) and the longest "
                         f"measured episode ({longest:.1f} h)")
    per = 0
    for k in range(epochs, 0, -1):
        r = replay(arms, scenarios, k, in_flight, minutes, draws=draws, seed=seed, setup_hours=setup_hours, **kw)
        if r["p95_hours"] <= time_limit_hours:
            per, fit = k, r
            break
    if per == 0:
        # Not even one epoch fits: chunks of one epoch, each cut by its admission deadline (the rest
        # is carried to the next job).
        per = 1
        fit = replay(arms, scenarios, 1, in_flight, minutes, draws=draws, seed=seed, setup_hours=setup_hours, **kw)
    chunks = []
    for a in range(1, epochs + 1, per):
        b = min(epochs, a + per - 1)
        chunks.append({"epochs": f"{a}-{b}" if b > a else str(a), "admit_for_hours": round(admit_for, 2)})
    return {"time_limit_hours": time_limit_hours, "epochs_per_job": per, "chunks": chunks,
            "longest_episode_hours": round(longest, 2), "p95_hours_per_job": round(fit["p95_hours"], 2),
            "one_epoch_fits": fit["p95_hours"] <= time_limit_hours}


def config_arms(cfg) -> list[tuple]:
    from ..task import NO_SIMULATOR

    return [(c, NO_SIMULATOR) if c == "C0" else (c, s)
            for c in cfg["conditions"] for s in ([None] if c == "C0" else cfg["simulators_for"][c])]


def format_replay(name, r, mode) -> str:
    return (f"{name} ({mode}): {r['episodes']} episodes, {r['draws']:,} replays. Hours {r['mean_hours']:.1f} "
            f"(p10-p90 {r['p10_hours']:.1f}-{r['p90_hours']:.1f}), units {r['mean_units']:.0f} "
            f"({r['p10_units']:.0f}-{r['p90_units']:.0f}) at {r['units_per_hour']} an hour, setup "
            f"{r['setup_hours']} h included; slots busy {100 * (r['slot_use'] or 0):.0f}% of the episode time")


def add_parser(sub) -> None:
    p = sub.add_parser("plan", help="estimate a run's GPU hours and units by replaying measured durations")
    p.add_argument("config", type=Path)
    p.add_argument("--durations", type=Path, required=True, help="e.g. explore/colab_probe/durations-g4.json")
    p.add_argument("--fail-rate", type=float, default=0.0, help="share of attempts that fail part way")
    p.add_argument("--draws", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261102)
    p.add_argument("--setup-hours", type=float, default=None, help="default: the durations file's")
    p.add_argument("--units-per-hour", type=float, default=None, help="default: the durations file's")
    p.add_argument("--compare", action="store_true", help="also replay the old order (one condition after another)")
    p.add_argument("--time-limit", type=float, default=None, help="a job's time limit in hours: split into chunks")
    p.add_argument("--json", action="store_true")


def main(args) -> None:
    from ..coding.run import load_config

    cfg = load_config(args.config)
    data = load_durations(args.durations)
    minutes = condition_minutes(data, cfg["conditions"])
    arms = config_arms(cfg)
    setup = args.setup_hours if args.setup_hours is not None else data.get("setup_hours", 0.25)
    uph = args.units_per_hour if args.units_per_hour is not None else data.get("units_per_hour", 10.05)
    kw = dict(fail_rate=args.fail_rate, retries=cfg["run"]["retry_attempts"])
    common = dict(draws=args.draws, seed=args.seed, setup_hours=setup, units_per_hour=uph, **kw)
    out = {"config": cfg["name"], "in_flight": cfg["run"]["max_connections"], "fail_rate": args.fail_rate,
           "queue": replay(arms, cfg["scenarios"], cfg["epochs"], cfg["run"]["max_connections"], minutes,
                           mode="queue", **common)}
    if args.compare:
        out["sequential"] = replay(arms, cfg["scenarios"], cfg["epochs"], cfg["run"]["max_connections"], minutes,
                                   mode="sequential", **common)
    if args.time_limit is not None:
        out.update(plan_chunks(arms, cfg["scenarios"], cfg["epochs"], cfg["run"]["max_connections"], minutes,
                               args.time_limit, setup_hours=setup, draws=min(args.draws, 2000), seed=args.seed, **kw))
    if args.json:
        print(json.dumps(out, indent=1))
        return
    print(f"Estimate only (measured durations replayed, no run): {args.durations}, attempt failure rate "
          f"{args.fail_rate:g}, {out['in_flight']} in flight.")
    print(format_replay(cfg["name"], out["queue"], "one queue"))
    if "sequential" in out:
        print(format_replay(cfg["name"], out["sequential"], "one condition after another"))
    if "chunks" in out:
        print(f"Jobs of {args.time_limit} h: {out['epochs_per_job']} epoch(s) per job (p95 {out['p95_hours_per_job']} h"
              f"{'' if out['one_epoch_fits'] else ', NOT within the limit: the deadline carries the rest'}), "
              f"start no episode after {out['chunks'][0]['admit_for_hours']} h (the longest measured episode is "
              f"{out['longest_episode_hours']} h).")
        for i, c in enumerate(out["chunks"], 1):
            print(f"  job {i}: simtest run {args.config} --epochs {c['epochs']} --admit-for {c['admit_for_hours']}")
