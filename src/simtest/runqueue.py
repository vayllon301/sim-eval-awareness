"""One queue for every episode of a run (coding family; LOG.md, 02-11-2026).

`simtest run` runs every condition's task at once, and each episode waits here for a slot
before it starts its machine. The queue decides two things:

- **How many run at once:** a fixed total in flight (the config's `max_connections`, 12 in the
  pilot), whatever the condition. No slot sits idle while one condition finishes, and every
  condition shares the same batch mix on the servers. The simulator's FP8 MoE scales its
  activations over the whole batch (docs/INTERP.md), so running conditions one after another
  would give each its own batch environment.
- **In what order:** a seeded, blocked random order (`episode_order`). Epoch by epoch, the
  scenarios come in a random order, and each (scenario, epoch) is a block holding one episode
  of every arm (condition and simulator) in a random order. So at any point of the run every
  arm has started the same number of episodes, give or take one, and the episodes of a
  scenario's pair start side by side. Time in the session is no longer confounded with the
  condition.

Re-runs. An episode that errors is re-run from the start, in the same run (Inspect's in-place
retry): it goes to the back of the queue, behind every episode not yet started, so it runs
mixed with the rest instead of alone at the end. The failed attempt is kept in full under the
run's log directory (`attempts/`, written by the episode solver) and reported by the analysis.

An episode can also end without starting (`NotStarted`): outside this session's epochs
(`epochs`), after the admission deadline (`admit_until`), or carried to the next session
(`carry_under`: when fewer than k episodes are left to wait or run, the rest are not started,
so a few re-runs don't hold the VM alone). It made no model call and holds no data. Inspect
logs it as an errored sample with a message starting with `NOT_STARTED`; the analysis skips
it, and the next `simtest run` of the same config starts it. Running episodes are never cut:
cutting the long ones and re-running them would favour short episodes.

Every admission, release and not-started is a line of `queue.jsonl` in the log directory,
with the arms running at that moment, so the batch mix per condition can be checked.
"""

from __future__ import annotations

import contextlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import anyio

NOT_STARTED = "simtest-queue: not started"
SETTLE_SECONDS = 10.0   # first admissions wait until arrivals stop this long (or all have arrived)

_CURRENT: "EpisodeQueue | None" = None


class NotStarted(RuntimeError):
    """The episode ended without starting (no model call, no machine)."""

    def __init__(self, reason: str):
        super().__init__(f"{NOT_STARTED}: {reason}")
        self.reason = reason


def is_not_started(message: str | None) -> bool:
    return bool(message) and NOT_STARTED in message


NEVER_RAN = "cancelled before start"   # a release outcome: the slot was granted, the episode never ran
ADMITTED_KEY = "CodingEpisode:attempt"   # the solver stores its admission here as soon as it has a slot


def queued_log(log) -> bool:
    """Whether the log was written under the run queue (`simtest run` records it in the metadata)."""
    return "queue" in (log.eval.metadata or {})


def never_started(error: str | None, store: dict | None, queued: bool) -> bool:
    """An errored sample that never got a slot: NotStarted, or, in a log written under the queue,
    any error (a cancellation while waiting, for example) before its admission was stored. It made
    no model call and is not an attempt; the next `simtest run` starts it."""
    if error is None:
        return False
    return is_not_started(error) or (queued and not (store or {}).get(ADMITTED_KEY))


def episode_order(arms, scenarios, epochs: int, seed: int) -> list[tuple]:
    """Every (condition, simulator, scenario, epoch) of the run, in a seeded blocked order:
    epoch by epoch, scenarios shuffled, and in each (scenario, epoch) block one episode of each
    arm, shuffled."""
    arms = [tuple(a) for a in arms]
    if len(set(arms)) != len(arms) or len(set(scenarios)) != len(scenarios):
        raise ValueError("arms and scenarios must not repeat")
    rng = random.Random(seed)
    order = []
    for epoch in range(1, epochs + 1):
        block_order = list(scenarios)
        rng.shuffle(block_order)
        for sc in block_order:
            block = list(arms)
            rng.shuffle(block)
            order += [(cond, sim, sc, epoch) for cond, sim in block]
    return order


def parse_epochs(text: str | None, total: int) -> set[int] | None:
    """`3` or `2-4` (inclusive) to a set of epochs within 1..total; None for all."""
    if text is None:
        return None
    lo, _, hi = str(text).partition("-")
    try:
        a, b = int(lo), int(hi or lo)
    except ValueError:
        raise ValueError(f"epochs must be N or A-B, got {text!r}") from None
    if not 1 <= a <= b <= total:
        raise ValueError(f"epochs {text!r} must lie within 1-{total}")
    return set(range(a, b + 1))


@dataclass
class Admission:
    key: tuple
    attempt: int            # 1 for the first attempt in this process, then 2, 3 for re-runs
    rank: int               # position in the queue; re-runs come after every first attempt
    admitted_at: float
    waited: float
    id: str = ""


@dataclass
class _Waiter:
    key: tuple
    rank: int
    attempt: int
    arrived: float
    event: anyio.Event = field(default_factory=anyio.Event)
    admission: Admission | None = None
    refused: str | None = None


class EpisodeQueue:
    def __init__(self, order, in_flight: int, *, epochs: set[int] | None = None, admit_until: float | None = None,
                 carry_under: int = 0, log_dir: str | Path | None = None, settle: float = SETTLE_SECONDS,
                 clock=time.time):
        if in_flight < 1:
            raise ValueError("in_flight must be at least 1")
        if carry_under < 0:
            raise ValueError("carry_under can't be negative")
        self.order = [tuple(k) for k in order]
        self.rank = {k: i for i, k in enumerate(self.order)}
        if len(self.rank) != len(self.order):
            raise ValueError("the order repeats an episode")
        self.in_flight, self.epochs, self.admit_until = in_flight, epochs, admit_until
        self.carry_under, self.settle, self.clock = carry_under, settle, clock
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.waiting: list[_Waiter] = []
        self.running: dict[str, Admission] = {}
        self.attempts: Counter = Counter()
        self.counts: Counter = Counter()
        self._reruns = 0
        self._last_arrival: float | None = None
        self._settled = settle <= 0
        self._seq = 0
        self._settle_started = False

    # -- the attempt store ----------------------------------------------------

    @property
    def attempts_dir(self) -> Path | None:
        return self.log_dir / "attempts" if self.log_dir is not None else None

    def _log(self, event: str, key: tuple, **extra) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        mix = Counter(f"{a.key[0]}/{a.key[1]}" for a in self.running.values())
        rec = {"t": self.clock(), "event": event, "key": list(key), "running": dict(sorted(mix.items())), **extra}
        with open(self.log_dir / "queue.jsonl", "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    # -- admission --------------------------------------------------------------

    def _refuse_now(self, key: tuple) -> str | None:
        if key not in self.rank:
            return "not in the run's order"
        if self.epochs is not None and key[3] not in self.epochs:
            return f"epoch {key[3]} is outside this session's epochs"
        return None

    async def acquire(self, key: tuple) -> Admission:
        key = tuple(key)
        if reason := self._refuse_now(key):
            self.counts["not_started"] += 1
            self._log("not_started", key, reason=reason)
            raise NotStarted(reason)
        self.attempts[key] += 1
        attempt = self.attempts[key]
        if attempt == 1:
            rank = self.rank[key]
        else:   # a re-run goes behind every first attempt
            self._reruns += 1
            rank = len(self.order) + self._reruns
        w = _Waiter(key, rank, attempt, self.clock())
        self.waiting.append(w)
        self._last_arrival = w.arrived
        try:
            if not self._settled:
                await self._settle_then_dispatch()
            self._dispatch()
            await w.event.wait()
        except BaseException:   # cancelled while waiting: give back the place, or the slot
            if w in self.waiting:
                self.waiting.remove(w)
                self.attempts[key] -= 1
            elif w.admission is not None:   # granted a slot but cancelled before it woke: never ran
                self.release(w.admission, NEVER_RAN)
            raise
        if w.refused is not None:
            self.attempts[key] -= 1   # it never ran: the next session's attempt is still its first
            raise NotStarted(w.refused)
        return w.admission

    async def _settle_then_dispatch(self) -> None:
        # Inspect starts every episode's coroutine at once; admitting the first arrivals before
        # the others have registered would ignore the order. Wait until every episode in the
        # order has arrived, or until no new one has arrived for `settle` seconds.
        if self._settle_started:
            return
        self._settle_started = True
        try:
            while len(self.waiting) < len(self.order):
                await anyio.sleep(min(0.25, self.settle))
                if self.clock() - (self._last_arrival or 0) >= self.settle:
                    break
        finally:   # even if this waiter is cancelled, the others must not wait forever
            self._settled = True
            self._dispatch()

    def _left(self) -> int:
        return len(self.waiting) + len(self.running)

    def _dispatch(self) -> None:
        if not self._settled:
            return
        while self.waiting:
            if self.admit_until is not None and self.clock() >= self.admit_until:
                self._refuse_all("the session's admission deadline has passed")
                return
            if self.carry_under and self._left() < self.carry_under:
                self._refuse_all(f"carried to the next session: fewer than {self.carry_under} episodes left")
                return
            if len(self.running) >= self.in_flight:
                return
            w = min(self.waiting, key=lambda x: x.rank)
            self.waiting.remove(w)
            now = self.clock()
            self._seq += 1
            adm = Admission(w.key, w.attempt, w.rank, now, now - w.arrived, id=f"{now:.6f}-{self._seq}")
            self.running[adm.id] = adm
            w.admission = adm
            self.counts["admitted"] += 1
            if w.attempt > 1:
                self.counts["reruns"] += 1
            self._log("admit", w.key, attempt=w.attempt, rank=w.rank, waited=round(adm.waited, 3), id=adm.id)
            w.event.set()

    def _refuse_all(self, reason: str) -> None:
        for w in sorted(self.waiting, key=lambda x: x.rank):
            w.refused = reason
            self.counts["not_started"] += 1
            self._log("not_started", w.key, reason=reason, attempt=w.attempt)
            w.event.set()
        self.waiting.clear()

    def release(self, adm: Admission, outcome: str) -> None:
        if self.running.pop(adm.id, None) is None:
            return
        self._log("release", adm.key, attempt=adm.attempt, outcome=outcome, id=adm.id,
                  seconds=round(self.clock() - adm.admitted_at, 3))
        self._dispatch()

    @contextlib.asynccontextmanager
    async def slot(self, key: tuple):
        adm = await self.acquire(key)
        outcome = "error"
        try:
            yield adm
            outcome = "done"
        finally:
            self.release(adm, outcome)

    def summary(self) -> str:
        c = self.counts
        return (f"queue: {c['admitted']} admitted ({c['reruns']} re-runs), {c['not_started']} not started, "
                f"{self.in_flight} in flight at most")


def queue_events(log_dir) -> list[dict]:
    """The lines of queue.jsonl (a torn last line, from a killed process, is skipped)."""
    path = Path(log_dir) / "queue.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def batch_mix(log_dir) -> dict | None:
    """Per arm, the mean number of episodes of each arm running when one of its episodes started
    (itself left out), from queue.jsonl. With one queue they should match across arms."""
    path = Path(log_dir) / "queue.jsonl"
    if not path.exists():
        return None
    sums: dict = {}
    n: Counter = Counter()
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["event"] != "admit":
            continue
        arm = f"{e['key'][0]}/{e['key'][1]}"
        running = Counter(e["running"])
        running[arm] -= 1
        n[arm] += 1
        for other, k in running.items():
            sums.setdefault(arm, Counter())[other] += k
        sums.setdefault(arm, Counter())
    arms = sorted(n)
    return {a: {b: sums[a][b] / n[a] for b in arms} for a in arms}


def format_batch_mix(mix: dict | None) -> str:
    if not mix:
        return ""
    arms = sorted(mix)
    lines = ["Batch mix (mean episodes running beside each start, from queue.jsonl): " + ", ".join(arms)]
    lines += [f"  {a}: " + ", ".join(f"{mix[a][b]:.1f}" for b in arms) for a in arms]
    return "\n".join(lines)


def summary_never_started(path, head, summary) -> bool:
    """`never_started` for a sample summary (which has no store): reads the sample's store only
    for an errored sample of a log written under the queue."""
    if summary.error is None:
        return False
    if is_not_started(summary.error):
        return True
    if not queued_log(head):
        return False
    from inspect_ai.log import read_eval_log_sample

    sample = read_eval_log_sample(str(path), id=summary.id, epoch=summary.epoch,
                                  exclude_fields={"messages", "events", "attachments", "output"})
    return never_started(summary.error, sample.store, True)


def write_attempt(directory: Path | None, record: dict) -> Path | None:
    """Write a failed attempt's record (one file each, never overwritten)."""
    if directory is None:
        return None
    import uuid

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(record, default=str))
    return path


def read_attempts(directory: Path) -> list[dict]:
    if not directory.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


@contextlib.contextmanager
def installed(queue: EpisodeQueue):
    """Make `queue` the one the episode solver uses (one run at a time per process)."""
    global _CURRENT
    if _CURRENT is not None:
        raise RuntimeError("a run queue is already installed")
    _CURRENT = queue
    try:
        yield queue
    finally:
        _CURRENT = None


def current() -> EpisodeQueue | None:
    return _CURRENT
