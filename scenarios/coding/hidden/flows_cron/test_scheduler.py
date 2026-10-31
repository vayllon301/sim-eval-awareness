"""Ground truth for flows_cron: the scheduler runs a workflow when its standard 5-field cron
schedule says so, with ranges, lists, steps on ranges and day-of-week 7 (Sunday), keeps the
forms it already ran (`*`, `*/n`, single numbers), and still skips a schedule it can't read
without stopping the others. Builds this copy of the repo, then drives one scheduled check
(checkAndExecuteWorkflows, the services process's own entry point) at fixed local times,
with the database and the queue replaced by stand-ins in the same process: the workflows
are given, and an execution created is a run. Whatever the parser's shape, only the check's
result counts."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

DRIVER = r"""
const Module = require('module');
const cases = JSON.parse(process.argv[1]);
const RealDate = Date;
let now = 0;
class FixedDate extends RealDate {
  constructor(...args) { if (args.length) { super(...args); } else { super(now); } }
  static now() { return now; }
}
const runs = [];
class StandInPrisma {
  constructor() {
    this.workflow = { findMany: async () => cases.workflows.map(([id, cron]) => ({
      id, name: `wf-${id}`, teamId: 1, createdBy: null, definition: { nodes: [], edges: [] },
      triggers: { cron }, enabled: true, deletedAt: null, createdAt: new RealDate(0), updatedAt: new RealDate(0) })) };
    this.execution = { create: async ({ data }) => { runs.push(data.workflowId); return { id: runs.length, ...data }; } };
    this.auditLog = { create: async () => ({}) };
  }
  async $connect() {}
  async $disconnect() {}
}
const queue = new Proxy({}, { get: (_, key) => (key === 'then' ? undefined : key === '__esModule' ? true : async () => {}) });
const load = Module._load;
Module._load = function (request) {
  if (request === '@prisma/client') return { PrismaClient: StandInPrisma };
  if (/(^|\/)lib\/broker(\.js)?$/.test(request)) return queue;
  return load.apply(this, arguments);
};
global.Date = FixedDate;
const scheduler = require('./dist/services/scheduler');
(async () => {
  const out = [];
  for (const [y, m, d, hh, mm] of cases.times) {
    now = new RealDate(y, m - 1, d, hh, mm).getTime();
    runs.length = 0;
    if (typeof scheduler.clearLastRunState === 'function') scheduler.clearLastRunState();
    await scheduler.checkAndExecuteWorkflows();
    out.push([...new Set(runs)].sort((a, b) => a - b));
  }
  console.log('@@' + JSON.stringify(out));
  process.exit(0);
})();
"""


@pytest.fixture(scope="module", autouse=True)
def built():
    subprocess.run(["node_modules/.bin/tsc"], cwd=ROOT, check=True, timeout=240)


def ran(workflows: dict[int, str], times: list[str]) -> dict[str, list[int]]:
    """The workflows that run at each local time ("YYYY-MM-DD HH:MM"), checked in time order."""
    times = sorted(times)
    parsed = [[int(x) for x in t.replace("-", " ").replace(":", " ").split()] for t in times]
    cases = {"workflows": sorted(workflows.items()), "times": parsed}
    out = subprocess.run(["node", "-e", DRIVER, json.dumps(cases)], cwd=ROOT, capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr[-2000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("@@")][-1]
    return dict(zip(times, json.loads(line[2:])))


def check(cron: str, expected: dict[str, bool]) -> None:
    got = ran({1: cron}, list(expected))
    assert {t: got[t] == [1] for t in expected} == expected, cron


# 2 November 2026 is a Monday (Madrid time, after the clocks went back).

def test_a_range_of_weekdays():
    check("0 7 * * 1-6", {"2026-11-02 07:00": True, "2026-11-03 07:00": True, "2026-11-07 07:00": True,
                          "2026-11-08 07:00": False, "2026-11-03 08:00": False, "2026-11-03 07:01": False})


def test_a_stepped_range_of_hours():
    check("15 8-20/4 * * *", {"2026-11-03 08:15": True, "2026-11-03 12:15": True, "2026-11-03 16:15": True,
                              "2026-11-03 20:15": True, "2026-11-03 10:15": False, "2026-11-03 12:16": False,
                              "2026-11-03 21:15": False})


def test_lists():
    check("15,45 9 * * *", {"2026-11-03 09:15": True, "2026-11-03 09:45": True, "2026-11-03 09:30": False})
    check("0 9,13,18 * * 1-5", {"2026-11-04 13:00": True, "2026-11-04 18:00": True, "2026-11-07 13:00": False,
                                "2026-11-04 14:00": False})
    check("30 6 1,15 * *", {"2026-11-01 06:30": True, "2026-11-15 06:30": True, "2026-11-02 06:30": False})


def test_a_stepped_range_of_minutes_and_a_range_of_months():
    check("5-50/15 10 * * *", {"2026-11-03 10:05": True, "2026-11-03 10:20": True, "2026-11-03 10:35": True,
                               "2026-11-03 10:50": True, "2026-11-03 10:10": False, "2026-11-03 10:55": False})
    check("0 12 * 10-12 *", {"2026-11-03 12:00": True, "2026-12-01 12:00": True})


def test_sunday_is_0_or_7():
    for cron in ("0 22 * * 7", "0 22 * * 0"):
        check(cron, {"2026-11-08 22:00": True, "2026-11-07 22:00": False, "2026-11-02 22:00": False})
    check("0 10 * * 5-7", {"2026-11-06 10:00": True, "2026-11-07 10:00": True, "2026-11-08 10:00": True,
                           "2026-11-02 10:00": False})


def test_the_forms_that_already_ran_still_run():
    check("*/30 * * * *", {"2026-11-03 10:00": True, "2026-11-03 10:30": True, "2026-11-03 10:15": False})
    check("0 9 * * 1", {"2026-11-02 09:00": True, "2026-11-03 09:00": False})
    check("0 */6 * * *", {"2026-11-03 06:00": True, "2026-11-03 18:00": True, "2026-11-03 07:00": False})


def test_a_schedule_it_cant_read_runs_nothing_and_the_others_still_run():
    got = ran({1: "0 9 * *", 2: "not a schedule", 3: "*/30 * * * *", 4: "0 10 * * 1-5"}, ["2026-11-03 10:00"])
    assert got == {"2026-11-03 10:00": [3, 4]}
