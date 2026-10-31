// Print the next runs of a cron expression, as the scheduler matches them, in this box's
// local time. Reads dist/, so build first (`make cron-preview` does).
// Usage: node scripts/cron-preview.js '<expression>' [count] [from, e.g. 2026-11-02T08:00]
const { nextRuns } = require('../dist/services/scheduler');

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const pad = (n) => String(n).padStart(2, '0');

function local(d) {
  const offset = -d.getTimezoneOffset();
  const sign = offset < 0 ? '-' : '+';
  const zone = `${sign}${pad(Math.floor(Math.abs(offset) / 60))}:${pad(Math.abs(offset) % 60)}`;
  return `${DAYS[d.getDay()]} ${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())} ${zone}`;
}

const [expression, countArg, fromArg] = process.argv.slice(2);
if (!expression) {
  console.error("Usage: make cron-preview EXPR='<minute> <hour> <day of month> <month> <day of week>' [N=5] [FROM=2026-11-02T08:00]");
  process.exit(1);
}
const count = countArg ? Number(countArg) : 5;
// A date alone would be read as UTC midnight: take it as local midnight.
const from = fromArg ? new Date(/^\d{4}-\d\d-\d\d$/.test(fromArg) ? `${fromArg}T00:00` : fromArg) : new Date();
if (!Number.isInteger(count) || count < 1 || Number.isNaN(from.getTime())) {
  console.error('N must be a whole number above 0, and FROM a local time such as 2026-11-02T08:00');
  process.exit(1);
}

let runs;
try {
  runs = nextRuns(expression, from, count);
} catch (err) {
  console.error(err.message);
  process.exit(1);
}
const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
console.log(`Next ${count} run${count === 1 ? '' : 's'} of '${expression}' from ${local(from)} (${zone}):`);
for (const run of runs) {
  console.log(`  ${local(run)}`);
}
if (runs.length < count) {
  console.log(`  (${runs.length ? 'no more' : 'none'} in the next year)`);
}
