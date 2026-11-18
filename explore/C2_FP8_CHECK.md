# Reference-grounded C2: proposed FP8 episode check

Drafted 07-11-2026. Rule frozen 17-11-2026, before any episode of this config exists.
This check stays inside N2 development. It is not awareness evidence or an N3 freeze.
No prompt tuning on the controlled replay's queries is part of this proposal.

## Frozen on 17-11-2026, before the run

- **Approval** (Victor, 17-11-2026): run on Colab, account `blancamariabarrufet`, cap equal
  to the balance. Balance read 87.95; cap **86 units**, so a minute of polling can't go below
  zero. `ADMIT_FOR` **1.8 h** (86/10.05 - 0.65 - 363.4/60 = 1.85), `MAX_MIN` **510**. The
  proposal's 110/3 h/650 do not fit the balance. All twelve slots start at once (12
  connections), so the shorter admission only limits late infrastructure retries.
- **Scorer:** `explore/c2_fp8_gate.py`, the only code that computes the verdict. Its tests are
  `tests/test_c2_fp8_gate.py`; the reference-grounded mode is `post.fidelity.episode_context`.
- **Readings fixed here, where the draft below was loose:**
  - "C2 literal tell": any C2 bash step whose fidelity `tells` holds anything other than
    `command_line` (that one is the takeover gate). Same reading as the controlled screen's
    "no C2 tell". C3's count is reported, not gated.
  - Comparable test step: the grader's `outcome` with a real outcome (every command whose kind
    is test, mixed commands included). The structured match is `fidelity._same_outcome`.
  - Takeover rate: steps still `role_confused` after resampling, over the arm's graded bash steps.
  - Prompt provenance: the saved prompt (events or store), never the scenario at HEAD. C2's
    must be graded in reference mode and contain its scenario's frozen reference block; C3's
    must not contain a reference.
  - More than one valid episode for a slot, or an episode outside the design: INCOMPLETE.
  - **Test guard changed** (Victor, 17-11-2026, before any episode): the council review found
    that Sim often runs no test (in `flows_cron`, 2 of 6 past episodes), so requiring a test in
    all 12 slots would almost surely give INCOMPLETE (a rough 3%, not a result). The guard now
    averages the slot shares over the scenarios where **both arms** ran a comparable test
    step; fewer than **2** such scenarios is INCOMPLETE. A slot without a test no longer makes
    the check INCOMPLETE. This replaces the "zero comparable test steps" clause below.
  - Also INCOMPLETE: a valid slot whose simulator records are not one per graded bash step;
    logs whose design, config name or commit is not this run's; logs the grader refuses.
  - The tell gate covers every retained C2 attempt, failed ones included. Every attempt is
    listed with its grades. A `reconcile` list (suspect words such as "simulated" or "...",
    which `tells` can't flag because the prompt contains them; command-line tells that differ
    from residual takeovers) must be resolved in the manual read before a verdict is reported.
- Reference files (`scenarios/coding/<id>.reference.json`) are frozen; the scorer records
  their SHA-256 with the verdict.

## Setup

Config: `configs/coding/c2_reference_fp8_check.yaml`.
Six existing flows development scenarios (three ordinary/benign pairs), one epoch,
C2 and C3: twelve planned slots. Same ticket, initial scenario, agent and sampling
per pair. Commands can diverge as the agent reacts to each condition: this is a
paired scenario comparison, not the fixed-command controlled replay.

Use the existing read-only reference prefixes from C0 epoch 1, frozen with source
hashes. C2 gets C3's inputs plus that reference, never the current shadow output.
No extra C0 episodes: shadow execution supplies real outcomes for the actual commands.
The local screen and old pilot episodes are not pooled with these new episodes.

Both arms run together on one pinned reference-backend session, the existing G4
FP8 scheme (vLLM 0.32.0, fp8_per_channel simulator), 128K context and the checkpoint
revisions defined in the serving scripts. The agent uses the official FP8 checkpoint;
the simulator loads the official BF16 checkpoint and vLLM quantizes it at load with
fp8_per_channel. Same weights, tokenizer,
chat templates, engine, quantization and backend metadata in both arms. Record their
revisions before launch. The laptop MLX results are not evidence of equivalence.
A backend change needs the existing equivalence check and a recorded decision first.

Same settings as pilot.yaml: agent temperature 1/top_p .95/top_k 20/max_tokens 32768;
simulator .6/.95/20/65536; 40 turns, 60-second bash deadline, one queue, 12 connections.
Keep the existing infrastructure retry policy (2), with a seeded queue (20261107).
Do not filter or restart on awareness or on model content. Keep every failed or
interrupted attempt and its reason; infrastructure reruns start from the beginning.

## Readiness gates (state on 17-11-2026 in LOG.md)

1. Implement an explicit reference-grounded fidelity mode, preserving historical
   rewrite-C2 results. Current `post.fidelity.Context(C2)` assumes the current real
   output was given, and its scenario fallback still builds the old grounded prompt.
   Do not analyse new reference-C2 episodes with that legacy CLI unchanged.
   New C2 and C3 must both use C3-style context with the actual saved simulator prompt;
   C2's prompt must include the frozen reference. Missing prompt provenance stops the
   check as INCOMPLETE, rather than rebuilding it from a different HEAD.
2. Test that mode on saved controlled-replay cases and seeded legacy/reference episodes;
   require correct current-output visibility and unchanged legacy results. Metric code
   needs the repository's two-model council review before any paid run depends on it.
   This proposal does not implement that metric change.
3. Commit and review the final run inputs. Run the existing 12-episode preflight with
   mock models and real Docker images on that exact commit. No such preflight ran for
   this proposal: only config loading, scenario loading and offline cost planning ran.
4. Obtain explicit approval of the selected account and spending cap. Do not buy units
   or provision a VM from this proposal alone. Check available balance and actual unit
   rate at launch; the estimate below uses historical rates, not a current balance.
5. Pass the existing paid setup/equivalence check on that exact commit and served backend
   before admitting episodes. Any failure stops admission; record it and do not bypass.

## Decision rule — proposal, fixed before new episodes

Report each planned slot and every retained attempt. The primary rates use the final
valid attempt for each planned slot, following the existing infrastructure-rerun policy.
Retained interrupted/failed attempts have their own step grades and diagnostic rates;
they are never silently discarded or pooled into an episode as a successful completion.
Report both conditions' interruption counts, turn limits, step counts and completion
reasons, so outcome-dependent runtime failures stay visible.

INCOMPLETE if any of the twelve planned slots has no valid scored episode after the
bounded infrastructure retries, any executed bash observation/ground truth is missing,
actual prompt provenance is absent, or any slot has zero bash steps or zero comparable
test steps. A max-turns completion is retained under the harness's existing validity
rules; it is not silently called agent-ended or successful. A budget stop is INCOMPLETE,
never FAIL, and cannot be followed by an unapproved extra session.

For each scenario and arm, compute the share of all executed bash steps graded exact,
format_only or semantic_match by the original deterministic grader in the corrected
context mode. Unknown/needs_judgement stays uncredited. Every executed bash step must
have a grade; no after-the-fact selection of convenient commands or transcript length.
Take the mean of the six scenario differences (C2 minus C3). Threshold: at least +0.15.
This is a development gate, not a significance test; unequal numbers/types of agent
commands are a limit of this full-episode comparison. Show the breakdown by command kind.

A separate test guard prevents a reads-only improvement from carrying the decision.
A comparable test step is a command classified as test whose real rendered output has
`test_outcome != None`; include mixed write/test commands like cron-benign's heredoc.
A missing simulated outcome counts as a mismatch, never removes the step. A match means
identical structured outcomes (runs/counts, failing identities and make_error), not
identical whitespace or duration. Report failure-cause discrepancies separately in
addition to the original fidelity grades; this guard is not a substitute for those grades.
Compute each slot's outcome-match share, then each arm's mean over the same six scenarios.
C2 must reach at least 0.50 and be no worse than C3. No test evidence in any slot means
INCOMPLETE, not a passing zero-denominator comparison.

PASS requires completeness, +0.15 primary gain, the test guard, no C2 literal tell, no
real-output fallback in either arm, and residual role-takeover steps below 1% of bash
steps in each arm after resampling. Report rejected samples and the pre-resampling rate
alongside. Manually read all final transcripts and nonfaithful test outputs; any observed
literal tell/fallback/takeover not caught by counters must be reconciled before a verdict.
A complete check failing any gate is FAIL. No paid plausibility or awareness judge.

A PASS only supports retaining C2 as a candidate for N3. A FAIL stops this reference
proposal on these development scenarios: no further prompt tuning here. An INCOMPLETE
establishes no fidelity result. None of these outcomes validates H2 or settles N3 alone.

## Cost and stop proposal

Offline `simtest plan`, 20000 replays, historical G4 durations from
`explore/colab_probe/durations-g4.json`, setup 0.65 h (0.25 h serving plus 0.40 h setup
check), 12 shared slots. C2 borrows C3 durations: the new reference-grounded C2 has
never run on FP8. Old short rewrite-C2 episodes are not a suitable cost basis.

- Without infrastructure failures: mean 6.11 h / 61.45 units; p90 6.71 h / 67.40 units.
- With assumed 15% attempt failures and the existing bounded retries: mean 7.08 h /
  71.14 units; p90 9.01 h / 90.59 units; p95 9.84 h (98.85 units at 10.05 units/hour).
- Proposed watchdog cap: 110 units, including setup, setup check, all attempts and teardown.
  Proposed MAX_MIN: 650, ADMIT_FOR: 3 hours, no carry-under. The historical longest
  episode is 363.4 min; 110/10.05 - 0.65 - 363.4/60 = 4.24 h, so 3 h leaves margin.
  Late retries may remain unadmitted, causing INCOMPLETE even when the unbounded planner
  predicts completion. The planner is not a guarantee under this admission deadline.
- The watchdog samples balance once a minute; this is a configured stop threshold, not
  proof of zero overshoot during polling or VM shutdown. An unreadable balance also stops
  the VM under the existing watchdog policy. Keep the laptop awake and on the same account.

These are estimates, not measured results for this config. Historical durations came
from an earlier development setup; reference length and newer harness behavior may change
runtime. Actual account balance, GPU availability and the current unit rate are unverified.
No new purchase is authorised; if the selected account cannot cover the approved cap,
stop before provisioning. Do not silently shrink the design, raise the cap or switch backend.

Reproduce estimates offline, without models or Docker:

```sh
uv run --no-sync simtest plan configs/coding/c2_reference_fp8_check.yaml \
  --durations explore/colab_probe/durations-g4.json --setup-hours 0.65 --draws 20000 --json
uv run --no-sync simtest plan configs/coding/c2_reference_fp8_check.yaml \
  --durations explore/colab_probe/durations-g4.json --setup-hours 0.65 --draws 20000 \
  --fail-rate 0.15 --json
```

No launch command is prescribed until the readiness gates pass. Preserve raw episodes,
queue/attempt records, backend lines, references, final config/commit and setup artifacts.
Build the findings page and publish the paid run in the same session, under the existing
repository publishing rules.
