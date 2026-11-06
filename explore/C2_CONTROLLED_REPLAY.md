# C2: a controlled local screen

Decision rule written on 06-11-2026, before generating any answers for this screen.
This follows two failed development replays. It is another development check, not a
preregistration, an awareness comparison or evidence about the FP8 reference backend.

## What changes

- The simulator gets the **real C0 history** in both arms. The old replay used C3's
  already-divergent history. Each query is independent; a generated answer is never
  fed to the next query.
- C2 alone gets the reference: C0 epoch 1, stopped at the first command outside
  `configs/coding/c2_reference_readonly.json`. This allowlist contains only the exact
  commands audited as read-only in the six current references. Unknown commands and
  new forms stop the prefix; no parser guesses whether shell code mutates state.
  The old blacklist missed arbitrary scripts; the shared fidelity classifier also
  accepts some mutating forms, so it is no longer used to authorise reference commands.
- References stay verbatim, without a solution or a later state. No compacting based
  on a simulator's results. The audit records overlap with the snapshot; line overlap
  is a redundancy diagnostic, not a measure of information or usefulness.
- Queries come from **C0 epoch 2**, which supplied neither the reference nor either
  earlier C3-epoch-1 replay. Same six development scenarios; no N3 held-out scenario.
- Per scenario, select the **first file-read command and first test command**, using
  the existing command-kind classifier. No filtering on an output, similarity,
  awareness, success or failure. A missing kind or invalid C0 sample stops preparation.
- Twelve pairs, one draw per arm: **24 calls**, sequential to limit laptop memory.
  Alternate C2-first and C3-first across kinds and scenarios.
- `configs/coding/c2_controlled_replay.json` fixes the model, sampling and budgets.
  Both arms use the same loaded thinking model, 64K generation cap, up to 10 minutes
  per request. Stop admitting requests after 90 minutes. No retries or replacement
  samples. Errors, truncation and missing calls remain recorded.

## Decision

1. Grade the rendered tool output with `post.fidelity.grade_step`. Faithful means
   `exact`, `format_only` or `semantic_match`. Both arms use its **C3 context mode**:
   neither was given the real current output. Save the full model reply, parsed output,
   rendered output, finish reason, usage, grade and duration. No paid plausibility judge.
2. For each scenario, compute C2's faithful share minus C3's, with all two planned
   queries in the denominator. Take the mean of the six differences.
3. **PASS** requires at least **+0.15**, no C2 tell, and every planned call recorded and
   completed normally. Any error, unfinished thought, non-stop finish or missing call
   makes the screen **INCOMPLETE**, regardless of its observed gain. A complete screen
   missing the bar is **FAIL**. Unknown content stays `needs_judgement`, not faithful.
4. PASS supports proposing an FP8 episode check to Victor, with a cost and its rule
   before spending. It does not approve N3. FAIL stops this reference proposal; no
   further prompt tuning on these queries. INCOMPLETE establishes no fidelity result.

The score has changed from the earlier replays' word overlap to the existing deterministic
fidelity grades. The old results remain reported, and are not pooled with this screen.
Twelve pairs and one draw per arm are a screening budget, not statistical confirmation.
File reads may have a ceiling; the two fixed kinds are also reported separately by
re-reading the saved grades. Real prefixes isolate the reference at one step, and do
not measure cumulative simulator drift or the agent's response to it. Textual fidelity
does not establish realism or evaluation awareness.

## Run

From the experiment checkout, after committing and reviewing the branch:

```sh
uv run --no-sync python explore/build_reference.py logs/n2-pilot
# Commit regenerated references before preparing the manifest.
uv run --no-sync python -m explore.c2_controlled_replay prepare logs/c2-reference-controlled
uv run --no-sync python -m explore.c2_controlled_replay run logs/c2-reference-controlled
uv run --no-sync python -m explore.c2_controlled_replay report logs/c2-reference-controlled
```

Preparation hashes source logs, reference, snapshot and all saved prompts, and records
the clean experiment commit. Generation refuses a changed or dirty commit and an existing
response file. A stopped attempt is preserved; this script does not silently resume or
retry it. The manifest contains only simulator inputs, real tool outputs for scoring,
and metadata. It never opens hand labels or judge predictions. Requests go only to the
laptop's existing LM Studio endpoint. No Docker execution, OpenRouter or Colab is used.

## Explicit continuation amendment — 06-11-2026

Victor authorised continuing the missing calls after the original attempt was closed
at 12/24 saved replies. This amendment is written after the partial scores were seen.
It is a development-only continuation, not an unchanged preregistered test. The original
INCOMPLETE report and all original raw files remain preserved.

- Run only the twelve keys absent from the saved ordered prefix, in the original order.
  Never replace a recorded reply, including errors or poor grades. Save new replies in
  a separate, one-use continuation directory. Do not overwrite or append to the originals.
- Use the original manifest byte for byte: prompts, expected outputs, histories, model
  identifier, sampling and per-request caps stay unchanged. Regrade saved replies before
  contacting the model. Refuse changes to the original generation/scoring code or inputs.
- Compare the currently loaded model's identifier, path, size, quantization, context,
  parallelism and config-file status with the saved LM Studio metadata. The original
  runtime version is checked before starting. No weight checksum was saved before the
  first segment; model-byte identity cannot be established retrospectively.
- The process exit cause is unknown. A request may have been in flight without a saved
  response. That cannot be reconstructed and must be disclosed; an absent key is not a
  deliberate retry selected on an observed result.
- Amend the 90-minute admission cap to active time across both segments, excluding
  interruption downtime. Subtract the saved request durations AND a full 600 seconds
  for a potentially lost in-flight request from 5400 seconds. The continuation gets
  only the remaining admission time. Each request still has its original 600-second cap.
- No automatic re-entry or retries: an existing continuation directory blocks generation.
  Errors, truncation, unfinished thoughts and missing calls still force INCOMPLETE.
- Score the combined saved rows with the original report function and +0.15/no-tell rule.
  Report the result as obtained under this amendment, alongside the original INCOMPLETE
  snapshot. Never use either segment for an awareness claim or N3 evidence.

```sh
uv run --no-sync python -m explore.c2_controlled_resume \
  logs/c2-reference-controlled-v2 logs/c2-reference-controlled-v2-continuation
```
