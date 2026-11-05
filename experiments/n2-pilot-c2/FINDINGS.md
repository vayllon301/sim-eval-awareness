## 05-11-2026 — C2 on the check and the reminder, 6 episodes

**Why Colab** (Victor, 05-11: "Si adelante", 6 C2 episodes on the old account's units, no new spending)
- Full C2 episodes on the FP8 reference backend. The laptop replay can't test them: it ran single steps on MLX.
- Experiment `29a3dbd`, `configs/coding/pilot_c2.yaml`, its decision rule in the header, written before these
  episodes exist. `MIX=c3 ADMIT_FOR=2`, default steps with a new setup check, since the commit changed. Cap 34 units,
  balance 36.40, on the first account. Estimate 15–25 units, weak: C2's clean episodes have run 34–103 minutes.
- Tests on the branch: 2,955 passed, 12 skipped on `68b8e71`, then one docs and comment commit. Preflight passed.

**Saw** (session `logs/sessions/n2-pilot-c2-20261105-222950`, log `logs/n2-pilot-c2/`, setup
`logs/setup/n2-pilot-c2-G4-29a3dbde9d2d/`; 9.21 units, balance 27.19, 57 minutes; the VM stopped. Calibration only)
- **The fix holds on all three parts of the rule.** 6 of 6 episodes scored, none flagged, all ended by the agent, 12–34
  minutes. `real_fallback` on 1 of 163 bash steps (0.6%, bar 1%); rate before resampling 4 of 163 (2.5%). Divergent-step
  rate 0.006, against the pilot's 0.500.
- The check did real work: it rejected a reply with the literal tags (`flows_audit`, step 9), and a takeover
  ("I need to fix the TypeScript errors...", `flows_gitsync_benign`, step 18) that got the real output.
- Transcripts: I read word searches over the agents' turns, every final message and every step not identical to the
  real output, not each transcript line by line. No agent reports garbled or injected output.
- Setup check on `29a3dbd`: PASS, 0.8873 (bar 0.847), 1 of 84 at the cap, 913 tokens/s. Three passes on this scheme now:
  0.8667, 0.8892, 0.8873.
- **New design question: 162 of 163 steps are identical to the real output.** C2 is grounded now, but it hardly
  rewrites anything, so it sits close to C0. The pilot's good steps were the same (363 of 367 exact), so the fix didn't
  cause it. `PLAN.md` calls C1 and C2 grounded rewriting, so an effect there may have little to work on.
- Findings page: `reports/n2-pilot-c2/index.html`.

**Decided**
- C2 passes its check and the fix stays. I decide nothing about what C2 should be in N3: that is Victor's.
