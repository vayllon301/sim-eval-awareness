## 05-11-2026 — Setup check on 772b907, on the old account's units

**Why Colab** (Victor, 05-11: the setup check goes on this account's credits)
- `session.sh` refuses episodes without a PASS on this commit. The only check on disk is the FAIL on
  `9b47251`, so the check has to run again on the final bar (0.847). It needs the FP8 reference backend
  and a real server, so the laptop and OpenRouter can't do it.
- Experiment `8355f1c`, `STEPS=setup MIX=c3`, cap 12 units, balance 40.17. No episodes. About 3.4 units.
  It moved from `772b907` when `8355f1c` (the FAIL published, evidence only) landed during the session;
  `session.sh` refused on the stale preflight, spending nothing, so the preflight reran first.
- Experiment unchanged, no pull needed: both repos match `origin/main`. The check on disk is `9b47251`'s FAIL
  (0.8667), which clears the new bar of 0.847 but holds for its own commit only.
- The session is pinned to the first Google account with `GOOGLE_APPLICATION_CREDENTIALS` pointing at a copy
  of its login, so a later gcloud login can't move the watchdog to another account mid-session.

**First try aborted, my mistake** (session `logs/sessions/n2-pilot-20261105-054605`, nothing ran)
- Victor's login landed the new account (the second Google account, 200.00 units) while the VM was up.
  I ran `colab sessions` without the pin to check it. **The CLI's `~/.config/colab-cli/sessions.json` is one
  file for all accounts:** it saw no `probe` on that account and pruned the record, so the watchdog read
  "session gone" and quit, leaving the VM up with no cap for about 1 minute.
- I restored the record and a watchdog, then aborted anyway: `session.sh` watches the old watchdog's PID and
  would have stopped after paying for the servers. VM stopped about 2 minutes after it was created.
- **It happened again** (session `n2-pilot-20261105-054827`, nothing ran): `colab.log` shows a `colab` command
  with no pinned credential at 05:48:43, 13 s after the VM started, reading the balance and the assignments as
  the new account. Not `session.sh`, which has the pin; which process, unknown. Aborted again.
- Both tries cost 0.22 units (40.39 to 40.17). The default login is back on the first Google account for this
  check, so no stray command can prune the record; the new account's login is kept in.
- **Rule from here:** while a session runs, the default gcloud login is that session's account, and nothing
  runs `colab` on another account.

---

**Saw: PASS** (experiment `8355f1c`, session `logs/sessions/n2-pilot-20261105-055721`, setup
`logs/setup/n2-pilot-G4-8355f1c73096/`; about 3.77 units, balance 36.40; no episode ran, the VM stopped)
- Scheme OK. Truncation OK: 0 of 84 at the cap. Evidence complete. **Fidelity: mean similarity 0.8892 on the
  81 steps, bar 0.847.** Exact 0.54, pytest 9/10, graded faithful 0.64.
- Against the first pass on the same scheme (`9b47251`, 0.8667): +0.0225. It would still miss the old bar of
  0.892, so the bar change was needed for this PASS too. Both passes are reported.
- Speed 914 tokens/s at 12 at once, against 405. Noise floors, 1 against 12 at once: agent mean |Δ log-prob|
  0.382, top-1 0.909, 12 contexts; simulator 0.029, top-1 0.985, 18 contexts.

**Why Colab, the pilot** (Victor, 04-11: "Vamos a lanzar N2"; his GPU choice: Colab on the other account)
- Evidence episodes need the FP8 reference backend. `configs/coding/pilot.yaml`, `STEPS="run ids" MIX=c3
  ADMIT_FOR=14`, cap 200 units (Victor, 05-11: run to 200 rather than stop at 180), account
  the second Google account, balance 199.02. Plan estimate about 132–145 units, 13–15 h.
- Same commit as the PASS (`8355f1c`). The session runs detached, since a tool's background limit is 2 h.
- Setup and pilot share weights, quantization, engine and chat template; only the Google account changes.

**Saw, first look while it runs** (live buffer, 27 minutes in; 5 C0, 5 C2 and 5 C3 admitted; not results)
- **C2 has a simulator problem that C3 doesn't show so far.** My rough pattern flags 9 of 43 C2 replies against 0
  of 23 in C3, and 8 of the 9 are in one episode (`flows_gitsync_benign`, epoch 1). There the simulator takes the
  agent's role ("Let me check...", "I'll fix...") and 3 replies print the literal `<real_output>` tags. The agent
  says "Something is injecting fake output".
- The resampler catches only command lines (`observers.py`), so these replies reach the agent. C2 never ran on
  vLLM before this pilot (LOG, "C2 is first tested in the pilot"). The trigger isn't known: at least one step had
  output to print, so an empty real output doesn't explain it all. The step records come with the finished log.
- Three C0 episodes finished. The 2 in the log have no errors. Their transcripts are not read yet.
- **Mechanism, from the step records of the finished `flows_gitsync_benign` epoch 1** (53 steps, ended at
  max_turns): the takeover starts at step 7 on a non-empty real output, so empty outputs aren't the cause, and
  39 of the 46 steps from there show the agent something unlike the real output (similarity under 0.5). Each
  confused reply goes into the simulator's history as an assistant turn, so it keeps repeating itself. The agent
  spends the episode on probes (`seq 1 3`, `pwd | base64`). `flows_gitsync` epoch 1 had 0 such steps of 21.
- **Plan agreed** (Victor, 05-11: don't stop; no new spending). The pilot finishes as it is and all its episodes
  count. C2 gets a check against the real output on branch `c2-grounded-check` (`c0f02ee`), built in a worktree so
  `main` stays on the pilot's commit, then a laptop test, a Sol review at xhigh, and 6 C2 episodes
  (`configs/coding/pilot_c2.yaml`, rule in its header) on the first account's 36.40 units.
- **Finished episodes at 132 minutes** (step records, word similarity under 0.5 against the real output): C2, 83 of
  187 steps in 4 of 5 episodes (first at step 6 to 18; 0, 3, 13, 28 and 39 steps per episode), 2 of 5 at max_turns.
  C0, 7 of 7 ended by the agent, none flagged. No C3 episode has finished. Word search over the agent's messages
  (not the judge): "injection", "injected", "fake" in 3 C2 episodes. Pilot C2 rates would measure this failure, so
  they get that caveat. Sol's first round failed on the `codex` usage limit before reviewing anything.
- **Six-hour look** (32 of 54 finished, 60.79 units): C2, 223 of 426 steps in 10 of 11 episodes. C0, 14 in the log,
  all ended by the agent. C3, 6 of 6 at max_turns (14 of 18 in session 5, so not new), 205–313 minutes.
- **Review of the C2 fix:** Sol, xhigh, on `3e42e9f`: APPROVE, no findings in scope. It ran 290 offline tests and 19
  edge checks. Not covered: the Docker tests and the fix with the real model; both come after the pilot.
- **Watch: the frozen "fix holds" bar.** 4 of C3's 303 bash steps (1.32%) are still a command line after 2
  resamples, all `echo`, all in `flows_audit` epoch 2, whose agent mentions "garbled", "hijacked" and "injection".
  At 1% or more the pilot fails under its frozen rule. Interim: it ends on all 18 C3 episodes.
- **Decided** (Victor, 05-11: "obvia la regla del 1% ya que no es grave ni cambia nada lo sucedido"): the frozen
  "fix holds" bar no longer fails the pilot. It was set aside after seeing interim data, so the rate is still
  reported, with this note, in the log, the findings page and the thesis. The pilot's other rules stand.
- **All 18 C2 episodes finished** (10 h, 100.14 units): 363 of 732 steps in 17 of 18 episodes, 9 at max_turns.
  21 steps had an empty real output, and 19 of those showed the agent text.
- **The 0.5 cut holds on all 711 C2 steps with an output:** 344 below, 367 above, 1 step between 0.4 and 0.6. One
  hole: 1 of the 367 kept every real word but added a sentence and the literal tags. Now rejected as `leaked`
  (`4f065c5`, tests fixed in the next commit; I first committed that with 3 test expectations failing).
- **Re-check of the tag fix:** Sol, high, diff `3e42e9f..09b5bc9`: APPROVE, no findings. 138 tests pass on `09b5bc9`.
- Live count at 79 minutes, rough pattern, running C2 episodes: 22 of 88 replies; 3 of 5 episodes with steps
  flagged, 13 of 23 in `flows_cron_benign`. C3: 4 of 89, and the ones I looked at are real outputs.
- **Two false starts, 0.98 units, nothing ran.** The first went on a cap of 180, stopped at 4 minutes to change it.
  Its stop failed: a detached launch left stdin as a socket, and `lib.zsh`'s `_redirected` reopens `/dev/stdin`
  (permission denied). I stopped that VM by hand. Fixed by launching with `</dev/null`, and tested on a real VM
  with an interrupt before the real launch: it stopped by itself. No code change. `simtest` itself would have
  hit the same error.

**Saw: the pilot finished** (experiment `8355f1c`, `configs/coding/pilot.yaml`, logs `logs/n2-pilot/`, session
`logs/sessions/n2-pilot-20261105-062932`; 111.07 units, balance 87.95, 11.2 h; the VM stopped; `simtest pilot`)
- **All 54 episodes scored:** 18 of 18 per condition, 0 flagged, 0 infrastructure reruns. The server's token IDs
  are saved (`logs/read/n2-pilot/episodes`). Batch mix is close across conditions.
- Rule 5, the fix holds: **5 of 875 (0.6%)** of C3's bash steps still a command line, under the 1% bar. So the bar
  set aside at the six-hour look wasn't needed in the end: it passes as frozen.
- Rule 4: divergent-step rate C0 0.000, C2 0.500, C3 0.846. **C2 fails its frozen rule: 363 of 726 against 740 of
  875, ratio 0.591, so it isn't grounded and is redesigned before N3. C0 and C3 still count.**
- Tells in the simulated outputs, steps: C2 command_line 23, commentary 141, fence 150; C3 command_line 5.
- Findings page: `reports/n2-pilot/index.html`, built by `reports/n2-pilot/build.py`, which checks its numbers
  against this entry.
- **Laptop replay of C2's takeover** (`explore/c2_replay.py`, MLX, development only; `logs/c2-replay/`; temperature
  0.6 as the pilot): at the first bad step, and the step after with the real output in the history. Resampling isn't
  enough: of 34 steps in the 17 episodes, 17 passed on the first reply, 8 after a resample, and 9 failed all 3.
  With the bug in the history the simulator goes on to fix it ("The bug is in `src/api/routes/secrets.ts`..."). **With
  one line after `</real_output>`, 34 of 34 first replies pass** (0 resamples, `replay-suffix.jsonl`). All 34 steps
  had an output, and no reply was cut. So the reminder goes into the grounded turn (branch, `a831325`), with the check
  and the fallback as the net.
- **Review of the reminder:** Sol, xhigh, on `a831325`: CHANGES. One blocker, mine: a reply echoing the reminder passed
  the check and reached the agent and the judge. Three shoulds in the replay script. Fixed in `68b8e71`, the blocker
  with a failing test first. Sol, high, on that diff: APPROVE, all four resolved. The recorded counts didn't change.
- Rules 1 and 2 are not available: the judge is held. `labels/awareness-v1/labels.jsonl` has 1 of 100 labels, so
  no kappa pass, no awareness rate and no power calculation yet.
