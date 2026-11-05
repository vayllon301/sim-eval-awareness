# C2 check, 05-11-2026

**The question.** Does the fix for C2 hold in full episodes on the reference backend?

**The answer.** Yes, on its written rule, but C2 now copies the real output on 162 of 163 steps, so it
sits close to C0. The pilot (`experiments/n2-pilot/`) showed the simulator taking the agent's role in C2.

This is a calibration, never evidence for a comparison. It has no C0 or C3 beside it.

## Setup

- **Episodes:** 6. The flows dev set in C2, 1 epoch. Config `configs/coding/pilot_c2.yaml`, with its
  decision rule in the header, as the pilot otherwise.
- **The fix:** each C1 or C2 reply is checked against the real output, resampled up to 2 times, then
  replaced by the real output. The grounded turn ends with a reminder of the simulator's role. See
  `docs/TESTBED.md`, "The grounded turn".
- **Backend:** Colab G4, simulator as `fp8_per_channel`. **Commit:** `29a3dbde9d2d`.
- **Cost:** 9.21 compute units, 57 minutes, setup check included.

## Files

| Path | What |
| --- | --- |
| `FINDINGS.md` | The log entry for this run |
| `report/index.html` | The findings page |
| `logs/` | The `.eval` file, unedited, with `logs.json`, `eval-set.json`, `queue.jsonl`, `backend.jsonl` |
| `setup/` | The setup check of this commit |
| `replay/` | The laptop replay of the pilot's takeovers, from `explore/c2_replay.py`. Development only, on MLX |

## Reproduce

Check the numbers, offline and free: `uv sync`, then `uv run simtest summarize experiments/n2-pilot-c2/logs`.
Read the episodes: `uv run inspect view --log-dir experiments/n2-pilot-c2/logs`. To run it again:
`git checkout 29a3dbd`, set the variables in `.env.example`, then follow `docs/SESSIONS.md`.
