# N2 pilot, 05-11-2026

**The question.** Does the setup run on the reference backend with nothing lost, and do C2 and C3
behave as designed, before the main experiment (N3)?

**The answer.** The run is clean and C3 holds, but C2 fails its frozen rule: the simulator takes the
agent's role, so C2 has to be fixed before N3. There is no awareness rate yet: the frozen judge reads
these episodes only after it passes kappa on the hand labels, and that hasn't happened.

This is a calibration, never evidence for a comparison.

## Setup

- **Episodes:** 54. The flows dev set (3 pairs, so 6 scenarios) in C0, C2 and C3, 3 epochs, one queue,
  12 at once, 40 turns. Config: `configs/coding/pilot.yaml`, with its decision rule in the header.
- **Models:** the agent is Qwen3.8-27B and the simulator Qwen-AgentWorld-35B-A3B, both on vLLM 0.32.0.
- **Backend:** Colab G4 (NVIDIA RTX PRO 6000 Blackwell), simulator served as `fp8_per_channel` with
  `--moe-backend triton`. The server lines and the setup check are in `logs/backend.jsonl` and `setup/`.
- **Commit:** `8355f1c730968b6f55aea9842f52d54455bf7537`.
- **Cost:** 111.07 compute units, 11.2 hours, plus about 3.77 units for the setup check.

## What it found

| Frozen rule | Result |
| --- | --- |
| 18 of 18 episodes scored per condition | Pass: 0 flagged, 0 infrastructure reruns |
| Under 1% of C3's bash steps still a command line | Pass: 5 of 875 (0.6%) |
| C2's divergent-step rate under half of C3's | **Fail:** 363 of 726 against 740 of 875, ratio 0.591 |
| The awareness floor and power | Not available: no judge results yet |

C2's numbers measure a failure and not realism, so read them with `FINDINGS.md`. The 1% bar was set
aside at an interim 1.32%, then passed as frozen at the end; `FINDINGS.md` says so. The setup check's
bar also moved after a first FAIL, published in `experiments/setup-checks/`.

## Files

| Path | What |
| --- | --- |
| `FINDINGS.md` | The log entry for this run. Only the Google accounts' addresses are taken out |
| `report/index.html` | The findings page, every episode included |
| `logs/` | Every `.eval` file, unedited, with `logs.json`, `eval-set.json`, `queue.jsonl`, `backend.jsonl` |
| `setup/` | The setup check of this commit: `check.json`, `server.txt`, the forced pass, the `equiv` files |

## Reproduce

1. **Check the numbers, offline and free:** `uv sync`, then
   `uv run simtest summarize experiments/n2-pilot/logs` and `uv run simtest pilot experiments/n2-pilot/logs`.
2. **Read the episodes:** `uv run inspect view --log-dir experiments/n2-pilot/logs`, or open
   `report/index.html`.
3. **Run it again:** `git checkout 8355f1c`, set the variables in `.env.example`, then follow
   `docs/SESSIONS.md`. It needs a GPU with about 96 GB. Models served elsewhere won't match episode for
   episode.
