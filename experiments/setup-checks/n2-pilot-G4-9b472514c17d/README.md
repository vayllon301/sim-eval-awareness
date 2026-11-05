# Setup check on a Colab G4, 05-11-2026: FAIL

The first real run of the setup check (`src/simtest/` and `explore/sim_quant.py`), before any
pilot episode. **Verdict: FAIL.** No episode ran and the VM stopped.

- **Commit:** `9b472514c17d964834044ad2acf46f5c0a777d99`
- **Backend:** Colab G4 (NVIDIA RTX PRO 6000 Blackwell), vLLM 0.32.0, simulator served as
  `fp8_per_channel` with `--moe-backend triton`, 12 requests at once.
- **Why it failed:** mean similarity 0.8667 on the 81 deterministic steps, against a bar of 0.892.
  The scheme (every layer `fp8_per_channel_static`), truncations (1 of 84 at the 16K cap) and
  evidence all passed.
- **What was decided afterwards:** the 0.8667 was accepted as sufficient and the bar moved to
  0.847 (0.867 minus 0.02) in `889cbc6`. **That bar was set after seeing this data.** The FAIL is
  published as it came out; it was not turned into a pass. A later check on a later commit runs in
  its own directory, and both are reported.

Also measured (recorded, no bar): speed 917 tokens/s at 12 at once; noise floors, 1 against 12 at
once: agent mean |Δ log-prob| 0.383, top-1 0.910; simulator 0.029, top-1 0.985.

## Files

| Path | What |
| --- | --- |
| `setup/check.json` | The verdict and every number behind it |
| `setup/scores.md`, `scores.jsonl` | The bake-off scores of the forced pass |
| `setup/sim-agentworld-think-vllm12-forced.jsonl` | The simulator's outputs for the 84 steps |
| `setup/real.jsonl` | The real outputs the simulator is compared with |
| `setup/speed.jsonl` | Throughput windows |
| `setup/equiv-c1.jsonl`, `equiv-c12.jsonl` | Log-probs on the same contexts at 1 and at 12 requests at once |
| `setup/server.txt`, `snapshot.txt` | The servers' command lines and start-up lines; the machine |
| `equiv/contexts.jsonl` | The contexts the `equiv` files are measured on (sha256 `bd65162b…65bf5` is in their headers) |
| `session/` | The session's server lines, `serve.log`, vLLM versions and GPU |

Unedited copies of `logs/setup/n2-pilot-G4-9b472514c17d/`. Scanned for keys and e-mail addresses;
none found.
