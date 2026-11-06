# C2 reference preparation

Development preparation for the controlled replay, not a model result.

- Protocol: [`explore/C2_CONTROLLED_REPLAY.md`](../../explore/C2_CONTROLLED_REPLAY.md).
- Configuration: [`configs/coding/c2_controlled_replay.json`](../../configs/coding/c2_controlled_replay.json).
- Code and reference commit: `9715e8bf29980e796d222707f50cbc062b50642b`.
- Source: the completed N2 pilot's real C0 logs; source-file SHA-256 hashes are in the audit.
- [`reference-audit.json`](reference-audit.json) records each reference's size, overlap with
  its initial snapshot, conservative command classification and content hashes.

The audit is generated from the same `prepare` function that freezes the replay inputs.
Line overlap is only a redundancy diagnostic. It does not measure whether information
is useful, whether an output is realistic, or whether an agent recognises an evaluation.
No new simulator answers or judge labels are part of this artifact.
