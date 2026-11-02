# Vendored judge code

Files here are kept exactly as published. Don't edit them.

| File | Source | Licence |
| --- | --- | --- |
| `scout_eval_awareness.py` | [`meridianlabs-ai/inspect_scout`](https://github.com/meridianlabs-ai/inspect_scout), `examples/docs/eval_awareness.py` at commit `2653f933d634f3f7e8f20c7d49ba3ff26f30fd16` (the same file at tag `0.6.2`), git blob `858cf99c747feb00530176ab4e594ae808ab298d` | MIT, Copyright (c) 2025 Meridian Labs: `LICENSE-inspect_scout` |

- Scout ships this scanner as a documentation example, not in the package, so it is copied
  here. The package itself (`inspect-scout`) is a dependency.
- `judge/scout.py` checks the file's git blob hash before every run and refuses a changed
  file. A test checks it too.
- How the second judge uses it: `docs/TESTBED.md`, "The second judge".
