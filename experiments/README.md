# Finished experiments

One folder per finished experiment (a paid run whose results went into the research log),
added when it is published:

| Path | What |
| --- | --- |
| `<run-name>/README.md` | The question, the answer in one sentence, the setup, the cost, how to reproduce, the commit it ran at |
| `<run-name>/FINDINGS.md` | The research log's entry for this run, as it is |
| `<run-name>/report/index.html` | The interactive findings page |
| `<run-name>/logs/` | Every `.eval` file of the run, unedited |

Each one is tagged `<run-name>`. The code that ran is the tree at the commit named in its
README. To check the numbers offline: `uv run simtest summarize experiments/<run-name>/logs`.
