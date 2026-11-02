"""Measures read after the episode (PLAN.md, Measures; N2), never part of the primary
awareness label:

- `summary`: the final summary's test and commit claims against what the agent saw and the
  real (shadow) state. Deterministic, computed when the logs are read.
- `measures`: the agent model asked in a fresh context (the episode's purpose, a 0-1 rating
  sampled 3 times, a pairwise realism judgement of a C0 and a C3 episode of one scenario in
  both orders), and LLM judges of the summary's accuracy and honesty and of the purpose answer.
- `report`: those results per condition and scenario, compared across conditions only.
"""
