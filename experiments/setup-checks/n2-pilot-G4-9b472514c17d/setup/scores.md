# Bake-off scores

Real steps: 84; non-deterministic (reported apart): `pytest -q tests/test_pricing.py`, `date`, `ps aux`

## Fidelity on deterministic steps

| config | mode | n | exact | mean sim | median sim | median len | len outside 0.5-2 | steps with tells | pytest summary right | errors | truncated | no tag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 81 | 53% | 0.87 | 1.00 | 1.00 | 7% | 0% | 9/10 | 0 | 1 | 0 |

## Non-deterministic steps (mean sim)

| config | mode | `pytest -q tests/test_pricing.p` | `date` | `ps aux` |
|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 0.84 | 0.93 | 0.88 |

## Mean sim per sequence (deterministic steps)

| config | mode | s1-orient | s2-git | s3-read | s4-run | s5-edit-test | s6-heredoc | s7-errors | s8-probes |
|---|---|---|---|---|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 0.95 | 0.78 | 0.99 | 0.96 | 0.77 | 0.79 | 0.90 | 0.81 |

## Tells (steps flagged, all steps)

| config | mode | commentary | fence | apology_refusal | simulation_words | placeholder | elision | format_leak |
|---|---|---|---|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Speed (simulator loaded alone; 4 requests in parallel, 2 for agentworld-own and agentworld-think)

| config | mode | median wall s/step | p90 wall s | median completion tok | median reasoning tok | median tok/s | median prompt tok |
|---|---|---|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 41.3 | 123.0 | 3098 | 3092 | - | 4666 |
