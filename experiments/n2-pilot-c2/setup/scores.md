# Bake-off scores

Real steps: 84; non-deterministic (reported apart): `pytest -q tests/test_pricing.py`, `date`, `ps aux`

## Fidelity on deterministic steps

| config | mode | n | exact | mean sim | median sim | median len | len outside 0.5-2 | steps with tells | pytest summary right | errors | truncated | no tag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 81 | 56% | 0.89 | 1.00 | 1.00 | 6% | 0% | 9/10 | 0 | 1 | 0 |

## Non-deterministic steps (mean sim)

| config | mode | `pytest -q tests/test_pricing.p` | `date` | `ps aux` |
|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 0.84 | 0.93 | 0.88 |

## Mean sim per sequence (deterministic steps)

| config | mode | s1-orient | s2-git | s3-read | s4-run | s5-edit-test | s6-heredoc | s7-errors | s8-probes |
|---|---|---|---|---|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 0.94 | 0.78 | 1.00 | 0.94 | 0.81 | 0.91 | 0.91 | 0.82 |

## Tells (steps flagged, all steps)

| config | mode | commentary | fence | apology_refusal | simulation_words | placeholder | elision | format_leak |
|---|---|---|---|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Speed (simulator loaded alone; 4 requests in parallel, 2 for agentworld-own and agentworld-think)

| config | mode | median wall s/step | p90 wall s | median completion tok | median reasoning tok | median tok/s | median prompt tok |
|---|---|---|---|---|---|---|---|
| agentworld-think-vllm12 | forced | 34.2 | 125.9 | 2701 | 2624 | - | 4666 |
