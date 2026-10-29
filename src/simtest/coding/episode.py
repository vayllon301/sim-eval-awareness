"""One coding episode = one Inspect sample: the agent loop as a custom solver.

At the start, after the machine starts and before the agent's first turn, the loop reads
`date` on the real box (harness work, in every condition) and stores it as `session_start`;
the C2 and C3 simulators get it in the snapshot's facts block (snapshot.session_snapshot).

Each assistant message, in two phases like the office testbed (src/simtest/episode.py):
  phase 1  every tool call is run on the real machine (shadow execution) and recorded
           before any simulator call, so a simulator failure can't erase an attempt;
  phase 2  each bash result is rendered by the condition's observer and sent back; the
           agent sees C0's real output or the simulator's.

bash runs on the shadow machine in every condition; send_message is recorded and answered
locally (nothing is sent anywhere). There is no submit tool: as in a real coding agent, the
episode ends when the agent replies without a tool call (end reason `no_tool_call`, the
normal finish), and that reply's visible text is stored as `final_message`, its summary for
the ticket. A call to any other tool (a hallucinated `submit` included) is answered as an
unknown tool. After the loop, the behaviour probe runs on the real machine and stores the raw
facts; the scorer turns them into metrics at analysis time, so the loop only records facts
and a verifier change can be re-applied to old logs (as in the office testbed).

The agent's earlier reasoning is sent back in full by the rawchat provider
(coding/provider.py). Nothing model-produced may raise out of the loop: an output that
isn't a clean finish is kept and flagged, like the office testbed.
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageSystem, ChatMessageTool, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import LimitExceededError, StoreModel, span
from pydantic import Field

from ..observers import IncompleteOutput, OutputError, incomplete_reason, is_infrastructure
from ..unicode import clean, has_lone_surrogate, sanitize_output
from .checks import baseline, probe
from .machine import MachineError, Machine
from .observers import make_observer
from .prompts import TOOL_NAMES
from .scenario import CodingScenario, load_scenario
from .snapshot import session_start

# How a valid episode ends: the agent replies without a tool call (its normal finish), or it
# reaches the turn cap. Anything else is flagged and excluded from rates. The paired
# difference is shared with the office analysis, whose valid end reasons include these
# (tested in test_coding_analysis).
FINISHED, MAX_TURNS = "no_tool_call", "max_turns"
VALID_END_REASONS = frozenset({FINISHED, MAX_TURNS})
MESSAGE_SENT = "Message sent."


class CodingEpisode(StoreModel):
    condition: str = ""
    observer: str = ""
    scenario: str = ""
    variant: str = ""
    machine: dict[str, Any] = Field(default_factory=dict)
    session_start: str = ""   # `date` on the real box at episode start; C2/C3 see it in the snapshot
    steps: list[dict[str, Any]] = Field(default_factory=list)   # per step: tool, args, real result, observation
    simulator_calls: list[dict[str, Any]] = Field(default_factory=list)
    checks_raw: dict[str, Any] = Field(default_factory=dict)    # raw facts from the probe; the scorer evaluates them
    final_message: str | None = None   # the visible text of the reply that ended the episode (not its reasoning)
    notes: list[str] = Field(default_factory=list)
    end_reason: str = ""
    error: str | None = None
    error_status: int | None = None


def loaded_scenarios(specs: list[str]) -> dict[str, CodingScenario]:
    out: dict[str, CodingScenario] = {}
    for spec in specs:
        sc = load_scenario(spec)
        if sc.id in out:
            raise ValueError(f"duplicate scenario id {sc.id!r}")
        out[sc.id] = sc
    return out


def _args(raw: str) -> tuple[dict | None, str | None]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        return None, f"could not parse JSON arguments: {exc}"
    return (value, None) if isinstance(value, dict) else (None, "arguments must be a JSON object")


def not_executed(stop_reason: str) -> str:
    kind = {"max_tokens": "truncated", "model_length": "truncated", "content_filter": "filtered"}
    return f"[not run: the reply was {kind.get(stop_reason, 'incomplete')}]"


@solver
def coding_agent(condition: str, scenarios: list[str], machine_factory, max_turns: int = 40,
                 deadline: int = 60) -> Solver:
    loaded = loaded_scenarios(scenarios)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        scenario = loaded[state.metadata["scenario"]]
        simulator = get_model(role="simulator", required=True) if condition != "C0" else None
        observer = None  # made once the session start is known (C2/C3 show it in the snapshot)
        agent = get_model()
        machine: Machine = machine_factory(scenario, deadline)

        episode = state.store_as(CodingEpisode)
        episode.condition = condition
        episode.scenario, episode.variant = scenario.id, scenario.variant
        steps: list[dict] = []
        notes: set[str] = set()

        def save(end_reason: str = "", error: str | None = None, status: int | None = None) -> None:
            episode.machine = clean(machine.info)
            episode.steps = clean([dict(s) for s in steps])
            episode.simulator_calls = clean(list(getattr(observer, "records", [])))
            episode.notes = sorted(notes)
            episode.end_reason, episode.error, episode.error_status = end_reason, clean(error), status

        state.messages.insert(0, ChatMessageSystem(content=scenario.system_prompt()))
        end_reason, output_error = MAX_TURNS, None
        started = False
        try:
            await machine.start()
            started = True
            base = await baseline(machine, scenario)
            episode.checks_raw = {"baseline_error": base.get("errors")} if base.get("errors") else {}
            episode.session_start = await session_start(machine, scenario)
            observer = make_observer(condition, simulator, scenario, session_start=episode.session_start)
            episode.observer = observer.name
            for turn in range(max_turns):
                try:
                    output = await agent.generate(input=state.messages, tools=scenario.tool_infos())
                except Exception as exc:
                    if is_infrastructure(exc):
                        raise
                    output_error = OutputError("agent", exc)
                    end_reason = output_error.end_reason
                    break
                if output.empty:
                    raise RuntimeError("agent returned no choices")
                output, changed = sanitize_output(output)
                if changed:
                    notes.add("agent:invalid_unicode")
                state.output = output
                state.messages.append(output.message)
                calls = list(output.message.tool_calls or [])
                if flag := incomplete_reason("agent", output.stop_reason):
                    for call in calls:
                        steps.append(_step(len(steps), turn, call, executed=False,
                                           result={"not_executed": output.stop_reason}))
                        state.messages.append(ChatMessageTool(content=not_executed(output.stop_reason),
                                                              tool_call_id=call.id, function=call.function))
                    end_reason = flag
                    break
                if not calls:
                    end_reason = FINISHED
                    episode.final_message = clean(output.message.text)
                    break

                # Phase 1: run every call on the real machine, record before any simulator call.
                batch = []
                for call in calls:
                    step = _step(len(steps), turn, call)
                    args, arg_error = _args(_raw_arguments(call))
                    real = None
                    if call.function == "send_message":
                        step.update(executed=True, result={"ok": arg_error is None, "error": arg_error})
                    elif call.function == "bash":
                        command = (args or {}).get("command")
                        if arg_error or not isinstance(command, str):
                            step.update(executed=False, result={"error": arg_error or "command must be a string"})
                        else:
                            real = await machine.run(command)
                            step.update(executed=True, command=command, result=real.to_dict())
                    else:
                        step.update(executed=False, result={"error": f"unknown tool {call.function!r}"})
                    steps.append(step)
                    batch.append((call, step, args, arg_error, real))
                save()

                # Phase 2: answer each call (bash through the observer).
                for call, step, args, arg_error, real in batch:
                    async with span(f"step {step['step']}: {call.function}", type="simtest_step"):
                        observation = await _answer(observer, call, step, args, arg_error, real, notes)
                        step["observation"] = observation
                        transcript().info(clean({k: step.get(k) for k in
                                                  ("step", "tool", "command", "args", "result", "observation")}),
                                          source="simtest")
                        state.messages.append(ChatMessageTool(content=observation, tool_call_id=call.id,
                                                              function=call.function))
                    save()
            save(end_reason, output_error.detail if output_error else None,
                 output_error.status if output_error else None)
            # The behaviour probe runs after the loop, on the real final state (any end reason).
            if started and not base.get("errors"):
                episode.checks_raw = clean(await probe(machine, scenario, base))
        except IncompleteOutput as exc:  # a simulator output that wasn't a clean finish
            if record := getattr(exc, "record", None):
                observer.records.append(record)
            end_reason = exc.end_reason
            output_error = exc if isinstance(exc, OutputError) else None
            save(end_reason, output_error.detail if output_error else None,
                 output_error.status if output_error else None)
        except (LimitExceededError, MachineError) as exc:
            save("error", repr(exc))
            raise
        except Exception as exc:
            save("error", repr(exc))
            raise
        finally:
            if started:
                try:
                    await machine.stop()
                except Exception:
                    pass
        save(end_reason, output_error.detail if output_error else None,
             output_error.status if output_error else None)
        return state

    return solve


def _raw_arguments(call) -> str:
    """The raw JSON string of a tool call's arguments (Inspect parses them to a dict)."""
    if isinstance(call.arguments, str):
        return call.arguments
    try:
        return json.dumps(call.arguments)
    except (TypeError, ValueError):
        return "{}"


def _step(n: int, turn: int, call, executed: bool = False, result: dict | None = None,
          command: str | None = None) -> dict:
    step = {"step": n, "turn": turn, "tool": call.function if call.function in TOOL_NAMES else "unknown",
            "tool_raw": call.function, "args": _safe_args(call.arguments), "executed": executed,
            "command": command, "result": result, "observation": None}
    if has_lone_surrogate(step["args"]):
        step["args_raw"] = ascii(call.arguments)
    return step


def _safe_args(args) -> dict:
    return clean(args) if isinstance(args, dict) else {"_value": clean(repr(args))}


async def _answer(observer, call, step, args, arg_error, real, notes: set[str]) -> str:
    if call.function == "send_message":
        return f"Error: {arg_error}" if arg_error else MESSAGE_SENT
    if call.function != "bash":
        return f"Error: {step['result'].get('error', 'unknown tool')}"
    if real is None:
        return f"Error: {step['result'].get('error', 'invalid arguments')}"
    try:
        observation, record = await observer.observe(step["command"], real)
    except (IncompleteOutput, OutputError) as exc:
        if record := getattr(exc, "record", None):
            observer.records.append(record)
        raise
    if record is not None:
        observer.records.append(record)
        if record.get("invalid_unicode"):
            notes.add("simulator:invalid_unicode")
    return observation
