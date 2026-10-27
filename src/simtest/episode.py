"""One episode = one Inspect sample: the agent loop as a custom solver.

Deliberately not `react()` (injects its own prompt and submit tool, executes calls
after submit) and not `execute_tools()` (truncates long outputs, validates
arguments with its own messages, aborts the batch on non-ToolError exceptions).

Per assistant message:
  phase 1  every tool call is executed against the SandboxEnv, in order, and
           recorded (ActionRecord + step) before any simulator call;
  phase 2  each true result is rendered by the condition's observer and sent
           back as a tool message (error=None, so the wire text is the observer's
           text in every condition).

After a submit, a scenario's follow-up question is put to the agent once (ask_followup);
the reply is recorded and nothing it calls is executed.

The loop only records facts: per step the true result and the exact observation.
The fidelity audit runs at analysis time (simtest summarize), never in the episode.
Per-episode state lives in the `Episode` store model, JSON only. The SandboxEnv
object never goes in the store (Inspect would log it as null); the final state
is stored as a JSON snapshot and the scorer rebuilds the env from it.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict
from typing import Any

from inspect_ai.log import transcript
from inspect_ai.model import ChatMessageSystem, ChatMessageTool, ChatMessageUser, Model, get_model
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolCall
from inspect_ai.util import LimitExceededError, StoreModel, span
from pydantic import Field

from .env import SandboxEnv, ToolResult
from .observers import IncompleteOutput, OutputError, incomplete_reason, is_infrastructure, make_observer
from .scenarios import Scenario, load_scenario
from .tools import TOOL_INFOS, TOOL_NAMES
from .unicode import clean, clean_text, has_lone_surrogate, sanitize_output
from .verifier import ActionRecord

NOT_EXECUTED_AFTER_SUBMIT = "not executed: the session was already submitted"


def not_executed(stop_reason: str) -> str:
    kind = {"max_tokens": "truncated", "model_length": "truncated", "content_filter": "filtered"}
    return f"not executed: the output was {kind.get(stop_reason, 'incomplete (stop reason ' + stop_reason + ')')}"

# Inspect parses provider tool arguments inside generate(); with Python's default limit
# (4300 digits) a longer JSON integer raises there, before we see the output. Raised to a
# large but finite limit (process-wide, set once): such a value parses and is an ordinary
# tool error (not a string). Longer still, the parser raises and the episode is flagged
# agent_output_error, never resampled.
INT_MAX_STR_DIGITS = 1_000_000


def ensure_int_digit_limit() -> None:
    current = sys.get_int_max_str_digits()
    if current != 0 and current < INT_MAX_STR_DIGITS:
        sys.set_int_max_str_digits(INT_MAX_STR_DIGITS)


# End reasons of a completed episode. Anything else (truncated/filtered/unknown-stop
# output of the agent or the simulator, verifier error, limit) is kept, flagged and
# excluded from rates. Infrastructure errors are Inspect sample errors (re-run).
VALID_END_REASONS = frozenset({"submitted", "no_tool_call", "max_turns"})


class Episode(StoreModel):
    condition: str = ""
    observer: str = ""
    actions: list[dict[str, Any]] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    final_state: dict[str, Any] = Field(default_factory=dict)
    simulator_calls: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # e.g. 'simulator:invalid_unicode'; never exclude
    followup: dict[str, Any] = Field(default_factory=dict)  # the follow-up question and reply, if asked
    end_reason: str = ""
    error: str | None = None
    error_status: int | None = None  # HTTP status of an output error, if any


def load_scenarios(specs: list[str]) -> dict[str, Scenario]:
    loaded: dict[str, Scenario] = {}
    for spec in specs:
        sc = load_scenario(spec)
        if sc.id in loaded:
            raise ValueError(f"duplicate scenario id {sc.id!r}")
        loaded[sc.id] = sc
    return loaded


def _repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"


def json_safe(value: Any, _depth: int = 0) -> Any:
    """JSON-only copy of parsed tool arguments; never raises.

    Inspect's lenient parser (YAML fallback) can yield dates, non-string keys, etc.
    Keys become strings (repr for non-strings); a value JSON can't hold becomes
    {"unserialisable": repr}, so the verifier still sees a non-string argument,
    consistent with the env rejecting it."""
    if isinstance(value, str):
        return clean_text(value)  # lone surrogates can't be serialised
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if _depth >= 50:
        return {"unserialisable": "<nested too deep>"}
    if isinstance(value, (list, tuple)):
        return [json_safe(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        return {clean_text(k) if isinstance(k, str) else _repr(k): json_safe(v, _depth + 1) for k, v in value.items()}
    return {"unserialisable": _repr(value)}


# Stable error texts: the true result is shown to the agent (C0) and to simulators, so
# it must not echo agent text (Inspect's parse diagnostic quotes the raw arguments; an
# unknown tool name is agent-authored). The raw diagnostic stays in the step record.
PARSE_ERROR = "invalid arguments: could not parse JSON arguments"
UNKNOWN_TOOL = "unknown tool"


def execute_call(env: SandboxEnv, call: ToolCall) -> ToolResult:
    """The current runner's rules, in this order."""
    if env.submitted is not None:
        # Logged (still an attempt if forbidden) but never executed or shown to a simulator.
        return ToolResult(False, error=NOT_EXECUTED_AFTER_SUBMIT)
    if call.parse_error:
        return ToolResult(False, error=PARSE_ERROR)
    if call.function not in TOOL_NAMES:
        return ToolResult(False, error=UNKNOWN_TOOL)
    return env.execute(call.function, call.arguments)


FOLLOWUP_PENDING = "the follow-up call did not complete"


def followup_pending(question: str) -> dict:
    """The record of a follow-up that was asked but has no reply yet. Saved before the call,
    so a sample cut off during it (a time limit; cancellation still propagates) counts the
    question as asked and unanswered."""
    return {"question": question, "reply": None, "submit_answers": [], "stop_reason": None, "tool_calls": [],
            "error": FOLLOWUP_PENDING, "error_status": None}


async def ask_followup(agent: Model, state: TaskState, question: str, notes: set[str]) -> dict:
    """Put one question to the agent after it submitted, and record the reply.

    The call offers the same tools as every other call: OpenRouter needs the tool
    definitions on every request whose history holds tool calls, and Anthropic rejects such
    a request without them (Inspect drops tools for tool_choice="none" on OpenRouter). But
    the episode is over, so nothing the agent calls is executed and the follow-up can't
    change a primary outcome. The record keeps the message text (`reply`) and the answers of
    any submit calls (`submit_answers`); the verifier reads the YES/NO from them. An
    infrastructure error is re-raised (sample error, re-run).
    Every other failure (a generate() error, an output with no choices, a sample limit
    reached) is recorded and makes the follow-up metrics NaN, without flagging the
    episode: its outcomes were fixed before the question, so a resample would only
    replace them."""
    record = {**followup_pending(question), "error": None}
    state.messages.append(ChatMessageUser(content=question))
    try:
        output = await agent.generate(input=state.messages, tools=TOOL_INFOS)
    except LimitExceededError as exc:  # Inspect control flow, but the episode is already complete
        return {**record, "error": f"LimitExceededError: {exc}"}
    except Exception as exc:
        if is_infrastructure(exc):
            raise
        error = OutputError("agent", exc)
        return {**record, "error": error.detail, "error_status": error.status}
    if output.empty:
        return {**record, "error": "agent returned no choices"}
    output, changed = sanitize_output(output)
    if changed:
        notes.add("agent:invalid_unicode")
    state.messages.append(output.message)
    calls = output.message.tool_calls or []
    answers = [c.arguments.get("answer") for c in calls if c.function == "submit" and isinstance(c.arguments, dict)]
    return {**record, "reply": output.message.text.strip(), "submit_answers": [a for a in answers if isinstance(a, str)],
            "stop_reason": output.stop_reason, "tool_calls": [c.function for c in calls]}


@solver
def simtest_agent(condition: str, scenarios: list[str], max_turns: int = 20) -> Solver:
    loaded = load_scenarios(scenarios)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        ensure_int_digit_limit()
        scenario = loaded[state.metadata["scenario"]]
        env = scenario.make_env()  # fresh for every attempt (retries reuse the sample uuid)
        simulator = get_model(role="simulator", required=True) if condition != "C0" else None
        observer = make_observer(condition, simulator)
        observer.reset(scenario, env)
        agent = get_model()

        episode = state.store_as(Episode)
        episode.condition, episode.observer = condition, observer.name
        actions: list[ActionRecord] = []
        steps: list[dict] = []
        notes: set[str] = set()
        followup: dict = {}

        def record(call: ToolCall, original: ToolCall, turn: int, result: ToolResult) -> dict:
            """Append the ActionRecord and step for a call (after it ran, if it ran).
            `call` is from the sanitised output; `original` is what the model returned
            (the env ran on its arguments), kept losslessly if it had to be cleaned."""
            n, args = len(actions), json_safe(call.arguments)
            # Keep the history sendable and loggable (e.g. dates from Inspect's YAML fallback).
            call.arguments = args
            actions.append(ActionRecord(n, call.function, args, result.ok, result.error))
            step = {"step": n, "turn": turn, "tool": call.function, "args": args,
                    "true_result": result.to_dict(), "observation": None}
            if call.parse_error:
                step["parse_error"] = call.parse_error  # private: never shown to anyone
            if result.gate:
                step["gate"] = result.gate  # private: the gate that rejected a write
            raw = {"id": original.id, "function": original.function, "arguments": original.arguments,
                   "parse_error": original.parse_error}
            if has_lone_surrogate(raw):
                step["call_raw"] = ascii(raw)  # lossless copy
            steps.append(step)
            return step

        def save(end_reason: str = "", error: str | None = None, error_status: int | None = None) -> None:
            # Final safety net: everything stored goes through the same string cleaner.
            episode.actions = clean([asdict(a) for a in actions])
            episode.steps = clean([dict(s) for s in steps])
            episode.final_state = clean(env.snapshot())
            episode.simulator_calls = clean(list(observer.log))
            episode.notes = sorted(notes)
            episode.followup = clean(followup)
            episode.end_reason, episode.error, episode.error_status = end_reason, clean(error), error_status

        state.messages.insert(0, ChatMessageSystem(content=scenario.system_prompt()))
        end_reason, output_error = "max_turns", None
        try:
            for turn in range(max_turns):
                try:
                    output = await agent.generate(input=state.messages, tools=TOOL_INFOS)
                except Exception as exc:
                    if is_infrastructure(exc):
                        raise
                    # generate() failed on the model's output (e.g. a provider-side parser):
                    # keep everything recorded so far and flag the episode, never resample.
                    output_error = OutputError("agent", exc)
                    end_reason = output_error.end_reason
                    break
                if output.empty:
                    raise RuntimeError("agent returned no choices")
                originals = list(output.message.tool_calls or [])  # what the env executes
                output, changed = sanitize_output(output)  # before anything else uses it
                if changed:
                    notes.add("agent:invalid_unicode")
                state.output = output
                state.messages.append(output.message)
                calls = list(zip(output.message.tool_calls or [], originals))
                if flag := incomplete_reason("agent", output.stop_reason):
                    # Kept and flagged. Its complete (parsed) calls are recorded as attempts
                    # but not executed; the episode ends without further generation.
                    # Unparseable calls stay in the raw transcript only: they can't be
                    # classified reliably, so they count as unknown.
                    for call, original in calls:
                        if original.parse_error is None:
                            record(call, original, turn, ToolResult(False, error=not_executed(output.stop_reason)))
                    end_reason = flag
                    break
                if not calls:
                    end_reason = "no_tool_call"
                    break

                # Phase 1: execute and record the whole batch, so a simulator failure
                # cannot erase attempts.
                batch = []
                for call, original in calls:
                    result = execute_call(env, original)  # original arguments: invalid UTF-8 is rejected
                    batch.append((call, result, record(call, original, turn, result)))
                save()

                # Phase 2: render observations, in order.
                for call, result, step in batch:
                    async with span(f"step {step['step']}: {call.function}", type="simtest_step"):
                        if result.error == NOT_EXECUTED_AFTER_SUBMIT:
                            observation = f"Error: {NOT_EXECUTED_AFTER_SUBMIT}"
                        else:
                            try:
                                observation = await observer.observe(call.function, call.arguments, result)
                            except OutputError as exc:
                                step["observer_error"] = exc.detail
                                raise
                            if observer.last_raw is not None:
                                step["observation_raw"] = observer.last_raw  # lossless copy
                                notes.add("simulator:invalid_unicode")
                        step["observation"] = observation
                        transcript().info(
                            clean({
                                "step": step["step"],
                                "tool": call.function,
                                "args": step["args"],
                                "true_result": step["true_result"],
                                "parse_error": step.get("parse_error"),
                                "observation": observation,
                                "observation_raw": step.get("observation_raw"),
                            }),
                            source="simtest",
                        )
                        state.messages.append(
                            ChatMessageTool(content=observation, tool_call_id=call.id, function=call.function)
                        )
                    save()
                if env.submitted is not None:
                    end_reason = "submitted"
                    break
            if end_reason == "submitted" and scenario.followup:
                # the outcomes are fixed before the question, and the question is on record as asked
                followup = followup_pending(scenario.followup["text"])
                save(end_reason)
                followup = await ask_followup(agent, state, scenario.followup["text"], notes)
        except IncompleteOutput as exc:  # simulator truncated/filtered/unknown stop/output error
            end_reason = exc.end_reason
            output_error = exc if isinstance(exc, OutputError) else None
        except LimitExceededError as exc:
            save("limit", str(exc))
            raise
        except TimeoutError as exc:
            # Inspect treats a bare TimeoutError reaching the sample as a completed
            # sample; make it an ordinary sample error so it is re-run.
            save("error", repr(exc))
            raise RuntimeError(f"timeout: {exc!r}") from exc
        except Exception as exc:  # infrastructure error: sample error, re-run on resume
            save("error", repr(exc))
            raise
        save(end_reason, output_error.detail if output_error else None, output_error.status if output_error else None)
        return state

    return solve
