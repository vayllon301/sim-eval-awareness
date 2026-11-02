"""An episode as the agent saw it: the transcript that hand labellers and the LLM judge read
(PLAN.md, Measures).

`episode_steps(sample)` returns the episode as an ordered list of steps, and
`render_text(steps)` the plain text with step ids that the judge will be given. Both read
only `sample.messages`, which is exactly what the agent was sent:

- the system prompt and the task message;
- per agent turn (one assistant message): its reasoning (the raw chain of thought), its
  visible text, its tool calls, and the tool output it got for each call.

In C1-C3 the tool output is the simulator's (the observation the loop sent back), never
the shadow machine's real output: the real output lives only in the store
(`CodingEpisode:steps[i].result.output`), which is never read here. Nothing else of the
sample is read either: no condition, run, scenario id, simulator, model name, score,
store, sample metadata, timestamps or tool-call ids (their format differs by backend).

Step ids are stable: `system`, `task`, then `t1`, `t2`, ... one per agent turn in order.
They depend only on the order of the messages. A later user message (none in this
testbed) would be its own step, `user2`, `user3`, ... Content this module doesn't know
(an image, say) raises TranscriptError rather than being dropped.
"""

from __future__ import annotations

import hashlib
import json

from inspect_ai.model import ContentReasoning, ContentText

from ..unicode import clean

RENDER_VERSION = 1   # bump when the steps or the text change; labels record it


class TranscriptError(ValueError):
    pass


def _text(content) -> str:
    """The text of a system, user or tool message."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if not isinstance(p, ContentText):
            raise TranscriptError(f"unsupported content in a message: {type(p).__name__}")
        parts.append(p.text or "")
    return "".join(parts)


def _assistant_parts(msg) -> tuple[str, str, bool]:
    """(reasoning, visible text, reasoning redacted). A redacted reasoning block is opaque
    provider data: only its summary, if any, is kept."""
    content = msg.content
    if isinstance(content, str):
        return "", content, False
    reasoning, text, redacted = [], [], False
    for p in content:
        if isinstance(p, ContentReasoning):
            if p.redacted:
                redacted = True
                if p.summary:
                    reasoning.append(p.summary)
            elif p.reasoning:
                reasoning.append(p.reasoning)
        elif isinstance(p, ContentText):
            text.append(p.text or "")
        else:
            raise TranscriptError(f"unsupported content in an agent turn: {type(p).__name__}")
    return "\n\n".join(r for r in reasoning if r), "".join(text), redacted


def _arguments(call) -> dict:
    args = call.arguments
    if isinstance(args, dict):
        return json.loads(json.dumps(args, default=str))
    return {"_value": str(args)}


def episode_steps(sample) -> list[dict]:
    """The episode as ordered steps (JSON-ready dicts):

        {"id": "system", "kind": "system", "text": ...}
        {"id": "task", "kind": "task", "text": ...}
        {"id": "t3", "kind": "turn", "turn": 3, "reasoning": ..., "reasoning_redacted": bool,
         "text": ..., "calls": [{"n": 1, "function": "bash", "arguments": {...},
                                 "parse_error": None, "output": ... or None}]}

    `output` is the tool message the agent got for that call (None if it got none, as when
    the episode ended after the call)."""
    steps: list[dict] = []
    users = turns = 0
    current: dict | None = None        # the latest agent turn
    by_id: dict[str, dict] = {}        # its calls, by tool-call id
    for m in sample.messages:
        if m.role == "system":
            if steps:
                raise TranscriptError("a system message after the start of the episode")
            steps.append({"id": "system", "kind": "system", "text": _text(m.content)})
        elif m.role == "user":
            users += 1
            first = users == 1
            steps.append({"id": "task" if first else f"user{users}", "kind": "task" if first else "user",
                          "text": _text(m.content)})
            current = None
        elif m.role == "assistant":
            turns += 1
            reasoning, text, redacted = _assistant_parts(m)
            calls = []
            by_id = {}
            for i, c in enumerate(m.tool_calls or [], start=1):
                call = {"n": i, "function": c.function, "arguments": _arguments(c),
                        "parse_error": c.parse_error or None, "output": None}
                calls.append(call)
                if c.id:
                    by_id[c.id] = call
            current = {"id": f"t{turns}", "kind": "turn", "turn": turns, "reasoning": reasoning,
                       "reasoning_redacted": redacted, "text": text, "calls": calls}
            steps.append(current)
        elif m.role == "tool":
            if current is None:
                raise TranscriptError("a tool message before any agent turn")
            call = by_id.get(m.tool_call_id) if m.tool_call_id else None
            if call is None:   # no id match: the first call still without an output
                call = next((c for c in current["calls"] if c["output"] is None), None)
            if call is None or call["output"] is not None:
                raise TranscriptError(f"turn {current['turn']}: a tool message with no call to answer")
            call["output"] = _text(m.content)
        else:
            raise TranscriptError(f"unknown message role {m.role!r}")
    return clean(steps)   # lone surrogates (none expected: the loop sanitises outputs) can't break the text


def step_ids(steps: list[dict]) -> list[str]:
    return [s["id"] for s in steps]


def turn_ids(steps: list[dict]) -> list[str]:
    """The agent turns' ids: where awareness can show (the onset step is one of them)."""
    return [s["id"] for s in steps if s["kind"] == "turn"]


_HEAD = {"system": "system prompt", "task": "task message", "user": "user message"}


def _call_body(call: dict) -> str:
    args = call["arguments"]
    if call["function"] == "bash" and set(args) == {"command"} and isinstance(args["command"], str):
        body = args["command"]
    else:
        body = json.dumps(args, ensure_ascii=False, indent=1)
    if call["parse_error"]:
        body += f"\n(arguments could not be parsed: {call['parse_error']})"
    return body


def render_text(steps: list[dict]) -> str:
    """The plain-text transcript, one block per step, each headed by its step id. Empty
    reasoning or visible text is left out; an empty tool output is shown as (empty)."""
    out = []
    for s in steps:
        if s["kind"] != "turn":
            out.append(f"=== [{s['id']}] {_HEAD[s['kind']]} ===\n{s['text']}")
            continue
        block = [f"=== [{s['id']}] agent turn {s['turn']} ==="]
        if s["reasoning_redacted"]:
            block.append("--- reasoning (redacted by the provider" + ("; its summary) ---" if s["reasoning"] else ") ---"))
            if s["reasoning"]:
                block.append(s["reasoning"])
        elif s["reasoning"]:
            block += ["--- reasoning ---", s["reasoning"]]
        if s["text"]:
            block += ["--- visible text ---", s["text"]]
        for c in s["calls"]:
            block += [f"--- tool call {c['n']}: {c['function']} ---", _call_body(c)]
            if c["output"] is not None:
                block += [f"--- tool output {c['n']} ---", c["output"] if c["output"] != "" else "(empty)"]
        out.append("\n".join(block))
    return "\n\n".join(out) + "\n"


def transcript_sha256(steps: list[dict]) -> str:
    return hashlib.sha256(render_text(steps).encode()).hexdigest()
