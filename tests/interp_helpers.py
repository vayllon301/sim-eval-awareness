"""Fabricated Inspect samples and fake-tokenizer readings for the reader's tests. Their shape is a
real coding sample's: the agent's model events carry the exact request (the rawchat wire format) in
`call.request` and the server's counts in `output.usage`; simulator calls carry role "simulator"."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser, ContentReasoning, ContentText
from inspect_ai.tool import ToolCall

TOOLS = [{"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
                                           "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}]
SYSTEM, TASK = "You are a coding assistant.", "FLOW-1: fix the export."


def _usage(prompt, completion):
    return NS(input_tokens=prompt, output_tokens=completion)


def _event(messages, reasoning, text, calls, prompt, completion, stop="tool_calls", role=None, error=None):
    wire = {"model": "agent", "messages": [dict(m) for m in messages], "tools": TOOLS, "max_tokens": 32768}
    return NS(event="model", role=role, error=error, call=NS(request=wire, response={"choices": []}),
              output=NS(usage=_usage(prompt, completion), stop_reason=stop, empty=False))


def _assistant_wire(reasoning, text, calls):
    out = {"role": "assistant", "content": text, "reasoning": reasoning}
    if calls:
        out["tool_calls"] = [{"id": f"call-{i}", "function": {"name": c[0], "arguments": json.dumps(c[1])},
                              "type": "function"} for i, c in enumerate(calls)]
    return out


def _assistant(reasoning, text, calls):
    return ChatMessageAssistant(content=[ContentReasoning(reasoning=reasoning), ContentText(text=text)], tool_calls=[
        ToolCall(id=f"call-{i}", function=c[0], arguments=c[1]) for i, c in enumerate(calls)] or None)


def sample(turns=None, end="no_tool_call", uuid="u1", unread_output=False):
    """`turns`: (reasoning, text, [(tool, args, output)...]). The last turn ends the episode: with
    `unread_output` its calls got outputs that no turn read (the turn cap), as the loop leaves them."""
    turns = turns or [("Let me look.", "", [("bash", {"command": "ls"}, "a.txt")]),
                      ("One more.", "Checking.", [("bash", {"command": "cat a.txt"}, "hello"),
                                                  ("bash", {"command": "pwd"}, "/tmp")]),
                      ("Done now.", "The export is fixed.", [])]
    wire = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TASK}]
    msgs = [ChatMessageSystem(content=SYSTEM), ChatMessageUser(content=TASK)]
    events = [NS(event="sample_init")]
    for k, (reasoning, text, calls) in enumerate(turns):
        spec = [(c[0], c[1]) for c in calls]
        events.append(_event(wire, reasoning, text, spec, 100 + 50 * k, 20 + k, "tool_calls" if calls else "stop"))
        if k:   # a simulator call between two agent calls, in C2 and C3
            events.append(NS(event="model", role="simulator", error=None, output=NS(usage=_usage(9, 9), stop_reason="stop"),
                             call=NS(request={"messages": [{"role": "user", "content": "sim"}]}, response={})))
        wire.append(_assistant_wire(reasoning, text, spec))
        msgs.append(_assistant(reasoning, text, spec))
        last = k == len(turns) - 1
        for i, c in enumerate(calls):
            msgs.append(ChatMessageTool(content=c[2], tool_call_id=f"call-{i}", function=c[0])) if not last or unread_output else None
            wire.append({"role": "tool", "content": c[2], "tool_call_id": f"call-{i}"})
    return NS(events=events, messages=msgs, uuid=uuid, id="flows_audit", epoch=1, metadata={"scenario": "flows_audit"},
              store={"CodingEpisode:end_reason": end}, attachments={})
