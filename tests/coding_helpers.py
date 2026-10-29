"""Offline helpers for the coding-family tests: a scripted agent, a scripted simulator and a
fake machine (no Docker)."""

from __future__ import annotations

import shutil
import subprocess

from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.tool import ToolCall

from simtest.coding.machine import ExecResult


def tool_call_output(name: str, calls: list[ToolCall]) -> ModelOutput:
    msg = ChatMessageAssistant(content="", tool_calls=calls, model=name, source="generate")
    return ModelOutput(model=name, choices=[ChatCompletionChoice(message=msg, stop_reason="tool_calls")])


class ScriptedAgent:
    """A list of turns; each turn a list of (tool, args). Then it answers with a fixed text."""

    def __init__(self, turns, name="mockllm/agent", done="done"):
        self.turns, self.name, self.done = turns, name, done

    def __call__(self, input, tools, tool_choice, config):
        turn = sum(isinstance(m, ChatMessageAssistant) for m in input)
        if turn >= len(self.turns):
            return ModelOutput.from_content(self.name, self.done)
        calls = [ToolCall(id=f"c{turn}_{i}", function=t, arguments=a) for i, (t, a) in enumerate(self.turns[turn])]
        return tool_call_output(self.name, calls)


class ScriptedSimulator:
    """Returns the next line of `outputs` for each call (reasoning wrapped in <think>)."""

    def __init__(self, outputs, name="mockllm/sim"):
        self.outputs, self.name, self.i = list(outputs), name, 0

    def __call__(self, input, tools, tool_choice, config):
        text = self.outputs[min(self.i, len(self.outputs) - 1)]
        self.i += 1
        return ModelOutput.from_content(self.name, text)


FAKE_DATE = "Thu Oct 29 10:15:02 CET 2026"


class FakeMachine:
    """Implements the Machine protocol with scripted bash outputs (no Docker). `run` pops the
    next (output, exit_code) from `outputs`; harness `exec` answers `date` with `date` and
    anything else with empty success. `events` logs start, exec and run calls in order."""

    def __init__(self, outputs=None, info=None, date=FAKE_DATE):
        self.outputs = list(outputs or [])
        self.info = info or {"image": "fake", "container": "fake"}
        self.date = date
        self.commands: list[str] = []
        self.events: list[str] = []
        self.i = 0

    async def start(self):
        self.info["started"] = True
        self.events.append("start")

    async def run(self, command):
        self.commands.append(command)
        self.events.append(f"run:{command}")
        out, code = self.outputs[self.i] if self.i < len(self.outputs) else ("", 0)
        self.i += 1
        return ExecResult(out, code, False, 0.1)

    async def exec(self, command, user="root", timeout=180, workdir=None):
        self.events.append(f"exec:{command}")
        if command == "date":
            return ExecResult(f"{self.date}\n" if self.date else "", 0, False, 0.0)
        return ExecResult("", 0, False, 0.0)

    async def copy_in(self, src, dest):
        pass

    async def docker_diff(self):
        return ""

    async def start_read_watcher(self, paths):
        self.watched = list(paths)

    async def collect_reads(self):
        return list(getattr(self, "reads", []))

    async def stop(self):
        pass


def mock(outputs, name, **kw):
    return get_model(name, custom_outputs=outputs, memoize=False, **kw)


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0
