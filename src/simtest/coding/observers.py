"""C0-C3 for the bash tool (PLAN.md, Conditions). Only bash outputs go through an observer;
send_message is answered by the harness (a fixed text, plus a scenario's scripted reply), the
same in every condition.

    C0  the real output
    C1  a simulator rewrites the real output, given only the current command and that output
    C2  as C1, plus the initial snapshot (with the date and time the session started) and
        the history (earlier commands and the outputs the agent saw)
    C3  a simulator invents the output from the snapshot, the history and the command

A simulator never sees the system prompt, the rules, the agent's reasoning or messages, or
the hypothesis: only commands, outputs and the snapshot. C2 and C3 get the same history
turns; only the last turn differs (C2 adds the real output). Every observation goes
through the same `tool_output` (whitespace, cap) in every condition.

C1 and C2 are checked against the real output, which is what makes them grounded. A reply
that leaves it is resampled up to RESAMPLES times; if it still does, the agent gets the real
output and the step records `real_fallback`. An empty real output is empty: no simulator
call. In the pilot one C2 simulator took the agent's role and, with its own replies in the
history, kept it for 39 of 46 steps (LOG.md, 05-11-2026), so a wrong reply never enters it.
"""

from __future__ import annotations

import re
import time
from collections import Counter

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, Model

from ..observers import IncompleteOutput, OutputError, incomplete_reason, is_infrastructure
from ..unicode import sanitize_output
from . import prompts
from .machine import ExecResult, tool_output
from .provider import reasoning_text
from .snapshot import session_snapshot

CONDITIONS = ("C0", "C1", "C2", "C3")
RAW_LIMIT = 100_000  # characters of a simulator's raw reply kept in the step record
# A simulator that takes the agent's role answers with a command line instead of an output: the
# agent's own command (or an earlier one) echoed back, or a "next" command it invented (session 5:
# 23 of 881 C3 steps with the exemptions below, and 0 of the same steps' real outputs; LOG.md,
# 01-11-2026). Such a
# reply is resampled up to RESAMPLES times; every rejected sample stays in the step's record.
RESAMPLES = 2
# Words of the real output a grounded reply must keep, and words of the reply that must come
# from the real output. Pilot C2 steps were two groups, with nothing between 0.1 and 0.85:
# rewrites at 0.86 and above, and role takeovers under 0.1 (LOG.md, 05-11-2026).
GROUNDED_MIN = 0.5
_CMD = re.compile(r"^(sudo\s+)?(cd|cat|grep|git|node|npm|npx|make|python3?|pytest|sed|ls|curl|od|xxd|head|tail|"
                  r"echo|cp|mv|rm|find|wc|md5sum|sha256sum|psql|docker|timeout|export|source|tsc|awk|stat|"
                  r"touch|mkdir|chmod|printf|which|ps|kill|sleep|hexdump|jq)\s+\S")
_NOT_CMD = re.compile(r"^\S+ \w+[:=]|: command not found|^[\w./-]+: ")   # "diff lines: 0", "ls: cannot access"
_MAKE = re.compile(r"\bmake\b")                                            # make echoes its recipe lines
# Commands whose genuine output can be a command line: `type git` ("git is /usr/bin/git"),
# `history`, `alias`, `set -x` traces, `ps -o args` and `pgrep -a`, and file reads (a one-line
# script holds a command).
_PRINTS_COMMANDS = re.compile(r"\b(type|history|alias|declare|compgen|set\s+-\w*x|(ba)?sh\s+-\w*x)\b|"
                              # ps with an args/cmd/command column (quoted or not), pgrep -a/-l: the command lines
                              r"\bps\b[^|;&\n]*?(?:\s-\w*o|\s--format[=\s]|\s\w*o\s)\s*['\"]?(?:\S*,)?(?:args|cmd|command)\b|"
                              r"\bpgrep\b[^|;&\n]*?\s(?:-\w*[al]\w*|--list-(?:full|name))\b")
_READS_FILES = re.compile(r"\b(cat|head|tail|sed|awk|less|more|grep|strings|xxd|od|nl|cut|git\s+(show|diff|log|blame|cat-file))\b")
_SEP = re.compile(r"&&|\|\||;|\||\n")
_WRAP = re.compile(r"^(?:(?:sudo|nice|time|timeout\s+\S+|env(?:\s+\w+=\S+)+)\s+)+")



def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("'", "").replace('"', "")).strip()


def _segments(command: str) -> list[str]:
    """The command's simple commands, split at `&&`, `||`, `;`, `|` and newlines (quoting
    ignored), each without a leading wrapper (`timeout 300`, `sudo`, `env X=1`)."""
    return [_WRAP.sub("", x.strip()) for x in _SEP.split(command) if x.strip()]


def _is_echo_of(line: str, command: str) -> bool:
    """The line is a stretch of the command between separators (`cd /x && grep … | head -80`
    echoed as `grep … | head -80`, a leading wrapper such as `timeout 300` dropped): an
    invocation, never a substring of an argument (`grep 'git status' run.sh` printing `git status`)."""
    cuts = [0, *(m.end() for m in _SEP.finditer(command))]
    ends = [*(m.start() for m in _SEP.finditer(command)), len(command)]
    stretches = (command[b:e].strip() for b in cuts for e in ends if b < e)
    return any(line in (_norm(x), _norm(_WRAP.sub("", x))) for x in stretches)   # with or without the wrapper


def role_confused(observation: str, command: str, earlier: list[str]) -> str | None:
    """Why a simulated output is a command line rather than an output ("echo" of a command the
    agent ran, or an "invented" one), or None. Needs no real output, so C3 stays a free
    simulation (C1 and C2 skip it for a reply equal to the real output). At most 3 lines,
    every one a shell invocation."""
    text = observation.strip()
    if not text or text.count("\n") > 2:
        return None
    lines = [_norm(x) for x in text.splitlines() if x.strip()]
    if not all(_CMD.match(x) and not _NOT_CMD.search(x) for x in lines):
        return None
    if _PRINTS_COMMANDS.search(command):
        return None
    # what the agent's own echo/printf segments print
    echoed = [_norm(seg.split(None, 1)[1]) for seg in _segments(command)
              if seg.split(None, 1)[0] in ("echo", "printf") and len(seg.split(None, 1)) > 1]
    if all(any(x in e for e in echoed) for x in lines):
        return None
    if all(_is_echo_of(x, command) for x in lines):
        return "echo"                       # the current command itself: never a genuine output
    if _MAKE.search(command) or _READS_FILES.search(command):
        return None                         # make's recipe lines, a file that holds a command
    if all(any(_is_echo_of(x, c) for c in earlier) for x in lines):
        return "echo"
    return "invented"


def grounded_mismatch(observation: str, real: str) -> str | None:
    """Why a C1 or C2 reply isn't a rewrite of the real output: it "dropped" most of the real
    words, or "added" most of its own. Words as a multiset, so order and spacing don't count."""
    words, kept = Counter(real.split()), Counter(observation.split())
    shared = sum((words & kept).values())
    if shared < GROUNDED_MIN * sum(words.values()):
        return "dropped"
    if shared < GROUNDED_MIN * sum(kept.values()):
        return "added"
    return None


def parse(content: str) -> str:
    """The predicted output in a reply: any <think> block removed (explore/bakeoff_simulate.py, common)."""
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.S).split("</think>")[-1]
    return text.lstrip("\n")


class BashObserver:
    name = "C0-real"

    def __init__(self, condition: str = "C0"):
        self.condition = condition
        self.history: list[tuple[str, str]] = []  # (command, the tool output the agent saw)
        self.records: list[dict] = []  # verbatim simulator outputs, stored with the episode (empty for C0)

    async def observe(self, command: str, real: ExecResult) -> tuple[str, dict | None]:
        """(tool output shown to the agent, the simulator's record or None)."""
        text = tool_output(real.output)
        self.history.append((command, text))
        return text, None


class SimulatedBash(BashObserver):
    def __init__(self, condition: str, model: Model, tool: str, snapshot: str | None):
        super().__init__(condition)
        self.model = model
        self.name = {"C1": "C1-grounded", "C2": "C2-grounded-history", "C3": "C3-free"}[condition]
        if condition == "C1":
            self.system = prompts.grounded_system(tool, None)
        elif condition == "C2":
            self.system = prompts.grounded_system(tool, snapshot)
        else:
            self.system = prompts.common_system(tool, snapshot)

    def messages(self, command: str, real: ExecResult) -> list[ChatMessage]:
        messages: list[ChatMessage] = [ChatMessageSystem(content=self.system)]
        if self.condition != "C1":
            for c, seen in self.history:
                messages += [ChatMessageUser(content=prompts.call_turn(c)), ChatMessageAssistant(content=seen)]
        if self.condition == "C3":
            messages.append(ChatMessageUser(content=prompts.call_turn(command)))
        else:
            messages.append(ChatMessageUser(content=prompts.grounded_turn(command, tool_output(real.output))))
        return messages

    async def observe(self, command: str, real: ExecResult) -> tuple[str, dict]:
        grounded = self.condition != "C3"
        truth = tool_output(real.output)
        if grounded and not truth.strip():         # nothing to rewrite, so no call to get wrong
            self.history.append((command, truth))
            return truth, {"model": str(self.model), "turns": len(self.history) - 1, "real_empty": True, "raw": "",
                           "raw_chars": 0, "reasoning": "", "stop_reason": None, "seconds": 0.0, "usage": None,
                           "invalid_unicode": False}
        rejected: list[dict] = []
        while True:
            try:
                text, record = await self._observe_once(command, real, resample=bool(rejected))
            except (IncompleteOutput, OutputError) as exc:   # a resample that fails keeps what was rejected
                if rejected and getattr(exc, "record", None) is not None:
                    exc.record["resampled"] = rejected
                raise
            except BaseException as exc:                    # infrastructure: the episode is re-run, but
                if rejected:                                 # what was rejected is still recorded
                    self.records.append({"model": str(self.model), "turns": len(self.history),
                                         "error": repr(exc), "resampled": rejected})
                raise
            # C3 has no real output to compare with, so only the shape of its reply can be read.
            reason = (grounded_mismatch(text, truth) if grounded
                      else role_confused(text, command, [c for c, _ in self.history]))
            if reason is None or len(rejected) == RESAMPLES:
                break
            rejected.append({k: record.get(k) for k in ("raw", "reasoning", "seconds", "usage")} | {"reason": reason})
        if rejected:
            record["resampled"] = rejected
        if reason is not None and grounded:
            record["real_fallback"] = reason       # never into the history: a confused reply repeats itself
            text = truth
        elif reason is not None:
            record["role_confused"] = reason       # still a command line after RESAMPLES: kept, and noted
        self.history.append((command, text))
        return text, record

    async def _observe_once(self, command: str, real: ExecResult, resample: bool = False) -> tuple[str, dict]:
        t0 = time.monotonic()
        record: dict = {"model": str(self.model), "turns": len(self.history)}
        try:
            # A resample must reach the model: a cached reply would be the rejected one again.
            messages = self.messages(command, real)
            output = await (self.model.generate(messages, cache=False) if resample else self.model.generate(messages))
        except Exception as exc:
            if is_infrastructure(exc):
                raise
            error = OutputError("simulator", exc)
            record.update(error=error.detail, seconds=round(time.monotonic() - t0, 2))
            error.record = record
            raise error from exc
        if output.empty:
            raise RuntimeError("simulator returned no choices")
        output, changed = sanitize_output(output)
        message = output.message
        raw = message.text
        record.update({
            "raw": raw[:RAW_LIMIT], "raw_chars": len(raw), "reasoning": reasoning_text(message)[:RAW_LIMIT],
            "stop_reason": output.stop_reason, "seconds": round(time.monotonic() - t0, 2),
            "usage": output.usage.model_dump(exclude_none=True) if output.usage else None,
            "invalid_unicode": changed,
        })
        if flag := incomplete_reason("simulator", output.stop_reason):
            error = IncompleteOutput(flag, output.stop_reason)
            error.record = record
            raise error
        return tool_output(parse(raw)), record


def make_observer(condition: str, simulator: Model | None, scenario, session_start: str | None = None) -> BashObserver:
    """The condition's observer. C2 and C3 get the scenario's snapshot with `session_start`
    (`date` on the real box at episode start) in its facts block; C1 gets no snapshot."""
    if condition == "C0":
        return BashObserver()
    if simulator is None:
        raise ValueError(f"{condition} needs a simulator model")
    if condition == "C1":
        return SimulatedBash(condition, simulator, scenario.tool_text(), None)
    if session_start is None:
        raise ValueError(f"{condition} needs the session start time (the snapshot's date line)")
    return SimulatedBash(condition, simulator, scenario.tool_text(), session_snapshot(scenario.snapshot, session_start))
