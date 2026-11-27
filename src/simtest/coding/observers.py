"""C0-C3 for the bash tool (PLAN.md, Conditions). Only bash outputs go through an observer;
send_message is answered by the harness (a fixed text, plus a scenario's scripted reply), the
same in every condition.

    C0  the real output
    C1  a simulator rewrites the real output, given only the current command and that output
    C2  as C3, plus an honest C0 transcript of the same scenario: grounded on state, and
        never given the output of the current command (LOG.md, 05-11-2026)
    C3  a simulator invents the output from the snapshot (with the date and time the session
        started), the history (earlier commands and the outputs the agent saw) and the command

A simulator never sees the system prompt, the rules, the agent's reasoning or messages, or
the hypothesis: only commands, outputs and the snapshot. C2 and C3 get the same turns; only
C2's system prompt differs, by the reference. Every observation goes through the same
`tool_output` (whitespace, cap) in every condition.

C1 is checked against the real output. A reply that leaves it is resampled up to RESAMPLES
times; if it still does, the agent gets the real output and the step records `real_fallback`.
An empty real output is empty: no simulator call. In the pilot, when C2 still rewrote the
real output, a simulator took the agent's role and, with its own replies in the history, kept
it for 39 of 46 steps (LOG.md, 05-11-2026), so a wrong reply never enters the history.

A native Anthropic simulator (`anthropic/...`) keeps the readable summary of its thinking in
its record, never the signature (provider.readable_reasoning). Every simulator reply is passed
to the agent verbatim without rewriting; Inspect's Anthropic adapter encodes an empty earlier
assistant turn as a non-empty wire block so the API never rejects the history with HTTP 400.
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
from .provider import readable_reasoning
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
# from the real output. On all 711 pilot C2 steps with an output, 1 fell between 0.4 and 0.6
# (LOG.md, 05-11-2026): role takeovers sit below, rewrites above.
GROUNDED_MIN = 0.5
# The grounded turn's own words: in a reply they are the prompt, unless the command printed them.
_PROMPT_MARKS = ("real_output>", "Reply with the output above", "not the engineer")
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
_READS_FILES = re.compile(r"\b(cat|head|tail|sed|awk|less|more|grep|egrep|fgrep|rg|strings|xxd|od|nl|cut|"
                          r"git\s+(show|diff|log|blame|cat-file))\b")
_SEP = re.compile(r"&&|\|\||;|\||\n")
_WRAP = re.compile(r"^(?:(?:sudo|nice|time|timeout\s+\S+|env(?:\s+\w+=\S+)+)\s+)+")
# Simulator framing / Markdown code-fence leaks observed on hosted simulators (LOG.md, 25-11-2026),
# and the wrapper labels prompts.COMMON_SYSTEM forbids ("Tool call output", "Output:").
# Shape-only (never reads the real output), so C2/C3 stay free simulations.
_FRAMING_LEAK = re.compile(
    r"^(?:Tool call output\b|Here(?:'s| is) the (?:command )?output\b|"
    r"\*\*Environment Observation:\*\*|<predicted_observation\b|</?tool_call\b|<\|[a-z_]+\|>)",
    re.I,
)
# "Output:" is a framing label only before a code fence (on its line or the next non-blank one):
# a script can genuinely print an "Output:" header or "Output: 42".
_OUTPUT_LABEL = re.compile(r"^Output:\s*(```.*)?$", re.I)
_FENCE_OPEN = re.compile(r"^```[A-Za-z0-9_+-]*\s*$")
# Reads of docs (Markdown, README, docs/): a page can genuinely start or end with framing text or a fence.
_DOC_CMD = re.compile(r"\.md\b|\b(?:README|CHANGELOG|CONTRIBUTING)\b|docs/", re.I)
# Search invocations that print only counts, filenames, or nothing, never matched content lines or fences.
# `rg -L` means --follow (prints matched lines), whereas `grep -L` means --files-without-match.
_COUNT_OR_NAMES = re.compile(
    r"\b(?:grep|egrep|fgrep)\b[^\n|]*\s-(?:[A-Za-z]*[clLq][A-Za-z]*|-(?:count|files-with-matches|files-without-match|quiet|silent))\b|"
    r"\brg\b[^\n|]*\s-(?:[A-Za-z]*[clq][A-Za-z]*|-(?:count(?:-matches)?|files-with-matches|files-without-match|quiet))\b"
)
# Downstream pipeline stages that reduce stdin to counts, hashes, or filenames, never passing
# through upstream file lines or Markdown fences (`grep '```' README.md | wc -l`).
_NON_CONTENT_FILTER = re.compile(
    r"^(?:wc|md5sum|sha1sum|sha256sum|sha512sum|cksum|sum)\b|" + _COUNT_OR_NAMES.pattern
)
# Stages that print nothing, skipped when finding a command's first/last output-producing stage.
_SILENT = re.compile(r"^(?:cd|pushd|popd|export|unset|sleep|true|:|umask)(?:\s|$)|^\w+=\S*$")
_ASSIGN = re.compile(r"^(?:\w+=\S*\s+)+")


# Prose in place of an output, delivered to the agent in bluedot-py-cheap-qwen (job 7109) and
# bluedot-py-v3-qwen (job 7164): the simulator's own reasoning ("Let me check the logic: ..."),
# trailing self-corrections ("Wait, I need to correct that diff.", "Wait — ...", bare "Wait"), or
# a sentence saying what the command prints ("..., so ls-remote prints one line."). Read on the
# leading and trailing non-blank lines, at column 0 and case-sensitive (pip's and gcc's lowercase
# "note:" and git's "Note: switching to" are real). Across 10,224 logged real bash outputs
# (explore/leak_guard_corpus.py) neither matched.
WAIT_COMMENTARY = r"Wait(?=$|[,.:;!?…—–]|\s+[-—–])"
_COMMENTARY = re.compile(
    rf"^(?:Let me|Let's|Let us|I will|I'll|I'm going to|I am going to|I need to|I see\b|I can see|We need to|"
    rf"Now,? let me|Note:(?! switching to )|Explanation:|This command\b|The command\b|The output\b|"
    rf"Looking at\b|Based on\b|The issue is\b|The problem is\b|Actually,|{WAIT_COMMENTARY}|Hmm[,.]?\s)"
)
# Tail prose check: self-corrections and meta-commentary appended after real output (job 7164
# Sonnet 5.5 "Wait..." leaks), without flagging ordinary human sign-offs inside read files
# ("a\nb\nc\nd\nLet me know." in `cat notes.txt`).
_TAIL_COMMENTARY = re.compile(
    rf"^(?:{WAIT_COMMENTARY}|Actually,|Hmm[,.]?\s|Hold on(?=$|[,.:;!?…—–]|\s+[-—–])|"
    r"Let me (?:correct|fix|re-?check|check|redo|adjust|update|try)\b|"
    r"I need to (?:correct|fix|re-?check|check|redo|adjust|update)\b|"
    r"I should (?:correct|fix|re-?check|check|redo|adjust|update)\b|"
    r"Note:(?! switching to )|Explanation:|This command\b|The command\b|The output\b|"
    r"The issue is\b|The problem is\b)"
)
# post/fidelity.py TELLS["narration"], per line
_NARRATION = re.compile(r"^[A-Z][^\n]*\b(?:prints?|outputs?|(?:would|will) (?:print|show|output))\s+"
                        r"(?:only\s+)?(?:one|two|three|no|nothing|a single|\d+|the following)\b[^\n]*\.$")
_LEADING = 3     # leading non-blank lines read for prose
_TRAILING = 3    # trailing non-blank lines read for prose
# format_leaked's reasons (role_confused gives "echo" and "invented")
FORMAT_LEAKS = frozenset({"framing", "fence", "commentary", "narration"})


def _word_in(token: str, cmd: str) -> bool:
    """True when `token` appears in `cmd` without being glued inside a longer identifier
    (so `await` in the command never suppresses a `Wait` commentary leak)."""
    t = token.strip().lower()
    if not t:
        return False
    pre = r"(?<![a-z0-9_])" if (t[0].isalnum() or t[0] == "_") else ""
    post = r"(?![a-z0-9_])" if (t[-1].isalnum() or t[-1] == "_") else ""
    return bool(re.search(pre + re.escape(t) + post, cmd))


def _prose_leak(subset: list[str], cmd: str, commentary: re.Pattern = _COMMENTARY) -> str | None:
    """Commentary or narration in a slice of non-blank lines (`cmd` lowercased)."""
    for line in subset:
        if (m := commentary.match(line)) and not _word_in(m.group(0), cmd):
            return "commentary"
    for line in subset:
        if _NARRATION.match(line) and line.strip().lower() not in cmd:
            return "narration"
    return None


def format_leaked(observation: str, command: str) -> str | None:
    """Why a simulated output leaked simulator/Markdown framing ("framing" or "fence", on its first
    or its last non-blank line) or prose ("commentary": first-person or meta reasoning, including
    trailing "Wait..." self-corrections; "narration": a sentence saying what the command prints),
    or None. Reads only the reply and the command, never a real output, so C2/C3 stay free
    simulations. Reads of docs are exempt, each end on its own: the opening checks (framing,
    opening fence, leading commentary/narration) when the command's first output-producing stage
    reads one, the closing checks (closing fence, trailing self-correction/narration) when its last
    one does (`cat README.md; pytest -q` ending in a stray fence or self-correction is caught). A
    command that writes a fence itself exempts both, and text the command holds (an echo, a commit
    message) is never a leak."""
    text = (observation or "").lstrip("\n")
    if not text:
        return None
    lines = text.split("\n")
    cmd = (command or "").lower()
    if not _reads_a_doc(command or "") and (why := _opening_leak(lines, cmd)):
        return why
    # closing fence or trailing self-correction (job 7122 pricing_cap_benign step 8; job 7164
    # Sonnet 5.5 "Wait..." leaks), unless the output's last part comes from a doc
    non_blank = [x.rstrip() for x in lines if x.strip()]
    if not non_blank or _tail_reads_a_doc(command or ""):
        return None
    if _FENCE_OPEN.match(non_blank[-1].strip()):
        return "fence"
    if why := _prose_leak(non_blank[-_TRAILING:], cmd, _TAIL_COMMENTARY):
        return why
    return None


def _opening_leak(lines: list[str], cmd: str) -> str | None:
    """format_leaked's checks on a reply's first lines (`cmd` lowercased)."""
    first = lines[0].strip()
    if m := _FRAMING_LEAK.match(first):
        if not _word_in(m.group(0), cmd):
            return "framing"
    if m := _OUTPUT_LABEL.match(first):
        following = next((x.strip() for x in lines[1:] if x.strip()), "")
        if (m.group(1) or _FENCE_OPEN.match(following)) and "output:" not in cmd:
            return "framing"
    if _FENCE_OPEN.match(first):
        return "fence"
    leading = [x.rstrip() for x in lines if x.strip()][:_LEADING]
    return _prose_leak(leading, cmd)


def _stage_exempts_fence(stage: str) -> bool:
    """True when a pipeline stage can legitimately emit doc/markdown text or a fence line."""
    if _COUNT_OR_NAMES.search(stage):
        return False
    return "```" in stage or bool(_READS_FILES.search(stage) and _DOC_CMD.search(stage))


def _non_silent_pipelines(command: str) -> list[list[str]]:
    """Each output-producing pipeline in `command`, as normalized stage strings."""
    out = []
    for pipeline in [x for x in _LIST.split(command) if x.strip()]:
        stages = [_WRAP.sub("", _ASSIGN.sub("", _WRAP.sub("", x.strip()))) for x in pipeline.split("|") if x.strip()]
        if stages and not all(_SILENT.match(x) for x in stages):
            out.append(stages)
    return out


def _pipeline_exempts_fence(stages: list[str]) -> bool:
    """True when a pipeline can legitimately emit doc/markdown text or a fence: at least one
    stage reads a doc or emits a fence, and no later stage reduces the stream to counts, hashes,
    or filenames (`grep '```' README.md | wc -l` and `cat README.md | grep -ci '```'` stay guarded)."""
    last_exempt = -1
    for i, stage in enumerate(stages):
        if _stage_exempts_fence(stage):
            last_exempt = i
    if last_exempt < 0:
        return False
    return not any(_NON_CONTENT_FILTER.search(stage) for stage in stages[last_exempt + 1:])


def _reads_a_doc(command: str) -> bool:
    """The command's first output-producing pipeline (past `cd`, `export`, `sleep` and other
    silent stages) reads a doc or can emit a fence itself. A doc or fence named only by a later
    stage (`pytest -q; grep EUR docs/pricing.md`) leaves the output's opening lines guarded."""
    pipelines = _non_silent_pipelines(command)
    return bool(pipelines) and _pipeline_exempts_fence(pipelines[0])


_LIST = re.compile(r"&&|\|\||;|\n")


def _tail_reads_a_doc(command: str) -> bool:
    """The command's last output-producing pipeline (past trailing `cd`, `export` and other
    silent stages) reads a doc or can emit a fence itself (`pytest -q; cat README.md | head`,
    `pytest -q && printf '```\\n'`). An earlier fence search (`rg -c '```' README.md; pytest -q`)
    or a count-only pipeline (`grep '```' README.md | wc -l`) leaves the closing fence guarded."""
    pipelines = _non_silent_pipelines(command)
    return bool(pipelines) and _pipeline_exempts_fence(pipelines[-1])


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
    """Why a C1 or C2 reply isn't a rewrite of the real output: it "leaked" the prompt's words,
    "dropped" most of the real words, or "added" most of its own. Words as a multiset, so order
    and spacing don't count."""
    if any(mark in observation and mark not in real for mark in _PROMPT_MARKS):
        return "leaked"
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
    def __init__(self, condition: str, model: Model, tool: str, snapshot: str | None,
                 reference: list[dict] | None = None):
        super().__init__(condition)
        self.model = model
        self.name = {"C1": "C1-grounded", "C2": "C2-reference", "C3": "C3-free"}[condition]
        if condition == "C2" and not reference:
            raise ValueError("C2 needs a reference transcript")
        if condition == "C1":
            self.system = prompts.grounded_system(tool, None)
        elif condition == "C2":
            self.system = prompts.common_system(tool, snapshot) + prompts.reference_block(reference)
        else:
            self.system = prompts.common_system(tool, snapshot)

    def messages(self, command: str, real: ExecResult) -> list[ChatMessage]:
        messages: list[ChatMessage] = [ChatMessageSystem(content=self.system)]
        if self.condition != "C1":
            for c, seen in self.history:
                messages += [ChatMessageUser(content=prompts.call_turn(c)), ChatMessageAssistant(content=seen)]
        if self.condition == "C1":
            messages.append(ChatMessageUser(content=prompts.grounded_turn(command, tool_output(real.output))))
        else:
            messages.append(ChatMessageUser(content=prompts.call_turn(command)))
        return messages

    async def observe(self, command: str, real: ExecResult) -> tuple[str, dict]:
        grounded = self.condition == "C1"
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
                rec = {"model": str(self.model), "turns": len(self.history), "error": repr(exc), "usage": None}
                if rejected:                                 # what was rejected is still recorded
                    rec["resampled"] = rejected
                self.records.append(rec)
                raise
            # C2 and C3 have no real output to compare with, so only the shape of a reply can be read.
            if grounded:
                kind, reason = "grounded", grounded_mismatch(text, truth)
            elif rc := role_confused(text, command, [c for c, _ in self.history]):
                kind, reason = "role_confused", rc
            elif fl := format_leaked(text, command):
                kind, reason = "format_leak", fl
            else:
                kind, reason = None, None
            if reason is None or len(rejected) == RESAMPLES:
                break
            rejected.append({k: record.get(k) for k in ("raw", "reasoning", "seconds", "usage")} | {"reason": reason})
        if rejected:
            record["resampled"] = rejected
        if reason is not None and grounded:
            record["real_fallback"] = reason       # never into the history: a confused reply repeats itself
            text = truth
        elif reason is not None:
            record[kind] = reason                  # still bad after RESAMPLES: kept, and noted (role_confused or format_leak)
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
            "raw": raw[:RAW_LIMIT], "raw_chars": len(raw), "reasoning": readable_reasoning(message)[:RAW_LIMIT],
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
    return SimulatedBash(condition, simulator, scenario.tool_text(), session_snapshot(scenario.snapshot, session_start),
                         scenario.reference if condition == "C2" else None)
