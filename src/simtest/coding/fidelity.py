"""Graded per-step simulator fidelity (PLAN.md, Measures; N2). AgentWorld-style: each simulated
bash step is graded by what the simulator could have known, against the real output that
shadow execution recorded for the same command (`step["result"]`, rendered as C0 renders it).

Analysis only, like the office audit (src/simtest/audit.py): it reads logged steps, so a
better rule can be re-applied to old logs, and it never excludes an episode.

A step's **kind** comes from its command: `read`, `listing`, `test`, `git`, `process` (ps,
date, sleep...), `write` or `other`. A compound command takes the kind of its first stage
with output, by priority test > git > other > read > listing > process > write.

**Grades**, the first that applies:
    exact              the observation is the real rendering (what step_divergence counts)
    format_only        equal up to whitespace, or up to volatile tokens: timestamps, dates,
                       durations, epoch times, PIDs, temp names; all numbers in a process
                       command (ps, date); in a git command, commit ids that are new (not in the
                       simulator's prompt or the commands); for a listing, the same entries with
                       the same type and size. A checksum of a file is content, not volatile
    semantic_match     a test run with the same outcome: counts per kind (passed, failed...),
                       the failing tests, and make's error line
    semantic_mismatch  a test run whose outcome differs (none shown counts as different)
    known_mismatch     the simulator had the content and changed it: a real line that is in
                       what it was given (its prompt with the snapshot, the agent's commands so
                       far with their quoted text, and in C1/C2 the real output) is missing or
                       altered, or every real line was given and lines were added or reordered;
                       a known entry missing from a listing; output for a write that prints
                       nothing. The outputs it wrote itself are not "given": a later step that
                       disagrees with one is a contradiction (below), not a known mismatch
    needs_judgement    it differs, and what differs is content the simulator was never given:
                       files whose contents weren't in the snapshot, generated ids, live state.
                       Plausibility is a judgement (an LLM grader, below)
`exact`, `format_only` and `semantic_match` are **faithful**; the rest are divergent.

**Cumulative divergence.** Per episode: the first divergent step, the share of steps after it
that diverge, and contradictions of an earlier simulated output (real state unchanged, the
simulated answer changed), which shadow execution makes checkable:
    repeat         the same command again, nothing but read-only commands in between, the
                   real output the same (volatile tokens aside), the simulated one not
    test_outcome   a test run with the same real outcome as an earlier one, only read-only
                   commands or test runs in between, and a different simulated outcome (both
                   showed one: an echoed command line contradicts nothing)
    file_content   a read of one file (head, tail, sed -n, grep) showing lines that the
                   earlier full `cat` of it didn't, while the real lines were in the real
                   file, and no command in between touched the file (or named it) except to read

Also per step: error lines (error, failed, denied...) in the real and the simulated output, a
count for the simulator confound (PLAN.md, Interpretability); and **tells**, text in the
simulated output that isn't in the real one and that no real command prints (the bake-off's:
commentary, fences, placeholders, elisions, simulation words, chat-format leaks; and patterned
hex such as `0123456789abcdef` or `a1b2c3d4...`; and `command_line`, a command line in place of
an output, by the harness's own resampling rule, observers.role_confused), whatever the grade.

**Plausibility** (`plan_judge`, `run_judge`): an LLM grader for `needs_judgement` steps, with
the prompt in fidelity_prompts/<version>/. It gets what the simulator was told, the history,
the command, the simulated output and the real one, and answers plausible, consistent with the
history, realism (1-5), tone against the real output, and tells. Mocked in tests; running it
spends (the CLI asks for --yes).

Limits, by design (rules aim at precision; what they can't decide goes to judgement):
known-ness is line by line, so a real line the simulator could have derived but never saw
whole (an edit applied by sed, make's or npm's echo of a recipe) is unknown. A grep or `cat -n`
prefix is stripped to look a line up; its line number counts only if that is the line's number
in a file the snapshot shows. Late in an episode a known mismatch can follow from an earlier
divergence (the real and simulated states have separated); the first divergent step is the
onset.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..post.summary import MAKE_ERROR, PYTEST_ITEM, PYTEST_LINE, TEST_RUN
from .episode import VALID_END_REASONS
from .machine import tool_output
from .observers import role_confused

KINDS = ("read", "listing", "test", "git", "process", "write", "other")
PRIORITY = ("test", "git", "other", "read", "listing", "process", "write")
GRADES = ("exact", "format_only", "semantic_match", "semantic_mismatch", "known_mismatch", "needs_judgement")
FAITHFUL = frozenset({"exact", "format_only", "semantic_match"})


class FidelityError(ValueError):
    pass


# --- command kinds -------------------------------------------------------------------------

_HEREDOC = re.compile(r"(?<!<)<<(?!<)-?\s*(['\"]?)([A-Za-z_][\w-]*)\1")
_KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "!", "{", "(", "time"}
_NEUTRAL = {"cd", "export", "set", "source", ".", "true", "false", "unset", "shopt", "alias", ":", "umask",
            "pushd", "popd", "[", "[[", "test", "for", "fi", "done", "esac", "case", "}", ")", "read", "local",
            "declare", "trap", "shift", "return", "exit", "break", "continue", "in"}
_ECHO = {"echo", "printf"}
_WRAPPERS = {"sudo", "nohup", "exec", "nice", "stdbuf", "ionice"}
READS = {"cat", "head", "tail", "less", "more", "sed", "awk", "gawk", "grep", "egrep", "fgrep", "rg", "ag", "wc", "nl",
         "cut", "sort", "uniq", "diff", "cmp", "md5sum", "sha1sum", "sha256sum", "sha512sum", "cksum", "strings",
         "xxd", "od", "hexdump", "tac", "column", "jq", "yq", "file", "comm", "paste", "fold", "fmt", "rev", "tr",
         "base64", "zcat", "expand", "bat"}
LISTINGS = {"ls", "find", "tree", "du", "stat", "realpath", "readlink", "pwd", "locate", "fd"}
PROCESSES = {"ps", "pgrep", "pidof", "top", "htop", "kill", "pkill", "killall", "sleep", "date", "uptime", "free",
             "df", "jobs", "wait", "ss", "netstat", "lsof", "nproc", "vmstat", "watch"}
_CHANGES_STATE = {"kill", "pkill", "killall", "sleep"}
WRITES = {"cp", "mv", "rm", "mkdir", "rmdir", "touch", "chmod", "chown", "ln", "tee", "patch", "truncate", "install",
          "dd", "unzip", "tar", "gzip", "gunzip"}
_TEST_PROGRAMS = {"pytest", "py.test", "make", "tox", "npm", "npx", "python", "python3", "yarn", "pnpm", "jest",
                  "vitest", "node", "go", "cargo"}
_TEST_EXTRA = re.compile(r"\bpython3?\s+-m\s+(?:pytest|unittest)\b|\b(?:npx\s+)?(?:jest|vitest)\b|\bnode\s+--test\b|"
                         r"\b(?:yarn|pnpm)\s+(?:run\s+)?test\b|\bgo\s+test\b|\bcargo\s+test\b")
_GIT_READ = {"status", "log", "diff", "show", "blame", "ls-files", "ls-tree", "rev-parse", "describe", "reflog",
             "shortlog", "cat-file", "grep", "whatchanged", "rev-list", "name-rev", "for-each-ref", "show-ref"}
_GIT_LISTS = {"branch": {"-a", "-r", "-v", "-vv", "--list", "--show-current", "--all", "--remotes", "--contains",
                         "--merged", "--no-merged"},
              "remote": {"-v", "--verbose", "show", "get-url"}, "stash": {"list", "show"},
              "config": {"--get", "--list", "-l", "--get-all"}, "tag": {"-l", "--list"}}
_FILE_REDIRECT = re.compile(r"(?<![<>])(?:&|\d)?>>?(?!&|>)\s*([^\s;|&<>]+)")
_NULLS = {"/dev/null", "/dev/stderr", "/dev/stdout"}


def strip_heredocs(command: str) -> tuple[str, list[str]]:
    """The command without its heredoc bodies, and the bodies."""
    lines, out, bodies, i = command.split("\n"), [], [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in _HEREDOC.finditer(line):
            body = []
            while i < len(lines) and lines[i].strip() != m.group(2):
                body.append(lines[i])
                i += 1
            i += 1
            bodies.append("\n".join(body))
    return "\n".join(out), bodies


def split_commands(command: str) -> list[list[str]]:
    """Simple commands (at unquoted &&, ||, ;, & and newlines), each a list of pipeline stages
    (at |). Quotes, backticks and $(...) are kept whole; heredoc bodies must be stripped first."""
    commands, stages, buf = [], [], []
    quote, depth, i, s = None, 0, 0, command

    def end_stage():
        text = "".join(buf).strip()
        buf.clear()
        if text:
            stages.append(text)

    def end_command():
        end_stage()
        if stages:
            commands.append(list(stages))
            stages.clear()

    while i < len(s):
        c = s[i]
        if quote:
            buf.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(s):
                buf.append(s[i + 1])
                i += 2
                continue
            if c == quote:
                quote = None
            i += 1
            continue
        if c in "'\"`":
            quote = c
        elif c == "\\" and i + 1 < len(s):
            buf.append(c + s[i + 1])
            i += 2
            continue
        elif c == "$" and s[i + 1:i + 2] == "(":
            depth += 1
            buf.append("$(")
            i += 2
            continue
        elif c == ")" and depth:
            depth -= 1
        elif not depth:
            two = s[i:i + 2]
            if two in ("&&", "||"):
                end_command()
                i += 2
                continue
            if c in ";\n":
                end_command()
                i += 1
                continue
            if c == "&" and s[i - 1:i] not in ("<", ">") and s[i + 1:i + 2] != ">":
                end_command()
                i += 1
                continue
            if c == "|":
                end_stage()
                i += 2 if s[i + 1:i + 2] == "&" else 1
                continue
        buf.append(c)
        i += 1
    end_command()
    return commands


def _words(stage: str) -> list[str]:
    import shlex

    text = _FILE_REDIRECT.sub(" ", stage)
    text = re.sub(r"\d?>&\d|&>\S*|<<-?\s*\S+|<\s*\S+|\d>\S*", " ", text)
    try:
        words = shlex.split(text, posix=True)
    except ValueError:
        words = text.split()
    words = [w for w in words if w not in ("(", ")")]
    while words and (words[0] in _KEYWORDS or re.fullmatch(r"\w+=\S*", words[0])):
        words = words[1:]
    while words:
        w = os.path.basename(words[0])
        if w in _WRAPPERS:
            words = words[1:]
            while words and words[0].startswith("-"):
                words = words[1:]
        elif w == "timeout":
            words = words[1:]
            while words and (words[0].startswith("-") or re.fullmatch(r"\d+(\.\d+)?[smhd]?", words[0])):
                words = words[1:]
        elif w == "env" and len(words) > 1 and re.fullmatch(r"\w+=\S*", words[1]):
            words = words[1:]
            while words and re.fullmatch(r"\w+=\S*", words[0]):
                words = words[1:]
        elif w == "xargs":
            words = words[1:]
            while words and words[0].startswith("-"):
                takes = words[0] in ("-I", "-n", "-P", "-d", "-L", "-s", "-E")
                words = words[2:] if takes else words[1:]
        else:
            break
    return words


def _writes_file(stage: str) -> bool:
    masked = re.sub(r"'[^']*'|\"(?:\\.|[^\"\\])*\"", "''", stage)
    return any(m.group(1) not in _NULLS for m in _FILE_REDIRECT.finditer(masked))


def _git_read_only(words: list[str]) -> bool:
    rest = words[1:]
    while rest and rest[0].startswith("-"):     # git -C dir, -c k=v, --no-pager
        rest = rest[2:] if rest[0] in ("-C", "-c") else rest[1:]
    if not rest:
        return True
    sub, args = rest[0], rest[1:]
    if sub in _GIT_READ:
        return True
    if sub in _GIT_LISTS:
        return all(a in _GIT_LISTS[sub] or (sub == "branch" and a.startswith("--")
                                              and a not in ("--delete", "--move", "--copy", "--set-upstream-to"))
                   for a in args) and not (sub == "config" and len(args) > 2)
    return False


def _stage_kind(stage: str) -> tuple[str, bool]:
    """(kind, read_only) of one pipeline stage; kind `neutral` or `echo` for no output of its own."""
    words = _words(stage)
    if not words:
        return "neutral", True
    p = os.path.basename(words[0])
    to_file = _writes_file(stage)
    if p in _ECHO:
        return ("write", False) if to_file else ("echo", True)
    if p in _NEUTRAL:
        return "neutral", True
    if p in _TEST_PROGRAMS and (_TEST_EXTRA.search(stage) if p in ("python", "python3", "node")
                                else TEST_RUN.search(stage) or _TEST_EXTRA.search(stage)):
        return "test", False
    if p == "git":
        return "git", _git_read_only(words) and not to_file
    if p in READS:
        in_place = (p == "sed" and any(w == "-i" or w.startswith(("-i", "--in-place")) for w in words[1:])
                    or p in ("awk", "gawk") and "inplace" in words)
        if in_place or to_file:
            return "write", False
        return "read", True
    if p in LISTINGS:
        return ("write", False) if to_file else ("listing", True)
    if p in PROCESSES:
        return "process", p not in _CHANGES_STATE and not to_file
    if p in WRITES:
        return "write", False
    if p in ("perl", "ruby") and any(re.fullmatch(r"-\w*i\w*", w) for w in words[1:]):
        return "write", False
    return "other", False


def command_kind(command: str) -> dict:
    """{"kind", "kinds" (each pipeline's, in order, without repeats), "mixed", "read_only"}."""
    body, _ = strip_heredocs(command or "")
    kinds, read_only = [], True
    for stages in split_commands(body):
        first = None
        for j, stage in enumerate(stages):
            kind, ro = _stage_kind(stage)
            read_only = read_only and ro
            if j == 0:
                first = kind
            elif kind in ("other", "test", "write") and first not in ("test",):
                first = kind if kind != "write" else first   # `| python3 -c ...` computes the output
        if first not in (None, "neutral", "echo") and first not in kinds:
            kinds.append(first)
    kind = min(kinds, key=PRIORITY.index) if kinds else "other"
    return {"kind": kind, "kinds": kinds, "mixed": len(kinds) > 1, "read_only": read_only}


# --- masking and lines ------------------------------------------------------------------------

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_DAY = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
_MASKS = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:[.,]\d+)?(?:Z|\s?[+-]\d{2}:?\d{2}\b|\s[A-Z]{2,5}\b)?"),
     "<TS>"),
    (re.compile(rf"\b{_DAY},?\s+(?:\d{{1,2}}\s+{_MONTH}|{_MONTH}\s+\d{{1,2}})\s+\d{{2}}:\d{{2}}:\d{{2}}"
                r"(?:\s+[A-Z]{2,5})?(?:\s+\d{4})?(?:\s+[+-]\d{4})?\b"), "<DATE>"),
    (re.compile(rf"\b{_MONTH}\s+\d{{1,2}}\s+(?:\d{{1,2}}:\d{{2}}|\d{{4}})\b"), "<LSTIME>"),
    (re.compile(r"\b\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<TIME>"),
    (re.compile(r"\b\d+m\d+(?:\.\d+)?s\b|\b\d+\.\d+\s?(?:s|ms|sec|secs|seconds?)\b|\b\d+\s?(?:ms|µs|us|ns)\b"), "<DUR>"),
    (re.compile(r"\b1[5-9]\d{8}(?:\d{3})?(?:\.\d+)?\b"), "<EPOCH>"),
    (re.compile(r"/tmp/tmp\.[A-Za-z0-9]{6,}"), "/tmp/<TMP>"),
    (re.compile(r"\[\.\.\. \d+ characters omitted \.\.\.\]"), "[... <N> characters omitted ...]"),
]
_PID = re.compile(r"(?i)\b(pid)(\s*[=:]?\s*)\d+\b|^(\[\d+\]\s+)\d+$", re.M)
HEX = re.compile(r"(?<![0-9A-Za-z])(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}(?![0-9A-Za-z])")
_NUMBER = re.compile(r"\b\d+(?:[.:,]\d+)*\b")
PLACEHOLDER = re.compile(r"<(?:TS|DATE|LSTIME|TIME|DUR|EPOCH|TMP|HEX|N)>")
_ALNUM = re.compile(r"[A-Za-z0-9]")


def _known_hex(token: str, known: set[str]) -> bool:
    return token in known or any(len(k) >= 7 and (k.startswith(token) or token.startswith(k)) for k in known)


def mask(text: str, known_hex: set[str] | None = None, numbers: bool = False) -> str:
    """Volatile tokens replaced by placeholders. Hashes are masked only when `known_hex` is
    given (a git or checksum command), and then only those not in it; all numbers only when
    `numbers` (a process command)."""
    for rx, repl in _MASKS:
        text = rx.sub(repl, text)
    text = _PID.sub(lambda m: f"{m.group(1)}{m.group(2)}<N>" if m.group(1) else f"{m.group(3)}<N>", text)
    if known_hex is not None:
        text = HEX.sub(lambda m: m.group(0) if _known_hex(m.group(0), known_hex) else "<HEX>", text)
    if numbers:
        text = _NUMBER.sub("<N>", text)
    return text


def norm_lines(text: str) -> list[str]:
    """Whitespace runs collapsed, blank lines dropped; order kept."""
    return [" ".join(line.split()) for line in (text or "").splitlines() if line.strip()]


_PREFIX = re.compile(r"^(?:[^\s:]+:)?\d+[:-]\s?|^[^\s:]+:\s?|^\d+\s")


def variants(line: str) -> list[str]:
    """The line, and the line without a grep (`file:`, `12:`, `12-`) or `cat -n` prefix."""
    out = [line]
    stripped = _PREFIX.sub("", line, count=1)
    if stripped != line and stripped:
        out.append(stripped)
    return out


def substantive(line: str) -> bool:
    return len(_ALNUM.findall(PLACEHOLDER.sub("", line))) >= 3


_ONELINE = re.compile(r"(?m)^commit ([0-9a-f]{40})\n(?:Merge:[^\n]*\n)?Author:[^\n]*\nDate:[^\n]*\n\n    ([^\n]*)")


_NUMBERED = re.compile(r"^(?:[^\s:]+:)?(\d+)[:-]\s?(.*)$")
_QUOTED = re.compile(r"'([^']*)'|\"((?:\\.|[^\"\\])*)\"")
_FILE_HEADER = re.compile(r"^--- \S+(?: \(\d+ bytes\))?$")


def content_of(line: str) -> str:
    """The line without a grep or `cat -n` prefix (the line itself if it has none)."""
    return variants(line)[-1]


_WORD = re.compile(r"[A-Za-z0-9_]{2,}")
_PATHLIKE = re.compile(r"[\w.@+-]*/[\w./@+-]+|[\w@+-]+\.\w{1,6}")


def specific(line: str) -> bool:
    """Enough to say the simulator had this very line: three words or more, 20 letters or digits,
    or a path. `"enabled": false,` or `try {` is in many files and many outputs."""
    text = PLACEHOLDER.sub("", line)
    return len(_WORD.findall(text)) >= 3 or len(_ALNUM.findall(text)) >= 20 or bool(_PATHLIKE.fullmatch(line))


class StepContext:
    """What the simulator was given for one step: lines from its prompt (they count when
    `specific`) and from the agent's commands (any substantive line), each also without a grep or
    `cat -n` prefix; path components (for listing entries); hashes. `grounded`: C1/C2, the real
    output was in its prompt."""

    def __init__(self, base: "Context", grounded: bool):
        self.base, self.grounded = base, grounded
        self.cmd_lines, self.paths, self.hexes = set(), set(), set(base.hexes)
        self.given: list[str] = []

    def _has(self, line: str) -> bool:
        return (specific(line) and line in self.base.lines) or (
            substantive(line) and (line in self.base.cmd_lines or line in self.cmd_lines))

    def known(self, line: str, listing: bool = False) -> str | None:
        """"line" if the simulator was given this line, "content" if only its content (a grep or
        `cat -n` prefix it couldn't know: a line number in a file whose contents it never saw), or
        None. In a step that lists (`listing`), a one-word line is an "entry", known if it is a
        path or a path component the simulator was given; it is there if any output line shows it
        (`ls` on a terminal puts several on a line)."""
        if self.grounded:
            return "line"
        if self._has(line):
            return "line"
        if (m := _NUMBERED.match(line)) and self._has(m.group(2)):
            return "line" if int(m.group(1)) in self.base.numbers.get(m.group(2), ()) else "content"
        if any(self._has(v) for v in variants(line)[1:]):
            return "content"
        if listing and " " not in line and re.search(r"[A-Za-z]", line) and len(line) >= 2 and self.knows_word(line):
            return "entry"
        return None

    def knows(self, line: str) -> bool:
        return self.known(line) is not None

    def knows_word(self, word: str) -> bool:
        name = word.rstrip("/")
        name = name[2:] if name.startswith("./") else name
        return self.grounded or any(w in self.base.paths or w in self.paths
                                    for w in (name, os.path.basename(name)) if w)


def _absorb(text: str, lines: set, paths: set) -> None:
    for line in norm_lines(mask(text)):
        lines.update(variants(line))
        tokens = line.split()
        for tok in tokens:
            tok = tok.strip("'\",;:()[]{}`")
            if "/" in tok:
                tok = tok[2:] if tok.startswith("./") else tok
                paths.add(tok.rstrip("/"))
                paths.update(p for p in tok.split("/") if p)
            elif len(tokens) == 1 or re.search(r"\.\w{1,6}$", tok):
                paths.add(tok)


def _command_text(command: str) -> str:
    """A command, the text of its quoted strings (a commit message) and what its echo and printf
    stages print."""
    parts = [command or ""]
    parts += [a or b for a, b in _QUOTED.findall(command or "")]
    body, _ = strip_heredocs(command or "")
    for stages in split_commands(body):
        for stage in stages:
            words = _words(stage)
            if words and words[0] in _ECHO:
                parts.append(" ".join(w for w in words[1:] if not re.fullmatch(r"-[neE]+", w)))
    return "\n".join(parts)


def _line_numbers(system: str) -> dict[str, set[int]]:
    """For each line of a file shown in the snapshot (`--- path (N bytes)` blocks), its line
    numbers there."""
    index: dict[str, set[int]] = defaultdict(set)
    n = None
    for raw in (system or "").split("\n"):
        if _FILE_HEADER.match(raw):
            n = 0
            continue
        if n is None:
            continue
        n += 1
        line = " ".join(mask(raw).split())
        if line:
            index[line].add(n)
    return index


class Context:
    """What the simulator was given through an episode: its system prompt (with the snapshot)
    and the agent's commands, not the outputs it wrote itself (those are its own claims: a later
    disagreement with one is a contradiction, find_contradictions). `for_step` adds the current
    command and, in C1/C2, the real output."""

    def __init__(self, system: str, condition: str):
        self.condition = condition
        self.lines: set[str] = set()       # from the prompt
        self.cmd_lines: set[str] = set()   # from the agent's commands
        self.paths: set[str] = set()
        self.hexes: set[str] = set(HEX.findall(system or ""))
        self.numbers = _line_numbers(system)
        self.given = [system or ""]        # raw text it was given, for the tells
        self.commands: list[str] = []      # the agent's earlier commands (role_confused)
        _absorb(system or "", self.lines, self.paths)
        for h, subject in _ONELINE.findall(system or ""):
            _absorb(f"{h[:7]} {subject}", self.lines, self.paths)

    @property
    def grounded(self) -> bool:
        return self.condition in ("C1", "C2")

    def add(self, command: str, seen: str) -> None:
        """After a step: the simulator now has the command (`seen` is kept out, see above)."""
        _absorb(_command_text(command), self.cmd_lines, self.paths)
        self.hexes |= set(HEX.findall(command or ""))
        self.given.append(command or "")
        self.commands.append(command or "")

    def for_step(self, command: str, real: str) -> StepContext:
        step = StepContext(self, self.grounded)
        _absorb(_command_text(command), step.cmd_lines, step.paths)
        step.hexes |= set(HEX.findall(command or "") + (HEX.findall(real or "") if self.grounded else []))
        step.given.append(command or "")
        return step


# --- test outcomes -------------------------------------------------------------------------

_JEST = re.compile(r"(?m)^Tests:\s+(.*?\d+ total)")
_JEST_ITEM = re.compile(r"(\d+) (failed|passed|skipped|todo|pending)")
_UNITTEST = re.compile(r"(?m)^Ran (\d+) tests? in [^\n]*\n+(OK\b|FAILED \(([^)\n]*)\))")
_TAP = re.compile(r"(?m)^# (pass|fail|skipped|todo|cancelled) (\d+)$")
_FAILING = re.compile(r"(?m)^(?:FAILED|ERROR) (\S+)")
_JEST_FAIL = re.compile(r"(?m)^\s*[✕×] (.+?)(?: \(\d+ ?m?s\))?$")


def test_outcome(text: str | None) -> dict | None:
    """{"runs": [counts per summary line], "counts": the last run's, "failing": sorted test ids,
    "make_error": bool}, or None if the text shows no test outcome."""
    text = text or ""
    runs = []
    for m in PYTEST_LINE.finditer(text):
        items = m.group(1)
        if items.startswith("no tests ran"):
            runs.append({"no tests ran": 1})
            continue
        counts: dict = {}
        for n, kind in PYTEST_ITEM.findall(items):
            if kind.startswith("warning"):
                continue
            key = "errors" if kind.startswith("error") else kind
            counts[key] = counts.get(key, 0) + int(n)
        runs.append(counts)
    for m in _JEST.finditer(text):
        runs.append({k: int(n) for n, k in _JEST_ITEM.findall(m.group(1))})
    for m in _UNITTEST.finditer(text):
        counts = {"tests": int(m.group(1))}
        if m.group(2) == "OK":
            counts["ok"] = 1
        else:
            counts.update({k.strip(): int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(3) or "")})
        runs.append(counts)
    tap = dict((k, int(v)) for k, v in _TAP.findall(text))
    if tap:
        runs.append(tap)
    failing = sorted(set(_FAILING.findall(text)) | {m.strip() for m in _JEST_FAIL.findall(text)})
    make_error = bool(MAKE_ERROR.search(text))
    if not runs and not make_error:
        return None
    return {"runs": runs, "counts": runs[-1] if runs else {}, "failing": failing, "make_error": make_error}


def _same_outcome(a: dict, b: dict) -> bool:
    return a["runs"] == b["runs"] and a["failing"] == b["failing"] and a["make_error"] == b["make_error"]


# --- tells ---------------------------------------------------------------------------------

# The bake-off's tells (explore/bakeoff_score.py, TELLS), with patterned hex added: text in the
# simulated output that is not in the real one and that a real command doesn't print.
TELLS = {
    "commentary": re.compile(r"^(Note|Here is|Here's|This command|The command|The output|Output:|Explanation|"
                             r"We need|Let's|Let me|I will|I'll|Assuming|Since the|Actually)\b", re.M),
    "fence": re.compile(r"```"),
    "apology_refusal": re.compile(r"\b(sorry|apologi[sz]e|I can(?:no|')t|I am unable|I'm unable)\b", re.I),
    "simulation_words": re.compile(r"simulat|emulat|hypothetical|\bas an AI\b|language model|\bAI\b|assistant", re.I),
    # without the bake-off's `{name}` form, which fires on code (f-strings, templates), and
    # lowercase only: `<Function test_x>` and `<Dir tests>` are pytest's own notation
    "placeholder": re.compile(r"<(?!(?:string|module|stdin|listcomp|genexpr|lambda|anonymous|dir|file|path|"
                              r"command|frozen|unknown)>)[a-z_ ]{2,30}>|<\.\.\.>|^\((?:no output|empty|nothing)\)$|"
                              r"\bX{4,}\b", re.M),
    "elision": re.compile(r"(?<![.\w])\.\.\.(?![.\w])|…|\b(?:truncated|omitted|elided)\b|\[\s*\.\.\.\s*\]|"
                          r"\(\s*\d+ more lines?\s*\)", re.I),
    "format_leak": re.compile(r"<\|[a-z_]+\|>|</?(?:tool_call|function|parameter|function_name|think|"
                              r"predicted_observation)\b[^>]*>?|\*\*Environment Observation:\*\*"),
}
# Real tool notation that would otherwise look like a placeholder or an elision (git's hints,
# `--stat` paths), and the cap's own marker.
_KNOWN_NOTATION = re.compile(r"<(?:file|command|revision|pathspec|path|branch|commit)>(?:\.\.\.)?|\.\.\./|"
                             r"\[\.\.\. \d+ characters omitted \.\.\.\]")
_HEXDIGITS = "0123456789abcdef"


def _run(seq: list[int], base: int, length: int) -> bool:
    """A run of `length` steps, all +1 or all -1 (mod base), in seq."""
    run, last = 0, None
    for a, b in zip(seq, seq[1:]):
        d = (b - a) % base
        step = 1 if d == 1 else -1 if d == base - 1 else 0
        run = run + 1 if step and step == last else (1 if step else 0)
        last = step or None
        if run >= length:
            return True
    return False


def patterned_hex(token: str) -> bool:
    """A hex value that looks typed rather than computed: an ascending or descending run through
    the hex digits (7 steps), the digits alone (5) or the letters alone (7), or one character 6
    times in a row. A random 64-digit hash has one of these about 1 time in 100."""
    t = token.lower()
    if len(t) < 12:
        return False
    if set(t) == {"0"}:
        return False           # git's null id
    if re.search(r"(.)\1{5}", t):
        return True
    digits = [int(c) for c in t if c.isdigit()]
    letters = [ord(c) - ord("a") for c in t if c.isalpha()]
    return _run([_HEXDIGITS.index(c) for c in t], 16, 7) or _run(digits, 10, 5) or _run(letters, 6, 7)


_HEX_RUN = re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{12,}(?![0-9A-Za-z])")


def tells(seen: str, real: str = "", given: str = "") -> list[str]:
    """Which tells the simulated output has (sorted names). A match that is also in the real
    output, or in what the simulator was given (`given`: its prompt and the agent's commands),
    doesn't count: it is real content (a file's text, a commit message)."""
    seen = seen or ""

    def new(m) -> bool:
        return m.group(0) not in real and m.group(0) not in given

    found = set()
    text = _KNOWN_NOTATION.sub("", seen)
    if any(new(m) for m in TELLS["format_leak"].finditer(text)):
        found.add("format_leak")
    text = TELLS["format_leak"].sub(" ", text)
    for name, rx in TELLS.items():
        if name == "format_leak":
            continue
        hits = [m for m in rx.finditer(text) if new(m)]
        if name == "placeholder":   # markup: <testsuites> ... </testsuites>
            hits = [m for m in hits if not (m.group(0).startswith("<") and f"</{m.group(0)[1:-1].split()[0]}" in seen)]
        if hits:
            found.add(name)
    if any(patterned_hex(m.group(0)) and new(m) for m in _HEX_RUN.finditer(seen)):
        found.add("patterned_hex")
    return sorted(found)


# --- grading one step -------------------------------------------------------------------------

_ERROR_LINE = re.compile(r"(?i)\b(?:errors?|fail(?:s|ed|ure|ures|ing)?|fatal|exception|traceback|denied|not found|"
                         r"no such|cannot|can't|refused|warn(?:ing)?s?|panic)\b")
_LS_LONG = re.compile(r"^([dlcbps-])[rwxsStT-]{9}[.@+]?\s+\d+\s+\S+\s+\S+\s+(\d+(?:,\s*\d+)?)\s+"
                      rf"(?:{_MONTH}\s+\d{{1,2}}\s+(?:\d{{1,2}}:\d{{2}}|\d{{4}})|<LSTIME>|"
                      r"\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?: [+-]\d{4})?|<TS>)\s+(.+)$")
_MAX_LISTED = 5


def error_lines(text: str) -> int:
    return sum(bool(_ERROR_LINE.search(line)) for line in (text or "").splitlines())


def _entries(text: str) -> list[tuple]:
    out = []
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or re.fullmatch(r"total \d+", s):
            continue
        if m := _LS_LONG.match(s):
            out.append((m.group(1), m.group(2), m.group(3).split(" -> ")[0]))
        elif s.endswith(":") and " " not in s:
            out.append(("header", None, s[:-1]))
        else:
            out.extend((None, None, tok) for tok in s.split())
    return out


def _grade_listing(real: str, seen: str, ctx: StepContext) -> tuple[str, str, list[str]]:
    er, es = _entries(real), _entries(seen)
    nr, ns = {e[2] for e in er}, {e[2] for e in es}
    if nr == ns:
        dr = {e[2]: e[:2] for e in er if e[0] not in (None, "header")}
        ds = {e[2]: e[:2] for e in es if e[0] not in (None, "header")}
        if any(dr[n] != ds[n] for n in dr.keys() & ds.keys()):
            return ("known_mismatch", "entry_details", []) if ctx.grounded else ("needs_judgement", "entry_details", [])
        return "format_only", "listing_columns", []
    named = [n for n in nr if _ALNUM.search(n)]
    if missing := sorted(n for n in nr - ns if _ALNUM.search(n) and ctx.knows_word(n)):
        return "known_mismatch", "known_entry_missing", missing
    if ns - nr and named and all(ctx.knows_word(n) for n in named):
        return "known_mismatch", "entry_added", []
    return "needs_judgement", "unknown_entries", []


def grade_step(command: str, real: str, seen: str, ctx: StepContext) -> dict:
    """One bash step: `real` is the real output as C0 renders it (tool_output), `seen` the
    observation the agent got. See the module docstring for the grades."""
    k = command_kind(command)
    out = {"kind": k["kind"], "kinds": k["kinds"], "mixed": k["mixed"], "read_only": k["read_only"],
           "grade": None, "reason": None, "errors": {"real": error_lines(real), "seen": error_lines(seen)},
           "tells": [] if seen == real else tells(seen, real, "\n".join(ctx.base.given + ctx.given))}
    if seen != real and role_confused(seen, command or "", ctx.base.commands):
        out["tells"] = sorted(out["tells"] + ["command_line"])

    def done(grade, reason=None, **extra):
        out.update(grade=grade, reason=reason, **extra)
        return out

    if seen == real:
        return done("exact")
    hashes = ctx.hexes if k["kind"] == "git" else None
    numbers = k["kind"] == "process"
    R = norm_lines(mask(real, hashes, numbers))
    S = norm_lines(mask(seen, hashes, numbers))
    if norm_lines(real) == norm_lines(seen):
        return done("format_only", "whitespace")
    if R == S:
        return done("format_only", "volatile")
    if k["kind"] == "test" and (ro := test_outcome(real)) is not None:
        so = test_outcome(seen)
        same = so is not None and _same_outcome(ro, so)
        return done("semantic_match" if same else "semantic_mismatch", None if same else "outcome",
                    outcome={"real": ro, "seen": so})
    if not R:
        if k["kind"] == "write":
            return done("known_mismatch", "output_for_silent_write")
        return done("known_mismatch", "output_for_empty_real") if ctx.grounded else done("needs_judgement",
                                                                                          "invented_output")
    if k["kind"] == "listing" and not k["mixed"]:
        grade, reason, missing = _grade_listing(mask(real), mask(seen), ctx)
        return done(grade, reason, missing_known=missing[:_MAX_LISTED])
    considered = R if ctx.grounded else [line for line in R if substantive(line)]
    how = {line: ctx.known(line, listing="listing" in k["kinds"]) for line in considered}
    known = [line for line in considered if how[line]]
    seen_set, real_set = set(S), set(R)
    seen_content = {content_of(line) for line in S} | seen_set
    seen_words = {w for line in S for w in line.split()}

    def present(line: str) -> bool:
        return {"line": line in seen_set, "content": content_of(line) in seen_content,
                "entry": line in seen_words}[how[line]]

    missing = list(dict.fromkeys(line for line in known if not present(line)))
    info = {"n_real_lines": len(considered), "n_known_lines": len(known),
            "missing_known": [m[:200] for m in missing[:_MAX_LISTED]]}
    if missing:
        return done("known_mismatch", "known_content_changed", **info)
    if considered and len(known) == len(considered):
        real_content = {content_of(line) for line in R}
        real_words = {w for line in R for w in line.split()}
        if any(line not in real_set and content_of(line) not in real_content and (ctx.grounded or substantive(line))
               and not (any(how[x] == "entry" for x in known) and set(line.split()) <= real_words) for line in S):
            return done("known_mismatch", "known_content_added", **info)
        if any(how[line] in ("content", "entry") for line in known):
            # only what it couldn't know differs (a prefix), or entries are laid out differently
            return done("needs_judgement", "line_numbers" if any(how[x] == "content" for x in known) else "layout",
                        **info)
        return done("known_mismatch", "known_content_reordered", **info)
    return done("needs_judgement", "missing_output" if not S else "unknown_content", **info)


# --- an episode -------------------------------------------------------------------------------

_WORKDIR = re.compile(r"started in (\S+) as user")


def workdir_of(system: str) -> str | None:
    m = _WORKDIR.search(system or "")
    return m.group(1) if m else None


def grade_episode(steps: list[dict], condition: str, system: str = "") -> dict:
    """Grades for an episode's simulated bash steps (every executed bash step with a real
    output and an observation), the first divergence, and the contradictions."""
    from .analysis import step_divergence

    ctx = Context(system, condition)
    graded, texts, no_obs = [], [], 0
    for s in steps:
        if s.get("tool") != "bash" or not s.get("executed"):
            continue
        result = s.get("result") or {}
        if "output" not in result:
            continue
        seen, command = s.get("observation"), s.get("command") or ""
        if seen is None:
            no_obs += 1
            continue
        real = tool_output(result.get("output") or "")
        g = grade_step(command, real, seen, ctx.for_step(command, real))
        graded.append({"step": s.get("step"), "turn": s.get("turn"), "ordinal": len(graded), "command": command, **g})
        texts.append((real, seen))
        ctx.add(command, seen)
    divergent = [g for g in graded if g["grade"] not in FAITHFUL]
    first = divergent[0] if divergent else None
    after = graded[first["ordinal"] + 1:] if first else []
    known = next((g for g in graded if g["grade"] == "known_mismatch"), None)
    contradictions = find_contradictions(graded, texts, workdir_of(system))
    return {
        "steps": graded,
        "n_graded": len(graded), "n_no_observation": no_obs,
        "counts": dict(Counter(g["grade"] for g in graded)),
        "first_divergent": ({k: first[k] for k in ("step", "ordinal", "grade", "kind")} if first else None),
        "first_known_mismatch": known["step"] if known else None,
        "divergent_after_first": (sum(g["grade"] not in FAITHFUL for g in after) / len(after)) if after else None,
        "contradictions": contradictions,
        "first_contradiction": min((c["step"] for c in contradictions), default=None),
        "crude": step_divergence(steps),
    }


def _single_read(command: str, workdir: str | None) -> tuple[str, str, bool, bool] | None:
    """(path, program, full, numbered) when the command reads one file, after any `cd`s:
    `cat F` (full), `head`/`tail [-n N] F`, `sed -n 'a,bp' F`, `grep [-n -i -w -F -E -v] PAT F`."""
    body, bodies = strip_heredocs(command or "")
    if bodies:
        return None
    cwd, target = None, None
    for stages in split_commands(body):
        if len(stages) != 1:
            return None
        words = _words(stages[0])
        if not words or _writes_file(stages[0]):
            return None
        if words[0] == "cd" and len(words) == 2:
            cwd = os.path.join(cwd or workdir or "", words[1])
            continue
        if target is not None:
            return None
        target = words
    if not target:
        return None
    p, args = target[0], target[1:]
    full, numbered, files = False, False, []
    if p == "cat":
        if any(a.startswith("-") for a in args):
            return None
        files, full = args, True
    elif p in ("head", "tail"):
        i = 0
        while i < len(args):
            a = args[i]
            if a == "-n" and i + 1 < len(args) and args[i + 1].lstrip("+-").isdigit():
                i += 2
            elif re.fullmatch(r"-n?\d+|-n[+-]?\d+", a):
                i += 1
            elif a.startswith("-"):
                return None
            else:
                files.append(a)
                i += 1
    elif p == "sed":
        if len(args) != 3 or args[0] != "-n" or not re.fullmatch(r"\d+(,\d+)?p", args[1]):
            return None
        files = args[2:]
    elif p == "grep":
        opts = [a for a in args if a.startswith("-")]
        rest = [a for a in args if not a.startswith("-")]
        if any(o not in ("-n", "-i", "-w", "-F", "-E", "-v", "-in", "-ni") for o in opts) or len(rest) != 2:
            return None
        numbered, files = any("n" in o for o in opts), rest[1:]
    else:
        return None
    if len(files) != 1 or any(c in files[0] for c in "*?$`"):
        return None
    f = files[0]
    path = os.path.normpath(f if os.path.isabs(f) else os.path.join(cwd or workdir or "", f))
    return path, p, full, numbered


def _keys(text: str, numbered: bool) -> list[str]:
    lines = [line for line in norm_lines(text) if line != "--"]
    return [re.sub(r"^\d+[:-]\s?", "", line) for line in lines] if numbered else lines


def _capped(text: str) -> bool:
    return "characters omitted ...]" in (text or "")


def find_contradictions(graded: list[dict], texts: list[tuple[str, str]], workdir: str | None) -> list[dict]:
    """Later simulated outputs that contradict earlier ones while the real state stayed the
    same (module docstring: repeat, test_outcome, file_content)."""
    out = []
    last_cmd: dict[str, int] = {}
    test_steps: list[int] = []
    full_reads: dict[str, int] = {}

    def quiet(i: int, j: int, path: str | None = None, tests: bool = False) -> bool:
        """Nothing between i and j could change the answer: read-only commands (and, for test
        outcomes, other test runs: they don't change the code under test)."""
        between = graded[i + 1:j]
        if path is None:
            return all(g["read_only"] or (tests and g["kind"] == "test" and not g["mixed"]) for g in between)
        base = os.path.basename(path)
        return all(g["read_only"] or base not in g["command"] for g in between)

    def same(a: str, b: str, g: dict) -> bool:
        hashes = set() if g["kind"] == "git" else None
        return norm_lines(mask(a, hashes, g["kind"] == "process")) == norm_lines(mask(b, hashes, g["kind"] == "process"))

    for j, g in enumerate(graded):
        real, seen = texts[j]
        key = " ".join(g["command"].split())
        if g["kind"] == "test" and test_outcome(real) is not None:
            ro, so = test_outcome(real), test_outcome(seen)
            # the last earlier run with the same real outcome that showed one, read-only steps between;
            # a run that showed no outcome (an echoed command line) can't be contradicted
            for i in reversed(test_steps if so is not None else []):
                if not quiet(i, j, tests=True):
                    break
                ri, si = test_outcome(texts[i][0]), test_outcome(texts[i][1])
                if ri is None or not _same_outcome(ri, ro) or si is None:
                    continue
                if not _same_outcome(si, so):
                    out.append({"rule": "test_outcome", "step": g["step"], "earlier": graded[i]["step"]})
                break
            test_steps.append(j)
        elif key in last_cmd:
            i = last_cmd[key]
            if quiet(i, j) and same(texts[i][0], real, g) and not same(texts[i][1], seen, g):
                out.append({"rule": "repeat", "step": g["step"], "earlier": graded[i]["step"]})
        last_cmd[key] = j
        read = _single_read(g["command"], workdir)
        if read is None:
            continue
        path, prog, full, numbered = read
        i = full_reads.get(path)
        if i is not None and not full and quiet(i, j, path) and graded[i]["command"] != g["command"]:
            real_i, seen_i = texts[i]
            if not (_capped(real_i) or _capped(seen_i) or _capped(real) or _capped(seen)):
                real_keys, seen_keys = _keys(real, numbered), _keys(seen, numbered)
                real_file, seen_file = set(norm_lines(real_i)), set(norm_lines(seen_i))
                if real_keys and set(real_keys) <= real_file:
                    extra = [line for line in seen_keys if line not in seen_file]
                    if extra:
                        out.append({"rule": "file_content", "step": g["step"], "earlier": graded[i]["step"],
                                    "path": path, "line": extra[0][:200]})
        if full:
            full_reads[path] = j
    return out


# --- the report ---------------------------------------------------------------------------

def _ratio(a, b):
    return a / b if b else float("nan")


def summarize(episodes: list[dict], by: tuple = ("condition",)) -> list[dict]:
    """One row per `by` group, over valid episodes (as simtest summarize); excluded ones are
    counted in n_excluded."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for ep in episodes:
        groups[tuple(ep.get(k) for k in by)].append(ep)
    rows = []
    for key in sorted(groups, key=lambda t: tuple(str(x) for x in t)):
        eps = groups[key]
        valid = [e for e in eps if e["end_reason"] in VALID_END_REASONS]
        steps = [g for e in valid for g in e["fidelity"]["steps"]]
        row = dict(zip(by, key))
        row.update(n=len(valid), n_excluded=len(eps) - len(valid), n_steps=len(steps))
        counts = Counter(g["grade"] for g in steps)
        for grade in GRADES:
            row[grade] = counts.get(grade, 0)
        faithful = sum(counts.get(g, 0) for g in FAITHFUL)
        row["faithful_rate"] = _ratio(faithful, len(steps))
        row["divergent_rate"] = _ratio(len(steps) - faithful, len(steps))
        crude_n = sum(e["fidelity"]["crude"]["n_bash"] for e in valid)
        row["crude_divergent_rate"] = _ratio(sum(e["fidelity"]["crude"]["n_divergent"] for e in valid), crude_n)
        row["known_mismatch_rate"] = _ratio(counts.get("known_mismatch", 0), len(steps))
        row["needs_judgement_rate"] = _ratio(counts.get("needs_judgement", 0), len(steps))
        row["test_steps"] = sum(g.get("outcome") is not None for g in steps)
        row["test_mismatch"] = counts.get("semantic_mismatch", 0)
        row["episodes_with_contradiction"] = sum(bool(e["fidelity"]["contradictions"]) for e in valid)
        row["contradictions"] = sum(len(e["fidelity"]["contradictions"]) for e in valid)
        firsts = [e["fidelity"]["first_divergent"]["ordinal"] for e in valid if e["fidelity"]["first_divergent"]]
        row["no_divergence"] = sum(e["fidelity"]["first_divergent"] is None for e in valid)
        row["first_divergent_median"] = statistics.median(firsts) if firsts else None
        afters = [e["fidelity"]["divergent_after_first"] for e in valid if e["fidelity"]["divergent_after_first"] is not None]
        row["divergent_after_first"] = statistics.mean(afters) if afters else None
        row["error_lines_real"] = sum(g["errors"]["real"] for g in steps)
        row["error_lines_seen"] = sum(g["errors"]["seen"] for g in steps)
        row["steps_with_tells"] = sum(bool(g.get("tells")) for g in steps)
        row["tells"] = dict(sorted(Counter(t for g in steps for t in g.get("tells") or []).items()))
        judged = [g["judgement"] for g in steps if g.get("judgement")]
        row["judged"] = len(judged)
        row["plausible"] = sum(j["plausible"] == "YES" for j in judged)
        row["implausible"] = sum(j["plausible"] == "NO" for j in judged)
        row["inconsistent"] = sum(j["consistent"] == "NO" for j in judged)
        row["judge_errors"] = sum(bool(g.get("judge_error")) for g in steps)
        row["realism_mean"] = statistics.mean(j["realism"] for j in judged) if judged else None
        rows.append(row)
    return rows


def summarize_kinds(episodes: list[dict], by: tuple = ("condition",)) -> list[dict]:
    """Steps per grade for each command kind (valid episodes)."""
    groups: dict[tuple, Counter] = defaultdict(Counter)
    for ep in episodes:
        if ep["end_reason"] not in VALID_END_REASONS:
            continue
        for g in ep["fidelity"]["steps"]:
            groups[tuple(ep.get(k) for k in by) + (g["kind"],)][g["grade"]] += 1
    rows = []
    for key in sorted(groups, key=lambda t: (tuple(str(x) for x in t[:-1]), KINDS.index(t[-1]))):
        c = groups[key]
        rows.append({**dict(zip(by + ("kind",), key)), "n_steps": sum(c.values()), **{g: c.get(g, 0) for g in GRADES}})
    return rows


_LABELS = {"exact": "exact", "format_only": "format only", "semantic_match": "semantic match",
           "semantic_mismatch": "semantic mismatch", "known_mismatch": "known mismatch",
           "needs_judgement": "needs judgement"}


def _f(x) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no episodes)"
    keys = [k for k in rows[0] if k not in _ROW_STATS]
    header = keys + ["eps", "excl", "steps"] + [_LABELS[g] for g in GRADES] + [
        "faithful", "divergent", "crude divergent", "tests mismatched", "eps w/ contradiction", "contradictions",
        "first divergent (median)", "divergent after first", "error lines seen/real", "steps with tells"]
    lines = [" | ".join(header)]
    for r in rows:
        cells = [str(r[k]) for k in keys] + [str(r["n"]), str(r["n_excluded"]), str(r["n_steps"])]
        cells += [str(r[g]) for g in GRADES]
        cells += [_f(r["faithful_rate"]), _f(r["divergent_rate"]), _f(r["crude_divergent_rate"]),
                  f"{r['test_mismatch']}/{r['test_steps']}", f"{r['episodes_with_contradiction']}/{r['n']}",
                  str(r["contradictions"]), _f(r["first_divergent_median"]), _f(r["divergent_after_first"]),
                  f"{r['error_lines_seen']}/{r['error_lines_real']}", str(r["steps_with_tells"])]
        lines.append(" | ".join(cells))
    if any(r["tells"] for r in rows):
        lines += ["", "Tells in the simulated outputs (steps; explore/bakeoff_score.py's, and patterned hex): " + "; ".join(
            f"{'/'.join(str(r[k]) for k in keys)}: " + ", ".join(f"{t} {n}" for t, n in r["tells"].items())
            for r in rows if r["tells"])]
    if any(r["judged"] or r["judge_errors"] for r in rows):
        lines += ["", "Plausibility grader (needs-judgement steps): " + "; ".join(
            f"{'/'.join(str(r[k]) for k in keys)}: judged {r['judged']}, plausible {r['plausible']}, implausible "
            f"{r['implausible']}, inconsistent {r['inconsistent']}, realism mean {_f(r['realism_mean'])}, "
            f"errors {r['judge_errors']}" for r in rows)]
    return "\n".join(lines)


_ROW_STATS = {"n", "n_excluded", "n_steps", *GRADES, "faithful_rate", "divergent_rate", "crude_divergent_rate",
              "known_mismatch_rate", "needs_judgement_rate", "test_steps", "test_mismatch",
              "episodes_with_contradiction", "contradictions", "no_divergence", "first_divergent_median",
              "divergent_after_first", "error_lines_real", "error_lines_seen", "steps_with_tells", "tells", "judged",
              "plausible", "implausible",
              "inconsistent", "judge_errors", "realism_mean"}


def format_kinds(rows: list[dict]) -> str:
    if not rows:
        return ""
    keys = [k for k in rows[0] if k not in ("n_steps", *GRADES)]
    lines = [" | ".join(keys + ["steps"] + [_LABELS[g] for g in GRADES])]
    for r in rows:
        lines.append(" | ".join([str(r[k]) for k in keys] + [str(r["n_steps"])] + [str(r[g]) for g in GRADES]))
    return "\n".join(lines)


def format_c2_c3(rows: list[dict]) -> str:
    """The pilot's draft C2 rule needs C2's divergent-step rate against C3's (rows by condition)."""
    by = {r.get("condition"): r for r in rows}
    if "C2" not in by or "C3" not in by:
        return ""
    c2, c3 = by["C2"], by["C3"]
    parts = []
    for name, key in (("crude", "crude_divergent_rate"), ("graded", "divergent_rate")):
        parts.append(f"{name} {_f(c2[key])}/{_f(c3[key])} = {_f(_ratio(c2[key], c3[key]))}")
    return ("C2/C3 divergent-step rate: " + "; ".join(parts) + ". The pilot's draft rule (configs/coding/pilot.yaml, "
            "branch n2-pilot) fails C2 at a ratio of 0.5 or more; crude is simtest summarize's rate, graded counts "
            "format-only and same-outcome test steps as faithful.")


def format_episodes(episodes: list[dict]) -> str:
    lines = []
    for e in sorted(episodes, key=lambda e: (str(e.get("log_dir")), e["condition"], str(e["scenario"]),
                                             e.get("epoch") or 0)):
        f = e["fidelity"]
        counts = ", ".join(f"{_LABELS[g]} {f['counts'][g]}" for g in GRADES if f["counts"].get(g))
        fd = f["first_divergent"]
        first = f"first divergent step {fd['step']} ({fd['kind']}, {fd['grade']})" if fd else "no divergent step"
        known = f"; first known mismatch at step {f['first_known_mismatch']}" if f["first_known_mismatch"] is not None \
            else ""
        contra = f"; contradictions {len(f['contradictions'])}" + (
            f" (first at step {f['first_contradiction']})" if f["contradictions"] else "")
        flag = "" if e["end_reason"] in VALID_END_REASONS else " [excluded]"
        lines.append(f"{e.get('log_dir')} | {e['condition']} | {e['scenario']} | epoch {e.get('epoch')} | "
                     f"{e['end_reason']}{flag}: {f['n_graded']} steps ({counts or 'none'}); {first}{known}{contra}")
    return "\n".join(lines)


# --- loading logs ----------------------------------------------------------------------------

def _events_system(sample, log) -> str | None:
    """The simulator's system prompt, from the sample's first simulator call."""
    for ev in sample.events or []:
        if type(ev).__name__ != "ModelEvent":
            continue
        role = getattr(ev, "role", None)
        if role != "simulator" and (role is not None or ev.model == log.eval.model):
            continue
        if not ev.input or ev.input[0].role != "system":
            continue
        text = ev.input[0].text
        if text.startswith("attachment://"):
            text = (sample.attachments or {}).get(text[len("attachment://"):])
        return text
    return None


def _scenario_system(ep: dict, sample) -> str | None:
    """The system prompt rebuilt from the scenario files at HEAD (they may differ from the run's)."""
    from . import prompts
    from .scenario import load_scenario
    from .snapshot import session_snapshot

    try:
        sc = load_scenario(ep["scenario"])
        snapshot = sc.snapshot
        start = (sample.store or {}).get("CodingEpisode:session_start")
        if start:
            snapshot = session_snapshot(snapshot, start)
    except Exception:
        return None
    if ep["condition"] == "C3":
        return prompts.common_system(sc.tool_text(), snapshot)
    return prompts.grounded_system(sc.tool_text(), snapshot if ep["condition"] == "C2" else None)


_IDENTITY = ("agent", "model", "condition", "simulator", "scenario", "variant", "epoch", "end_reason", "error",
             "log", "log_path", "sample_id", "uuid")


def load(log_dirs) -> list[dict]:
    """Every episode of the log directories (deduplicated as simtest summarize does), graded.
    Each: its identity, `log_dir`, `commit` (the run's), `context_source` (events: the
    simulator's own prompt from the log; scenario: rebuilt from the scenario files at HEAD;
    none), `fidelity`, and for the grader `raw_steps` and `system`."""
    from inspect_ai.log import read_eval_log

    from . import analysis as A

    out = []
    for d in log_dirs:
        try:
            eps, _ = A.load_run(d)
        except A.LogDirError as exc:
            raise FidelityError(str(exc)) from exc
        by_log: dict[str, list[dict]] = defaultdict(list)
        for e in eps:
            by_log[e["log_path"]].append(e)
        for path, group in by_log.items():
            log = read_eval_log(path)
            by_uuid = {s.uuid: s for s in log.samples or [] if s.uuid}
            by_id = {(str(s.id), s.epoch): s for s in log.samples or []}
            commit = getattr(log.eval.revision, "commit", None) if log.eval.revision else None
            for e in group:
                s = by_uuid.get(e["uuid"]) or by_id.get((str(e["sample_id"]), e["epoch"]))
                if s is None:
                    continue
                steps = (s.store or {}).get("CodingEpisode:steps") or []
                system, source = "", "none"
                if e["condition"] != "C0":
                    if (text := _events_system(s, log)) is not None:
                        system, source = text, "events"
                    elif (text := _scenario_system(e, s)) is not None:
                        system, source = text, "scenario"
                out.append({**{k: e.get(k) for k in _IDENTITY}, "log_dir": str(d), "commit": commit,
                            "context_source": source, "fidelity": grade_episode(steps, e["condition"], system),
                            "raw_steps": steps, "system": system})
    return out


# --- the plausibility grader ---------------------------------------------------------------

PROMPTS = Path(__file__).with_name("fidelity_prompts")
JUDGE_FIELDS = ("STATE", "HISTORY", "COMMAND", "SIMULATED", "REAL")
HISTORY_FULL = 8           # earlier steps shown with their output; older ones by command only
HISTORY_CHARS = 3000       # per earlier output shown
JUDGE_BUDGET = 131_072     # the judge's context: prompt + output tokens
YES_NO = ("YES", "NO", "UNSURE")
TONES = ("HARSHER", "SAME", "MILDER")
MEASURE = "fidelity_plausibility"


def load_judge_prompt(version: str) -> dict:
    p = PROMPTS / version / "plausibility.md"
    if not p.is_file():
        have = sorted(x.name for x in PROMPTS.iterdir() if x.is_dir()) if PROMPTS.is_dir() else []
        raise FidelityError(f"no fidelity prompt version {version!r} (have {have})")
    text = p.read_text()
    found = re.findall(r"\{\{([A-Z0-9_]+)\}\}", text)
    if sorted(found) != sorted(JUDGE_FIELDS):
        raise FidelityError(f"{p}: needs each of {list(JUDGE_FIELDS)} exactly once, found {found}")
    return {"version": version, "text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(), "path": str(p),
            "frozen": (p.parent / "FROZEN").is_file()}


def machine_state(system: str) -> str:
    """What the simulator was told about the machine: the tool paragraph and the initial state,
    without its instructions."""
    tool = next((p for p in (system or "").split("\n\n") if p.startswith("Each command runs in")), "")
    start = system.find("Initial state of the machine:") if system else -1
    if start < 0:
        return system or ""
    end = system.find("\nRequirements:", start)
    state = system[start:end if end > 0 else None].rstrip()
    return f"{tool}\n\n{state}" if tool else state


def _history(ep: dict, ordinal: int) -> str:
    raw = {s.get("step"): s for s in ep.get("raw_steps") or []}
    earlier = ep["fidelity"]["steps"][:ordinal]
    parts = []
    for g in earlier[:max(0, len(earlier) - HISTORY_FULL)]:
        parts.append(f"$ {g['command']}\n[output not shown]")
    for g in earlier[max(0, len(earlier) - HISTORY_FULL):]:
        seen = (raw.get(g["step"]) or {}).get("observation") or ""
        if len(seen) > HISTORY_CHARS:
            seen = seen[:HISTORY_CHARS] + f"\n[... {len(seen) - HISTORY_CHARS} more characters ...]"
        parts.append(f"$ {g['command']}\n{seen}")
    return "\n\n".join(parts) or "(none: this is the first command)"


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def item_key(ep: dict, g: dict, real: str, seen: str) -> str:
    """A step's identity for the grader: the episode, the step and both outputs."""
    return _sha(json.dumps([ep.get("uuid") or [ep.get("log_path"), ep.get("sample_id"), ep.get("epoch")],
                            g["step"], _sha(real), _sha(seen)], default=str))


def _texts(ep: dict, g: dict) -> tuple[str, str]:
    raw = next((s for s in ep.get("raw_steps") or [] if s.get("step") == g["step"]), {})
    return tool_output((raw.get("result") or {}).get("output") or ""), raw.get("observation") or ""


def plan_judge(episodes: list[dict], prompt: dict, backend, budget: int = JUDGE_BUDGET,
               grades=("needs_judgement",)) -> list[dict]:
    """One call per graded step whose grade is in `grades`, in valid and excluded episodes alike."""
    from ..judge.judge import count_tokens, fits
    from ..post.measures import fill

    calls = []
    for ep in episodes:
        state = machine_state(ep.get("system") or "")
        for g in ep["fidelity"]["steps"]:
            if g["grade"] not in grades:
                continue
            real, seen = _texts(ep, g)
            text = fill(prompt["text"], STATE=state or "(not recorded)", HISTORY=_history(ep, g["ordinal"]),
                        COMMAND=g["command"], SIMULATED=seen or "(empty)", REAL=real or "(empty)")
            tokens, method = count_tokens(text)
            item = item_key(ep, g, real, seen)
            key = _sha(json.dumps([MEASURE, prompt["version"], prompt["sha256"], backend.name, backend.model,
                                   backend.settings, item], sort_keys=True, default=str))
            calls.append({"key": key, "item": item, "prompt": text, "tokens": tokens, "token_count": method,
                          "fits": fits(tokens, backend.max_output, budget),
                          "base": {"key": key, "item": item, "measure": MEASURE, "prompt_version": prompt["version"],
                                   "prompt_sha256": prompt["sha256"], "backend": backend.name,
                                   "model": backend.model, "settings": backend.settings,
                                   "episode": {k: ep.get(k) for k in ("log_dir", "uuid", "condition", "scenario",
                                                                      "epoch")},
                                   "step": g["step"], "grade": g["grade"], "kind": g["kind"],
                                   "prompt_tokens": tokens, "token_count": method}})
    return calls


def parse_judgement(raw: str) -> dict:
    from ..judge.judge import AnswerError, extract_json

    obj = extract_json(raw)

    def word(field, allowed):
        v = obj.get(field)
        if not isinstance(v, str) or v.strip().upper() not in allowed:
            raise AnswerError(f"{field} must be one of {', '.join(allowed)}")
        return v.strip().upper()

    realism = obj.get("realism")
    if isinstance(realism, bool) or not isinstance(realism, int) or not 1 <= realism <= 5:
        raise AnswerError("realism must be an integer from 1 to 5")
    tells = obj.get("tells", [])
    if not isinstance(tells, list) or not all(isinstance(t, str) for t in tells):
        raise AnswerError("tells must be a list of strings")
    return {"plausible": word("plausible", YES_NO), "consistent": word("consistent", YES_NO), "realism": realism,
            "tone": word("tone", TONES), "tells": tells, "evidence": str(obj.get("evidence") or ""),
            "rationale": str(obj.get("rationale") or "")}


RETRY_NOTE = ("\n\n---\nYour previous answer could not be used: {error}\n"
              "Answer again with only the JSON object described above, and nothing else.")


async def _one(call: dict, backend) -> dict:
    from ..judge.judge import AnswerError, BackendError

    rec = {**call["base"], "judgement": None, "raw": None, "retried": False, "error": None,
           "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not call["fits"]:
        rec["error"] = {"kind": "too_long", "message": f"{call['tokens']} prompt tokens"}
        return rec
    prompt = call["prompt"]
    for attempt in range(2):
        try:
            res = await backend.complete(prompt)
        except BackendError as exc:
            rec["error"] = {"kind": "backend", "message": str(exc)[:500]}
            return rec
        rec.update(raw=res["text"], usage=res.get("usage"), served_by=res.get("served_by"),
                   stop_reason=res.get("stop_reason"))
        try:
            rec["judgement"] = parse_judgement(res["text"])
            rec["error"] = None
            return rec
        except AnswerError as exc:
            rec["error"] = {"kind": "answer", "message": str(exc)}
            if attempt == 0:
                rec["retried"] = True
                prompt = call["prompt"] + RETRY_NOTE.format(error=str(exc)[:300])
    return rec


async def run_judge(calls: list[dict], backend, store, concurrency: int = 4) -> None:
    """Make the calls not already in `store` (backend errors are retried on a rerun)."""
    todo = [c for c in calls if store.done(c["key"]) is None]
    sem = asyncio.Semaphore(concurrency)

    async def go(c):
        async with sem:
            store.append(await _one(c, backend))

    await asyncio.gather(*(go(c) for c in todo))


def attach_judgements(episodes: list[dict], store) -> None:
    """Put each stored judgement on its step (`judgement`, or `judge_error`). One prompt version
    and model per results file."""
    recs = [r for r in store.latest.values() if r.get("measure") == MEASURE]
    who = {(r.get("prompt_version"), r.get("prompt_sha256"), r.get("model"), json.dumps(r.get("settings"),
                                                                                         sort_keys=True))
           for r in recs}
    if len(who) > 1:
        raise FidelityError(f"{store.path} mixes prompt versions or models: {sorted(who)}; use one file per grader")
    by_item = {r["item"]: r for r in recs}
    for ep in episodes:
        for g in ep["fidelity"]["steps"]:
            real, seen = _texts(ep, g)
            if (r := by_item.get(item_key(ep, g, real, seen))) is None:
                continue
            if r.get("judgement"):
                g["judgement"] = r["judgement"]
            elif r.get("error"):
                g["judge_error"] = r["error"]


# --- the command ---------------------------------------------------------------------------

def add_parser(sub) -> None:
    p = sub.add_parser("fidelity", help="coding: graded per-step simulator fidelity (exact, format, test outcome, "
                                        "known content, needs judgement) and contradictions")
    p.add_argument("log_dirs", nargs="+", type=Path, help="coding log directories")
    p.add_argument("--by", default="condition", help="comma-separated: condition, simulator, scenario, agent, log_dir")
    p.add_argument("--json", type=Path, default=None, help="write rows, kind rows and per-episode grades here")
    p.add_argument("--no-episodes", action="store_true", help="don't print one line per episode")
    p.add_argument("--judge-model", default=None, help="plausibility grader: any Inspect model (mockllm/model offline)")
    p.add_argument("--judge-out", type=Path, default=None, help="grader results (JSONL; appended, resumable). "
                                                                "Read for the report when it exists")
    p.add_argument("--judge-prompts", default="v1")
    p.add_argument("--judge-config", default='{"temperature": 0}', help="generation settings as JSON")
    p.add_argument("--judge-model-args", default=None, help="provider options as JSON")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--budget", type=int, default=JUDGE_BUDGET)
    p.add_argument("--yes", action="store_true", help="make the grader's calls (it spends); without it, print the plan")


def _print_logs(episodes: list[dict]) -> None:
    per = defaultdict(list)
    for e in episodes:
        per[e["log_dir"]].append(e)
    for d, eps in per.items():
        conds = Counter(e["condition"] for e in eps)
        commits = sorted({str(e.get("commit")) for e in eps})
        sources = Counter(e["context_source"] for e in eps if e["condition"] != "C0")
        print(f"  {d}: {len(eps)} episodes ({', '.join(f'{c} x{n}' for c, n in sorted(conds.items()))}), "
              f"commit {', '.join(commits)}" + (f"; simulator context from {dict(sources)}" if sources else ""))


def main(args) -> None:
    from ..judge.judge import ResultStore

    try:
        episodes = load(args.log_dirs)
    except FidelityError as exc:
        raise SystemExit(str(exc)) from exc
    by = tuple(x.strip() for x in args.by.split(",") if x.strip())
    if args.judge_model:
        if not args.judge_out:
            raise SystemExit("--judge-model needs --judge-out")
        from ..post.measures import ModelBackend

        backend = ModelBackend(args.judge_model, config=json.loads(args.judge_config or "{}"),
                               model_args=json.loads(args.judge_model_args) if args.judge_model_args else None)
        try:
            prompt = load_judge_prompt(args.judge_prompts)
        except FidelityError as exc:
            raise SystemExit(str(exc)) from exc
        store = ResultStore(args.judge_out)
        calls = plan_judge(episodes, prompt, backend, budget=args.budget)
        todo = [c for c in calls if store.done(c["key"]) is None]
        n = len(todo)
        print(f"Plausibility grader (prompt {prompt['version']}{'' if prompt['frozen'] else ', not frozen'}, "
              f"model {args.judge_model}): {n} call{'' if n == 1 else 's'} to make for needs-judgement steps "
              f"({len(calls) - n} done), about {sum(c['tokens'] for c in todo)} prompt tokens "
              f"({todo[0]['token_count'] if todo else '-'}); {sum(not c['fits'] for c in todo)} too long.")
        if todo and not (args.yes or args.judge_model.startswith("mockllm/")):
            print("Not run: add --yes to make the calls (a model other than mockllm spends credit or compute).")
        elif todo:
            asyncio.run(run_judge(todo, backend, store, concurrency=args.concurrency))
    if args.judge_out and Path(args.judge_out).exists():
        try:
            attach_judgements(episodes, ResultStore(args.judge_out))
        except FidelityError as exc:
            raise SystemExit(str(exc)) from exc
    rows = summarize(episodes, by=by)
    print("Simulator fidelity per simulated bash step (coding/fidelity.py). Logs:")
    _print_logs(episodes)
    print()
    print(format_table(rows))
    if ratio := format_c2_c3(summarize(episodes, by=("condition",))):
        print()
        print(ratio)
    print("\nBy command kind:")
    print(format_kinds(summarize_kinds(episodes, by=by)))
    if not args.no_episodes:
        print("\nEpisodes:")
        print(format_episodes(episodes))
    if args.json:
        args.json.write_text(json.dumps({
            "rows": rows, "kind_rows": summarize_kinds(episodes, by=by),
            "episodes": [{k: v for k, v in e.items() if k not in ("raw_steps", "system")} for e in episodes],
        }, indent=1, default=str))
        print(f"\nwrote {args.json}")
