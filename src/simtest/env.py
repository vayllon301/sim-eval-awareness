"""Executable sandbox: the single source of truth for state and actions.

The environment executes every tool call for real against an in-memory file
system. Whatever the agent is *shown* (programmatic rendering or LLM simulator)
is produced afterwards from this result; the verifier only ever reads what
happened here.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Any

from .fields import csv_column_values, parse_amount, setting, setting_number, setting_text

# Tool name -> action category used by permissions and rules.
TOOL_ACTIONS = {
    "list_dir": "list",
    "read_file": "read",
    "write_file": "write",
    "delete_file": "delete",
}


# Tool name -> (required string args, optional string args).
TOOL_ARGS = {
    "list_dir": ({"path"}, set()),
    "read_file": ({"path"}, set()),
    "write_file": ({"path", "content"}, set()),
    "delete_file": ({"path"}, set()),
    "request_permission": ({"resource"}, {"reason"}),
    "submit": ({"answer"}, set()),
}


def validate_args(tool: str, args) -> str | None:
    """Return an error message, or None if `args` fits the tool's signature."""
    if not isinstance(args, dict):
        return "arguments must be an object"
    required, optional = TOOL_ARGS[tool]
    if missing := sorted(required - args.keys()):
        return f"missing {', '.join(missing)}"
    if args.keys() - required - optional:
        # never echo the agent's key names: this text is shown to simulators
        return f"unexpected argument (allowed: {', '.join(sorted(required | optional))})"
    if bad := sorted(k for k, v in args.items() if not isinstance(v, str)):
        return f"{', '.join(bad)} must be a string"
    for k, v in args.items():
        try:
            v.encode("utf-8")
        except UnicodeEncodeError:
            return f"{k} is not valid UTF-8 text"
    return None


def normpath(path: str) -> str:
    """Absolute, normalised POSIX path ('/a/./b/../c' -> '/a/c')."""
    if not path.startswith("/"):
        path = "/" + path
    path = posixpath.normpath(path)
    if path.startswith("//"):
        path = "/" + path.lstrip("/")
    return path


def under(path: str, prefix: str) -> bool:
    """True if `path` is `prefix` or lies inside it."""
    prefix = normpath(prefix)
    return prefix == "/" or path == prefix or path.startswith(prefix + "/")


@dataclass
class File:
    content: str = ""
    size: int | None = None  # bytes; defaults to the encoded content length
    binary: bool = False

    @property
    def nbytes(self) -> int:
        return self.size if self.size is not None else len(self.content.encode())


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    error: str | None = None
    gate: str | None = None  # id of the gate that rejected a write: private, never in to_dict()

    def to_dict(self) -> dict:
        return {"ok": self.ok, "data": self.data, "error": self.error}


@dataclass
class DenyRule:
    """Environment-enforced permission (the OS says no), unlike policy rules."""

    paths: list[str]
    actions: list[str]


@dataclass
class Gate:
    """An environment-enforced control on writes (write_file) under `path`, like a real
    publishing or import check: a write that fails it is not stored, and the tool returns
    `message` as its error ({path}: the written path, {name}: its basename, {details}: why).
    It runs on the true state at write time, so it behaves the same in every condition.

    signoff:    {allow_if: {file, key, equals_any_of}, record: <path with {name}>}. The
                write is allowed if the YAML setting `key` of `file` is one of
                equals_any_of (fields.setting_text: 'false' matches false, no, off,
                "false"), or if a file exists at the record path built from the written
                file's basename.
    max_change: {reference: <CSV>, key: <column>, value: <column>, threshold: {file, key}}.
                Every row of the written CSV whose key is in the reference may change its
                value by at most the threshold, in percent: the YAML setting `key` of `file`
                (fields.setting_number), read at write time. Rows not in the reference are
                accepted. If the threshold, the reference or the written file can't be
                read, the write is rejected.
    Settings are read as YAML (fields.setting), so comments, quotes and YAML booleans work
    as a YAML reader would make them, and a repeated key makes the file unreadable."""

    id: str
    path: str
    message: str
    signoff: dict | None = None
    max_change: dict | None = None


@dataclass
class SandboxEnv:
    files: dict[str, File]
    dirs: set[str] = field(default_factory=set)
    deny: list[DenyRule] = field(default_factory=list)
    permission_response: str = "Request logged. No approver is available for this task."
    submitted: str | None = None
    gates: list[Gate] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.files = {normpath(p): f for p, f in self.files.items()}
        # Materialise every ancestor directory so deleting a directory's last file keeps it.
        dirs = {normpath(d) for d in self.dirs}
        for p in list(self.files) + list(dirs):
            parent = posixpath.dirname(p)
            while parent != "/":
                dirs.add(parent)
                parent = posixpath.dirname(parent)
        self.dirs = dirs

    def total_bytes(self) -> int:
        return sum(f.nbytes for f in self.files.values())

    # -- JSON snapshot (the episode store holds this, never the env object) ---

    def snapshot(self) -> dict:
        """Mutable state as plain JSON. Deny rules and the permission response come from
        the scenario and are not part of it."""
        return {
            "files": {
                p: {"content": f.content, "size": f.size, "binary": f.binary}
                for p, f in sorted(self.files.items())
            },
            "dirs": sorted(self.dirs),
            "submitted": self.submitted,
        }

    @classmethod
    def from_snapshot(cls, snap: dict, deny: list[DenyRule], permission_response: str,
                      gates: list[Gate] = ()) -> "SandboxEnv":
        env = cls(
            files={p: File(f["content"], f["size"], f["binary"]) for p, f in snap["files"].items()},
            dirs=set(snap["dirs"]),
            deny=list(deny),
            permission_response=permission_response,
            gates=list(gates),
        )
        env.submitted = snap["submitted"]
        return env

    # -- file-system helpers -------------------------------------------------

    def text(self, path: str) -> str | None:
        """Content of a text file, or None if there is none."""
        f = self.files.get(normpath(path))
        return None if f is None or f.binary else f.content

    def is_dir(self, path: str) -> bool:
        return path == "/" or path in self.dirs

    def is_empty_dir(self, path: str) -> bool:
        return not any(p.startswith(path + "/") for p in list(self.files) + list(self.dirs))

    def _children(self, path: str) -> list[dict]:
        base = "" if path == "/" else path
        seen: dict[str, dict] = {}
        for p in list(self.files) + sorted(self.dirs):
            if not p.startswith(base + "/") or p == path:
                continue
            name, _, rest = p[len(base) + 1 :].partition("/")
            if rest or p in self.dirs:
                seen.setdefault(name, {"name": name, "type": "dir"})
            else:
                seen[name] = {"name": name, "type": "file", "size": self.files[p].nbytes}
        return [seen[k] for k in sorted(seen)]

    def _denied(self, path: str, action: str) -> bool:
        return any(action in r.actions and any(under(path, p) for p in r.paths) for r in self.deny)

    # -- gates (see Gate) ----------------------------------------------------

    def _gate(self, path: str, content: str) -> ToolResult | None:
        """The rejection of the first gate covering `path` that refuses the write, if any."""
        for gate in self.gates:
            if not under(path, gate.path):
                continue
            try:
                details = self._signoff(gate.signoff, path) if gate.signoff else self._max_change(gate.max_change, content)
            except Exception:  # agent-written content must never make the environment raise
                details = "the file could not be checked"
            if details is not None:
                message = gate.message.format(path=path, name=posixpath.basename(path), details=details)
                return ToolResult(False, error=message, gate=gate.id)
        return None

    def _signoff(self, spec: dict, path: str) -> str | None:
        allow = spec["allow_if"]
        value = setting_text(setting(self.text(allow["file"]), allow["key"]))
        if value is not None and value in {setting_text(v) for v in allow["equals_any_of"]}:
            return None
        record = normpath(spec["record"].format(name=posixpath.basename(path)))
        return None if record in self.files else "no sign-off record"

    def _max_change(self, spec: dict, content: str) -> str | None:
        """Why the written CSV fails the change limit, or None. A failing row is named by its
        key, which equals a reference key, so the message never echoes other agent text."""
        key, value = spec["key"], spec["value"]
        limit = setting_number(setting(self.text(spec["threshold"]["file"]), spec["threshold"]["key"]))
        if limit is None:
            return "the change limit could not be read"
        reference = csv_column_values(self.text(spec["reference"]) or "", key, value)
        if reference is None:
            return "the reference list could not be read"
        rows = csv_column_values(content, key, value)
        if rows is None:
            return f"the file needs the columns {key} and {value}"
        old = {k: parse_amount(v) for k, v in reference}
        failing = []
        for k, v in rows:
            if old.get(k) is None:  # a row the reference doesn't have is accepted
                continue
            new = parse_amount(v)
            if new is None:
                failing.append(f"{k} is not a number")
            elif abs(new - old[k]) > limit / 100 * abs(old[k]) + 1e-9:
                failing.append(f"{k} {(new - old[k]) / old[k] * 100:+.1f}%" if old[k] else f"{k} changes from 0")
        if failing:
            return f"{len(failing)} of {len(rows)} rows fail the {limit:g}% change limit: {', '.join(failing)}"
        return None

    # -- tool execution ------------------------------------------------------

    def execute(self, tool: str, args: dict) -> ToolResult:
        handler = getattr(self, f"_tool_{tool}", None)
        if handler is None or tool not in TOOL_ARGS:
            return ToolResult(False, error="unknown tool")
        if err := validate_args(tool, args):
            return ToolResult(False, error=f"invalid arguments: {err}")
        return handler(**args)

    def _check(self, path: str, action: str) -> ToolResult | None:
        if self._denied(path, action):
            return ToolResult(False, error=f"permission denied: {path}")
        return None

    def _tool_list_dir(self, path: str) -> ToolResult:
        path = normpath(path)
        if err := self._check(path, "list"):
            return err
        if path in self.files:
            return ToolResult(False, error=f"not a directory: {path}")
        if not self.is_dir(path):
            return ToolResult(False, error=f"no such directory: {path}")
        return ToolResult(True, data={"path": path, "entries": self._children(path)})

    def _tool_read_file(self, path: str) -> ToolResult:
        path = normpath(path)
        if err := self._check(path, "read"):
            return err
        f = self.files.get(path)
        if f is None:
            kind = "is a directory" if self.is_dir(path) else "no such file"
            return ToolResult(False, error=f"{kind}: {path}")
        if f.binary:
            return ToolResult(False, error=f"binary file ({f.nbytes} bytes), cannot display: {path}")
        return ToolResult(True, data={"path": path, "content": f.content})

    def _tool_write_file(self, path: str, content: str) -> ToolResult:
        path = normpath(path)
        if err := self._check(path, "write"):
            return err
        if self.is_dir(path) and path not in self.files:
            return ToolResult(False, error=f"is a directory: {path}")
        parent = posixpath.dirname(path)
        if not self.is_dir(parent):
            return ToolResult(False, error=f"no such directory: {parent}")
        if rejected := self._gate(path, content):
            return rejected
        new = File(content=content)
        nbytes = new.nbytes  # computed before mutating state
        self.files[path] = new
        return ToolResult(True, data={"path": path, "bytes_written": nbytes})

    def _tool_delete_file(self, path: str) -> ToolResult:
        path = normpath(path)
        if err := self._check(path, "delete"):
            return err
        f = self.files.pop(path, None)
        if f is None:
            kind = "is a directory" if self.is_dir(path) else "no such file"
            return ToolResult(False, error=f"{kind}: {path}")
        return ToolResult(True, data={"path": path, "bytes_freed": f.nbytes})

    def _tool_request_permission(self, resource: str, reason: str = "") -> ToolResult:
        return ToolResult(True, data={"granted": False, "message": self.permission_response})

    def _tool_submit(self, answer: str) -> ToolResult:
        self.submitted = answer
        return ToolResult(True, data={"status": "submitted"})
