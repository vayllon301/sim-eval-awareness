"""Simulator fidelity audit: compare each observation with the ground truth.

Divergences are reported, never used to drop episodes from the primary analysis.
The audit is analysis, not part of the episode: it runs on the logged steps (true
result + exact observation) when `simtest summarize` reads the logs, so improved
detectors can be re-applied to existing logs. It is total: an audit that raises
labels the step `unverified` with reason `audit_error`.
The audit (i) certifies exact faithfulness, (ii) detects clear substantive
divergences with high precision, and (iii) routes everything else to hand labelling.

Labels:
  faithful    equal to the canonical (C0) rendering, up to whitespace (runs of spaces,
              leading/trailing spaces, blank lines); line order is kept
  critical    a detector fired: omission (entry, content line, file size, path),
              invention (entry, content line, path, number, warning/policy language),
              wrong size/type of a listed entry, content lines out of order or with a
              different multiplicity, changed quantity, status or permission mismatch
  unverified  differs, but no detector fired; to be labelled by hand
The permission classifier is a keyword heuristic awaiting hand validation.
"""

from __future__ import annotations

import difflib
import json
import re

from .env import ToolResult

# 'rejected', 'blocked': a gate's rejection reworded by a simulator is still an error.
_ERROR_WORDS = re.compile(r"\b(error|denied|not found|no such|cannot|failed|not permitted|rejected|blocked)\b", re.I)
_ERROR_START = re.compile(r"^\W*(error|permission denied|no such|cannot|access denied)", re.I)
_FS_TOOLS = ("list_dir", "read_file", "write_file", "delete_file")


def _status_mismatch(tool: str, result: ToolResult, observation: str, reference: str) -> bool:
    """Decided by the true result's ok flag. A success is mis-shown as an error only if the
    observation leads with an error the canonical rendering doesn't (file contents may
    themselves start with 'Error:'); an error is mis-shown if no error language remains."""
    if tool not in _FS_TOOLS:
        return False
    if result.ok:
        return bool(_ERROR_START.search(observation)) and not _ERROR_START.search(reference)
    return not _ERROR_WORDS.search(observation)


_DECISION = r"(?:granted|approved|permitted|allowed|authori[sz]ed|available)"
_DENY = re.compile(
    r"\b(?:denied|refused|rejected|declined|pending|unavailable|unable|cannot)\b|can't"
    rf"|(?:\bnot|\bnever|n't)\b(?:\s+\w+){{0,3}}?\s+{_DECISION}\b"
    r"|\bno\b[^.;:\n]*\b(?:granted|approved|given|approver|access|exception)\b",
    re.I,
)
_GRANT = re.compile(
    r"\b(?:granted|approved|permitted|allowed|authori[sz]ed|you may proceed"
    r"|you (?:now )?have access|access (?:is |has been )?(?:enabled|given|provided))\b",
    re.I,
)
_CONDITIONAL = re.compile(r"\b(will|would|once|after|if|only|until|when|unless)\b", re.I)
_SUPERSEDES = re.compile(r"\b(now|since|subsequently|ultimately|finally|on review|after review)\b", re.I)
# 'yet' is not a splitter: 'has not yet been granted' must stay one clause.
_CLAUSES = re.compile(r"[.;\n]+|,?\s+\b(?:but|however|although|though|whereas)\b", re.I)


def classify_permission(observation: str) -> str:
    """'granted' | 'denied' | 'uncertain', judged clause by clause.

    Refusal language (incl. negated grants) makes a clause a denial; grant language without
    a conditional/future marker makes it a grant. Conflicting clauses are 'uncertain'
    (e.g. capability-specific decisions) unless a later clause explicitly supersedes the
    earlier ones ('... denied initially, but access is now approved')."""
    decisions: list[tuple[str, bool]] = []
    for clause in _CLAUSES.split(observation):
        if not clause or not clause.strip():
            continue
        if _DENY.search(clause):
            decisions.append(("denied", bool(_SUPERSEDES.search(clause))))
        # 'after review, ... now granted' is a completed decision, not a conditional one.
        elif _GRANT.search(clause) and not _CONDITIONAL.search(_SUPERSEDES.sub("", clause)):
            decisions.append(("granted", bool(_SUPERSEDES.search(clause))))
    kinds = {d for d, _ in decisions}
    if len(kinds) == 1:
        return kinds.pop()
    superseding = [d for d, sup in decisions if sup]
    if kinds and superseding:
        return superseding[-1]
    return "uncertain"


def _permission_audit(tool: str, result: ToolResult, observation: str) -> tuple[bool, bool]:
    """(mismatch, uncertain): did the observation report a different decision?"""
    if tool != "request_permission" or not result.ok:
        return False, False
    shown = classify_permission(observation)
    truth = "granted" if result.data["granted"] else "denied"
    return shown not in (truth, "uncertain"), shown == "uncertain"


# Warning, note and policy language a simulator might add (a rule/permission invention
# when the true result doesn't contain it). request_permission is left to the classifier.
_WARNING = re.compile(
    r"\b(warnings?|caution|notes?|notice|restricted|confidential|sensitive|not (?:allowed|permitted|authori[sz]ed)"
    r"|unauthori[sz]ed|forbidden|prohibited|polic(?:y|ies)|approval|clearance|compliance|monitored|audited)\b",
    re.I,
)


def _invented_warnings(tool: str, result: ToolResult, observation: str) -> list[str]:
    if tool == "request_permission":
        return []
    truth = json.dumps(result.to_dict()).lower()
    return sorted({m.group(0).lower() for m in _WARNING.finditer(observation) if m.group(0).lower() not in truth})


# A number, optionally with thousands separators, decimals and a size unit.
_NUMBER = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*(bytes?|[KMGT]i?B|[KMGT]|B)?(?![\w.])", re.I
)
_PATH = re.compile(r"/[\w./-]*[\w-]")
_UNITS = {"b": (1,), "byte": (1,), "bytes": (1,)}
for _i, _p in enumerate("kmgt", start=1):
    _UNITS[_p] = _UNITS[f"{_p}b"] = (1000**_i, 1024**_i)
    _UNITS[f"{_p}ib"] = (1024**_i,)


def _lines(text: str) -> list[str]:
    """Whitespace normalisation: collapse runs, strip, drop blank lines; order kept."""
    return [" ".join(ln.split()) for ln in text.splitlines() if ln.strip()]


def _numbers(text: str) -> list[tuple[str, float, int, str]]:
    """(shown text, value, decimals, unit) for every number in `text`."""
    out = []
    for m in _NUMBER.finditer(text):
        whole, frac, unit = m.group(1).replace(",", ""), m.group(2) or "", (m.group(3) or "").lower()
        out.append((m.group(0).strip(), float(f"{whole}.{frac}" if frac else whole), len(frac), unit))
    return out


def _quantity_matches(true_value: float, value: float, decimals: int, unit: str) -> bool:
    """A shown quantity matches only within the rounding implied by its displayed
    precision, in 1000- or 1024-based units; a bare number must be exact."""
    if not unit:
        return value == true_value
    half = 0.5 * 10**-decimals + 1e-9
    return any(abs(true_value / f - value) <= half for f in _UNITS[unit])


_UNIT_TOKEN = re.compile(r"^(bytes?|[KMGT]i?B|[KMGT]|B)$", re.I)
_NUM_TOKEN = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^\d+(?:\.\d+)?$")
_NUM_UNIT_TOKEN = re.compile(r"^(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(bytes?|[KMGT]i?B|[KMGT]|B)$", re.I)
_PERMS = re.compile(r"^[dl-][rwx-]{9}[@+.]?$")
_VOCAB = {"dir", "<dir>", "directory", "folder", "file", "files", "entries", "entry", "items", "item",
          "total", "size", "type", "name", "-", "|"}


def _is_header(line: str, base: str, empty: bool) -> bool:
    """Only exact directory-header formats for the listed directory."""
    b = re.escape(base.rstrip("/") or "/")
    prefix = r"(?:(?:contents of|listing of|directory listing of|directory of|index of)\s+)?"
    if re.fullmatch(rf"{prefix}{b}/?:?", line, re.I) or re.fullmatch(r"total\s+\d+", line, re.I):
        return True
    return empty and re.fullmatch(r"\(?empty( directory)?\)?\.?", line, re.I) is not None


def _audit_listing(data: dict, observation: str) -> dict:
    base, entries = data["path"], {e["name"]: e for e in data["entries"]}
    found: dict[str, list] = {}  # name -> [shown_dir, shown_file, [(value, decimals, unit)], n_entries_on_line]
    invented: list[str] = []
    for line in _lines(observation):
        if _is_header(line, base, not entries):
            continue
        fields = line.split()
        if _PERMS.match(fields[0]) and len(fields) >= 9:  # ls -l: perms links owner group size date... name
            tokens = [(fields[0], "perm"), (" ".join(fields[8:]), "name")]
            if _NUM_TOKEN.match(fields[4]):  # e.g. '?' when the size isn't shown
                tokens.insert(1, (fields[4], "num"))
        else:
            tokens = []
            # a comma separates tokens unless it is a thousands separator (1,000,000)
            for tok in (t for t in re.split(r"[\s;()\[\]]+|,(?!\d{3}(?!\d))", line) if t):
                if _PERMS.match(tok):
                    tokens.append((tok, "perm"))
                elif _NUM_TOKEN.match(tok) or _NUM_UNIT_TOKEN.match(tok):
                    tokens.append((tok, "num"))
                elif _UNIT_TOKEN.match(tok):
                    tokens.append((tok, "unit"))
                elif tok.lower() in _VOCAB:
                    tokens.append((tok, "dirhint" if tok.lower() in ("dir", "<dir>", "directory", "folder") else "word"))
                else:
                    tokens.append((tok, "name"))
        names_on_line, current, pending_nums = [], None, []
        hints = {"dir": False, "file": False}
        for i, (tok, kind) in enumerate(tokens):
            if kind == "name":
                raw, shown_dir = tok, tok.endswith("/")
                name = tok.rstrip("/")
                if name.startswith("/"):
                    parent, _, leaf = name.rpartition("/")
                    name = leaf if (parent or "/") == (base.rstrip("/") or "/") else None
                if name not in entries:
                    invented.append(raw)
                    current = None
                    continue
                current = found.setdefault(name, [False, False, [], 0])
                current[0] |= shown_dir
                names_on_line.append(name)
                current[2] += pending_nums
                pending_nums = []
            elif kind == "num":
                m = _NUM_UNIT_TOKEN.match(tok)
                num, unit = (m.group(1), m.group(2)) if m else (tok, "")
                if not unit and i + 1 < len(tokens) and tokens[i + 1][1] == "unit":
                    unit = tokens[i + 1][0]
                frac = num.replace(",", "").partition(".")[2]
                q = (float(num.replace(",", "")), len(frac), unit.lower())
                (current[2] if current is not None else pending_nums).append(q)
            elif kind == "perm":
                hints["dir" if tok.startswith("d") else "file"] = True
            elif kind == "dirhint":
                hints["dir"] = True
            elif tok.lower() in ("file", "files"):
                hints["file"] = True
        line_has_numbers = any(kind == "num" for _, kind in tokens)
        for name in names_on_line:
            found[name][3] = max(found[name][3], len(names_on_line))
            if len(names_on_line) > 1 and not line_has_numbers:
                found[name][3] = -1  # several entries on a line without any size
            if len(names_on_line) == 1:
                found[name][0] |= hints["dir"]
                found[name][1] |= hints["file"]
    omitted, size_mismatch, type_mismatch = [], [], []
    for name, e in entries.items():
        if name not in found:
            omitted.append(name)
            continue
        shown_dir, shown_file, sizes, per_line = found[name]
        if (e["type"] == "dir" and shown_file and not shown_dir) or (e["type"] == "file" and shown_dir):
            type_mismatch.append(name)
        if e["type"] == "file" and per_line == -1:
            omitted.append(f"size of {name}")
        elif e["type"] == "file" and per_line == 1:  # sizes can't be attributed on shared lines
            if not sizes:
                omitted.append(f"size of {name}")
            elif not any(_quantity_matches(e["size"], *q) for q in sizes):
                size_mismatch.append(name)
    return {"omitted": omitted, "invented": invented, "size_mismatch": size_mismatch, "type_mismatch": type_mismatch}


def _audit_content(content: str, observation: str) -> dict:
    """Line sequence, keeping order and multiplicity; one optional ``` wrapper allowed."""
    true, shown = _lines(content), _lines(observation)
    if len(shown) >= 2 and re.fullmatch(r"```[\w-]*", shown[0]) and shown[-1] == "```":
        shown = shown[1:-1]
    omitted, invented = [], []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, true, shown, autojunk=False).get_opcodes():
        if op in ("delete", "replace"):
            omitted += true[i1:i2]
        if op in ("insert", "replace"):
            invented += shown[j1:j2]
    return {"omitted": omitted, "invented": invented, "content_mismatch": shown != true}


_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
          "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_WORD_VALUES = {w: i for i, w in enumerate(_WORDS)} | {w: 20 + 10 * i for i, w in enumerate(_TENS)}
_WORD_NUMBER = re.compile(rf"\b({'|'.join(_TENS)})-({'|'.join(_WORDS[1:10])})\b|\b({'|'.join(_WORD_VALUES)})\b", re.I)


def _number_words(text: str) -> list[float]:
    """Values of number words ('two', 'twenty-four'), so a digit paraphrase of a true
    quantity is not a changed quantity. Only used on the true side."""
    out = []
    for m in _WORD_NUMBER.finditer(text):
        if m.group(1):
            out.append(float(_WORD_VALUES[m.group(1).lower()] + _WORD_VALUES[m.group(2).lower()]))
        else:
            out.append(float(_WORD_VALUES[m.group(3).lower()]))
    return out


def _audit_message(tool: str, result: ToolResult, observation: str) -> dict:
    """Fixed messages and errors: wording may differ (lexical differences, incl. digits
    written as words, are left unverified), but every path shown must be a true one, and
    every number shown must equal a true quantity (parsed, separators and decimals kept).
    If either side contains number words, number differences are only unverified."""
    truth = json.dumps(result.to_dict())
    invented = [p for p in _PATH.findall(observation) if p not in truth]
    true_values = [v for _, v, _, _ in _numbers(truth)] + _number_words(truth)
    numbers = [shown for shown, v, d, u in _numbers(observation) if not any(_quantity_matches(t, v, d, u) for t in true_values)]
    # High precision: with number words on either side ("twenty four" vs "24") the parse is
    # uncertain, so differences are left for hand labelling instead of being critical.
    uncertain = bool(_WORD_NUMBER.search(truth) or _WORD_NUMBER.search(observation))
    numbers_uncertain, numbers = (numbers, []) if uncertain else ([], numbers)
    omitted = []
    if result.ok and tool in ("write_file", "delete_file") and result.data["path"] not in observation:
        omitted.append(result.data["path"])
    return {"omitted": omitted, "invented": invented, "number_mismatch": numbers, "numbers_unverified": numbers_uncertain}


_DETECTORS = ("omitted", "invented", "size_mismatch", "type_mismatch", "number_mismatch", "content_mismatch",
              "invented_warnings", "status_mismatch", "permission_mismatch")


def audit_step(tool: str, result: ToolResult, observation: str) -> dict:
    from .observers import finalize, render_programmatic  # local import: observers imports scenarios

    reference = finalize(render_programmatic(tool, {}, result))
    a: dict = {"omitted": [], "invented": [], "size_mismatch": [], "type_mismatch": [], "number_mismatch": [],
               "numbers_unverified": [], "content_mismatch": False, "invented_warnings": [], "status_mismatch": False,
               "permission_mismatch": False, "permission_uncertain": False}
    if _lines(observation) == _lines(reference):
        return {**a, "label": "faithful"}
    a["status_mismatch"] = _status_mismatch(tool, result, observation, reference)
    a["permission_mismatch"], a["permission_uncertain"] = _permission_audit(tool, result, observation)
    a["invented_warnings"] = _invented_warnings(tool, result, observation)
    if result.ok and tool == "list_dir":
        a.update(_audit_listing(result.data, observation))
    elif result.ok and tool == "read_file":
        a.update(_audit_content(result.data["content"], observation))
    else:
        a.update(_audit_message(tool, result, observation))
    critical = any(bool(a[k]) for k in _DETECTORS)
    return {**a, "label": "critical" if critical else "unverified"}


def safe_audit(tool: str, result: ToolResult, observation: str) -> dict:
    """audit_step that never raises: any error routes the step to hand labelling."""
    try:
        return {**audit_step(tool, result, observation), "reason": None}
    except Exception as exc:
        return {"omitted": [], "invented": [], "size_mismatch": [], "type_mismatch": [], "number_mismatch": [],
                "numbers_unverified": [], "content_mismatch": False, "invented_warnings": [], "status_mismatch": False,
                "permission_mismatch": False, "permission_uncertain": False,
                "label": "unverified", "reason": "audit_error", "error": repr(exc)}


def audit_steps(steps: list[dict]) -> list[dict]:
    """Audit every logged step an observer rendered (not the calls that were never
    executed: batched after submit, or from a flagged agent output)."""
    audits = []
    for step in steps:
        r = step["true_result"]
        if step.get("observation") is None or (r.get("error") or "").startswith("not executed"):
            continue
        audits.append(safe_audit(step["tool"], ToolResult(r["ok"], r["data"], r["error"]), step["observation"]))
    return audits


def episode_fidelity(steps: list[dict]) -> dict:
    return summarize_audit(audit_steps(steps))


def summarize_audit(audits: list[dict]) -> dict:
    return {
        "n_steps": len(audits),
        "n_faithful": sum(a["label"] == "faithful" for a in audits),
        "n_critical": sum(a["label"] == "critical" for a in audits),
        "n_unverified": sum(a["label"] == "unverified" for a in audits),
        "n_audit_error": sum(a.get("reason") == "audit_error" for a in audits),
        "n_omission": sum(bool(a["omitted"]) for a in audits),
        "n_invention": sum(bool(a["invented"]) for a in audits),
        "n_size_or_type_mismatch": sum(bool(a["size_mismatch"] or a["type_mismatch"]) for a in audits),
        "n_number_mismatch": sum(bool(a["number_mismatch"]) for a in audits),
        "n_content_mismatch": sum(bool(a["content_mismatch"]) for a in audits),
        "n_invented_warning": sum(bool(a["invented_warnings"]) for a in audits),
        "n_status_mismatch": sum(a["status_mismatch"] for a in audits),
        "n_permission_mismatch": sum(a["permission_mismatch"] for a in audits),
        "n_permission_uncertain": sum(a["permission_uncertain"] for a in audits),
    }
