"""Parsers for what an agent writes, shared by the verifier (success checks, flags) and
the environment (gates): `key: value` report lines (read as Markdown, with markdown-it),
dotenv assignments and amounts, and the field conditions built on them, amounts mentioned
in free text, CSV columns, and the YAML settings files gates read.

Nothing here may raise on agent-written text: a value that doesn't parse simply doesn't match.
"""

from __future__ import annotations

import csv
import io
import math
import re
from functools import lru_cache
from typing import NamedTuple

import yaml
from markdown_it import MarkdownIt

# Whole-value grammar: an optional '+' and currency (EUR, euro(s), €) before the number, an
# unsigned number, an optional "million", an optional currency after it, an optional note in
# parentheses, an optional footnote marker ('[1]', '[^1]', '*') and an optional final period.
# A grouped number has one kind of thousands separator and may end in a fraction marked with
# the other one: '4,843,230.00', '4.843.230,00', '4 843 230,00'.
_CURRENCY = r"(?:euros?|eur|€)"
_AMOUNT = re.compile(
    rf"(?:\+\s*{_CURRENCY}?|{_CURRENCY}\s*\+?)?\s*"
    r"(?P<num>\d{1,3}(?:(?P<sep>[,. \u00a0\u202f])\d{3})(?:(?P=sep)\d{3})*(?:(?P<dec>[.,])\d+)?|\d+(?:[.,]\d+)?)"
    rf"\s*(?P<unit>million|mn|m)?\s*{_CURRENCY}?\s*(?:\([^)]*\))?\s*(?:\[\^?\w{{1,4}}\]|\*{{1,3}}|[†‡])?\.?",
    re.I,
)


def parse_amount(text: str) -> float | None:
    """Parse a whole field value such as '4,840,000', '4,843,230.00', 'EUR 4.84 million',
    '4 840 000 €', '+4,843,230 euros', '4,843,230 [1]'.

    Returns None for anything else (signs, other units, a fraction marked like the thousands
    ('4,843,230,00'), ambiguous '4,840,000 million')."""
    m = _AMOUNT.fullmatch(text.strip())
    if not m:
        return None
    num, sep, dec, unit = m.group("num"), m.group("sep"), m.group("dec"), m.group("unit")
    if sep and dec and (dec == sep or unit):
        return None
    if sep and unit:  # '4.812 million' is a decimal; '4,840,000 million' is ambiguous
        return float(num.replace(",", ".")) * 1e6 if num.count(sep) == 1 and sep in ",." else None
    if sep:  # thousands separators, then an optional fraction
        whole, _, fraction = num.rpartition(dec) if dec else (num, "", "")
        return float(re.sub(r"[,. \u00a0\u202f]", "", whole) + (f".{fraction}" if fraction else ""))
    value = float(num.replace(",", "."))
    return value * 1e6 if unit else value


# As in sh: an inline comment needs whitespace before '#'; 'x#y' is a literal value.
_DOTENV_QUOTED = re.compile(r"""(['"])(.*?)\1(?:\s+#.*)?\s*""")
_DOTENV_PLAIN = re.compile(r"""([^\s'"]*)(?:\s+#.*)?\s*""")


def _dotenv_value(raw: str) -> str | None:
    """Value as a shell would set it, or None if the assignment is malformed."""
    raw = raw.strip()
    m = _DOTENV_QUOTED.fullmatch(raw) or _DOTENV_PLAIN.fullmatch(raw)
    return m.group(2) if m and m.re is _DOTENV_QUOTED else (m.group(1) if m else None)


# -- report lines, read as Markdown ----------------------------------------------------

_MARKDOWN = MarkdownIt("commonmark")


class MarkdownText(NamedTuple):
    """Text as read through Markdown: `plain` without emphasis, strong, code-span or link
    markup (link labels and code kept verbatim, so underscores in paths stay), and for each
    link where its label starts in `plain` and its destination."""

    plain: str
    links: tuple[tuple[int, str], ...] = ()


def _plain(children) -> MarkdownText:
    plain, links = "", []
    for tok in children or []:
        if tok.type == "link_open":
            links.append((len(plain), str(tok.attrGet("href") or "")))
        elif tok.type in ("text", "code_inline", "html_inline", "image"):  # an image: its alt text
            plain += tok.content
        elif tok.type in ("softbreak", "hardbreak"):
            plain += " "
    return MarkdownText(plain, tuple(links))


def markdown_text(source: str) -> MarkdownText:
    """A value read as inline Markdown."""
    try:
        return _plain(_MARKDOWN.parseInline(source.strip())[0].children)
    except Exception:  # agent-written text must never raise
        return MarkdownText(source.strip())


@lru_cache(maxsize=256)
def colon_fields(text: str) -> tuple[tuple[str, MarkdownText], ...]:
    """(key, value) per 'Key: value' report line, in order. Each line is read as Markdown
    (block markers such as bullets, numbers, headings and quotes, and the inline markup, are
    removed; see MarkdownText) and split at the first colon of its plain text, so
    '**Figures status:** final' and '**Figures status**: final' are the same field. Keys
    are lower-cased with whitespace collapsed."""
    out = []
    for line in text.splitlines():
        try:
            tokens = _MARKDOWN.parse(line.strip())
        except Exception:  # agent-written text must never raise
            tokens = []
        for tok in tokens:
            if tok.type != "inline":
                continue
            line_text = _plain(tok.children)
            i = line_text.plain.find(":")
            if i < 0:
                continue
            rest = line_text.plain[i + 1:]
            start = i + 1 + len(rest) - len(rest.lstrip())
            links = tuple((max(at - start, 0), href) for at, href in line_text.links if at > i)
            out.append((" ".join(line_text.plain[:i].split()).lower(), MarkdownText(rest.strip(), links)))
    return tuple(out)


def parse_field_lines(text: str, fmt: str) -> list[tuple[str, str | None]]:
    """(key, value) per assignment line, in order.

    dotenv: keys and values are exact (case kept); comments outside quotes ignored;
            a malformed assignment (unmatched quote, trailing text) maps to None.
    colon:  'Key: value' report lines read as Markdown (colon_fields); the plain values."""
    if fmt == "colon":
        return [(key, value.plain) for key, value in colon_fields(text)]
    if fmt != "dotenv":
        raise ValueError(f"unknown field format {fmt!r}")
    out: list[tuple[str, str | None]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        out.append((key.strip(), _dotenv_value(value)))
    return out


def parse_fields(text: str, fmt: str) -> dict[str, str | None]:
    """Key -> value; the last assignment wins (see parse_field_lines)."""
    return dict(parse_field_lines(text, fmt))


# A note after a report value: 'East (preliminary)', 'East; ...', 'East — ...', 'East - ...',
# 'East, ...'. In an amount a bare comma is a thousands separator, so only ', ' starts a
# note there ('(...)' right after an amount is also part of the amount grammar).
_NOTE = re.compile(r"\s*\(|;|\s[—–-]\s|,")
_AMOUNT_NOTE = re.compile(r"\s*\(|;|\s[—–-]\s|,\s")
# A template line left unfilled ('Best region: <region name>') or an empty value; a path or
# a URI in angle brackets ('</data/x.csv>', '<https://...>') is a value.
_PLACEHOLDER = re.compile(r"(?:<(?!/|[a-zA-Z][\w+.-]*:)[^<>]*>)?\.?")


def leading(value: MarkdownText | str, amount: bool = False) -> tuple[str, str]:
    """(value, citation) of a field value's leading part: its plain text (Markdown removed,
    link labels kept) up to the first note separator (see _NOTE).
    value:    that text; used by equals_any_of and amounts, so '*final*', '[final](#s)' and
              '[final (as requested)](#s)' all read 'final'.
    citation: that text and the destinations of the links that start in it; used by
              contains and not_contains, so '[Reconciled (final)](/data/restricted/x)' cites
              /data/restricted, while a link inside the note stays out."""
    text = markdown_text(value) if isinstance(value, str) else value
    note = (_AMOUNT_NOTE if amount else _NOTE).search(text.plain)
    end = note.start() if note else len(text.plain)
    lead = " ".join(text.plain[:end].split())
    hrefs = [href for at, href in text.links if note is None or at < end]
    return lead, " ".join([lead, *hrefs]).strip()


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


# A field condition has at most one of these; none means the field just has to be present.
FIELD_CONDITIONS = ("equals_any_of", "amount_any_of", "is_amount", "contains", "not_contains")


def _strings(value) -> list[str] | None:
    """A non-empty string or non-empty list of non-empty strings, as a list (None if neither)."""
    values = [value] if isinstance(value, str) else value
    if isinstance(values, list) and values and all(isinstance(v, str) and v for v in values):
        return values
    return None


def validate_field_spec(spec: dict, where: str = "field check", file: bool = True) -> None:
    """Fail at scenario load time, not mid-experiment, on a malformed field condition.
    `file`: the condition names its file (False for a flag's `when`, which uses the flag's)."""
    if not isinstance(spec, dict):
        raise ValueError(f"{where} must be a mapping: {spec!r}")
    allowed = {"key", "format", "rel_tol", "negate", *FIELD_CONDITIONS} | ({"file"} if file else set())
    if unknown := set(spec) - allowed:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)} in {spec}")
    if file and not (isinstance(spec.get("file"), str) and spec["file"].startswith("/")):
        raise ValueError(f"{where} needs an absolute 'file': {spec}")
    if not isinstance(spec.get("key"), str) or not spec["key"]:
        raise ValueError(f"{where} needs a 'key': {spec}")
    if spec.get("format", "colon") not in ("colon", "dotenv"):
        raise ValueError(f"{where}: unknown field format in {spec}")
    if len([c for c in FIELD_CONDITIONS if c in spec]) > 1:
        raise ValueError(f"{where} takes at most one of {', '.join(FIELD_CONDITIONS)}: {spec}")
    if "equals_any_of" in spec:
        vals = spec["equals_any_of"]
        if not (isinstance(vals, list) and vals and all(isinstance(v, (str, int)) for v in vals)):
            raise ValueError(f"{where}: equals_any_of must be a non-empty list of strings: {spec}")
    if "amount_any_of" in spec:
        vals = spec["amount_any_of"]
        if not (isinstance(vals, list) and vals and all(_is_number(v) and v > 0 for v in vals)):
            raise ValueError(f"{where}: amount_any_of must be a non-empty list of positive numbers: {spec}")
    if "is_amount" in spec and spec["is_amount"] is not True:
        raise ValueError(f"{where}: is_amount takes only true (the field has a readable amount): {spec}")
    if "rel_tol" in spec and not (_is_number(spec["rel_tol"]) and 0 <= spec["rel_tol"] <= 0.1):
        raise ValueError(f"{where}: rel_tol must be a number in [0, 0.1]: {spec}")
    for cond in ("contains", "not_contains"):
        if cond in spec and _strings(spec[cond]) is None:
            raise ValueError(f"{where}: {cond} must be a non-empty string or list of strings: {spec}")
    if not isinstance(spec.get("negate", False), bool):
        raise ValueError(f"{where}: negate must be true or false: {spec}")


def _normalised(spec: dict, value: MarkdownText | str | None):
    """The comparable form of one value under the spec's condition (None: no value)."""
    if value is None:
        return None
    if spec.get("format", "colon") == "dotenv":
        return parse_amount(value) if "amount_any_of" in spec else value
    if "amount_any_of" in spec or "is_amount" in spec:
        return parse_amount(leading(value, amount=True)[0])
    if "equals_any_of" in spec:
        return leading(value)[0].lower().rstrip(".").strip()
    if "contains" in spec or "not_contains" in spec:
        return leading(value)[1]  # the citation: with link destinations, not a note
    return value.plain


def field_value(spec: dict, content: str):
    """The field's one value, normalised for the spec's condition, or None if it is missing
    or ambiguous. dotenv: the last assignment wins, as in a shell. colon: lines whose value
    is a template placeholder or empty are ignored; repeats of the same value are fine,
    two different values are ambiguous."""
    if spec.get("format", "colon") == "dotenv":
        values = [v for k, v in parse_field_lines(content, "dotenv") if k == spec["key"]][-1:]
    else:
        key = " ".join(spec["key"].split()).lower()
        values = [v for k, v in colon_fields(content) if k == key and not _PLACEHOLDER.fullmatch(v.plain)]
    normalised = {_normalised(spec, v) for v in values}
    return normalised.pop() if len(normalised) == 1 else None


def field_holds(spec: dict, content: str | None) -> bool:
    """A field condition on a file's content (None: the file doesn't exist): the field has
    one readable value (field_value) and it meets the condition. equals_any_of: the leading
    value, case-insensitive (dotenv: exact). amount_any_of: exact (up to float rounding)
    unless the spec sets rel_tol. contains / not_contains: the leading value (as for
    equals_any_of: the primary citation, before a note) contains one / none of the strings,
    case-insensitive. No condition: the value is not empty.
    negate: true inverts the check on the value, never the value's existence: a missing,
    ambiguous or unreadable field ('Total: NOT AVAILABLE' for an amount) fails either way.
    So amount_any_of: [4813430] with negate reads 'a total other than 4,813,430'."""
    if content is None:
        return False
    value = field_value(spec, content)
    return value is not None and _meets(spec, value) is not bool(spec.get("negate", False))


def _meets(spec: dict, value) -> bool:
    if "equals_any_of" in spec:
        if spec.get("format", "colon") == "dotenv":
            return value in {str(v) for v in spec["equals_any_of"]}
        return value in {str(v).lower() for v in spec["equals_any_of"]}
    if "amount_any_of" in spec:
        return any(same_amount(value, t, spec.get("rel_tol", 0)) for t in spec["amount_any_of"])
    if "is_amount" in spec:  # the value is an amount: field_value read it as one
        return True
    if "contains" in spec:
        return any(x.lower() in value.lower() for x in _strings(spec["contains"]))
    if "not_contains" in spec:
        return not any(x.lower() in value.lower() for x in _strings(spec["not_contains"]))
    return bool(value.strip())


def same_amount(a: float | None, b: float | None, rel_tol: float = 0) -> bool:
    """Equal amounts; 1e-9 covers float rounding ('4.84323 million')."""
    return a is not None and b is not None and abs(a - b) <= max(rel_tol, 1e-9) * abs(b)


# Thousands separators an amount may be written with in free text.
_SEP = r"[,. \u00a0\u202f']"


def mentions_amount(text: str, amount: int) -> bool:
    """The text mentions the whole number `amount`, plain or with one kind of thousands
    separator ('4813430', '4,813,430', '4.813.430', '4 813 430'). A longer number doesn't
    count ('14,813,430', '4,813,430,000'); a comma after a plain amount does (CSV)."""
    digits = str(amount)
    groups = [digits[max(0, i - 3):i] for i in range(len(digits), 0, -3)][::-1]
    pattern = rf"(?<!\d){digits}(?!\d)"
    if len(groups) > 1:
        sep = groups[0] + "".join(rf"(?P=sep){g}" if i else rf"(?P<sep>{_SEP}){g}" for i, g in enumerate(groups[1:]))
        pattern += rf"|(?<!\d){sep}(?!\d|(?P=sep)\d)"
    return re.search(pattern, text) is not None


def contains_any(text: str, strings: list[str], amounts: list[int]) -> bool:
    """The text contains one of the strings (case-insensitive) or mentions one of the amounts."""
    low = text.lower()
    return any(s.lower() in low for s in strings) or any(mentions_amount(text, a) for a in amounts)


def csv_column_values(content: str, key: str, value: str) -> list[tuple[str, str]] | None:
    """(key, value) per data row of a CSV whose header has both columns (names compared after
    stripping whitespace), or None if the text isn't such a CSV. Cells are stripped; rows
    with an empty key are skipped."""
    try:
        rows = [r for r in csv.reader(io.StringIO(content)) if any(c.strip() for c in r)]
    except csv.Error:
        return None
    header = [c.strip() for c in rows[0]] if rows else []
    if key not in header or value not in header:
        return None
    i, j = header.index(key), header.index(value)
    return [(r[i].strip(), r[j].strip() if j < len(r) else "") for r in rows[1:] if i < len(r) and r[i].strip()]


# -- YAML settings (gates) ----------------------------------------------------------


_MERGE = "tag:yaml.org,2002:merge"


def _repeats_a_key(root: yaml.Node) -> bool:
    """Some mapping in the composed YAML graph repeats a key: merged mappings included, merge
    keys counted as keys, checked before any merge is resolved and without changing a node
    (PyYAML's own loader keeps the last of repeated keys silently)."""
    constructor = yaml.constructor.SafeConstructor()
    seen, todo = set(), [root]
    while todo:
        node = todo.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, yaml.MappingNode):
            keys = ["<<" if k.tag == _MERGE else repr(constructor.construct_object(k)) if isinstance(k, yaml.ScalarNode)
                    else id(k) for k, _ in node.value]
            if len(keys) != len(set(keys)):
                return True
            todo += [n for pair in node.value for n in pair]
        elif isinstance(node, yaml.SequenceNode):
            todo += node.value
    return False


def setting(content: str | None, key: str):
    """A top-level setting of a YAML file, as a YAML 1.1 reader sees it: comments dropped,
    true/false, yes/no and on/off (any case) booleans, numbers numbers, merge keys merged
    (an explicit key overrides a merged one). A quoted scalar is read as if unquoted, so
    '"false"' is false and '"20"' is 20. None if the file is missing, isn't one YAML
    document or a mapping, repeats a key in any mapping (a merged one included), or lacks
    the key (YAML keys are exact)."""
    if content is None:
        return None
    try:
        root = yaml.compose(content, Loader=yaml.SafeLoader)
        if root is None or _repeats_a_key(root):
            return None
        doc = yaml.safe_load(content)  # resolves merge keys as YAML does
    except Exception:  # agent-written text: malformed YAML, several documents, deep nesting
        return None
    if not isinstance(doc, dict) or key not in doc:
        return None
    value = doc[key]
    if isinstance(value, str):
        try:
            unquoted = yaml.safe_load(value)
        except Exception:
            unquoted = value
        if isinstance(unquoted, (bool, int, float)):
            value = unquoted
    return value


def setting_text(value) -> str | None:
    """A setting as text to compare: booleans 'true'/'false', numbers as written by Python,
    strings stripped and lower-cased; None for anything else (null, lists, mappings)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return value.strip().lower() if isinstance(value, str) else None


def setting_number(value) -> float | None:
    """A setting as a finite number: a YAML number, or a string amount with an optional '%'
    ('20%', '20 %', '12.5'); None for booleans and anything else."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        number = parse_amount(value.strip().removesuffix("%").strip())
    else:
        return None
    return number if number is not None and math.isfinite(number) else None
