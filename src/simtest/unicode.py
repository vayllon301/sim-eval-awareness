"""Model-produced text must never be able to make an episode raise.

A lone surrogate ("\\ud800", which the OpenAI SDK accepts from an escaped JSON
response) can't be encoded as UTF-8, so Inspect would fail when serialising the
sample, and eval_set would resample the episode. One generic mechanism handles it:
`sanitize_output` round-trips the whole ModelOutput through model_dump() /
model_validate() with every string (keys and values, any depth) cleaned, without
knowing about individual fields. `clean` is the same cleaner for anything we store.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.model import ModelOutput


def clean_text(text: str) -> str:
    """Replace lone surrogates with U+FFFD (valid pairs are kept)."""
    return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def clean(value: Any) -> Any:
    """Recursively clean every string, including dict keys; other values unchanged."""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, dict):
        return {clean(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clean(v) for v in value)
    return value


def has_lone_surrogate(value: Any) -> bool:
    return clean(value) != value


def sanitize_output(out: ModelOutput) -> tuple[ModelOutput, bool]:
    """(sanitised output, changed). Rebuilt from a cleaned model_dump(); the fields are
    then moved onto the original object, because Inspect's transcript event holds that
    same object and must become serialisable too. Untouched when nothing needs cleaning."""
    dump = out.model_dump()
    cleaned = clean(dump)
    if cleaned == dump:
        return out, False
    fresh = ModelOutput.model_validate(cleaned)
    for name in type(out).model_fields:
        setattr(out, name, getattr(fresh, name))
    return out, True
