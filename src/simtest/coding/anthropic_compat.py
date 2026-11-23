"""Inspect 0.3.281's Anthropic provider with the anthropic SDK 2.4.0: the pinned pair.

Inspect builds every message with `MessageParam(role=..., content=...)`. `MessageParam` was a
TypedDict, and calling one makes a plain dict. SDK 2.4.0 made it a union of three TypedDicts
(user, assistant, system), which can't be called, so with the pinned pair every request fails
before it is sent: `TypeError: 'typing.Union' object is not callable`
(tests/test_coding_anthropic.py pins the failure and this fix).

`apply()` gives the provider module the call it expects, with the same result: a plain dict of
the keyword arguments. It changes nothing when `MessageParam` can be called, so it stops
mattering once either package is fixed. `coding.run.build_model` applies it before it builds a
native Anthropic model.
"""

from __future__ import annotations


def _message_param(**fields) -> dict:
    """What calling the TypedDict made: a dict of its fields."""
    return dict(fields)


def apply() -> bool:
    """Make Inspect's `MessageParam(...)` calls work. True if it had to change anything."""
    import inspect_ai.model._providers.anthropic as provider

    if callable(provider.MessageParam):
        return False
    provider.MessageParam = _message_param
    return True
