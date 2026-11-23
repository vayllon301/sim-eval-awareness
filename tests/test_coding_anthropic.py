"""A native Anthropic simulator (`anthropic/claude-sonnet-5-5`), offline: the request Inspect's own
provider would send for the shipped config, what the simulator's record keeps of the reply, and a
command that prints nothing. The HTTP transport is fake (anthropic_helpers.py); nothing is sent."""

from __future__ import annotations

import anyio
import pytest
from anthropic.types import MessageParam as SDK_MESSAGE_PARAM
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

import inspect_ai.model._providers.anthropic as provider
from simtest.coding import anthropic_compat
from simtest.coding.machine import ExecResult
from simtest.coding.observers import SimulatedBash
from simtest.coding.run import build_model, load_config
from anthropic_helpers import FAKE_KEY, FakeAnthropic, reply

TOOL, SNAP = "bash tool text", "INITIAL-SNAPSHOT"


def result(output):
    return ExecResult(output, 0, False, 0.1)


@pytest.fixture(autouse=True)
def sdk_as_installed(monkeypatch):
    """Every test starts from the SDK's own `MessageParam`, and the provider module gets it back
    afterwards: `build_model` patches the module for the whole process."""
    monkeypatch.setattr(provider, "MessageParam", SDK_MESSAGE_PARAM)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)


def sonnet55(fake: FakeAnthropic):
    """The shipped config's simulator, on the fake transport."""
    spec = load_config("configs/coding/bluedot_cheap_qwen.yaml")["simulators"]["sonnet55"]
    return build_model({**spec, "args": {**spec["args"], "http_client": fake.client}})


def observe(obs, *steps):
    async def run():
        return [await obs.observe(command, result(real)) for command, real in steps]

    return anyio.run(run)


def test_the_pinned_inspect_and_sdk_disagree_on_message_param_and_the_shim_fixes_it():
    if callable(SDK_MESSAGE_PARAM):
        pytest.skip("the SDK's MessageParam can be called again: anthropic_compat is no longer needed")
    with pytest.raises(TypeError, match="not callable"):
        anyio.run(provider.message_param, ChatMessageUser(content="hi"))
    assert anthropic_compat.apply() is True
    assert anyio.run(provider.message_param, ChatMessageUser(content="hi")) == {"role": "user", "content": "hi"}
    assert anthropic_compat.apply() is False    # nothing left to change


def test_anthropic_compat_is_a_noop_if_upstream_removes_message_param(monkeypatch):
    monkeypatch.delattr(provider, "MessageParam", raising=False)
    assert anthropic_compat.apply() is False


def test_inspect_sends_an_empty_assistant_turn_as_a_non_empty_wire_block():
    """Anthropic rejects empty text blocks; Inspect encodes empty assistant turns on the wire."""
    anthropic_compat.apply()
    sent = anyio.run(provider.message_param, ChatMessageAssistant(content=""))
    assert sent == {"role": "assistant", "content": [{"type": "text", "text": "(no content)"}]}


def test_the_shipped_simulator_sends_medium_effort_summarized_thinking_and_caches():
    fake = FakeAnthropic([reply("file_a\nfile_b"), reply("ok")])
    obs = SimulatedBash("C3", sonnet55(fake), TOOL, SNAP)
    observe(obs, ("ls", "x"), ("cat a", "y"))
    assert fake.urls == ["https://api.anthropic.com/v1/messages"] * 2
    headers, body = fake.requests[0]
    assert headers["authorization"] == f"Bearer {FAKE_KEY}"    # ANTHROPIC_API_KEY, as SDK 2.4.0 sends it
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 32768
    assert body["output_config"] == {"effort": "medium"}
    assert body["thinking"]["type"] == "adaptive" and body["thinking"]["display"] == "summarized"
    assert not {"temperature", "top_p", "top_k", "fallbacks"} & set(body)
    # prompt caching: the system prompt (the snapshot) is a cache breakpoint, and the turns are cached
    assert body["system"][-1]["cache_control"] == {"type": "ephemeral"} and SNAP in body["system"][-1]["text"]
    assert body["cache_control"] == {"type": "ephemeral"}
    # the second call sends the history as plain turns: the commands and the outputs, no thinking
    turns = fake.bodies[1]["messages"]
    assert [t["role"] for t in turns] == ["user", "assistant", "user"]
    assert [b["type"] for b in turns[1]["content"]] == ["text"] and turns[1]["content"][0]["text"] == "file_a\nfile_b"
    assert fake.bodies[1]["system"] == body["system"]    # the cached prefix doesn't move


def test_the_record_keeps_the_thinking_summary_the_visible_text_and_the_cache_tokens():
    usage = {"input_tokens": 5, "cache_creation_input_tokens": 20, "cache_read_input_tokens": 3000,
             "output_tokens": 9}
    fake = FakeAnthropic([reply("total 8\nfile_a", thinking="ls prints the two files.", usage=usage)])
    obs = SimulatedBash("C3", sonnet55(fake), TOOL, SNAP)
    [(text, record)] = observe(obs, ("ls -l", "x"))
    assert text == "total 8\nfile_a"
    assert record["raw"] == "total 8\nfile_a"                  # the visible text only
    assert record["reasoning"] == "ls prints the two files."   # the readable summary...
    assert "SIG-abc" not in str(record)                        # ...never the signature
    assert record["model"] == "anthropic/claude-sonnet-5-5" and record["stop_reason"] == "stop"
    assert record["usage"]["input_tokens"] == 5 and record["usage"]["output_tokens"] == 9
    assert record["usage"]["input_tokens_cache_read"] == 3000
    assert record["usage"]["input_tokens_cache_write"] == 20


def test_literal_no_content_and_genuinely_empty_replies_are_preserved_without_corrupting_output():
    # Turn 1: genuine literal `(no content)` output (no prior empty turn) must stay intact.
    # Turn 2: genuinely empty reply (thinking only, no text) must stay empty (`""`).
    # Turn 3: literal `(no content)` after an empty turn must STILL stay intact (never silently stripped).
    # Turn 4: normal output after both; outbound history has non-empty text blocks on every assistant turn.
    fake = FakeAnthropic([reply("(no content)"), reply(None), reply("(no content)"), reply("a\nb")])
    obs = SimulatedBash("C3", sonnet55(fake), TOOL, SNAP)
    seen = observe(obs, ("printf '(no content)'", "(no content)"), ("touch a", ""),
                   ("printf '(no content)'", "(no content)"), ("ls", "y"))
    assert [text for text, _ in seen] == ["(no content)", "", "(no content)", "a\nb"]
    assert obs.history == [
        ("printf '(no content)'", "(no content)"), ("touch a", ""),
        ("printf '(no content)'", "(no content)"), ("ls", "a\nb"),
    ]
    assert seen[0][1]["raw"] == "(no content)" and "empty_placeholder" not in seen[0][1]
    assert seen[1][1]["raw"] == "" and "empty_placeholder" not in seen[1][1]
    assert seen[2][1]["raw"] == "(no content)" and "empty_placeholder" not in seen[2][1]
    # Outbound Anthropic history: every assistant message has a non-empty text block (no HTTP 400).
    assert all(b["text"].strip() for body in fake.bodies for t in body["messages"] if t["role"] == "assistant"
               for b in t["content"])


def test_c1_has_no_history_so_it_never_reads_the_placeholder_back():
    fake = FakeAnthropic([reply("(no content)")] * 3)
    obs = SimulatedBash("C1", sonnet55(fake), TOOL, None)
    [(text, record)] = observe(obs, ("echo hi", "hi"))
    assert "empty_placeholder" not in record
    assert text == "hi" and record["real_fallback"] == "dropped"   # C1's own check, unchanged
