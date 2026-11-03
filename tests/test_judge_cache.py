"""Prompt caching for the primary judge on Anthropic models through OpenRouter (judge.py,
backends.py, command.py): the blocks sent concatenate to the exact prompt, only the shared
prefix (the header and the transcript) carries `cache_control`, the settings and the cache key
record it, and every other model is called exactly as before. Offline: Inspect's request is captured where it would
go on the wire (after OpenRouter's provider adds its markers) and answered locally."""

from __future__ import annotations

import argparse
import asyncio
import json

import pytest
from inspect_ai.model import get_model
from inspect_ai.model._providers import openai_compatible as OC
from openai.types.chat import ChatCompletion

from simtest.judge import backends as B
from simtest.judge import command as C
from simtest.judge import judge as J
from simtest.judge import transcript as T
from judge_helpers import synthetic_sample

HAIKU = "openrouter/anthropic/claude-haiku-5.5"
PIN = {"provider": {"order": ["anthropic"], "allow_fallbacks": False}}
# A valid answer to every label: AMBIGUOUS is a choice of each choice label, and metagaming reads the score.
ANY = json.dumps({"verdict": "AMBIGUOUS", "score": 0, "quotes": [], "first_step": "none", "rationale": "No sign."})
EPHEMERAL = {"type": "ephemeral"}


def steps():
    return T.episode_steps(synthetic_sample())


@pytest.fixture
def wire(monkeypatch):
    """The messages Inspect's OpenAI-compatible providers send (with OpenRouter's cache markers),
    answered offline with a valid verdict."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-offline")
    sent = []

    async def fake(self, request, config):
        sent.append(json.loads(json.dumps(request["messages"])))
        return ChatCompletion.model_validate({
            "id": "gen-offline", "object": "chat.completion", "created": 0, "model": "anthropic/claude-haiku-5.5",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ANY}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
                      "prompt_tokens_details": {"cached_tokens": 900, "cache_write_tokens": 0}}})

    monkeypatch.setattr(OC.OpenAICompatibleAPI, "_generate_completion", fake)
    return sent


def haiku(**kw):
    return B.InspectBackend(HAIKU, temperature=None, reasoning_effort="high", model_args=PIN, **kw)


def v3_calls(backend, n_eps=1):
    eps = [{"key": f"a.eval::s{i}::1", "log": "a.eval", "sample_id": f"s{i}", "epoch": 1} for i in range(n_eps)]
    calls, refused = J.plan_calls(eps, lambda ep: synthetic_sample(), J.load_prompts("v3"), "v3", backend, 200_000)
    assert not refused
    return calls


# --- the shared prefix -------------------------------------------------------------------

def test_v3_prompts_share_the_header_and_the_transcript_then_differ():
    text = T.render_text(steps())
    heads = set()
    for label, pr in J.load_prompts("v3").items():
        prompt = J.build_prompt(pr["text"], text)
        n = J.cache_prefix_chars(pr["text"], text)
        heads.add(prompt[:n])
        assert prompt[n:].startswith("## The question"), label
    assert len(heads) == 1                       # one cache entry per episode serves all five labels
    assert heads.pop().endswith(text.rstrip("\n") + "\n</transcript>\n\n")
    # The prefix ends after the first blank line that follows the transcript (and any newlines after it).
    assert J.cache_prefix_chars("A\n{{TRANSCRIPT}}\n\n\nQ", "T\n") == len("A\nT\n\n\n")
    assert J.cache_prefix_chars("A\n{{TRANSCRIPT}}\nQ", "T") == 0          # no blank line: no split


# --- what goes on the wire ---------------------------------------------------------------

def test_the_blocks_sent_are_the_exact_prompt_and_only_the_prefix_is_marked(wire, tmp_path):
    be = haiku()
    calls = v3_calls(be)
    recs = asyncio.run(J.run_calls(calls, be, J.ResultStore(tmp_path / "r.jsonl"), progress=None))
    assert len(recs) == len(wire) == 5 and all(r["error"] is None for r in recs)
    prompts = {c["prompt"]: c for c in calls}
    for messages in wire:
        assert len(messages) == 1 and messages[0]["role"] == "user"
        head, tail = messages[0]["content"]
        assert head["type"] == tail["type"] == "text"
        assert head["text"] + tail["text"] in prompts            # byte for byte the rendered prompt
        assert head["cache_control"] == EPHEMERAL and "cache_control" not in tail
        assert len(head["text"]) == prompts[head["text"] + tail["text"]]["cache_prefix"]
    assert len({m[0]["content"][0]["text"] for m in wire}) == 1   # the same prefix for all five
    for r in recs:
        assert r["settings"]["prompt_cache"] == "prefix"
        assert r["attempts"][0]["cache_prefix_chars"] == calls[0]["cache_prefix"]
        assert r["attempts"][0]["usage"]["input_tokens_cache_read"] == 900


def test_the_retry_sends_the_same_prefix():
    class Rec:
        name, model, settings, max_output, prompt_cache = "fake", "fake", {"prompt_cache": "prefix"}, 0, True

        def __init__(self):
            self.seen = []

        async def complete(self, prompt, cache_prefix=None):
            self.seen.append((prompt, cache_prefix))
            return {"text": "not json" if len(self.seen) == 1 else ANY, "stop_reason": "stop"}

    be = Rec()
    base = {"key": "k", "label": "eval_awareness", "episode": {"key": "e"}, "transcript_sha256": "x"}
    rec = asyncio.run(J.judge_call(be, "HEAD\n\n## Q", "eval_awareness", steps(), base, cache_prefix=6))
    assert rec["value"] == "REAL" and [n for _, n in be.seen] == [6, 6]
    assert be.seen[1][0].startswith(be.seen[0][0])


def test_other_models_and_caching_off_send_one_string_as_before(wire, tmp_path):
    for i, be in enumerate((B.InspectBackend("openrouter/openai/gpt-6.1", reasoning_effort="high"),
                            haiku(prompt_cache=False))):
        wire.clear()
        calls = v3_calls(be)
        asyncio.run(J.run_calls(calls, be, J.ResultStore(tmp_path / f"r{i}.jsonl"), progress=None))
        assert sorted(m[0]["content"] for m in wire) == sorted(c["prompt"] for c in calls)
        assert all(isinstance(m[0]["content"], str) for m in wire)


def test_a_backend_without_prompt_caching_is_never_given_a_prefix():
    class Plain:                                 # the interface every other backend has: complete(prompt)
        name, model, settings, max_output = "fake", "fake", {}, 0

        async def complete(self, prompt):
            return {"text": ANY, "stop_reason": "stop"}

    base = {"key": "k", "label": "eval_awareness", "episode": {"key": "e"}, "transcript_sha256": "x"}
    rec = asyncio.run(J.judge_call(Plain(), "HEAD\n\n## Q", "eval_awareness", steps(), base, cache_prefix=6))
    assert rec["value"] == "REAL" and "cache_prefix_chars" not in rec["attempts"][0]


# --- settings and the cache key ----------------------------------------------------------

def test_caching_is_on_by_default_for_openrouter_anthropic_models_only(wire):
    on = haiku()
    assert on.prompt_cache and on.settings["prompt_cache"] == "prefix" and on.config.cache_prompt is True
    before = {"temperature": None, "max_tokens": 16000, "reasoning_effort": "high", "seed": None, "model_args": PIN}
    off = haiku(prompt_cache=False)
    assert not off.prompt_cache and off.settings == before and off.config.cache_prompt is None
    for model in ("openrouter/openai/gpt-6.1", "openrouter/google/gemini-4-argon", "openrouter/qwen/qwen3.8-27b"):
        be = B.InspectBackend(model)
        assert not be.prompt_cache and "prompt_cache" not in be.settings and be.config.cache_prompt is None
    assert not B.InspectBackend(get_model("mockllm/model")).prompt_cache
    assert B.prompt_cache_default(HAIKU) and not B.prompt_cache_default("anthropic/claude-haiku-5-5")


def test_results_with_and_without_caching_never_share_a_cache_key(wire, tmp_path):
    on, off = v3_calls(haiku()), v3_calls(haiku(prompt_cache=False))
    assert not {c["key"] for c in on} & {c["key"] for c in off}
    store = J.ResultStore(tmp_path / "r.jsonl")
    asyncio.run(J.run_calls(off, haiku(prompt_cache=False), store, progress=None))
    assert all(store.done(c["key"]) for c in off) and not any(store.done(c["key"]) for c in on)


def test_the_cli_turns_caching_off_and_the_plan_says_so(wire, tmp_path):
    p = argparse.ArgumentParser()
    C.add_parser(p.add_subparsers(dest="cmd"))

    def args(*extra):
        return p.parse_args(["judge", "tune", "spec.yaml", "--backend", "inspect", "--model", HAIKU, "--prompts", "v3",
                             "--out", "r.jsonl", "--temperature", "none", "--reasoning-effort", "high",
                             "--model-args", json.dumps(PIN), *extra])

    be = C._backend(args())
    assert be.prompt_cache and be.settings["prompt_cache"] == "prefix"
    assert "prompt cache" in C._plan_summary(v3_calls(be), J.ResultStore(tmp_path / "r.jsonl"), be)
    off = C._backend(args("--no-prompt-cache"))
    assert not off.prompt_cache and "prompt cache" not in C._plan_summary(v3_calls(off), J.ResultStore(
        tmp_path / "r.jsonl"), off)
