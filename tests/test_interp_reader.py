"""The post-hoc reader on the fake tokenizer and backend: the colab-probe adapter, role
masks from placeholder rendering, projections, SAE latents, saved readings."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from simtest.interp import directions as D
from simtest.interp.fake import FakeBackend, FakeTokenizer
from simtest.interp.reader import (R, ROLES, Episode, Reading, RenderError, colab_probe_prompts, from_colab_probe,
                                   load_colab_probe, prompt_checks, read, render, summarize)
from simtest.interp.sae import TopKSAE
from simtest.interp.vectors import VectorSet

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "tests" / "interp_fixtures" / "colab-probe-smoke.jsonl"  # logs/colab-probe/smoke-or-20261027.jsonl
PROMPTS = colab_probe_prompts(ROOT / "explore")


def rows():
    return [json.loads(line) for line in SMOKE.read_text().splitlines()]


def episode(**kw):
    return from_colab_probe(rows(), **PROMPTS, **kw)


def test_prompts_come_from_the_episode_script():
    assert PROMPTS["system"].startswith("You are a software engineer at Larkspur Analytics working in the repository "
                                        "/home/dev/pricing-service.")
    assert PROMPTS["ticket"].startswith("PRICING-412: CI on main is red")
    assert [t["function"]["name"] for t in PROMPTS["tools"]] == ["bash", "submit"]
    assert "bash -lc" in PROMPTS["tools"][0]["function"]["description"]


def test_colab_probe_adapter():
    ep = episode()
    roles = [m["role"] for m in ep.messages]
    # 4 turns with 2, 2, 2 and 1 calls; the last turn's output ("Submitted.") was never read
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant", "tool", "tool",
                     "assistant", "tool", "tool", "assistant"]
    assert ep.turns == [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    assert ep.meta["unread_outputs_dropped"] == 1 and ep.meta["outcome"] == "submitted"
    assert ep.meta["prompt_tokens"] == [572, 1358, 3062, 4264]
    a = ep.messages[2]
    assert a["reasoning_content"].startswith("Let's start by examining the repository")
    assert a["tool_calls"][0]["function"]["arguments"] == {
        "command": "ls -la /home/dev/pricing-service && cat /home/dev/pricing-service/README* 2>/dev/null | head -50"}
    assert ep.messages[3]["content"].startswith("total 44")
    kept = episode(drop_unread_outputs=False)
    assert kept.messages[-1] == {"role": "tool", "content": "Submitted."} and kept.meta["unread_outputs_dropped"] == 0


def test_older_rows_without_tool_calls_use_calls():
    rs = rows()
    for r in rs:
        r.pop("tool_calls", None)
    assert from_colab_probe(rs, **PROMPTS).messages == episode().messages


def test_bad_arguments_and_gaps_are_errors():
    rs = rows()
    rs[0]["tool_calls"][0]["arguments"] = "{not json"
    with pytest.raises(ValueError, match="not JSON"):
        from_colab_probe(rs, **PROMPTS)
    rs = rows()
    del rs[1]
    with pytest.raises(ValueError, match="consecutive"):
        from_colab_probe(rs, **PROMPTS)


def test_load_records_the_source_hash():
    ep = load_colab_probe(SMOKE, **PROMPTS)
    assert len(ep.meta["source_sha256"]) == 64 and ep.meta["source"] == str(SMOKE)


def decoded(rd, tok, role):
    """The text of the tokens with this role (one character per token in the fake tokenizer)."""
    return "".join(c for c, r in zip(rd.text, rd.role) if r == R[role])


def test_render_labels_every_segment():
    tok, ep = FakeTokenizer(), episode()
    rd = render(ep, tok)
    assert rd.text == tok.apply_chat_template(ep.messages, tools=ep.tools, tokenize=False)
    assert len(rd.ids) == len(rd.role) == len(rd.turn) == len(rd.text)  # character tokenizer
    reasoning = decoded(rd, tok, "reasoning")
    for m in ep.messages:
        if m["role"] == "assistant" and m["reasoning_content"].strip():
            assert m["reasoning_content"].strip() in reasoning
    outputs = decoded(rd, tok, "tool_output")
    assert all(m["content"].strip() in outputs for m in ep.messages if m["role"] == "tool")
    assert "<tool_response>" not in outputs and "<|im_start|>" not in reasoning
    assert decoded(rd, tok, "user") == PROMPTS["ticket"].strip()
    assert "git log --oneline -10" in decoded(rd, tok, "tool_call")
    content = decoded(rd, tok, "content")  # smoke turns 0-2 had blank content; turn 3 had text
    assert content.startswith("I've diagnosed the failure")
    system = decoded(rd, tok, "system")  # the tools block and the system prompt
    assert '"name": "bash"' in system and PROMPTS["system"].strip() in system
    assert set(np.unique(rd.role)) <= set(range(len(ROLES)))


def test_render_turns():
    rd = render(episode(), FakeTokenizer())
    for role in ("reasoning", "tool_call", "tool_output"):
        assert set(rd.turn[rd.role == R[role]].tolist()) <= {0, 1, 2, 3}
    assert set(rd.turn[rd.role == R["system"]].tolist()) == {-1}
    assert set(rd.turn[rd.role == R["tool_output"]].tolist()) == {0, 1, 2}  # turn 3's output dropped
    assert rd.turn[-1] == 3


class DroppingTokenizer(FakeTokenizer):
    """A template that leaves out earlier reasoning."""

    def apply_chat_template(self, messages, tools=None, tokenize=True, **kw):
        ms = copy.deepcopy(messages)
        for m in ms[:-1]:
            m.pop("reasoning_content", None)
        return super().apply_chat_template(ms, tools=tools, tokenize=tokenize, **kw)


class DoublingTokenizer(FakeTokenizer):
    """A template that renders the user message twice."""

    def apply_chat_template(self, messages, tools=None, tokenize=True, **kw):
        ms = copy.deepcopy(messages)
        ms.insert(2, copy.deepcopy(ms[1]))
        return super().apply_chat_template(ms, tools=tools, tokenize=tokenize, **kw)


def test_render_refuses_templates_it_cannot_map():
    with pytest.raises(RenderError, match="dropped"):
        render(episode(), DroppingTokenizer())
    with pytest.raises(RenderError, match="more than once"):
        render(episode(), DoublingTokenizer())
    ep = episode()
    ep.messages[3]["content"] += " @@SEG00001@@"
    with pytest.raises(RenderError, match="reserved"):
        render(ep, FakeTokenizer())


def test_prompt_checks_grow_with_each_turn():
    checks = prompt_checks(episode(), FakeTokenizer())
    assert [c["recorded"] for c in checks] == [572, 1358, 3062, 4264]
    lengths = [c["rendered"] for c in checks]
    assert lengths == sorted(lengths) and len(set(lengths)) == 4


def fake_vectors(b, layers=(2, 4), names=("desperate", "calm")):
    rng = np.random.default_rng(1)
    vs = VectorSet(meta={"model": dict(b.info), "offset": {"calm": {"2": 0.25}}})
    for n in names:
        for L in layers:
            vs.add(n, L, D.unit(rng.standard_normal(b.d_model)))
    return vs


def test_read_projects_at_every_token(tmp_path):
    b = FakeBackend()
    ep, vs = episode(), fake_vectors(b)
    r = read(b, ep, vs, [2, 4])
    rd = render(ep, b.tokenizer)
    acts = b.activations(rd.ids.tolist(), [2, 4])
    assert r.proj.shape == (len(rd.ids), 2, 2)
    assert np.allclose(r.proj[:, 1, 0], acts[4] @ vs.get("desperate", 4), atol=1e-5)
    assert np.array_equal(r.role, rd.role) and np.array_equal(r.token_ids, rd.ids)
    assert r.meta["episode"]["outcome"] == "submitted" and r.meta["backend"]["engine"] == "fake"
    assert [c["recorded"] for c in r.meta["prompt_checks"]] == [572, 1358, 3062, 4264]
    out = r.save(tmp_path / "ep.npz")
    back = Reading.load(out)
    assert np.allclose(back.proj, r.proj) and back.layers == [2, 4] and back.names == ["desperate", "calm"]
    summary = json.loads((tmp_path / "ep.summary.json").read_text())
    g = summary["by_group"]["calm@2"]
    agent = np.isin(r.role, [R[x] for x in ("reasoning", "content", "tool_call")])
    assert np.isclose(g["agent"]["mean"], r.proj[agent, 0, 1].mean(), atol=1e-5)
    assert np.isclose(g["agent"]["mean_centred"], g["agent"]["mean"] - 0.25)
    assert g["tool_output"]["n"] == int((r.role == R["tool_output"]).sum())
    assert [t["turn"] for t in summary["by_turn"]["calm@2"]] == [0, 1, 2, 3]
    assert summary["by_turn"]["calm@2"][3]["tool_output"] is None


def test_prompt_ends_are_the_last_token_of_each_turn_prompt(tmp_path):
    b = FakeBackend()
    ep, vs = episode(), fake_vectors(b)
    r = read(b, ep, vs, [2, 4])
    lengths = [c["rendered"] for c in prompt_checks(ep, b.tokenizer)]
    assert r.prompt_end.dtype == np.int32 and r.prompt_end.tolist() == [n - 1 for n in lengths]
    assert r.meta["prompt_end_not_prefix"] == []
    for k, pos in enumerate(r.prompt_end.tolist()):  # the fake prompt ends with "<think>\n"; the reasoning follows
        assert r.turn[pos] == k and r.role[pos] == R["template"] and r.role[pos + 1] == R["reasoning"]
    summary = summarize(r)
    for li, L in enumerate(r.layers):
        for vi, name in enumerate(r.names):
            for row in summary["by_turn"][f"{name}@{L}"]:
                assert row["prompt_end"] == pytest.approx(float(r.proj[r.prompt_end[row["turn"]], li, vi]))
    row = summary["by_turn"]["calm@2"][1]
    assert row["prompt_end_centred"] == pytest.approx(row["prompt_end"] - 0.25)
    assert summary["by_turn"]["desperate@2"][1]["prompt_end_centred"] is None  # no offset in the file
    back = Reading.load(r.save(tmp_path / "ep.npz"))
    assert np.array_equal(back.prompt_end, r.prompt_end) and back.prompt_end.dtype == np.int32


class ForgetfulPromptTokenizer(FakeTokenizer):
    """Turn prompts leave out earlier reasoning (as Qwen templates do without
    preserve_thinking); the full rendering keeps it."""

    def apply_chat_template(self, messages, tools=None, tokenize=True, add_generation_prompt=False, **kw):
        if add_generation_prompt:
            messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in messages]
        return super().apply_chat_template(messages, tools=tools, tokenize=tokenize,
                                           add_generation_prompt=add_generation_prompt, **kw)


def test_prompt_end_is_missing_where_the_prompt_is_not_a_prefix():
    b = FakeBackend()
    b.tokenizer = ForgetfulPromptTokenizer()
    r = read(b, episode(), fake_vectors(b), [2])
    assert r.prompt_end[0] > 0 and r.prompt_end[1:].tolist() == [-1, -1, -1]  # turn 0 has no earlier reasoning
    assert r.meta["prompt_end_not_prefix"] == [1, 2, 3]
    rows_ = summarize(r)["by_turn"]["calm@2"]
    assert rows_[0]["prompt_end"] is not None and rows_[0]["prompt_end_centred"] is not None
    assert all(row["prompt_end"] is None and row["prompt_end_centred"] is None for row in rows_[1:])


def test_older_readings_without_prompt_end_still_load(tmp_path):
    b = FakeBackend()
    path = read(b, episode(), fake_vectors(b), [2]).save(tmp_path / "ep.npz")
    z = dict(np.load(path))
    z.pop("prompt_end")
    np.savez_compressed(tmp_path / "old.npz", **z)
    old = Reading.load(tmp_path / "old.npz")
    assert old.prompt_end is None and old.proj.shape[0] == len(old.role)
    assert all(row["prompt_end"] is None for row in summarize(old)["by_turn"]["calm@2"])


def test_read_refuses_vectors_from_another_model():
    b = FakeBackend()
    vs = fake_vectors(b)
    vs.meta["model"]["fingerprint"] = "fp1:other"
    with pytest.raises(ValueError, match="built on fp1:other"):
        read(b, episode(), vs, [2])
    assert read(b, episode(), vs, [2], allow_other_model=True).proj.shape[1] == 1


def test_read_stores_sae_top_k(tmp_path):
    b = FakeBackend()
    rng = np.random.default_rng(2)
    sae = TopKSAE(W_enc=rng.standard_normal((40, 16)).astype(np.float32), b_enc=np.zeros(40, np.float32),
                  W_dec=rng.standard_normal((16, 40)).astype(np.float32), b_dec=np.zeros(16, np.float32), k=5,
                  info={"repo": "fake"})
    r = read(b, episode(), fake_vectors(b), [2], saes={4: sae})
    assert r.layers == [2, 4]
    idx, val = r.sae[4]
    assert idx.shape == (len(r.role), 5) and (val[:, :-1] >= val[:, 1:]).all()
    back = Reading.load(r.save(tmp_path / "x.npz"))
    assert np.array_equal(back.sae[4][0], idx) and back.meta["sae"]["4"]["repo"] == "fake"
