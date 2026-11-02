"""The transcript renderer (judge/transcript.py): what the agent saw, with stable step ids, and
nothing that reveals the condition, run, scenario, simulator, scores or metadata."""

from __future__ import annotations

import json

import pytest
from inspect_ai.log import read_eval_log_sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ContentImage, ContentReasoning, ContentText
from inspect_ai.tool import ToolCall

from simtest.judge import transcript as T
from judge_helpers import LEAKS, patch_machine, run_episodes, synthetic_sample


def test_steps_follow_the_messages_with_stable_ids():
    steps = T.episode_steps(synthetic_sample())
    assert T.step_ids(steps) == ["system", "task", "t1", "t2"]
    assert T.turn_ids(steps) == ["t1", "t2"]
    t1, t2 = steps[2], steps[3]
    assert t1["reasoning"] == "Let me run the tests first." and t1["text"] == ""
    assert t1["calls"] == [{"n": 1, "function": "bash", "arguments": {"command": "make test"},
                            "parse_error": None, "output": "SIMULATED: 3 passed"}]
    assert t2["reasoning"] == "This smells like an eval scenario." and t2["calls"] == []
    assert t2["text"] == "Fixed the parser; make test passes."


def test_the_text_has_every_step_headed_by_its_id():
    text = T.render_text(T.episode_steps(synthetic_sample()))
    assert text.index("=== [system] system prompt ===") < text.index("=== [task] task message ===") \
        < text.index("=== [t1] agent turn 1 ===") < text.index("=== [t2] agent turn 2 ===")
    assert "--- reasoning ---\nLet me run the tests first." in text
    assert "--- tool call 1: bash ---\nmake test\n--- tool output 1 ---\nSIMULATED: 3 passed" in text
    assert "--- visible text ---\nFixed the parser; make test passes." in text
    assert "--- visible text ---" not in text.split("[t2]")[0]   # an empty visible text is left out


def test_nothing_but_what_the_agent_saw_is_rendered():
    steps = T.episode_steps(synthetic_sample())
    for blob in (T.render_text(steps), json.dumps(steps)):
        for leak in LEAKS:
            assert leak not in blob, leak


def test_bookkeeping_changes_nothing():
    a = synthetic_sample()
    b = synthetic_sample(id="other", epoch=3, uuid="x", metadata={"condition": "C0"}, store={}, scores={})
    assert T.episode_steps(a) == T.episode_steps(b)
    assert T.transcript_sha256(T.episode_steps(a)) == T.transcript_sha256(T.episode_steps(b))
    c = synthetic_sample()
    c.messages[3].content = "SIMULATED: 2 passed, 1 failed"
    assert T.transcript_sha256(T.episode_steps(c)) != T.transcript_sha256(T.episode_steps(a))


def test_outputs_match_calls_by_id_then_in_order():
    s = synthetic_sample()
    calls = [ToolCall(id="a", function="bash", arguments={"command": "ls"}),
             ToolCall(id="b", function="send_message", arguments={"channel": "#ops", "text": "hi"})]
    s.messages[2] = ChatMessageAssistant(content="", tool_calls=calls)
    s.messages[3:4] = [ChatMessageTool(content="Message sent.", tool_call_id="b", function="send_message"),
                       ChatMessageTool(content="", tool_call_id=None, function="bash")]
    t1 = T.episode_steps(s)[2]
    assert [c["output"] for c in t1["calls"]] == ["", "Message sent."]
    text = T.render_text(T.episode_steps(s))
    assert "--- tool output 1 ---\n(empty)" in text
    assert '"channel": "#ops"' in text


def test_redacted_reasoning_keeps_only_its_summary():
    s = synthetic_sample()
    s.messages[4] = ChatMessageAssistant(content=[
        ContentReasoning(reasoning="ENCRYPTED-BLOB", redacted=True, summary="Checked the tests."),
        ContentText(text="Done.")])
    t2 = T.episode_steps(s)[3]
    assert t2["reasoning"] == "Checked the tests." and t2["reasoning_redacted"]
    assert "ENCRYPTED-BLOB" not in T.render_text(T.episode_steps(s))


def test_unknown_content_raises_instead_of_being_dropped():
    s = synthetic_sample()
    s.messages[4] = ChatMessageAssistant(content=[ContentImage(image="data:image/png;base64,AAAA")])
    with pytest.raises(T.TranscriptError):
        T.episode_steps(s)


def test_c3_shows_the_simulated_output_and_survives_the_log(monkeypatch, tmp_path):
    # A real offline C3 episode: the shadow machine prints REAL, the simulator INVENTED.
    patch_machine(monkeypatch, [("REAL pytest output", 1)])
    log = run_episodes(tmp_path / "c3", "C3", ["pricing_cap"])
    s = log.samples[0]
    assert s.store["CodingEpisode:steps"][0]["result"]["output"] == "REAL pytest output"
    steps = T.episode_steps(s)
    text = T.render_text(steps)
    assert steps[2]["calls"][0]["output"] == "INVENTED output"
    assert "REAL pytest output" not in text
    for leak in ("pricing_cap", "C3", "mockllm", "mock", "max_turns", "no_tool_call"):
        assert leak not in text, leak
    # The same steps when the sample is read back from the .eval file.
    again = read_eval_log_sample(log.location, id=s.id, epoch=s.epoch)
    assert T.episode_steps(again) == steps
