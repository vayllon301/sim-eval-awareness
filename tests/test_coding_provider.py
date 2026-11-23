"""The rawchat provider sends earlier reasoning back in the `reasoning` field (as the Colab
probe and vLLM's renderer expect), and sends top_k."""

from __future__ import annotations

import anyio
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.tool import ToolCall

from simtest.coding.provider import RawChatAPI, reasoning_text


def api():
    return RawChatAPI("agent", base_url="http://localhost:18000/v1", config=GenerateConfig())


def test_reasoning_text_joins_reasoning_blocks():
    msg = ChatMessageAssistant(content=[ContentReasoning(reasoning="step one"), ContentText(text="answer")])
    assert reasoning_text(msg) == "step one"
    assert reasoning_text(ChatMessageAssistant(content="plain")) == ""


def test_assistant_message_carries_reasoning_and_tool_calls():
    a = api()
    msg = ChatMessageAssistant(
        content=[ContentReasoning(reasoning="I will list the dir"), ContentText(text="")],
        tool_calls=[ToolCall(id="c1", function="bash", arguments={"command": "ls"})])
    out = anyio.run(a.messages_to_openai, [ChatMessageSystem(content="sys"), ChatMessageUser(content="go"), msg])
    assistant = out[-1]
    assert assistant["role"] == "assistant"
    assert assistant["reasoning"] == "I will list the dir"      # the field vLLM's renderer reads
    assert assistant["tool_calls"][0]["function"]["name"] == "bash"


def test_top_k_is_sent():
    a = api()
    params = a.completion_params(GenerateConfig(temperature=1.0, top_p=0.95, top_k=20, max_tokens=100), tools=True)
    assert params["extra_body"]["top_k"] == 20
    assert params["temperature"] == 1.0 and params["top_p"] == 0.95


def test_reasoning_history_is_kept_in_full():
    assert api().force_reasoning_history() == "all"


# --- DeepSeek's own API: reasoning in `reasoning_content`, both ways -------------------------------

def deepseek():
    return RawChatAPI("deepseek-v4-pro", base_url="https://api.deepseek.com/v1", api_key="x",
                      reasoning_field="reasoning_content")


def turn():
    return ChatMessageAssistant(
        content=[ContentReasoning(reasoning="I will list the dir"), ContentText(text="")],
        tool_calls=[ToolCall(id="c1", function="bash", arguments={"command": "ls"})])


def completion(**message):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "c", "object": "chat.completion", "created": 0, "model": "deepseek-v4-pro",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "", **message,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]}}]})


def test_the_reasoning_field_is_reasoning_unless_a_config_says_otherwise():
    out = anyio.run(api().messages_to_openai, [turn()])
    assert out[0]["reasoning"] == "I will list the dir" and "reasoning_content" not in out[0]
    assert api().reasoning_field == "reasoning"


def test_deepseek_gets_earlier_reasoning_back_in_reasoning_content_with_the_tool_calls():
    out = anyio.run(deepseek().messages_to_openai, [ChatMessageUser(content="go"), turn()])
    assert out[-1]["reasoning_content"] == "I will list the dir" and "reasoning" not in out[-1]
    assert out[-1]["tool_calls"][0]["function"]["name"] == "bash" and out[-1]["content"] == ""


def test_an_unknown_reasoning_field_is_refused():
    import pytest
    from inspect_ai._util.error import PrerequisiteError

    with pytest.raises(PrerequisiteError, match="reasoning_field"):
        RawChatAPI("m", base_url="http://x/v1", reasoning_field="thinking")


def test_reasoning_content_round_trips_from_a_reply_to_the_next_request():
    a = deepseek()
    [choice] = a.chat_choices_from_completion(completion(reasoning_content="The ticket asks for X."), tools=[])
    assert reasoning_text(choice.message) == "The ticket asks for X."
    assert choice.message.tool_calls[0].function == "bash"
    out = anyio.run(a.messages_to_openai, [choice.message])
    assert out[0]["reasoning_content"] == "The ticket asks for X."     # unchanged, no added newline


def test_plain_reasoning_content_is_read_ahead_of_reasoning_details():
    details = [{"type": "reasoning.text", "text": "structured", "format": "x", "index": 0}]
    [choice] = deepseek().chat_choices_from_completion(
        completion(reasoning_content="plain text", reasoning_details=details), tools=[])
    assert reasoning_text(choice.message) == "plain text"


def test_deepseek_request_carries_the_thinking_switch_and_no_sampling():
    from simtest.coding.run import load_config

    spec = load_config("configs/coding/bluedot_cheap_deepseek.yaml")["agents"]["deepseek_v4_pro"]
    config = GenerateConfig(**spec["config"])
    params = deepseek().completion_params(config, tools=True)
    assert params["model"] == "deepseek-v4-pro" and params["max_tokens"] == 32768
    assert params["extra_body"] == {"thinking": {"type": "enabled"}}
    assert not {"temperature", "top_p", "reasoning_effort"} & set(params)


def test_a_record_reads_a_signed_summary_not_its_signature():
    from simtest.coding.provider import readable_reasoning

    signed = ChatMessageAssistant(content=[
        ContentReasoning(reasoning="SIGNATURE-1", summary="first thought", redacted=True),
        ContentReasoning(reasoning="OPAQUE", redacted=True),            # redacted, no summary
        ContentReasoning(reasoning="SIGNATURE-2", summary="second thought", redacted=True),
        ContentText(text="answer")])
    assert readable_reasoning(signed) == "first thought\n\nsecond thought"
    plain = ChatMessageAssistant(content=[ContentReasoning(reasoning="step one"), ContentReasoning(reasoning=" two"),
                                          ContentText(text="answer")])
    assert readable_reasoning(plain) == reasoning_text(plain) == "step one two"     # rawchat: unchanged
    assert readable_reasoning(ChatMessageAssistant(content="plain")) == ""
