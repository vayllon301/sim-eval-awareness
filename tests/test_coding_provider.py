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
