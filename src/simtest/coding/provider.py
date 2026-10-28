"""`rawchat/<model>`: an Inspect model API for OpenAI-compatible chat servers (vLLM, OpenRouter)
that sends earlier reasoning back exactly as the Colab probe does (explore/colab_probe/episode.py).

Why not Inspect's own providers (checked in inspect-ai 0.3.281, and pinned by tests):
- `openai-api/...` and `vllm/...` replay reasoning in the field it arrived in, but prefer
  `reasoning_content` over `reasoning` when a server sends both, and prefix the replayed
  content with a newline (`_openai.py: openai_assistant_content`).
- `openrouter/...` replays OpenRouter's structured `reasoning_details`, not the plain text.
- None of them sends `top_k` (it is not an OpenAI parameter).

This provider keeps everything else of Inspect (retries, logging of the exact request,
model roles) and changes three things:
1. Assistant messages go out as {"role": "assistant", "content": <text>, "reasoning":
   <reasoning text>, "tool_calls": [...]}: the field vLLM 0.32's renderer reads (LOG.md,
   26-10-2026), the content unchanged. OpenRouter accepts the same field.
2. The reasoning read from a response is the plain-text `reasoning` (or
   `reasoning_content`) field when present, ahead of `reasoning_details`.
3. `top_k` in GenerateConfig is sent as a body field. `reasoning_effort` is sent only if
   set (the coding runner refuses it for agents), and the reasoning history is always
   kept in full.

Model args: `base_url` (required, e.g. http://localhost:18000/v1) and `api_key_env` (the
name of an environment variable holding the key, e.g. OPENROUTER_API_KEY; none for a
local vLLM). The key itself never appears in a config or a log.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from inspect_ai._util.error import PrerequisiteError
from inspect_ai.model import ChatMessage, ContentReasoning, GenerateConfig, modelapi
from inspect_ai.model._openai import chat_choices_from_openai, openai_chat_message, openai_chat_tool_call_param
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

REASONING_FIELD = "reasoning"


def reasoning_text(message) -> str:
    """The reasoning of an Inspect assistant message, as one string."""
    if isinstance(message.content, str):
        return ""
    return "".join(c.reasoning for c in message.content if isinstance(c, ContentReasoning))


class RawChatAPI(OpenAICompatibleAPI):
    def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                 config: GenerateConfig = GenerateConfig(), api_key_env: str | None = None, **model_args: Any):
        if not base_url:
            raise PrerequisiteError("rawchat models need a base_url model arg (e.g. http://localhost:18000/v1)")
        if api_key is None:
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if not api_key:
                    raise PrerequisiteError(f"{api_key_env} is not set (rawchat api_key_env)")
            else:
                api_key = "EMPTY"  # a local vLLM without --api-key accepts any key
        super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config,
                         service="rawchat", strict_tools=False, stream=False, **model_args)

    def service_model_name(self) -> str:
        return self.model_name  # the served name as is (no service prefix to strip)

    def force_reasoning_history(self) -> Literal["none", "all", "last"] | None:
        return "all"

    async def messages_to_openai(self, input: list[ChatMessage]) -> list:
        out = []
        for message in input:
            if message.role == "assistant":
                turn: dict[str, Any] = {"role": "assistant", "content": message.text,
                                        REASONING_FIELD: reasoning_text(message)}
                if message.tool_calls:
                    turn["tool_calls"] = [openai_chat_tool_call_param(c) for c in message.tool_calls]
                out.append(turn)
            else:
                out.append(await openai_chat_message(message))
        return out

    def completion_params(self, config: GenerateConfig, tools: bool) -> dict[str, Any]:
        params = super().completion_params(config, tools)
        if config.top_k is not None:
            params.setdefault("extra_body", {})["top_k"] = config.top_k
        return params

    def chat_choices_from_completion(self, completion, tools):
        for choice in completion.choices:
            extra = choice.message.model_extra or {}
            text = extra.get("reasoning") or extra.get("reasoning_content")
            if isinstance(text, str) and text:
                # Inspect's parser reads reasoning_details first; keep the plain text instead.
                extra.pop("reasoning_details", None)
        return chat_choices_from_openai(completion, tools)


@modelapi(name="rawchat")
def rawchat():
    return RawChatAPI
