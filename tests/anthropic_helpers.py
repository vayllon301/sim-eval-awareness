"""Inspect's own Anthropic provider, offline: a fake HTTP transport under the real SDK client, so
a test sees the request that would go out and the provider parses a real streamed reply. No
network and no key (the tests pass a made-up one)."""

from __future__ import annotations

import json

import anthropic
import httpx2

FAKE_KEY = "test-key-not-a-secret"
USAGE = {"input_tokens": 12, "cache_creation_input_tokens": 3000, "cache_read_input_tokens": 0, "output_tokens": 40}


def reply(text, thinking="The directory holds two files.", usage=None, signature="SIG-abc") -> list[dict]:
    """The stream events of one Messages reply: a thinking block (unless None), then a text block
    (unless None)."""
    usage = usage or USAGE
    events = [{"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {**usage, "output_tokens": 1}}}]
    index = 0
    if thinking is not None:
        events += [
            {"type": "content_block_start", "index": index,
             "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": thinking}},
            {"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": signature}},
            {"type": "content_block_stop", "index": index}]
        index += 1
    if text is not None:
        events += [
            {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": index}]
    return events + [{"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                      "usage": usage}, {"type": "message_stop"}]


class FakeAnthropic:
    """Answers `/v1/messages` with the next scripted reply, and `/v1/messages/count_tokens` (which
    Inspect calls to size the thinking) with a fixed count. `requests` keeps each Messages request
    as (headers, body)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[tuple[dict, dict]] = []
        self.urls: list[str] = []
        self.client = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self._handle))

    def _handle(self, request):
        body = json.loads(request.content)
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 7})
        self.requests.append((dict(request.headers), body))
        self.urls.append(str(request.url))
        assert body.get("stream") is True, "the provider streams a request of this size"
        events = self.replies.pop(0)
        stream = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=stream)

    @property
    def bodies(self) -> list[dict]:
        return [body for _, body in self.requests]
