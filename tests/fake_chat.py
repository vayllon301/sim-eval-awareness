"""A local OpenAI-compatible chat server for offline end-to-end runs. Each request goes to a
Python function (a scripted agent or simulator), never to a model, so a run config's
`rawchat/` models work unchanged: the provider builds, sends and logs each request as in a
real run (the request Inspect logs is what `simtest equiv contexts` reads).

    with FakeChat(respond) as server:      # respond(body) -> {"content", "reasoning", "tool_calls"}
        server.url                         # http://127.0.0.1:<port>/v1
        server.requests                    # every request body, in arrival order

A reply's `tool_calls` is a list of (name, arguments) pairs. `finish_reason` defaults to
"tool_calls" when there are calls, else "stop". An exception in `respond` is a 500, which
Inspect retries and then reports as the sample's error.
"""

from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeChat:
    def __init__(self, respond, name: str = "fake"):
        self.respond, self.name = respond, name
        self.requests: list[dict] = []
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._n = 0

    def __enter__(self) -> "FakeChat":
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):   # quiet
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                try:
                    if not self.path.endswith("/chat/completions"):
                        raise ValueError(f"unexpected path {self.path}")
                    payload = owner.completion(body)
                    status = 200
                except Exception:
                    owner.errors.append(traceback.format_exc())
                    payload, status = {"error": {"message": owner.errors[-1][-500:]}}, 500
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def completion(self, body: dict) -> dict:
        with self._lock:
            self.requests.append(body)
            self._n += 1
            n = self._n
        reply = self.respond(body)
        calls = [{"id": f"call_{self.name}_{n}_{i}", "type": "function",
                  "function": {"name": name, "arguments": json.dumps(args)}}
                 for i, (name, args) in enumerate(reply.get("tool_calls") or [])]
        message = {"role": "assistant", "content": reply.get("content") or "", "reasoning": reply.get("reasoning") or ""}
        if calls:
            message["tool_calls"] = calls
        prompt = len(json.dumps(body.get("messages"))) // 4
        completion = (len(message["content"]) + len(message["reasoning"])) // 4 + 1
        return {"id": f"chatcmpl-{self.name}-{n}", "object": "chat.completion", "created": 0,
                "model": body.get("model"),
                "choices": [{"index": 0, "message": message,
                             "finish_reason": reply.get("finish_reason") or ("tool_calls" if calls else "stop")}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}
