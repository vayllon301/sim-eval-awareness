"""The labelling page: a local server (standard library only) for blind hand labels.

    simtest label serve <manifest.json> [--labels labels.jsonl] [--port 8766]

It serves one episode at a time in the manifest's order. The browser gets only the opaque
label id and the transcript (transcript.py); the condition, run, log and scenario stay in
the manifest, on the server. Each answer is autosaved to the labels file (labels.py). The
page starts at the first episode whose label is incomplete.

Endpoints: `GET /` (the page, read from label_page.html on each request, so an edit shows
on reload), `GET /api/state`, `GET /api/episode/<position>`, `POST /api/label` (JSON only,
which a cross-site form can't send). Requests must be addressed to localhost.
"""

from __future__ import annotations

import json
import re
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import pool as P
from . import transcript as T
from .labels import FIELDS, LabelError, LabelStore, is_complete, missing, resume_position, validate

PAGE = Path(__file__).with_name("label_page.html")
CACHE = 8          # rendered transcripts kept in memory
MAX_BODY = 1 << 20


class LabelApp:
    def __init__(self, manifest_path: Path, labels_path: Path | None = None, read=None):
        self.manifest_path = Path(manifest_path)
        m = json.loads(self.manifest_path.read_text())
        self.name = m["name"]
        self.root = (self.manifest_path.parent / m["root"]).resolve()
        self.episodes = sorted(m["episodes"], key=lambda e: e["order"])
        self.order = [e["label_id"] for e in self.episodes]
        if len(set(self.order)) != len(self.order):
            raise ValueError(f"{manifest_path}: repeated label ids")
        self.labels_path = Path(labels_path) if labels_path else self.manifest_path.with_name("labels.jsonl")
        self.store = LabelStore(self.labels_path)
        self._read = read or (lambda ep: P.read_sample(self.root, ep))
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._lock = threading.Lock()

    def steps(self, pos: int) -> list[dict]:
        ep = self.episodes[pos]
        with self._lock:
            if ep["label_id"] in self._cache:
                self._cache.move_to_end(ep["label_id"])
                return self._cache[ep["label_id"]]
        steps = T.episode_steps(self._read(ep))
        with self._lock:
            self._cache[ep["label_id"]] = steps
            while len(self._cache) > CACHE:
                self._cache.popitem(last=False)
        return steps

    def prefetch(self, pos: int) -> None:
        if 0 <= pos < len(self.episodes):
            threading.Thread(target=lambda: self._safe_steps(pos), daemon=True).start()

    def _safe_steps(self, pos: int) -> None:
        try:
            self.steps(pos)
        except Exception:
            pass   # the error shows when the episode is opened

    def state(self) -> dict:
        latest = self.store.latest()
        return {"name": self.name, "n": len(self.order), "order": self.order,
                "complete": [is_complete(latest.get(i)) for i in self.order],
                "started": [i in latest for i in self.order],
                "resume": resume_position(self.order, latest), "labels_file": self.labels_path.name}

    def episode(self, pos: int) -> dict:
        """Only what the labeller may see: the label id, the transcript and their own label."""
        ep = self.episodes[pos]
        steps = self.steps(pos)
        sha = T.transcript_sha256(steps)
        rec = self.store.get(ep["label_id"])
        self.prefetch(pos + 1)
        return {"pos": pos, "n": len(self.order), "label_id": ep["label_id"], "steps": steps,
                "turn_ids": T.turn_ids(steps), "transcript_sha256": sha,
                "transcript_changed": bool(ep.get("transcript_sha256")) and ep["transcript_sha256"] != sha,
                "label": {k: rec.get(k) for k in FIELDS} if rec else None,
                "saved_at": rec.get("saved_at") if rec else None, "missing": missing(rec)}

    def save(self, payload: dict) -> dict:
        if not isinstance(payload, dict) or not isinstance(payload.get("fields"), dict):
            raise LabelError("expected {label_id, fields}")
        label_id = payload.get("label_id")
        if label_id not in self.order:
            raise LabelError(f"unknown label id {label_id!r}")
        steps = self.steps(self.order.index(label_id))
        fields = validate(payload["fields"], T.step_ids(steps), T.turn_ids(steps))
        rec = self.store.append(label_id, fields, manifest=self.name, render_version=T.RENDER_VERSION,
                                transcript_sha256=T.transcript_sha256(steps))
        return {"label_id": label_id, "saved_at": rec["saved_at"], "missing": missing(rec),
                "complete": is_complete(rec)}


_LOCAL = {"127.0.0.1", "localhost", "[::1]"}


def make_handler(app: LabelApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "simtest-label"

        def log_message(self, fmt, *args):   # quiet; errors are answered as JSON
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def _local(self) -> bool:
            host = self.headers.get("Host") or ""
            name = host[: host.find("]") + 1] if host.startswith("[") else host.split(":")[0]
            if name in _LOCAL:
                return True
            self._json(403, {"error": "requests must be addressed to localhost"})
            return False

        def do_GET(self):
            if not self._local():
                return
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self._json(200, app.state())
                elif m := re.fullmatch(r"/api/episode/(\d+)", path):
                    pos = int(m.group(1))
                    if pos >= len(app.order):
                        self._json(404, {"error": f"no episode at position {pos}"})
                    else:
                        self._json(200, app.episode(pos))
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:   # shown on the page; the server keeps running
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_POST(self):
            if not self._local():
                return
            if urlparse(self.path).path != "/api/label":
                return self._json(404, {"error": "not found"})
            if not (self.headers.get("Content-Type") or "").startswith("application/json"):
                return self._json(415, {"error": "JSON only"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._json(413, {"error": "too large"})
            try:
                payload = json.loads(self.rfile.read(n) or b"null")
                self._json(200, app.save(payload))
            except (LabelError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def make_server(app: LabelApp, host: str = "127.0.0.1", port: int = 8766) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(app))


def serve(manifest: Path, labels: Path | None = None, host: str = "127.0.0.1", port: int = 8766) -> None:
    app = LabelApp(manifest, labels)
    httpd = make_server(app, host, port)
    st = app.state()
    print(f"Labelling {app.name}: http://{host}:{httpd.server_address[1]}/  "
          f"({sum(st['complete'])} of {st['n']} complete; labels: {app.labels_path})", flush=True)
    if app.store.skipped_lines:
        print(f"WARNING: {app.store.skipped_lines} unreadable line(s) in {app.labels_path} were ignored", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
