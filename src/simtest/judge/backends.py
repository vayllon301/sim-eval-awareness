"""Where the judge's calls go. Each backend has a `name`, a `model`, the `settings` that go
into the cache key, `max_output`, and `async complete(prompt) -> {"text", ...}`.

- `InspectBackend`: Inspect's model API, so any provider works ("openrouter/<org>/<model>",
  "mockllm/model" in tests). Fixed, recorded settings: temperature, max tokens, reasoning
  effort, seed. The frozen judge runs here, through OpenRouter.
- `CodexBackend`: `codex exec` with the ChatGPT login, for free tuning passes. The prompt
  goes on stdin, from an empty temporary directory, read-only, with web search, the shell
  and the other optional tools turned off, no session saved, and the user's config.toml
  ignored. Codex still adds its own system prompt and the user's `~/.codex/AGENTS.md`, so it
  is never the judge whose kappa is reported. Any tool call in its event stream makes the
  answer unusable (the judge must answer from the transcript alone).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile

from .judge import BackendError

CODEX_DISABLE = ("shell_tool", "unified_exec", "apps", "plugins", "memories", "multi_agent", "image_generation",
                 "browser_use", "computer_use", "in_app_browser", "goals", "skill_search", "hooks", "view_image",
                 "tool_suggest")
CODEX_ITEMS_OK = {"agent_message", "reasoning"}


class InspectBackend:
    name = "inspect"

    def __init__(self, model, temperature: float | None = 0.0, max_tokens: int = 16000,
                 reasoning_effort: str | None = None, seed: int | None = None, timeout: int = 900,
                 max_retries: int = 5):
        from inspect_ai.model import GenerateConfig, get_model

        self.settings = {"temperature": temperature, "max_tokens": max_tokens, "reasoning_effort": reasoning_effort,
                         "seed": seed}
        self.max_output = max_tokens
        cfg = {k: v for k, v in self.settings.items() if v is not None}
        self.config = GenerateConfig(**cfg, timeout=timeout, max_retries=max_retries)
        self._model = model if not isinstance(model, str) else get_model(model, config=self.config)
        self.model = model if isinstance(model, str) else str(self._model)

    async def complete(self, prompt: str) -> dict:
        from inspect_ai.model import ChatMessageUser

        try:
            out = await self._model.generate([ChatMessageUser(content=prompt)], config=self.config)
        except Exception as exc:   # provider and network errors: a rerun retries them
            raise BackendError(f"{type(exc).__name__}: {exc}") from exc
        if out.error:
            raise BackendError(f"model error: {out.error}")
        usage = out.usage.model_dump(exclude_none=True) if out.usage else None
        return {"text": out.completion, "stop_reason": out.stop_reason, "served_by": out.model, "usage": usage}


def codex_version(exe: str = "codex") -> str | None:
    try:
        return subprocess.run([exe, "--version"], capture_output=True, text=True, check=True,
                              timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def codex_logged_in(exe: str = "codex") -> bool:
    """`codex login status`, read for a yes/no only (never printed: it can name the account)."""
    try:
        r = subprocess.run([exe, "login", "status"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and "logged in" in (r.stdout + r.stderr).lower()


def parse_codex_events(stdout: str) -> dict:
    """The final message, the usage and any tool use from `codex exec --json`'s events."""
    text, usage, tools, errors = None, None, [], []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t in ("item.started", "item.completed", "item.updated"):
            item = ev.get("item") or {}
            kind = item.get("type") or item.get("item_type")
            if kind == "agent_message" and t == "item.completed":
                text = item.get("text", "")
            elif kind and kind not in CODEX_ITEMS_OK and kind not in tools:
                tools.append(kind)
        elif t == "turn.completed":
            usage = ev.get("usage")
        elif t in ("turn.failed", "error"):
            errors.append(json.dumps(ev.get("error", ev))[:500])
    return {"text": text, "usage": usage, "tools_used": tools, "errors": errors}


class CodexBackend:
    name = "codex"

    def __init__(self, model: str = "gpt-6.1-sol", reasoning_effort: str = "high", exe: str = "codex",
                 timeout: int = 1800, version: str | None = None):
        self.exe = shutil.which(exe) or exe
        self.version = version or codex_version(self.exe)
        if not self.version:
            raise BackendError(f"{exe} is not installed (codex --version failed)")
        self.model = model
        self.settings = {"reasoning_effort": reasoning_effort, "cli": self.version}
        self.max_output = 0     # the CLI sets it; the model's context is far above our prompts
        self.timeout = timeout

    def command(self, workdir: str) -> list[str]:
        cmd = [self.exe, "exec", "--json", "-m", self.model,
               "-c", f'model_reasoning_effort="{self.settings["reasoning_effort"]}"',
               "-c", 'web_search="disabled"']
        for f in CODEX_DISABLE:
            cmd += ["--disable", f]
        return cmd + ["-s", "read-only", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
                      "--ignore-rules", "--color", "never", "-C", workdir, "-"]

    async def complete(self, prompt: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="simtest-judge-") as workdir:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *self.command(workdir), cwd=workdir, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "NO_COLOR": "1"})
            except OSError as exc:
                raise BackendError(f"codex exec could not start: {exc}") from exc
            try:
                out, err = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise BackendError(f"codex exec timed out after {self.timeout} s") from None
        ev = parse_codex_events(out.decode(errors="replace"))
        if proc.returncode != 0 or ev["errors"] or ev["text"] is None:
            tail = err.decode(errors="replace").strip().splitlines()[-3:]
            raise BackendError(f"codex exec exit {proc.returncode}: {ev['errors'] or tail}"[:2000])
        return {"text": ev["text"], "usage": ev["usage"], "tools_used": ev["tools_used"]}
