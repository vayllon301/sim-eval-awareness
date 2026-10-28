"""Run a coding-family config with Inspect's eval_set (resumable, one log dir per config).

Config format (configs/coding/*.yaml):

    family: coding
    name: coding-dev-c0
    scenarios: [pricing_cap, pricing_cap_benign]
    conditions: [C0]
    epochs: 10
    max_turns: 40
    deadline: 60                    # in-container command deadline (seconds)
    agents:
      qwen27b:
        model: rawchat/agent        # served name on the vLLM/OpenRouter endpoint
        base_url: http://localhost:18000/v1
        config: {temperature: 1.0, top_p: 0.95, top_k: 20, max_tokens: 32768}
    simulators:                     # the panel; required if a condition is C1-C3
      agentworld:
        model: rawchat/simulator
        base_url: http://localhost:18001/v1
        config: {temperature: 0.6, top_p: 0.95, top_k: 20, max_tokens: 16384}
    simulators_for: {C3: [agentworld]}    # optional; default: every simulator for every C1-C3
    run: {max_connections: 4, retry_on_error: 2, retry_attempts: 2}

The agent and the simulator run on OpenAI-compatible endpoints: `rawchat/<served-name>`
with a base_url (vLLM on Colab through the tunnel, or OpenRouter for a smoke test only).
`api_key_env` names an env var holding the key; the key never goes in the config or a log.
Agents refuse `reasoning_effort` (the template default, xhigh, is the plan's setting) and
must send the raw chain of thought back (the rawchat provider does).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml
from inspect_ai import eval_set
from inspect_ai.model import GenerateConfig, Model, get_model

from ..task import NO_SIMULATOR, OPERATIONAL_CONFIG_KEYS
from .images import build_all
from .observers import CONDITIONS
from .scenario import load_scenario
from .snapshot import check as check_snapshot
from .task import SIMULATOR_NAME, make_design, panel_task

REPO_DIR = Path(__file__).resolve().parents[3]
CONFIG_KEYS = {"family", "name", "scenarios", "conditions", "epochs", "max_turns", "deadline",
               "agents", "simulators", "simulators_for", "run"}
MODEL_KEYS = {"model", "base_url", "api_key_env", "config", "args"}
RUN_KEYS = {"max_connections", "retry_on_error", "retry_attempts"}
AGENT_CONFIG_DEFAULTS = {"max_retries": 5, "timeout": 1800, "attempt_timeout": 1200}
SIMULATOR_CONFIG_DEFAULTS = {"max_retries": 5, "timeout": 1200, "attempt_timeout": 900}
FORBIDDEN_AGENT_CONFIG = {"reasoning_effort"}  # the plan keeps the template default (xhigh)


class ConfigError(ValueError):
    pass


def _keys(where, d, allowed, required=frozenset()):
    if not isinstance(d, dict):
        raise ConfigError(f"{where} must be a mapping")
    if unknown := sorted(set(d) - set(allowed)):
        raise ConfigError(f"{where}: unknown keys {unknown} (allowed: {sorted(allowed)})")
    if missing := sorted(set(required) - set(d)):
        raise ConfigError(f"{where}: missing keys {missing}")


def _positive_int(where, v, minimum=1):
    if not (isinstance(v, int) and not isinstance(v, bool) and v >= minimum):
        raise ConfigError(f"{where} must be an integer >= {minimum}, got {v!r}")
    return v


def _model_spec(where, spec, agent):
    _keys(where, spec, MODEL_KEYS, {"model", "base_url"})
    model = spec["model"]
    if not (isinstance(model, str) and model.startswith(("rawchat/", "mockllm/"))):
        raise ConfigError(f"{where}.model must start with rawchat/ or mockllm/, got {model!r}")
    config = spec.get("config") or {}
    _keys(f"{where}.config", config, set(GenerateConfig.model_fields))
    if agent and (bad := sorted(FORBIDDEN_AGENT_CONFIG & set(config))):
        raise ConfigError(f"{where}.config: an agent must not set {bad} (the plan keeps the xhigh template default)")
    try:
        GenerateConfig(**config)
    except Exception as exc:
        raise ConfigError(f"{where}.config: {exc}") from exc
    args = dict(spec.get("args") or {})
    if spec.get("base_url"):
        args["base_url"] = spec["base_url"]
    if spec.get("api_key_env"):
        args["api_key_env"] = spec["api_key_env"]
    defaults = AGENT_CONFIG_DEFAULTS if agent else SIMULATOR_CONFIG_DEFAULTS
    return {"model": model, "config": {**defaults, **config}, "args": args}


def _simulators(raw, conditions):
    llm = [c for c in conditions if c != "C0"]
    panel = raw.get("simulators")
    if not llm:
        if panel or "simulators_for" in raw:
            raise ConfigError("simulators given but every condition is C0")
        return {}, {}
    if not panel:
        raise ConfigError(f"simulators: is required for {', '.join(llm)}")
    specs = {}
    for name, spec in panel.items():
        if not (isinstance(name, str) and SIMULATOR_NAME.fullmatch(name) and name != NO_SIMULATOR):
            raise ConfigError(f"simulator name {name!r} must be a simple identifier other than {NO_SIMULATOR!r}")
        specs[name] = _model_spec(f"simulators.{name}", spec, agent=False)
    restrict = raw.get("simulators_for") or {}
    if not isinstance(restrict, dict):
        raise ConfigError("simulators_for must be a mapping condition -> [names]")
    for cond, names in restrict.items():
        if cond not in llm:
            raise ConfigError(f"simulators_for: {cond!r} is not an LLM condition in {conditions}")
        if not (isinstance(names, list) and names and len(set(names)) == len(names)):
            raise ConfigError(f"simulators_for.{cond} must be a non-empty list without duplicates")
        if unknown := [n for n in names if n not in specs]:
            raise ConfigError(f"simulators_for.{cond}: unknown simulators {unknown}")
    per = {c: list(restrict.get(c, specs)) for c in llm}
    if unused := [n for n in specs if not any(n in v for v in per.values())]:
        raise ConfigError(f"simulators {unused} are not used by any condition")
    return specs, per


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    _keys("config", raw, CONFIG_KEYS, {"name", "scenarios", "conditions", "agents"})
    if raw.get("family") not in (None, "coding"):
        raise ConfigError(f"family must be 'coding', got {raw.get('family')!r}")
    name = raw["name"]
    if not (isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9._-]+", name)):
        raise ConfigError(f"name must be a simple identifier, got {name!r}")
    scenarios = raw["scenarios"]
    if not (isinstance(scenarios, list) and scenarios and all(isinstance(s, str) for s in scenarios)):
        raise ConfigError("scenarios must be a non-empty list of scenario ids")
    for s in scenarios:
        load_scenario(s)
    conditions = raw["conditions"]
    if not (isinstance(conditions, list) and conditions and len(set(conditions)) == len(conditions)):
        raise ConfigError("conditions must be a non-empty list without duplicates")
    if bad := [c for c in conditions if c not in CONDITIONS]:
        raise ConfigError(f"unknown conditions {bad}; expected some of {list(CONDITIONS)}")
    agents = raw["agents"]
    if not (isinstance(agents, dict) and agents):
        raise ConfigError("agents must be a non-empty mapping")
    agents = {k: _model_spec(f"agents.{k}", v, agent=True) for k, v in agents.items()}
    if len({a["model"] for a in agents.values()}) != len(agents):
        raise ConfigError("each agent must use a different served model name")
    simulators, simulators_for = _simulators(raw, conditions)
    run = raw.get("run") or {}
    _keys("run", run, RUN_KEYS)
    return {
        "name": name, "scenarios": scenarios, "conditions": conditions,
        "epochs": _positive_int("epochs", raw.get("epochs", 1)),
        "max_turns": _positive_int("max_turns", raw.get("max_turns", 40)),
        "deadline": _positive_int("deadline", raw.get("deadline", 60)),
        "agents": agents, "simulators": simulators, "simulators_for": simulators_for,
        "run": {"max_connections": _positive_int("run.max_connections", run.get("max_connections", 4)),
                "retry_on_error": _positive_int("run.retry_on_error", run.get("retry_on_error", 2), 0),
                "retry_attempts": _positive_int("run.retry_attempts", run.get("retry_attempts", 2), 0)},
    }


def request_config(cfg, simulator=None):
    def request(spec):
        config = {k: v for k, v in sorted(spec["config"].items()) if k not in OPERATIONAL_CONFIG_KEYS}
        args = {k: v for k, v in spec["args"].items() if k != "api_key_env"}  # the env var name isn't a request setting
        return {"model": spec["model"], "config": config, "args": args}

    return {"agents": {k: request(s) for k, s in cfg["agents"].items()},
            "simulator": request(cfg["simulators"][simulator]) if simulator else None}


def build_model(spec) -> Model:
    config = GenerateConfig(**spec["config"], fail_on_refusal=False)
    return get_model(spec["model"], config=config, memoize=False, **spec["args"])


def git_dirty(repo: Path = REPO_DIR) -> bool:
    try:
        out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                             check=True)
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.stdout.strip())


def run_config(path, allow_dirty=False, log_root="logs", display=None, build=True, check_snapshots=True):
    cfg = load_config(path)
    if git_dirty() and not allow_dirty:
        raise SystemExit("Refusing to run from a dirty git working tree: commit first, or pass --allow-dirty.")
    api_key_envs = {s["args"].get("api_key_env") for s in [*cfg["agents"].values(), *cfg["simulators"].values()]}
    for env in sorted(filter(None, api_key_envs)):
        if not os.environ.get(env):
            raise SystemExit(f"{env} is not set. Run with: uv run --env-file ../.env simtest run ...")
    scenarios = [load_scenario(s) for s in cfg["scenarios"]]
    if build:
        build_all()
    if check_snapshots:
        for sc in scenarios:
            if diff := check_snapshot(sc):
                raise SystemExit(f"{sc.id}: the frozen snapshot differs from a fresh container. Re-run "
                                 f"`simtest snapshot {sc.id}` and review:\n" + "\n".join(diff[:40]))
    agents = {k: build_model(s) for k, s in cfg["agents"].items()}
    simulators = {n: build_model(s) for n, s in cfg["simulators"].items()}
    log_dir = Path(log_root) / cfg["name"]
    design = make_design(cfg["conditions"], cfg["simulators_for"],
                         {k: s["model"] for k, s in cfg["agents"].items()},
                         [sc.id for sc in scenarios], cfg["epochs"])
    common = {"scenarios": cfg["scenarios"], "max_turns": cfg["max_turns"], "epochs": cfg["epochs"],
              "deadline": cfg["deadline"], "design": design}
    tasks = []
    for c in cfg["conditions"]:
        if c == "C0":
            tasks.append(panel_task(c, request_config=request_config(cfg, None), **common))
        else:
            tasks += [panel_task(c, simulator=s, simulator_model=simulators[s],
                                 request_config=request_config(cfg, s), **common)
                      for s in cfg["simulators_for"][c]]
    success, logs = eval_set(
        tasks=tasks, model=list(agents.values()), log_dir=str(log_dir),
        retry_on_error=cfg["run"]["retry_on_error"], retry_attempts=cfg["run"]["retry_attempts"],
        max_connections=cfg["run"]["max_connections"], log_model_api=True, display=display,
        metadata={"family": "coding", "config_name": cfg["name"], "config_file": str(path),
                  "agents": {k: s["model"] for k, s in cfg["agents"].items()},
                  "simulators": {n: s["model"] for n, s in cfg["simulators"].items()}},
    )
    return success, logs, log_dir
