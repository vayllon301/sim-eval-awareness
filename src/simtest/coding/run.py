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
    run: {max_connections: 4, retry_attempts: 2, order_seed: 20261102}

`run.max_connections` is the total number of episodes in flight, over every condition, and the
connection limit of each model. Every condition's task runs at once, and the episodes share one
queue in a seeded, blocked order (`run.order_seed`; src/simtest/runqueue.py). An episode that
errors is re-run from the start up to `run.retry_attempts` times, at the back of the same queue.

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
# retry_on_error is accepted for the older configs and unused: re-runs follow retry_attempts.
RUN_KEYS = {"max_connections", "retry_on_error", "retry_attempts", "order_seed"}
AGENT_CONFIG_DEFAULTS = {"max_retries": 5, "timeout": 1800, "attempt_timeout": 1200}
# The simulator's calls in the flows world write up to about 21K tokens; with its KV cache full
# (preemptions) that took more than 900 s, and the 1200 s total left no room for a retry
# (LOG.md, 01-11-2026). The same limits in every simulated condition.
SIMULATOR_CONFIG_DEFAULTS = {"max_retries": 5, "timeout": 3600, "attempt_timeout": 2400}
# Re-runs (PLAN.md, Design): an errored episode is re-run from the start, whatever it contained,
# and the failed attempt is kept and reported. They are Inspect's in-place sample retries
# (`retry_on_error`, set from the config's `retry_attempts`): the retry gets a fresh state and
# goes to the back of the run's queue (src/simtest/runqueue.py), so it runs mixed with the rest,
# never alone at the end (session 5: two re-runs held the VM alone for 80 minutes; LOG.md,
# 01-11-2026). The episode solver keeps each failed attempt in full under `attempts/`, because
# Inspect keeps only its error. No task-level retries (retry_attempts=0): those wait until the task
# has drained. An episode left over (errored after its re-runs, or not started) runs on the next
# `simtest run` of the same config, mixed into that session's queue. retry_cleanup=False keeps
# every attempt log, and load_run aggregates them. continue_on_fail: one episode's error no
# longer cancels the others (session 5 lost 12 at 75 minutes; LOG.md, 01-11-2026).
EVAL_SET_OPTIONS = {"retry_attempts": 0, "retry_cleanup": False, "continue_on_fail": True}
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
                "retry_attempts": _positive_int("run.retry_attempts", run.get("retry_attempts", 2), 0),
                "order_seed": _positive_int("run.order_seed", run.get("order_seed", 0), 0)},
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


def run_config(path, allow_dirty=False, log_root="logs", display=None, build=True, check_snapshots=True,
               epochs=None, admit_for=None, carry_under=0, machine="docker", models=None):
    """Run a config (resumable). `epochs` ("2" or "1-3") limits this session to those epochs;
    `admit_for` (hours) stops starting episodes after that long; `carry_under` leaves the rest
    for the next session when fewer than that many episodes are left (src/simtest/runqueue.py).
    `machine` and `models` (a function from a model spec to a Model) are for offline runs:
    `simtest preflight` and the tests."""
    import time

    from .. import runqueue

    cfg = load_config(path)
    if git_dirty() and not allow_dirty:
        raise SystemExit("Refusing to run from a dirty git working tree: commit first, or pass --allow-dirty.")
    if models is None:
        api_key_envs = {s["args"].get("api_key_env") for s in [*cfg["agents"].values(), *cfg["simulators"].values()]}
        for env in sorted(filter(None, api_key_envs)):
            if not os.environ.get(env):
                raise SystemExit(f"{env} is not set. Run with: uv run --env-file ../.env simtest run ...")
    allowed = runqueue.parse_epochs(epochs, cfg["epochs"])
    scenarios = [load_scenario(s) for s in cfg["scenarios"]]
    if build:
        build_all()
    if check_snapshots:
        for sc in scenarios:
            if diff := check_snapshot(sc):
                raise SystemExit(f"{sc.id}: the frozen snapshot differs from a fresh container. Re-run "
                                 f"`simtest snapshot {sc.id}` and review:\n" + "\n".join(diff[:40]))
    make = models or build_model
    agents = {k: make(s) for k, s in cfg["agents"].items()}
    simulators = {n: make(s) for n, s in cfg["simulators"].items()}
    log_dir = Path(log_root) / cfg["name"]
    design = make_design(cfg["conditions"], cfg["simulators_for"],
                         {k: s["model"] for k, s in cfg["agents"].items()},
                         [sc.id for sc in scenarios], cfg["epochs"])
    common = {"scenarios": cfg["scenarios"], "max_turns": cfg["max_turns"], "epochs": cfg["epochs"],
              "deadline": cfg["deadline"], "design": design, "machine": machine}
    tasks, arms = [], []
    for c in cfg["conditions"]:
        if c == "C0":
            tasks.append(panel_task(c, request_config=request_config(cfg, None), **common))
            arms.append((c, NO_SIMULATOR))
        else:
            tasks += [panel_task(c, simulator=s, simulator_model=simulators[s],
                                 request_config=request_config(cfg, s), **common)
                      for s in cfg["simulators_for"][c]]
            arms += [(c, s) for s in cfg["simulators_for"][c]]
    if len(agents) > 1:   # Inspect would pair each task with each agent; the queue's keys have no agent
        raise ConfigError("the run queue supports one agent per config")
    run = cfg["run"]
    order = runqueue.episode_order(arms, cfg["scenarios"], cfg["epochs"], run["order_seed"])
    queue = runqueue.EpisodeQueue(order, run["max_connections"], epochs=allowed, carry_under=carry_under,
                                  admit_until=time.time() + admit_for * 3600 if admit_for is not None else None,
                                  log_dir=log_dir, settle=runqueue.SETTLE_SECONDS)
    per_task = len(cfg["scenarios"]) * cfg["epochs"]
    with runqueue.installed(queue):
        success, logs = eval_set(
            tasks=tasks, model=list(agents.values()), log_dir=str(log_dir),
            retry_on_error=run["retry_attempts"], **EVAL_SET_OPTIONS,
            # Every task at once, and every episode admitted to Inspect: the queue holds them.
            max_tasks=len(tasks), max_samples=per_task,
            max_connections=run["max_connections"], log_model_api=True, display=display,
            metadata={"family": "coding", "config_name": cfg["name"], "config_file": str(path),
                      "agents": {k: s["model"] for k, s in cfg["agents"].items()},
                      "simulators": {n: s["model"] for n, s in cfg["simulators"].items()},
                      "queue": {"in_flight": run["max_connections"], "order_seed": run["order_seed"],
                                "order": "blocked: epoch, then scenario, then arm (src/simtest/runqueue.py)"}},
        )
    print(queue.summary())
    return success, logs, log_dir
