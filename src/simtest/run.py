"""Run an experiment config with Inspect's eval_set (resumable, one log dir per config).

Config format (see configs/*.yaml):

    name: m1-baseline
    scenarios: [report_q3]
    conditions: [C0]
    epochs: 10
    max_turns: 20
    agents:
      <key>: {model: openrouter/..., config: {<GenerateConfig fields>}, args: {<provider args>}}
    simulators:                       # the panel; only if a condition is C1-C3
      <name>: {model: ..., config: {...}, args: {...}}
    simulators_for: {C1: [<name>, ...]}   # optional; default: every simulator for every C1-C3
    run: {max_connections: 4, retry_on_error: 2, retry_attempts: 2}

C0 runs once, without a simulator. Every other condition runs once per simulator in
its `simulators_for` list, as its own task with that simulator bound at task level.

Keys are validated strictly: a typo in model args would otherwise only surface as
a TypeError deep inside the OpenAI client. The API key is read by Inspect from
OPENROUTER_API_KEY in the environment (`uv run --env-file .env ...`), never from
the config.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import yaml
from inspect_ai import eval_set
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig, Model, get_model

from .observers import CONDITIONS
from .scenarios import load_scenario
from .task import NO_SIMULATOR, OPERATIONAL_CONFIG_KEYS, PACKAGE_DIR, SIMULATOR_NAME, make_design, panel_task

REPO_DIR = PACKAGE_DIR.parents[1]

CONFIG_KEYS = {"name", "scenarios", "conditions", "epochs", "max_turns", "agents", "simulators", "simulators_for", "run"}
MODEL_KEYS = {"model", "config", "args"}
RUN_KEYS = {"max_connections", "retry_on_error", "retry_attempts"}
# Model args accepted per provider (anything else would reach the provider client).
PROVIDER_ARGS = {
    "openrouter": {"provider", "models", "transforms", "reasoning_enabled", "strict_tools"},
    "mockllm": set(),
}
# OpenRouter provider-routing object (https://openrouter.ai/docs/features/provider-routing).
OPENROUTER_ROUTING_KEYS = {
    "order", "allow_fallbacks", "require_parameters", "data_collection", "only", "ignore",
    "quantizations", "sort", "max_price", "zdr",
}
# fail_on_refusal makes Inspect raise on a content_filter stop before our loop can flag
# it, and the retry then replaces the filtered episode: never allowed, always forced off.
FORBIDDEN_CONFIG_KEYS = {"fail_on_refusal"}
# Claude 5.x on OpenRouter: the endpoints list no sampling parameters, so a temperature
# was silently dropped (M1/M1b), and with provider.require_parameters: true the request
# fails. Rejected at load time instead. Both naming orders (claude-opus-5.5,
# claude-5-opus, bare claude-5), dotted or dashed minor, dated ids and :suffixes; the
# major version must be exactly 5 (not claude-3-5-sonnet, claude-haiku-4.5, claude-50).
CLAUDE_5 = re.compile(r"(?:^|/)anthropic/claude-(?:[a-z]+-)?5(?=$|[.:-])")
NO_SAMPLING_KEYS = {"temperature", "top_p", "top_k"}
# Defaults so failures surface instead of retrying forever. cache_prompt is off because
# OpenRouter's Anthropic cache markers don't appear in the logs. Role models (the
# simulators) don't inherit the eval's sampling config, so they get their own:
# temperature 1.0 (the Anthropic default), except Claude 5.x, which takes none.
AGENT_CONFIG_DEFAULTS = {"cache_prompt": False, "max_retries": 5, "timeout": 900, "attempt_timeout": 300}
SIMULATOR_CONFIG_DEFAULTS = {"cache_prompt": False, "max_retries": 5, "timeout": 600, "attempt_timeout": 180}
SIMULATOR_SAMPLING_DEFAULTS = {"temperature": 1.0}
AGENT_ARG_DEFAULTS = {"openrouter": {"strict_tools": False}}


class ConfigError(ValueError):
    pass


def _check_keys(where: str, d, allowed: set[str], required: set[str] = set()) -> None:
    if not isinstance(d, dict):
        raise ConfigError(f"{where} must be a mapping")
    if unknown := sorted(set(d) - allowed):
        raise ConfigError(f"{where}: unknown keys {unknown} (allowed: {sorted(allowed)})")
    if missing := sorted(required - set(d)):
        raise ConfigError(f"{where}: missing keys {missing}")


def _positive_int(where: str, v, minimum: int = 1) -> int:
    if not (isinstance(v, int) and not isinstance(v, bool) and v >= minimum):
        raise ConfigError(f"{where} must be an integer >= {minimum}, got {v!r}")
    return v


def _model_spec(where: str, spec: dict, agent: bool) -> dict:
    _check_keys(where, spec, MODEL_KEYS, {"model"})
    model = spec["model"]
    provider = model.split("/", 1)[0] if isinstance(model, str) and "/" in model else None
    if provider not in PROVIDER_ARGS:
        raise ConfigError(f"{where}.model must start with one of {sorted(PROVIDER_ARGS)}/, got {model!r}")
    config = spec.get("config") or {}
    _check_keys(f"{where}.config", config, set(GenerateConfig.model_fields) - FORBIDDEN_CONFIG_KEYS)
    try:
        GenerateConfig(**config)
    except Exception as exc:
        raise ConfigError(f"{where}.config: {exc}") from exc
    no_sampling = bool(CLAUDE_5.search(model))
    if no_sampling and (bad := sorted(NO_SAMPLING_KEYS & set(config))):
        raise ConfigError(
            f"{where}.config: {model} is a Claude 5.x model and takes no {bad}: its OpenRouter endpoints "
            "list no temperature, top_p or top_k, so the value would be silently dropped, and with "
            "provider.require_parameters: true the request fails. Remove it (the provider default applies)."
        )
    args = spec.get("args") or {}
    _check_keys(f"{where}.args", args, PROVIDER_ARGS[provider])
    if "provider" in args:
        _check_keys(f"{where}.args.provider", args["provider"], OPENROUTER_ROUTING_KEYS)
    if agent and args.get("strict_tools", False) is not False:
        raise ConfigError(f"{where}.args.strict_tools must be false (tool schemas must match TOOL_SCHEMAS)")
    defaults = AGENT_CONFIG_DEFAULTS if agent else {
        **SIMULATOR_CONFIG_DEFAULTS, **({} if no_sampling else SIMULATOR_SAMPLING_DEFAULTS)}
    arg_defaults = AGENT_ARG_DEFAULTS.get(provider, {}) if agent else {}
    return {"model": model, "config": {**defaults, **config}, "args": {**arg_defaults, **args}}


def _simulators(raw: dict, conditions: list[str]) -> tuple[dict, dict]:
    """The validated panel (name -> spec) and, per LLM condition, the simulators it runs with."""
    llm_conditions = [c for c in conditions if c != "C0"]
    panel = raw.get("simulators")
    if not llm_conditions:
        if panel or "simulators_for" in raw:
            raise ConfigError("simulators given but every condition is C0 (C0 never uses a simulator)")
        return {}, {}
    if not panel:
        raise ConfigError(f"simulators: is required for {', '.join(llm_conditions)} (a mapping name -> model spec)")
    if not isinstance(panel, dict):
        raise ConfigError("simulators must be a mapping name -> {model, config, args}")
    specs = {}
    for name, spec in panel.items():
        if not (isinstance(name, str) and SIMULATOR_NAME.fullmatch(name)):
            raise ConfigError(f"simulator name {name!r} must be a simple identifier (letters, digits, _)")
        if name == NO_SIMULATOR:
            raise ConfigError(f"simulator name {NO_SIMULATOR!r} is reserved for C0 (no simulator)")
        specs[name] = _model_spec(f"simulators.{name}", spec, agent=False)
    seen: dict[str, str] = {}
    for name, spec in specs.items():
        key = json.dumps([spec["model"], spec["args"]], sort_keys=True)
        if key in seen:
            raise ConfigError(f"simulators {seen[key]} and {name} have the same model and args: "
                              "a panel member must differ in model or provider routing")
        seen[key] = name
    restrict = raw.get("simulators_for") or {}
    if not isinstance(restrict, dict):
        raise ConfigError("simulators_for must be a mapping condition -> list of simulator names")
    for cond, names in restrict.items():
        if cond == "C0":
            raise ConfigError("simulators_for: C0 never uses a simulator")
        if cond not in llm_conditions:
            raise ConfigError(f"simulators_for: {cond!r} is not in conditions {conditions}")
        if not (isinstance(names, list) and names):
            raise ConfigError(f"simulators_for.{cond} must be a non-empty list of simulator names")
        if len(set(map(str, names))) != len(names):
            raise ConfigError(f"simulators_for.{cond} has duplicates: {names}")
        if unknown := [n for n in names if n not in specs]:
            raise ConfigError(f"simulators_for.{cond}: unknown simulators {unknown} (panel: {list(specs)})")
    per_condition = {c: list(restrict.get(c, specs)) for c in llm_conditions}
    if unused := [n for n in specs if not any(n in names for names in per_condition.values())]:
        raise ConfigError(f"simulators {unused} are not used by any condition (see simulators_for)")
    return specs, per_condition


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if isinstance(raw, dict) and "simulator" in raw:
        raise ConfigError("the single 'simulator:' block was replaced by a panel: use 'simulators:' "
                          "(a mapping name -> {model, config, args}), e.g. simulators: {qwen27b: {model: ...}}")
    _check_keys("config", raw, CONFIG_KEYS, {"name", "scenarios", "conditions", "agents"})
    name = raw["name"]
    if not (isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9._-]+", name)):
        raise ConfigError(f"name must be a simple identifier, got {name!r}")
    scenarios = raw["scenarios"]
    if not (isinstance(scenarios, list) and scenarios and all(isinstance(s, str) for s in scenarios)):
        raise ConfigError("scenarios must be a non-empty list of scenario ids")
    for s in scenarios:
        load_scenario(s)  # fails now, not mid-run
    conditions = raw["conditions"]
    if not (isinstance(conditions, list) and conditions and len(set(conditions)) == len(conditions)):
        raise ConfigError("conditions must be a non-empty list without duplicates")
    if bad := [c for c in conditions if c not in CONDITIONS]:
        raise ConfigError(f"unknown conditions {bad}; expected some of {list(CONDITIONS)}")
    agents = raw["agents"]
    if not (isinstance(agents, dict) and agents):
        raise ConfigError("agents must be a non-empty mapping")
    agents = {k: _model_spec(f"agents.{k}", v, agent=True) for k, v in agents.items()}
    models = [a["model"] for a in agents.values()]
    if len(set(models)) != len(models):
        raise ConfigError("each agent must use a different model id")
    simulators, simulators_for = _simulators(raw, conditions)
    run = raw.get("run") or {}
    _check_keys("run", run, RUN_KEYS)
    return {
        "name": name,
        "scenarios": scenarios,
        "conditions": conditions,
        "epochs": _positive_int("epochs", raw.get("epochs", 1)),
        "max_turns": _positive_int("max_turns", raw.get("max_turns", 20)),
        "agents": agents,
        "simulators": simulators,
        "simulators_for": simulators_for,
        "run": {
            "max_connections": _positive_int("run.max_connections", run.get("max_connections", 4)),
            "retry_on_error": _positive_int("run.retry_on_error", run.get("retry_on_error", 2), minimum=0),
            # eval_set task retries (Inspect's default is 10)
            "retry_attempts": _positive_int("run.retry_attempts", run.get("retry_attempts", 2), minimum=0),
        },
    }


def request_config(cfg: dict, simulator: str | None = None) -> dict:
    """The effective request configuration of every agent and of one task's simulator
    (model, config incl. cache_prompt, args incl. provider routing, reasoning,
    strict_tools); `simulator` None for C0. It is part of that task's fingerprint, so
    any change refuses to resume a log dir. Only the task's own simulator counts: the
    shared C0 task and the other simulators' tasks don't change when one panel member
    does."""

    def request(spec: dict) -> dict:
        config = {k: v for k, v in sorted(spec["config"].items()) if k not in OPERATIONAL_CONFIG_KEYS}
        return {"model": spec["model"], "config": config, "args": spec["args"]}

    return {
        "agents": {key: request(spec) for key, spec in cfg["agents"].items()},
        "simulator": request(cfg["simulators"][simulator]) if simulator else None,
    }


def build_model(spec: dict) -> Model:
    # explicit False, so a changed upstream default can't turn refusals into retries
    config = GenerateConfig(**spec["config"], fail_on_refusal=False)
    return get_model(spec["model"], config=config, memoize=False, **spec["args"])


def git_dirty(repo: Path = REPO_DIR) -> bool:
    """True if the working tree has uncommitted changes or untracked files (or no git)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(out.stdout.strip())


def run_config(
    path: str | Path, allow_dirty: bool = False, log_root: str | Path = "logs", display: str | None = None
) -> tuple[bool, list[EvalLog], Path]:
    cfg = load_config(path)
    if git_dirty() and not allow_dirty:
        raise SystemExit(
            "Refusing to run from a dirty git working tree: every result must map to a commit. "
            "Commit first, or pass --allow-dirty for a throwaway run."
        )
    models = [s["model"] for s in [*cfg["agents"].values(), *cfg["simulators"].values()]]
    if any(m.startswith("openrouter/") for m in models) and not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set. Run with: uv run --env-file .env simtest run ...")
    agents = {key: build_model(spec) for key, spec in cfg["agents"].items()}
    simulators = {name: build_model(spec) for name, spec in cfg["simulators"].items()}
    log_dir = Path(log_root) / cfg["name"]
    # Every task carries the whole intended design, so the analysis knows what a complete
    # run has even if some tasks never started or logged nothing.
    design = make_design(cfg["conditions"], cfg["simulators_for"],
                         {key: spec["model"] for key, spec in cfg["agents"].items()},
                         [load_scenario(s).id for s in cfg["scenarios"]], cfg["epochs"])
    common = {"scenarios": cfg["scenarios"], "max_turns": cfg["max_turns"], "epochs": cfg["epochs"], "design": design}
    tasks = []
    for c in cfg["conditions"]:
        if c == "C0":  # once, shared by every simulator; no simulator role
            tasks.append(panel_task(c, **common, request_config=request_config(cfg, None)))
        else:  # one task per simulator, bound at task level
            tasks += [panel_task(c, **common, simulator=s, simulator_model=simulators[s],
                                 request_config=request_config(cfg, s))
                      for s in cfg["simulators_for"][c]]
    # No eval-level model_roles: they would override every task's own simulator.
    success, logs = eval_set(
        tasks=tasks,
        model=list(agents.values()),
        log_dir=str(log_dir),
        retry_on_error=cfg["run"]["retry_on_error"],
        retry_attempts=cfg["run"]["retry_attempts"],
        max_connections=cfg["run"]["max_connections"],
        log_model_api=True,
        display=display,
        metadata={
            "config_name": cfg["name"],
            "config_file": str(path),
            "agents": {key: spec["model"] for key, spec in cfg["agents"].items()},
            "simulators": {name: spec["model"] for name, spec in cfg["simulators"].items()},
        },
    )
    return success, logs, log_dir
