"""The Inspect task: one sample per scenario, one task per condition and simulator.

C0 is one task with no simulator role (`simulator="none"`). Each LLM condition runs
once per panel simulator: `panel_task` binds that simulator at task level
(`Task.model_roles`), so eval_set never needs eval-level roles, which would override
every task's binding. The panel name is a task arg, so it is part of eval_set's task
identity, the fingerprint, the task and sample metadata, and the logs.

`simtest run` also gives every task the run's intended design (`make_design`) as a task
arg: conditions, the simulators of each LLM condition, agents, scenario ids and epochs.
The analysis uses it instead of inferring the panel from whatever episodes exist, so an
interrupted run reports missing cells instead of a smaller panel.

The fingerprint (scenario files, package source incl. prompts and tool schemas,
dependency lock, task arguments incl. the request configuration of every model) is
the task `version`. eval_set's task identity covers `version` but not source,
scenario or prompt changes, nor request settings such as cache_prompt, so this is
what makes resuming a log directory refuse after any of those changed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from inspect_ai import Epochs, Task, task, task_with
from inspect_ai.dataset import Sample
from inspect_ai.model import Model

from .episode import load_scenarios, simtest_agent
from .observers import CONDITIONS
from .scenarios import scenario_path
from .scorer import simtest_scorer

PACKAGE_DIR = Path(__file__).resolve().parent
LOCKFILE = PACKAGE_DIR.parents[1] / "uv.lock"
NO_SIMULATOR = "none"  # the simulator name of C0 (shared baseline, no simulator role)
# Operational settings: they don't change what is sent to a model, so they stay out of
# the fingerprint and of the log-directory checks (everything else in a model's config
# and args is fingerprinted and compared).
OPERATIONAL_CONFIG_KEYS = frozenset({
    "max_retries", "timeout", "attempt_timeout", "stream_idle_timeout", "max_connections", "adaptive_connections",
})
SIMULATOR_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def make_design(conditions: list[str], simulators_for: dict[str, list[str]], agents: dict[str, str],
                scenarios: list[str], epochs: int) -> dict:
    """The intended design of a run: what a complete run has episodes for."""
    return {
        "conditions": list(conditions),
        "simulators": {c: list(simulators_for[c]) for c in conditions if c != "C0"},
        "agents": dict(agents),
        "scenarios": list(scenarios),
        "epochs": epochs,
    }


def check_design(condition: str, simulator: str, design: dict | None) -> None:
    if design is None:
        return
    if condition not in design["conditions"]:
        raise ValueError(f"{condition} is not in the design's conditions {design['conditions']}")
    if condition != "C0" and simulator not in design["simulators"].get(condition, []):
        raise ValueError(f"simulator {simulator!r} is not in the design's panel for {condition}: "
                         f"{design['simulators'].get(condition)}")


def check_simulator_name(condition: str, simulator: str) -> None:
    """C0 takes no simulator; every LLM condition names its panel simulator."""
    if condition == "C0":
        if simulator != NO_SIMULATOR:
            raise ValueError(f"C0 has no simulator: simulator must be {NO_SIMULATOR!r}, got {simulator!r}")
    elif simulator == NO_SIMULATOR or not (isinstance(simulator, str) and SIMULATOR_NAME.fullmatch(simulator)):
        raise ValueError(f"{condition} needs a simulator name (a simple identifier other than "
                         f"{NO_SIMULATOR!r}), got {simulator!r}")


def fingerprint_parts(task_args: dict) -> dict:
    return {
        "task_args": task_args,
        "scenarios": {s: scenario_path(s).read_text() for s in task_args["scenarios"]},
        "source": {p.name: p.read_text() for p in sorted(PACKAGE_DIR.glob("*.py"))},
        "lock": LOCKFILE.read_text() if LOCKFILE.exists() else None,
    }


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def fingerprint(task_args: dict) -> str:
    return _sha(fingerprint_parts(task_args))


@task
def simtest(
    condition: str,
    scenarios: list[str],
    max_turns: int = 20,
    epochs: int = 1,
    request_config: dict | None = None,
    simulator: str = NO_SIMULATOR,
    design: dict | None = None,
) -> Task:
    """`design`: the run's intended design (`make_design`), the same for every task of a
    run; None for tasks built outside `simtest run` and the demo. `simulator`: panel name of the simulator this task runs with ("none" for C0); the
    model itself is bound with `panel_task`. `request_config`: effective request settings
    of the agents and this task's simulator (see run.request_config); only fingerprinted,
    so a changed model setting refuses to resume."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    check_simulator_name(condition, simulator)
    check_design(condition, simulator, design)
    task_args = {"condition": condition, "scenarios": list(scenarios), "max_turns": max_turns, "epochs": epochs,
                 "request_config": request_config, "simulator": simulator, "design": design}
    parts = fingerprint_parts(task_args)
    fp = _sha(parts)
    loaded = load_scenarios(scenarios)
    return Task(
        dataset=[
            Sample(
                id=sc.id,
                input=sc.task_message(),
                metadata={"scenario": sc.id, "split": sc.split, "title": sc.title, "simulator": simulator},
            )
            for sc in loaded.values()
        ],
        solver=simtest_agent(condition=condition, scenarios=list(scenarios), max_turns=max_turns),
        scorer=simtest_scorer(scenarios=list(scenarios)),
        epochs=Epochs(epochs, []),  # no reducer: every episode keeps its own score
        version=fp,
        display_name=f"simtest-{condition}" + (f"-{simulator}" if simulator != NO_SIMULATOR else ""),
        metadata={
            "condition": condition,
            "simulator": simulator,
            "fingerprint": fp,
            "fingerprint_parts": {
                "scenarios": {k: _sha(v) for k, v in parts["scenarios"].items()},
                "source": {k: _sha(v) for k, v in parts["source"].items()},
                "lock": _sha(parts["lock"]),
            },
        },
    )


def panel_task(
    condition: str,
    scenarios: list[str],
    max_turns: int = 20,
    epochs: int = 1,
    simulator: str = NO_SIMULATOR,
    simulator_model: Model | None = None,
    request_config: dict | None = None,
    design: dict | None = None,
) -> Task:
    """A simtest task with its panel simulator bound as a Task-level model role.

    Task-level roles are merged under eval-level ones (eval-level wins), so callers
    must not pass eval-level `model_roles`. The role stays required in the solver
    (`get_model(role="simulator", required=True)`): an unbound role is a sample error,
    never the agent model."""
    if condition == "C0" and simulator_model is not None:
        raise ValueError("C0 has no simulator role: don't bind a simulator model to it")
    t = simtest(condition=condition, scenarios=scenarios, max_turns=max_turns, epochs=epochs,
                request_config=request_config, simulator=simulator, design=design)
    if simulator_model is not None:
        task_with(t, model_roles={"simulator": simulator_model})
    return t
