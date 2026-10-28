"""The Inspect task for the coding family: one sample per scenario, one task per condition
and simulator, mirroring the office task (src/simtest/task.py).

C0 is one task with no simulator role; each LLM condition runs once per panel simulator,
bound at task level with `task_with` so eval_set needs no eval-level roles. The simulator
name is a task arg, so it is part of eval_set's identity, the fingerprint and the logs.
The fingerprint (scenario files incl. the snapshot and hidden tests, the coding package
source incl. prompts, the lockfile, the task args incl. request settings) is the task
`version`, which makes resuming refuse after any change.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from inspect_ai import Epochs, Task, task, task_with
from inspect_ai.dataset import Sample
from inspect_ai.model import Model

from ..task import NO_SIMULATOR
from .episode import coding_agent, loaded_scenarios
from .machine import MACHINES
from .observers import CONDITIONS
from .scenario import load_scenario
from .scorer import coding_scorer

CODING_DIR = Path(__file__).resolve().parent
LOCKFILE = CODING_DIR.parents[2] / "uv.lock"
SIMULATOR_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
CONDITIONS_SET = set(CONDITIONS)


def make_design(conditions, simulators_for, agents, scenarios, epochs) -> dict:
    return {"conditions": list(conditions),
            "simulators": {c: list(simulators_for[c]) for c in conditions if c != "C0"},
            "agents": dict(agents), "scenarios": list(scenarios), "epochs": epochs}


def check_simulator_name(condition: str, simulator: str) -> None:
    if condition == "C0":
        if simulator != NO_SIMULATOR:
            raise ValueError(f"C0 has no simulator: got {simulator!r}")
    elif simulator == NO_SIMULATOR or not SIMULATOR_NAME.fullmatch(str(simulator)):
        raise ValueError(f"{condition} needs a simulator name, got {simulator!r}")


def check_design(condition: str, simulator: str, design: dict | None) -> None:
    if design is None:
        return
    if condition not in design["conditions"]:
        raise ValueError(f"{condition} is not in the design's conditions {design['conditions']}")
    if condition != "C0" and simulator not in design["simulators"].get(condition, []):
        raise ValueError(f"simulator {simulator!r} is not in the design's panel for {condition}")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fingerprint_parts(task_args: dict) -> dict:
    files = {}
    for sid in task_args["scenarios"]:
        sc = load_scenario(sid)
        for f in sc.files():
            files[str(f.relative_to(CODING_DIR.parents[2]))] = f.read_text()
    return {"task_args": task_args, "scenario_files": files,
            "source": {p.name: p.read_text() for p in sorted(CODING_DIR.glob("*.py"))},
            "lock": LOCKFILE.read_text() if LOCKFILE.exists() else None}


def fingerprint(task_args: dict) -> str:
    return _sha(fingerprint_parts(task_args))


@task
def coding(condition: str, scenarios: list[str], max_turns: int = 40, epochs: int = 1, deadline: int = 60,
           machine: str = "docker", request_config: dict | None = None, simulator: str = NO_SIMULATOR,
           design: dict | None = None) -> Task:
    if condition not in CONDITIONS_SET:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    check_simulator_name(condition, simulator)
    check_design(condition, simulator, design)
    if machine not in MACHINES:
        raise ValueError(f"unknown machine {machine!r}; expected one of {sorted(MACHINES)}")
    task_args = {"condition": condition, "scenarios": list(scenarios), "max_turns": max_turns, "epochs": epochs,
                 "deadline": deadline, "machine": machine, "request_config": request_config,
                 "simulator": simulator, "design": design}
    parts = fingerprint_parts(task_args)
    fp = _sha(parts)
    loaded = loaded_scenarios(scenarios)
    return Task(
        dataset=[Sample(id=sc.id, input=sc.task_message(),
                        metadata={"scenario": sc.id, "split": sc.split, "variant": sc.variant,
                                  "title": sc.title, "simulator": simulator})
                 for sc in loaded.values()],
        solver=coding_agent(condition=condition, scenarios=list(scenarios), machine_factory=MACHINES[machine],
                            max_turns=max_turns, deadline=deadline),
        scorer=coding_scorer(scenarios=list(scenarios)),
        epochs=Epochs(epochs, []),
        version=fp,
        display_name=f"coding-{condition}" + (f"-{simulator}" if simulator != NO_SIMULATOR else ""),
        metadata={"family": "coding", "condition": condition, "simulator": simulator, "fingerprint": fp,
                  "fingerprint_parts": {"scenario_files": {k: _sha(v) for k, v in parts["scenario_files"].items()},
                                        "source": {k: _sha(v) for k, v in parts["source"].items()},
                                        "lock": _sha(parts["lock"])}},
    )


def panel_task(condition: str, scenarios: list[str], simulator: str = NO_SIMULATOR,
               simulator_model: Model | None = None, **kwargs) -> Task:
    if condition == "C0" and simulator_model is not None:
        raise ValueError("C0 has no simulator role")
    t = coding(condition=condition, scenarios=scenarios, simulator=simulator, **kwargs)
    if simulator_model is not None:
        task_with(t, model_roles={"simulator": simulator_model})
    return t
