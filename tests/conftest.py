"""Shared fixtures: run one episode through Inspect `eval()` with mockllm models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from inspect_ai import eval
from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import ModelOutput

from simtest.demo import EchoSimulator, ScriptedAgent, mock_model
from simtest.scenarios import scenario_path
from simtest.scorer import SCORER_NAME
from simtest.task import simtest


@dataclass
class Episode:
    log: EvalLog

    @property
    def sample(self) -> EvalSample:
        assert self.log.samples and len(self.log.samples) == 1
        return self.log.samples[0]

    @property
    def score(self):
        return (self.sample.scores or {}).get(SCORER_NAME)

    @property
    def end_reason(self) -> str:
        return self.score.metadata["end_reason"] if self.score else "error"

    @property
    def metrics(self) -> dict:
        return self.score.metadata["metrics"]

    @property
    def fidelity(self) -> dict:
        from simtest.audit import episode_fidelity  # computed at analysis time

        return episode_fidelity(self.steps)

    @property
    def store(self) -> dict:
        prefix = "Episode:"
        return {k[len(prefix) :]: v for k, v in self.sample.store.items() if k.startswith(prefix)}

    @property
    def steps(self) -> list[dict]:
        return self.store["steps"]

    @property
    def observations(self) -> list:
        return [s["observation"] for s in self.steps]


@pytest.fixture
def run(tmp_path):
    """run(agent, condition, scenario, sim) -> Episode.

    agent: a list of turns [(tool, args), ...] for a ScriptedAgent, a list of
    ModelOutput, or any custom_outputs callable. sim: a custom_outputs callable
    (default: a fresh EchoSimulator) or None for no simulator role. followup: the
    ScriptedAgent's reply to a follow-up question."""

    def _run(agent, condition="C0", scenario="report_q3", sim="echo", max_turns=20, followup=None,
             **eval_kwargs) -> Episode:
        if isinstance(agent, list) and agent and isinstance(agent[0], ModelOutput):
            outputs = agent
        elif isinstance(agent, list):
            outputs = ScriptedAgent(agent, followup=followup)
        else:
            outputs = agent
        roles = None
        if sim is not None:
            roles = {"simulator": mock_model(EchoSimulator() if sim == "echo" else sim, "mockllm/simulator")}
        logs = eval(
            simtest(condition=condition, scenarios=[str(scenario)], max_turns=max_turns,
                    simulator="none" if condition == "C0" else "mock"),
            model=mock_model(outputs, "mockllm/agent", strict_tools=False),
            model_roles=roles,
            limit=1,
            display="none",
            log_dir=str(tmp_path / "logs"),
            **eval_kwargs,
        )
        return Episode(logs[0])

    return _run


@pytest.fixture
def variant(tmp_path):
    """variant(scenario_id, mutate) -> path of a modified copy of the scenario YAML."""
    counter = iter(range(1000))

    def _variant(scenario_id: str, mutate) -> str:
        raw = yaml.safe_load(scenario_path(scenario_id).read_text())
        mutate(raw)
        path = Path(tmp_path) / f"{scenario_id}_{next(counter)}.yaml"
        path.write_text(yaml.safe_dump(raw, sort_keys=False))
        return str(path)

    return _variant
