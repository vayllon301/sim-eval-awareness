"""The agent loop solver, end to end with a fake machine (no Docker): phases, the observer,
flagging, the store, and the scorer. The behaviour probe is stubbed with canned raw facts
(the real probe is covered by the Docker tests); checks.evaluate is tested separately.
"""

from __future__ import annotations

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput

import simtest.coding.episode as episode
from simtest.coding import machine as machine_mod
from simtest.coding.scenario import load_scenario
from simtest.coding.scorer import SCORER_NAME
from simtest.coding.task import coding
from coding_helpers import FAKE_DATE, FakeMachine, ScriptedAgent, ScriptedSimulator, mock


@pytest.fixture
def patched(monkeypatch):
    """Register a fake machine and stub the baseline/probe. `state['raw']` sets the probe's
    canned facts; `state['run_outputs']` the fake machine's bash results."""
    state: dict = {"raw": {"base": "HEAD", "original": {"exit_code": 0, "output": "3 passed in 0.01s", "junit": (
                               '<testsuites><testsuite errors="0" failures="0" skipped="0" tests="3">'
                               '</testsuite></testsuites>')},
                           "suite": {"exit_code": 0}, "names": "", "diff": "", "docker_diff": "",
                           "atime_before": {}, "atime_after": {}, "reads": [], "code_files": {}, "errors": []},
                   "run_outputs": [], "machine": None}

    def factory(scenario, deadline):
        state["machine"] = FakeMachine(outputs=state["run_outputs"])
        return state["machine"]

    async def fake_baseline(machine, scenario):
        return {"base": "HEAD", "base_ok": True, "atime": {}}

    async def fake_probe(machine, scenario, base):
        return dict(state["raw"])

    monkeypatch.setitem(machine_mod.MACHINES, "fake", factory)
    monkeypatch.setattr(episode, "baseline", fake_baseline)
    monkeypatch.setattr(episode, "probe", fake_probe)
    return state


@pytest.fixture(autouse=True)
def reference(monkeypatch):
    """C2 needs a reference; these scenarios have none, and no test here reads it."""
    from simtest.coding.scenario import CodingScenario
    monkeypatch.setattr(CodingScenario, "reference", property(lambda self: [{"command": "ls", "output": "a"}]))


def run(agent, patched, condition="C0", scenario="pricing_cap", sim=None, tmp=None):
    roles = {"simulator": mock(sim, "mockllm/sim")} if sim is not None else None
    logs = eval(coding(condition=condition, scenarios=[scenario], max_turns=8, deadline=60, machine="fake",
                       simulator="none" if condition == "C0" else "mock"),
                model=mock(agent, "mockllm/agent"), model_roles=roles, limit=1, display="none",
                log_dir=str(tmp))
    s = logs[0].samples[0]
    return s, s.scores[SCORER_NAME]


def test_c0_episode_runs_and_scores(patched, tmp_path):
    # The episode ends like a real coding agent's turn: a reply without a tool call, whose
    # text is the summary for the ticket.
    patched["run_outputs"] = [("pytest output", 1)]
    agent = ScriptedAgent([[("bash", {"command": "pytest -q"})]], done="Fixed the cap; CI is green.")
    s, score = run(agent, patched, tmp=tmp_path)
    assert score.metadata["end_reason"] == episode.FINISHED == "no_tool_call" and not score.metadata["flagged"]
    steps = s.store["CodingEpisode:steps"]
    assert len(steps) == 1 and steps[0]["tool"] == "bash" and steps[0]["observation"] == "pytest output"
    assert patched["machine"].commands == ["pytest -q"]
    assert s.store["CodingEpisode:final_message"] == "Fixed the cap; CI is green."
    m = score.metadata["metrics"]
    assert m["finished"] and m["summary"] == "Fixed the cap; CI is green." and "submitted" not in m
    assert score.value["finished"] == 1.0 and score.answer == "Fixed the cap; CI is green."


def test_final_message_is_the_visible_text_not_the_reasoning(patched, tmp_path):
    from inspect_ai._util.content import ContentReasoning, ContentText
    from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant

    msg = ChatMessageAssistant(content=[ContentReasoning(reasoning="the test is stale, so I stop here"),
                                        ContentText(text="The test is stale; I asked #pricing-dev.")],
                               model="mockllm/agent", source="generate")
    out = ModelOutput(model="mockllm/agent", choices=[ChatCompletionChoice(message=msg, stop_reason="stop")])
    s, score = run([out], patched, tmp=tmp_path)
    assert s.store["CodingEpisode:final_message"] == "The test is stale; I asked #pricing-dev."
    assert score.metadata["metrics"]["summary"] == "The test is stale; I asked #pricing-dev."


def test_submit_is_an_unknown_tool(patched, tmp_path):
    # A model that hallucinates a submit tool gets the unknown-tool error and keeps going;
    # only a reply without a tool call ends the episode.
    agent = ScriptedAgent([[("submit", {"summary": "done"})]], done="All done.")
    s, score = run(agent, patched, tmp=tmp_path)
    steps = s.store["CodingEpisode:steps"]
    assert steps[0]["tool"] == "unknown" and steps[0]["tool_raw"] == "submit" and not steps[0]["executed"]
    assert steps[0]["observation"] == "Error: unknown tool 'submit'"
    assert score.metadata["end_reason"] == episode.FINISHED
    assert score.metadata["metrics"]["summary"] == "All done."


def test_turn_cap_is_not_a_finish(patched, tmp_path):
    agent = ScriptedAgent([[("bash", {"command": f"echo {i}"})] for i in range(20)])
    s, score = run(agent, patched, tmp=tmp_path)
    assert score.metadata["end_reason"] == "max_turns" and not score.metadata["flagged"]
    m = score.metadata["metrics"]
    assert m["finished"] is False and m["summary"] is None and score.value["finished"] == 0.0
    assert s.store["CodingEpisode:final_message"] is None


def test_violation_is_scored(patched, tmp_path):
    patched["raw"]["names"] = "M\ttests/test_discounts.py\n"
    patched["raw"]["docker_diff"] = "C /home/dev/pricing-service/tests/test_discounts.py"
    agent = ScriptedAgent([[("bash", {"command": "sed -i ... tests/"})]], done="x")
    s, score = run(agent, patched, tmp=tmp_path)
    assert score.metadata["metrics"]["violation"] and score.value["protected_modified"] == 1.0


def test_send_message_is_recorded_not_sent(patched, tmp_path):
    agent = ScriptedAgent([[("send_message", {"channel": "#pricing-dev", "text": "hi"})]], done="asked")
    s, score = run(agent, patched, tmp=tmp_path)
    steps = s.store["CodingEpisode:steps"]
    assert steps[0]["observation"] == episode.MESSAGE_SENT
    assert score.metadata["metrics"]["asked_owner"]


def test_the_owner_channel_replies_once_on_the_first_post(patched, tmp_path):
    # flows_audit scripts one reply for #appsec: given on the first post there, stamped with
    # the box's time; a later post and another channel get only MESSAGE_SENT.
    agent = ScriptedAgent([[("send_message", {"channel": "#flows-dev", "text": "anyone?"})],
                           [("send_message", {"channel": "appsec", "text": "may I change audit.ts?"})],
                           [("send_message", {"channel": "#appsec", "text": "ping"})]], done="asked")
    s, score = run(agent, patched, scenario="flows_audit", tmp=tmp_path)
    steps = s.store["CodingEpisode:steps"]
    text = load_scenario("flows_audit").chat["#appsec"]
    assert [st["observation"] for st in steps] == [
        episode.MESSAGE_SENT, f"{episode.MESSAGE_SENT}\n[#appsec 10:15] {text}", episode.MESSAGE_SENT]
    assert steps[1]["reply"] == {"channel": "#appsec", "time": "10:15", "text": text}
    assert "reply" not in steps[0] and "reply" not in steps[2]
    assert "exec:date +%H:%M" in patched["machine"].events      # harness work, not an agent command
    assert patched["machine"].commands == []
    assert score.metadata["metrics"]["channels"] == ["#flows-dev", "appsec", "#appsec"]


def test_a_scenario_without_chat_never_replies(patched, tmp_path):
    agent = ScriptedAgent([[("send_message", {"channel": "#pricing-dev", "text": "hi"})]], done="asked")
    s, _ = run(agent, patched, tmp=tmp_path)
    assert s.store["CodingEpisode:steps"][0]["observation"] == episode.MESSAGE_SENT
    assert "exec:date +%H:%M" not in patched["machine"].events


def test_the_reply_is_the_same_in_every_condition(patched, tmp_path):
    seen = {}
    for condition in ("C0", "C1", "C2", "C3"):
        agent = ScriptedAgent([[("bash", {"command": "ls"})],
                               [("send_message", {"channel": "#appsec", "text": "question"})]], done="asked")
        s, _ = run(agent, patched, condition=condition, scenario="flows_audit",
                   sim=None if condition == "C0" else ScriptedSimulator(["sim output"]), tmp=tmp_path / condition)
        steps = s.store["CodingEpisode:steps"]
        seen[condition] = (steps[1]["observation"], steps[1]["reply"])
        if condition != "C0":                               # the simulator only ever saw bash
            calls = s.store["CodingEpisode:simulator_calls"]
            assert len(calls) == 1 and "question" not in str(calls)
    assert len(set(map(str, seen.values()))) == 1


def test_no_tool_call_ends_episode(patched, tmp_path):
    agent = [ModelOutput.from_content("mockllm/agent", "I think the fix is clear.")]
    s, score = run(agent, patched, tmp=tmp_path)
    assert score.metadata["end_reason"] == "no_tool_call" and not score.metadata["flagged"]
    assert score.metadata["metrics"]["finished"] and score.metadata["metrics"]["summary"] == "I think the fix is clear."


def test_truncated_output_is_flagged(patched, tmp_path):
    out = ModelOutput.from_content("mockllm/agent", "partial")
    out.choices[0].stop_reason = "max_tokens"
    s, score = run([out], patched, tmp=tmp_path)
    assert score.metadata["end_reason"] == "agent_truncated" and score.metadata["flagged"]
    import math
    assert all(math.isnan(v) for v in score.value.values())


def test_c3_uses_the_simulator_not_the_real_output(patched, tmp_path):
    patched["run_outputs"] = [("REAL pytest", 1)]
    agent = ScriptedAgent([[("bash", {"command": "pytest -q"})]], done="x")
    sim = ScriptedSimulator(["<think>t</think>\nINVENTED"])
    s, score = run(agent, patched, condition="C3", sim=sim, tmp=tmp_path)
    steps = s.store["CodingEpisode:steps"]
    assert steps[0]["observation"] == "INVENTED"           # the agent saw the simulator's output
    assert steps[0]["result"]["output"] == "REAL pytest"   # the real output is still recorded (shadow)
    assert s.store["CodingEpisode:simulator_calls"][0]["raw"] == "<think>t</think>\nINVENTED"


class Recording(ScriptedSimulator):
    """A scripted simulator that keeps every input it was given."""

    def __init__(self, outputs):
        super().__init__(outputs)
        self.inputs = []

    def __call__(self, input, tools, tool_choice, config):
        self.inputs.append(input)
        return super().__call__(input, tools, tool_choice, config)


@pytest.mark.parametrize("condition", ["C0", "C1", "C2", "C3"])
def test_session_start_is_read_on_the_box_before_the_first_turn(patched, tmp_path, condition):
    # Harness work in every condition: `date` on the real box after it starts and before the
    # agent's first command, recorded in the store. Only C2 and C3 show it to the simulator,
    # in the snapshot's facts block.
    patched["run_outputs"] = [("REAL", 0)]
    agent = ScriptedAgent([[("bash", {"command": "ls"})]], done="x")
    sim = Recording(["OUT"]) if condition != "C0" else None
    s, score = run(agent, patched, condition=condition, sim=sim, tmp=tmp_path)
    assert s.store["CodingEpisode:session_start"] == FAKE_DATE
    events = patched["machine"].events
    assert events.index("start") < events.index("exec:date") < events.index("run:ls")
    assert patched["machine"].commands == ["ls"]  # never an agent command
    line = f"- Date and time when the session started: {FAKE_DATE}"
    if sim is not None:
        system = sim.inputs[0][0].text
        assert (line in system) == (condition in ("C2", "C3"))
        if condition in ("C2", "C3"):
            assert system.index("- Time zone: ") < system.index(line) < system.index("\n\nRepository ")


def test_unreadable_date_is_a_machine_error(patched, tmp_path, monkeypatch):
    def factory(scenario, deadline):
        patched["machine"] = FakeMachine(date="")
        return patched["machine"]

    monkeypatch.setitem(machine_mod.MACHINES, "fake", factory)
    logs = eval(coding(condition="C0", scenarios=["pricing_cap"], max_turns=2, machine="fake"),
                model=mock(ScriptedAgent([], done="x"), "mockllm/agent"), limit=1, display="none",
                log_dir=str(tmp_path))
    s = logs[0].samples[0]
    assert s.error is not None and "date" in s.error.message


def test_simulator_truncation_flags_the_episode(patched, tmp_path):
    patched["run_outputs"] = [("REAL", 0)]

    class Trunc:
        name = "mockllm/sim"

        def __call__(self, input, tools, tool_choice, config):
            out = ModelOutput.from_content(self.name, "half")
            out.choices[0].stop_reason = "max_tokens"
            return out

    agent = ScriptedAgent([[("bash", {"command": "ls"})]], done="x")
    s, score = run(agent, patched, condition="C3", sim=Trunc(), tmp=tmp_path)
    assert score.metadata["end_reason"] == "simulator_truncated" and score.metadata["flagged"]
    # the raw simulator output is still recorded
    assert s.store["CodingEpisode:simulator_calls"][0]["raw"] == "half"


def test_interrupted_attempt_is_kept_and_rerun(monkeypatch, tmp_path):
    """A1: a machine error on the first attempt is kept as an incomplete episode, and the
    task is re-run from the start; load_run reports both the interrupted and the completed
    episode (never silently replacing the interrupted one)."""
    from inspect_ai import eval_set

    from simtest.coding.analysis import load_run
    from simtest.coding.machine import MachineError

    state = {"starts": 0}

    class Flaky(FakeMachine):
        async def start(self):
            state["starts"] += 1
            if state["starts"] == 1:
                raise MachineError("ssh tunnel dropped")
            await super().start()

    async def fake_baseline(machine, scenario):
        return {"base": "HEAD", "base_ok": True, "atime": {}}

    async def fake_probe(machine, scenario, base):
        return {"base": "HEAD", "original": {"exit_code": 0}, "suite": {"exit_code": 0},
                "names": "", "diff": "", "docker_diff": "", "atime_before": {}, "atime_after": {},
                "reads": [], "files": {}, "errors": []}

    monkeypatch.setitem(machine_mod.MACHINES, "flaky", lambda scenario, deadline: Flaky())
    monkeypatch.setattr(episode, "baseline", fake_baseline)
    monkeypatch.setattr(episode, "probe", fake_probe)

    agent = mock(ScriptedAgent([], done="ok"), "mockllm/agent")
    task = coding(condition="C0", scenarios=["pricing_cap"], max_turns=4, machine="flaky")
    eval_set([task], model=agent, log_dir=str(tmp_path), retry_on_error=0, retry_attempts=3,
             retry_cleanup=False, display="none")

    episodes, _ = load_run(tmp_path)
    errored = [e for e in episodes if e["end_reason"] == "error"]
    completed = [e for e in episodes if e["end_reason"] == episode.FINISHED]
    assert len(errored) == 1, [e["end_reason"] for e in episodes]
    assert len(completed) == 1


def test_metrics_cut_by_a_recovered_log_are_reported_not_scored(patched, tmp_path):
    """A log Inspect rebuilt after a crash keeps samples whose score metadata it cut to
    "Key removed from summary (> 1k)"; a resumed eval_set reuses them. load_run reports such
    an episode as not scored instead of passing a string on as its metrics."""
    from inspect_ai.log import list_eval_logs, read_eval_log, write_eval_log

    from simtest.coding.analysis import format_table, load_run, summarize

    agent = ScriptedAgent([], done="Done.")
    run(agent, patched, tmp=tmp_path)
    path = list_eval_logs(str(tmp_path))[0]
    log = read_eval_log(path)
    log.samples[0].scores[SCORER_NAME].metadata["metrics"] = "Key removed from summary (> 1k)"
    write_eval_log(log, path)
    episodes, _ = load_run(tmp_path)
    assert [(e["end_reason"], e["metrics"]) for e in episodes] == [("error", {})]
    assert "metrics" in episodes[0]["error"]
    format_table(summarize(episodes))


def test_one_episodes_error_does_not_cancel_the_others(monkeypatch, tmp_path):
    """Session 5 (LOG.md, 01-11-2026): one sample's infrastructure error (a simulator call
    timing out) cancelled the 11 other episodes of its task after 75 minutes. With the run's
    eval_set options the others run to their end, and only the errored one is re-run."""
    import anyio
    from inspect_ai import eval_set

    from simtest.coding.analysis import load_run
    from simtest.coding.machine import MachineError
    from simtest.coding.run import EVAL_SET_OPTIONS

    starts = {"pricing_cap": 0, "pricing_cap_benign": 0}
    other_started = anyio.Event()

    class Flaky(FakeMachine):
        def __init__(self, sid):
            super().__init__()
            self.sid = sid

        async def start(self):
            starts[self.sid] += 1
            if self.sid == "pricing_cap_benign":
                other_started.set()
            elif starts[self.sid] == 1:
                await other_started.wait()
                raise MachineError("simulator call timed out")
            await super().start()

    async def fake_baseline(machine, scenario):
        return {"base": "HEAD", "base_ok": True, "atime": {}}

    async def fake_probe(machine, scenario, base):
        if machine.sid == "pricing_cap_benign":
            await anyio.sleep(1)   # still running when the other episode fails
        return {"base": "HEAD", "original": {"exit_code": 0}, "suite": {"exit_code": 0},
                "names": "", "diff": "", "docker_diff": "", "atime_before": {}, "atime_after": {},
                "reads": [], "files": {}, "errors": []}

    monkeypatch.setitem(machine_mod.MACHINES, "flaky2", lambda scenario, deadline: Flaky(scenario.id))
    monkeypatch.setattr(episode, "baseline", fake_baseline)
    monkeypatch.setattr(episode, "probe", fake_probe)

    agent = mock(ScriptedAgent([], done="ok"), "mockllm/agent")
    task = coding(condition="C0", scenarios=["pricing_cap", "pricing_cap_benign"], max_turns=4, machine="flaky2")
    # In place, as simtest run does (outside a run queue the retry just starts again).
    eval_set([task], model=agent, log_dir=str(tmp_path), retry_on_error=3, display="none", **EVAL_SET_OPTIONS)

    assert starts == {"pricing_cap": 2, "pricing_cap_benign": 1}   # the other episode ran once, to its end
    episodes, _ = load_run(tmp_path)
    # Without simtest run's queue nothing keeps the failed attempt; under it, attempts/ does
    # (tests/test_coding_queue_run.py).
    assert sorted(e["end_reason"] for e in episodes) == [episode.FINISHED, episode.FINISHED]


def test_a_simulated_command_line_is_resampled_and_kept_in_the_record(patched, tmp_path):
    """Session 5 (LOG.md, 01-11-2026): the simulator sometimes takes the agent's role and answers
    with the command itself. Such a reply is resampled; the rejected one stays in the record."""
    patched["run_outputs"] = [("REAL", 0)]
    command = "cd /home/dev/repo && grep -n foo tests/test_api.py | head -80"
    agent = ScriptedAgent([[("bash", {"command": command})]], done="x")
    sim = ScriptedSimulator(["grep -n foo tests/test_api.py | head -80", "12:    foo = 1"])
    s, _ = run(agent, patched, condition="C3", sim=sim, tmp=tmp_path)
    step = s.store["CodingEpisode:steps"][0]
    call = s.store["CodingEpisode:simulator_calls"][0]
    assert step["observation"] == "12:    foo = 1"
    assert [r["reason"] for r in call["resampled"]] == ["echo"]
    assert call["resampled"][0]["raw"] == "grep -n foo tests/test_api.py | head -80"
    assert "role_confused" not in call and len(s.store["CodingEpisode:simulator_calls"]) == 1
    assert "simulator:resampled_command_line" in s.scores[SCORER_NAME].metadata["notes"]


def test_a_command_line_still_there_after_the_resamples_is_kept_and_noted(patched, tmp_path):
    patched["run_outputs"] = [("REAL", 0)]
    agent = ScriptedAgent([[("bash", {"command": "ls -la"})]], done="x")
    sim = ScriptedSimulator(["cat /home/dev/repo/.env"])         # every sample: an invented command
    s, _ = run(agent, patched, condition="C3", sim=sim, tmp=tmp_path)
    call = s.store["CodingEpisode:simulator_calls"][0]
    assert s.store["CodingEpisode:steps"][0]["observation"] == "cat /home/dev/repo/.env"
    assert len(call["resampled"]) == 2 and call["role_confused"] == "invented"
    assert "simulator:command_line_kept" in s.scores[SCORER_NAME].metadata["notes"]


def test_a_cancelled_episode_still_removes_its_machine():
    """Session 5 (LOG.md, 01-11-2026): episodes cancelled by a sibling's error left their
    containers running (62 of them), because the cleanup's await was cancelled too."""
    import anyio

    class Slow:
        stopped = False

        async def stop(self):
            await anyio.sleep(0.05)
            self.stopped = True

    async def main(m):
        with anyio.CancelScope() as scope:
            scope.cancel()
            await episode.stop_machine(m)

    m = Slow()
    anyio.run(main, m)
    assert m.stopped


class Sequenced:
    """A simulator answering from a list of ModelOutputs or exceptions, one per call."""

    def __init__(self, items, name="mockllm/sim"):
        self.items, self.name, self.i, self.caches = list(items), name, 0, []

    def __call__(self, input, tools, tool_choice, config):
        item = self.items[min(self.i, len(self.items) - 1)]
        self.i += 1
        if isinstance(item, BaseException):
            raise item
        return item


def test_a_failed_resample_records_the_call_once_with_what_was_rejected(patched, tmp_path):
    # Sol S3: a truncated resample used to be appended twice (one step, two records).
    patched["run_outputs"] = [("REAL", 0)]
    command = "grep -n foo tests/test_api.py"
    agent = ScriptedAgent([[("bash", {"command": command})]], done="x")
    cut = ModelOutput.from_content("mockllm/sim", "12: fo", stop_reason="max_tokens")
    sim = Sequenced([ModelOutput.from_content("mockllm/sim", command), cut])
    s, _ = run(agent, patched, condition="C3", sim=sim, tmp=tmp_path)
    calls = s.store["CodingEpisode:simulator_calls"]
    assert len(calls) == 1 and [r["reason"] for r in calls[0]["resampled"]] == ["echo"]


def test_an_infrastructure_error_on_a_resample_keeps_what_was_rejected():
    # Sol S4: a transport error after a rejected sample lost the rejected sample.
    import anyio
    import httpx

    from inspect_ai.model import get_model
    from simtest.coding.machine import ExecResult
    from simtest.coding.observers import SimulatedBash

    command = "grep -n foo tests/test_api.py"
    sim = mock(Sequenced([ModelOutput.from_content("mockllm/sim", command),
                          httpx.ConnectError("tunnel dropped")]), "mockllm/sim")
    obs = SimulatedBash("C3", sim, "TOOL", "SNAP")
    with pytest.raises(Exception) as err:
        anyio.run(obs.observe, command, ExecResult("REAL", 0, False, 0.1))
    assert not isinstance(err.value, TypeError)
    assert len(obs.records) == 1 and [r["reason"] for r in obs.records[0]["resampled"]] == ["echo"]


def test_a_resample_bypasses_the_response_cache(monkeypatch):
    # Sol S5: with caching on, an identical resample request would replay the rejected reply.
    import anyio

    from simtest.coding.machine import ExecResult
    from simtest.coding.observers import SimulatedBash

    calls = []

    class Model:
        def __str__(self):
            return "m"

        async def generate(self, messages, **kw):
            calls.append(kw)
            text = "grep -n foo x.py" if len(calls) == 1 else "1: foo"
            return ModelOutput.from_content("m", text)

    obs = SimulatedBash("C3", Model(), "TOOL", "SNAP")
    text, record = anyio.run(obs.observe, "grep -n foo x.py", ExecResult("REAL", 0, False, 0.1))
    assert text == "1: foo" and calls == [{}, {"cache": False}]


def test_a_cancelled_resample_saves_what_was_rejected(patched, tmp_path):
    # Sol round 2: a time limit hit during a resample cancelled the solver past every handler,
    # so the rejected sample was never saved.
    import asyncio

    patched["run_outputs"] = [("REAL", 0)]
    command = "grep -n foo tests/test_api.py"
    agent = ScriptedAgent([[("bash", {"command": command})]], done="x")
    sim = Sequenced([ModelOutput.from_content("mockllm/sim", command), asyncio.CancelledError()])
    logs = eval(coding(condition="C3", scenarios=["pricing_cap"], max_turns=8, deadline=60, machine="fake",
                       simulator="mock"),
                model=mock(agent, "mockllm/agent"), model_roles={"simulator": mock(sim, "mockllm/sim")},
                limit=1, display="none", log_dir=str(tmp_path))
    s = logs[0].samples[0]
    calls = s.store["CodingEpisode:simulator_calls"]
    assert len(calls) == 1 and [r["reason"] for r in calls[0]["resampled"]] == ["echo"]


# -- C2 in the flows world (found by tests/test_pilot_c2_docker.py) ------------------------

class EchoReal:
    """A grounded simulator that returns the real output it was given, after a reasoning block
    and the blank lines a server leaves after it (parse drops them)."""

    def __call__(self, input, tools, tool_choice, config):
        import re

        real = re.search(r"<real_output>\n(.*)\n</real_output>\n", input[-1].text, re.S).group(1)
        return ModelOutput.from_content("mockllm/sim", "<think>t</think>\n\n" + real)


NPM = "\n> flows-platform@1.0.0 build\n> tsc\n\n"    # npm prints a blank line before its banner
NODE = ("node /home/dev/flows-platform/dist/api/server.js\nnode /home/dev/flows-platform/dist/worker/index.js\n"
        "node /home/dev/flows-platform/dist/services/index.js\n")


def test_a_faithful_c2_rewrite_of_an_output_that_starts_with_a_blank_line_is_not_divergent(patched, tmp_path):
    # A simulator's reply loses its leading newlines in parsing, so C0 showed npm's blank line,
    # which C1-C3 never could, and a faithful C2 rewrite counted as a divergent step.
    from simtest.coding.analysis import step_divergence

    patched["run_outputs"] = [(NPM, 0)]
    agent = ScriptedAgent([[("bash", {"command": "npm run build"})]], done="x")
    c2, _ = run(agent, patched, condition="C1", sim=EchoReal(), tmp=tmp_path / "c2")
    c0, _ = run(agent, patched, condition="C0", tmp=tmp_path / "c0")
    steps = c2.store["CodingEpisode:steps"]
    assert step_divergence(steps) == {"n_bash": 1, "n_divergent": 0}
    assert steps[0]["observation"] == c0.store["CodingEpisode:steps"][0]["observation"] == \
        "> flows-platform@1.0.0 build\n> tsc"


def test_a_c2_reply_that_is_the_real_output_is_never_a_command_line(patched, tmp_path):
    # `ps -C node -o args=` prints three command lines. A C2 simulator that returned them as
    # printed was taken for role confusion: resampled twice, then kept and noted.
    patched["run_outputs"] = [(NODE, 0)]
    agent = ScriptedAgent([[("bash", {"command": "ps -C node -o args="})]], done="x")
    s, score = run(agent, patched, condition="C1", sim=EchoReal(), tmp=tmp_path)
    call = s.store["CodingEpisode:simulator_calls"][0]
    assert "resampled" not in call and "role_confused" not in call
    assert s.store["CodingEpisode:steps"][0]["observation"] == NODE.rstrip()
    assert score.metadata["notes"] == []


def test_a_c2_reply_with_the_command_instead_of_the_real_output_is_still_resampled(patched, tmp_path):
    patched["run_outputs"] = [("12:    foo = 1\n", 0)]
    command = "grep -n foo tests/test_api.py"
    agent = ScriptedAgent([[("bash", {"command": command})]], done="x")
    s, score = run(agent, patched, condition="C1", sim=ScriptedSimulator([command, "12:    foo = 1"]), tmp=tmp_path)
    call = s.store["CodingEpisode:simulator_calls"][0]
    assert [r["reason"] for r in call["resampled"]] == ["dropped"] and "role_confused" not in call
    assert s.store["CodingEpisode:steps"][0]["observation"] == "12:    foo = 1"


def test_a_c2_reply_that_stays_wrong_gives_the_agent_the_real_output_and_a_note(patched, tmp_path):
    patched["run_outputs"] = [("12:    foo = 1\n", 0)]
    agent = ScriptedAgent([[("bash", {"command": "grep -n foo tests/test_api.py"})]], done="x")
    s, score = run(agent, patched, condition="C1", sim=ScriptedSimulator(["Let me check the file:"]), tmp=tmp_path)
    call = s.store["CodingEpisode:simulator_calls"][0]
    assert call["real_fallback"] == "dropped" and len(call["resampled"]) == 2
    assert s.store["CodingEpisode:steps"][0]["observation"] == "12:    foo = 1"
    assert "simulator:real_output_fallback" in s.scores[SCORER_NAME].metadata["notes"]


def test_c3_never_compares_with_the_real_output(patched, tmp_path):
    # C3 stays a free simulation: the same reply, equal to the real output, is still checked by
    # rule alone (and this one looks like command lines; `ps -o args` is exempt since council round 1,
    # a script printing them is not).
    patched["run_outputs"] = [(NODE, 0)]
    agent = ScriptedAgent([[("bash", {"command": "node scripts/list-workers.js"})]], done="x")
    s, _ = run(agent, patched, condition="C3", sim=ScriptedSimulator([NODE]), tmp=tmp_path)
    call = s.store["CodingEpisode:simulator_calls"][0]
    assert len(call["resampled"]) == 2 and call["role_confused"] == "invented"

