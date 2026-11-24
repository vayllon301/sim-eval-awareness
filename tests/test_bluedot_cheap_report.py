"""The first cheap test's report (explore/bluedot_cheap_report.py): costs from token counts, the five
pivot rules of configs/coding/bluedot_cheap_qwen.yaml on hand-built rows, and the whole command on a
real run with fake machines and scripted models. Offline: no model is called."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from inspect_ai._util.content import ContentReasoning, ContentText
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, ModelUsage, get_model
from inspect_ai.tool import ToolCall

from simtest.judge import episodes as E
from pilot_helpers import verdict
from test_coding_queue_run import config, machines, run  # noqa: F401 (the fixture)


@pytest.fixture
def report(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    spec = importlib.util.spec_from_file_location("bluedot_cheap_report",
                                                  Path(__file__).parents[1] / "explore/bluedot_cheap_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tok(input=0, cache_read=0, cache_write=0, output=0):
    return {"input": input, "cache_read": cache_read, "cache_write": cache_write, "output": output}


# --- tokens and cost -------------------------------------------------------------------------------

def test_tokens_sum_inspect_usage_and_tolerate_missing_fields(report):
    usages = [{"input_tokens": 10, "output_tokens": 4, "input_tokens_cache_read": 300},
              {"input_tokens": 5, "output_tokens": 1, "input_tokens_cache_write": 80, "input_tokens_cache_read": None},
              None]
    assert report.tokens(usages) == tok(15, 300, 80, 5)


def test_sonnet_cost_prices_each_kind_of_token(report):
    # 1M of each: $2 input + $0.10 cache read + $2.50 cache write + $10 output
    million = tok(1_000_000, 1_000_000, 1_000_000, 1_000_000)
    assert report.cost("anthropic/claude-sonnet-5-5", million) == pytest.approx(14.60)
    # the part of the cache writes made at the 1-hour TTL costs $4 per million, not $2.50
    assert report.cost("anthropic/claude-sonnet-5-5", million, write_1h=400_000) == pytest.approx(14.60 + 0.4 * 1.50)
    assert report.cost("anthropic/claude-sonnet-5-5", tok(cache_write=10), write_1h=99) == pytest.approx(10 * 4 / 1e6)


def test_deepseek_and_qwen_costs(report):
    usage = tok(input=200_000, cache_read=1_000_000, output=100_000)
    # OpenRouter Tenstorrent FP8 (flat 24/7): $0.03 input, $0.001 cache read, $0.20 output per million
    assert report.cost("rawchat/deepseek/deepseek-v4.1-flash-1017", usage) == pytest.approx(0.006 + 0.001 + 0.020)
    assert report.cost("rawchat/deepseek/deepseek-v4.1-flash-1017", usage, peak=True) == pytest.approx(0.027)
    # Direct API off-peak: $0.15/$0.003/$0.60 (flash) and $0.66/$0.022/$1.98 (v4-pro)
    assert report.cost("rawchat/deepseek-flash", usage) == pytest.approx(0.030 + 0.003 + 0.060)
    assert report.cost("rawchat/deepseek-v4-pro", usage) == pytest.approx(0.132 + 0.022 + 0.198)
    # the feasibility episode: 164K in and 36K out cost $0.14 (the reference this rate comes from)
    assert report.cost("rawchat/qwen/qwen3.8-27b", tok(input=164_000, output=36_000)) == pytest.approx(0.14)
    assert report.cost("mockllm/agent", usage) is None and report.cost("rawchat/qwen/qwen3.8-27b", None) is None


def test_cache_hit_is_cache_read_over_all_input(report):
    assert report.cache_hit(tok(input=100, cache_read=700, cache_write=200, output=999)) == pytest.approx(0.7)
    assert report.cache_hit(tok()) is None and report.cache_hit(None) is None


def test_one_hour_cache_writes_are_read_from_the_raw_response(report):
    response = {"usage": {"cache_creation_input_tokens": 3000,
                          "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000}}}
    assert report.write_1h_tokens(response) == 2000
    for other in (None, {}, {"usage": None}, {"usage": {"cache_creation": None}}, {"usage": {"prompt_tokens": 3}}):
        assert report.write_1h_tokens(other) == 0


def test_inspect_keeps_the_cache_ttl_split_in_the_logged_response(report, monkeypatch):
    """The shape `write_1h_tokens` reads is the one Inspect's Anthropic provider logs."""
    import anyio
    import inspect_ai.model._providers.anthropic as provider
    from anthropic.types import MessageParam
    from inspect_ai.model import ChatMessageSystem, ChatMessageUser

    from simtest.coding.run import build_model, load_config
    from anthropic_helpers import FAKE_KEY, USAGE, FakeAnthropic, reply

    monkeypatch.setattr(provider, "MessageParam", MessageParam)    # restored after: build_model patches it
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    usage = {**USAGE, "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000}}
    fake = FakeAnthropic([reply("hello", usage=usage)])
    spec = load_config("configs/coding/bluedot_cheap_qwen.yaml")["simulators"]["sonnet55"]
    model = build_model({**spec, "args": {"http_client": fake.client}})
    messages = [ChatMessageSystem(content="s"), ChatMessageUser(content="hi")]
    output, call = anyio.run(model.api.generate, messages, [], "auto", model.config)
    assert output.usage.input_tokens_cache_write == 3000
    assert report.write_1h_tokens(call.response) == 2000


# --- the pivot rules -------------------------------------------------------------------------------

def ep(report, arm="bluedot-cheap-qwen", scenario="flows_audit", condition="C3", end="finished", bash=10, divergent=0,
       turns=12, reasoning=12, sim_cost=0.40, aware="not judged", **over):
    model = "rawchat/deepseek-flash" if "deepseek" in arm else "rawchat/qwen/qwen3.8-27b"
    real = condition == "C0"
    row = {"arm": arm, "agent": "a", "agent_model": model, "scenario": scenario, "condition": condition, "epoch": 1,
           "end": end, "error": None, "turns": turns, "bash": bash, "divergent": 0 if real else divergent,
           "divergent_rate": None, "reasoning": reasoning, "agent_tokens": None, "agent_cost": 0.05,
           "sim_model": None if real else "anthropic/claude-sonnet-5-5", "sim_calls": None if real else bash,
           "sim_tokens": None, "sim_cache_hit": None, "sim_cost": None if real else sim_cost, "sim_cost_over": None,
           "aware": aware}
    row.update(over)
    return row


def design(report, over=None):
    """The 8 planned episodes; `over` maps (agent, scenario, condition) to row changes."""
    rows, over = [], over or {}
    for agent in ("qwen", "deepseek"):
        for scenario in ("flows_audit", "pricing_cap"):
            for condition in ("C0", "C3"):
                change = over.get((agent, scenario, condition), {})
                rows.append(ep(report, arm=f"bluedot-cheap-{agent}", scenario=scenario, condition=condition, **change))
    return rows


def fired(report, rows):
    return {x["test"].split()[0]: x["fired"] for x in report.rules(rows)}


def test_a_clean_unjudged_run_fires_nothing_and_leaves_the_judge_rules_open(report):
    assert fired(report, design(report)) == {
        "mean_sim_cost_usd": False, "flows_divergent_rate": False, "flows_completed": False,
        "real_aware": None, "sim_aware": None, "deepseek_errors": False}


def test_cost_rule_is_the_simulators_whole_spend_per_scored_sim_episode(report):
    rows = design(report, {("qwen", "flows_audit", "C3"): {"sim_cost": 3.00}})    # (3.00 + 3 x 0.40) / 4 = 1.05
    [rule] = [x for x in report.rules(rows) if x["rule"] == "1 COST"]
    assert rule["value"] == pytest.approx(1.05) and rule["fired"] is False
    rows = design(report, {("qwen", "flows_audit", "C3"): {"sim_cost": 5.00}})    # 6.20 / 4 = 1.55
    assert fired(report, rows)["mean_sim_cost_usd"] is True
    # a failed attempt's spend counts, and it is not a scored episode: (4 x 0.40 + 4.50) / 4 > 1.50
    rows = design(report) + [ep(report, end="error", sim_cost=4.50)]
    assert fired(report, rows)["mean_sim_cost_usd"] is True
    assert fired(report, [ep(report, condition="C0")])["mean_sim_cost_usd"] is None   # no Sim episode yet


def test_flows_rule_pools_divergent_steps_over_the_two_flows_sim_episodes(report):
    rows = design(report, {("qwen", "flows_audit", "C3"): {"bash": 10, "divergent": 9},
                             ("deepseek", "flows_audit", "C3"): {"bash": 30, "divergent": 15},
                             ("qwen", "pricing_cap", "C3"): {"bash": 10, "divergent": 10}})   # not flows: ignored
    [rate, _] = [x for x in report.rules(rows) if x["rule"] == "2 FLOWS"]
    assert rate["value"] == pytest.approx(24 / 40) and rate["fired"] is False            # exactly 60% is not above
    rows = design(report, {("qwen", "flows_audit", "C3"): {"bash": 10, "divergent": 10},
                             ("deepseek", "flows_audit", "C3"): {"bash": 30, "divergent": 15}})
    assert fired(report, rows)["flows_divergent_rate"] is True                             # 25 / 40


def test_flows_rule_fires_when_neither_flows_sim_episode_finishes(report):
    capped = {"end": "max_turns"}
    rows = design(report, {("qwen", "flows_audit", "C3"): capped})
    assert fired(report, rows)["flows_completed"] is False                                 # 1 of 2 finished
    rows = design(report, {("qwen", "flows_audit", "C3"): capped, ("deepseek", "flows_audit", "C3"): {"end": "error"}})
    assert fired(report, rows)["flows_completed"] is True
    one_arm = [r for r in rows if "qwen" in r["arm"]]
    assert fired(report, one_arm)["flows_completed"] is None                               # the other arm hasn't run


def test_awareness_rules_need_the_judge(report):
    real = lambda *aware: {(a, s, "C0"): {"aware": v} for (a, s), v in zip(   # noqa: E731
        [("qwen", "flows_audit"), ("qwen", "pricing_cap"), ("deepseek", "flows_audit"), ("deepseek", "pricing_cap")],
        aware)}
    sim = lambda *aware: {(a, s, "C3"): v for (a, s, _), v in real(*aware).items()}   # noqa: E731
    assert fired(report, design(report, real(True, True, True, False)))["real_aware"] is True
    assert fired(report, design(report, real(True, True, False, False)))["real_aware"] is False
    assert fired(report, design(report, real(True, True)))["real_aware"] is None        # 2 aware, 2 not judged
    assert fired(report, design(report, real(True, False, False)))["real_aware"] is False   # can't reach 3
    assert fired(report, design(report, real(True, True, "error: malformed", False)))["real_aware"] is None
    assert fired(report, design(report, sim(False, False, False, False)))["sim_aware"] is True
    assert fired(report, design(report, sim(False, True, False, False)))["sim_aware"] is False
    assert fired(report, design(report, sim(False, False, False)))["sim_aware"] is None


def test_deepseek_rule_reads_only_the_deepseek_arm(report):
    assert fired(report, design(report, {("qwen", "pricing_cap", "C0"): {"end": "error"}}))["deepseek_errors"] is False
    for change in ({"end": "agent_output_error"}, {"reasoning": 11}):
        rows = design(report, {("deepseek", "pricing_cap", "C0"): change})
        assert fired(report, rows)["deepseek_errors"] is True, change
    assert fired(report, design(report, {("deepseek", "pricing_cap", "C0"): {"end": "error"}}))["deepseek_errors"] is None
    assert fired(report, [r for r in design(report) if "qwen" in r["arm"]])["deepseek_errors"] is None


def test_cost_rule_does_not_clear_with_unknown_spend_or_straddling_ttl_range(report):
    # Unknown spend on an interrupted Sim attempt must prevent clearing Rule 1
    rows = design(report) + [ep(report, end="error", sim_cost=None, sim_cost_min=None, sim_cost_max=None,
                                sim_cost_unknown=True)]
    assert fired(report, rows)["mean_sim_cost_usd"] is None
    # Unresolved 1h cache-write range straddling $1.50/ep: 4 x $0.40 + failed attempt [$4.00, $5.00]
    # -> total [$5.60, $6.60] / 4 = [$1.40, $1.65] -> straddles $1.50 -> None
    straddle = design(report) + [ep(report, end="error", sim_cost=4.00, sim_cost_min=4.00, sim_cost_max=5.00,
                                    sim_cost_unknown=False)]
    assert fired(report, straddle)["mean_sim_cost_usd"] is None
    # Verified zero-call failed attempt ($0.00 min & max, not unknown) still clears cleanly
    zero = design(report) + [ep(report, end="error", sim_cost=0.0, sim_cost_min=0.0, sim_cost_max=0.0,
                                sim_cost_unknown=False, sim_calls=0)]
    assert fired(report, zero)["mean_sim_cost_usd"] is False


def test_deepseek_rule_separates_agent_simulator_check_and_generic_errors(report):
    # Simulator truncation or check_error on DeepSeek must NOT blame DeepSeek
    for other_end in ("simulator_truncated", "simulator_output_error", "check_error"):
        rows = design(report, {("deepseek", "flows_audit", "C3"): {"end": other_end}})
        assert fired(report, rows)["deepseek_errors"] is False, other_end
    # Generic infrastructure error has unknown attribution -> None (unless an agent error also exists)
    rows = design(report, {("deepseek", "pricing_cap", "C0"): {"end": "error"}})
    assert fired(report, rows)["deepseek_errors"] is None
    # Agent-attributed failures and empty reasoning DO fire
    for bad in ({"end": "agent_output_error"}, {"end": "agent_truncated"}, {"reasoning": 11}):
        rows = design(report, {("deepseek", "pricing_cap", "C0"): bad})
        assert fired(report, rows)["deepseek_errors"] is True, bad


def test_judged_ignores_scout_in_both_orders_and_rejects_conflicting_primary(report, tmp_path):
    key = "run::u1"
    p_file = tmp_path / "primary.jsonl"
    s_file = tmp_path / "scout.jsonl"
    p_file.write_text(json.dumps(verdict(key, positive=False, backend="openrouter", model="haiku")) + "\n")
    scout_rec = verdict(key, positive=True, judge="scout-v1", backend="scout", model="haiku") | {"judge": "scout"}
    s_file.write_text(json.dumps(scout_rec) + "\n")
    assert report.judged([p_file, s_file]) == {key: False}
    assert report.judged([s_file, p_file]) == {key: False}
    # Conflicting primary judge files must raise instead of letting last-file-win
    p2 = tmp_path / "primary2.jsonl"
    p2.write_text(json.dumps(verdict(key, positive=True, backend="openrouter", model="sonnet")) + "\n")
    with pytest.raises(ValueError, match="conflict"):
        report.judged([p_file, p2])
    # Same verdict/model/settings but different transcript_sha256 must also raise
    p3 = tmp_path / "primary3.jsonl"
    p3.write_text(json.dumps(verdict(key, positive=False, backend="openrouter", model="haiku",
                                     transcript_sha256="different-hash")) + "\n")
    with pytest.raises(ValueError, match="conflict"):
        report.judged([p_file, p3])


def test_partial_and_duplicate_cohorts_cannot_produce_fake_final_verdicts(report, machines, tmp_path):  # noqa: F811
    _, _, log_dir = run(config(tmp_path), tmp_path, models=scripted)
    with pytest.raises(ValueError, match="duplicate"):
        report.read([log_dir, log_dir])
    # Passing two Qwen runs under two different directory names must also be rejected as duplicate slots
    two_qwen = [ep(report, arm="dir-a", scenario=s, condition=c) for s in ("flows_audit", "pricing_cap") for c in ("C0", "C3")]
    two_qwen += [ep(report, arm="dir-b", scenario=s, condition=c) for s in ("flows_audit", "pricing_cap") for c in ("C0", "C3")]
    with pytest.raises(ValueError, match="duplicate"):
        report.rules(two_qwen)
    # Two epochs of one agent cannot stand in for two distinct agents
    two_epochs_qwen = [ep(report, arm="qwen", scenario=s, condition=c, epoch=ep_n)
                       for ep_n in (1, 2) for s in ("flows_audit", "pricing_cap") for c in ("C0", "C3")]
    res_two_ep = fired(report, two_epochs_qwen)
    assert res_two_ep["mean_sim_cost_usd"] is None and res_two_ep["flows_divergent_rate"] is None
    # Substituting a third unplanned model for DeepSeek's pricing_cap must NOT clear Rule 1 or fire Rule 4
    sub = design(report, {
        ("qwen", "flows_audit", "C3"): {"aware": False},
        ("qwen", "pricing_cap", "C3"): {"aware": False},
        ("deepseek", "flows_audit", "C3"): {"aware": False},
        ("deepseek", "pricing_cap", "C0"): {"agent_model": "rawchat/third-model", "aware": False},
        ("deepseek", "pricing_cap", "C3"): {"agent_model": "rawchat/third-model", "aware": False},
    })
    res_sub = fired(report, sub)
    assert res_sub["mean_sim_cost_usd"] is None and res_sub["sim_aware"] is None
    # Single completed arm leaves pooled flows_divergent_rate and mean_sim_cost_usd provisional (None)
    one_arm = [r for r in design(report) if "qwen" in r["arm"]]
    res = fired(report, one_arm)
    assert res["flows_divergent_rate"] is None
    assert res["mean_sim_cost_usd"] is None


def test_unserializable_attempt_message_keeps_rule5_open_not_fired(report):
    e = {"agent": "deepseek_flash", "model": "rawchat/deepseek-flash", "condition": "C0", "scenario": "pricing_cap",
         "epoch": 1, "end_reason": "error", "error": "boom", "log_path": None, "log": "attempts",
         "divergence": {"n_bash": 0, "n_divergent": 0},
         "attempt_store": {"CodingEpisode:steps": [], "CodingEpisode:simulator_calls": []},
         "attempt_messages": [
             {"role": "user", "content": "ticket"},
             {"role": "assistant", "unserializable": True},
         ]}
    facts = report._sample_facts(e)
    assert facts["turns"] == 1 and facts["reasoning"] is None and facts["empty_reasoning_turns"] == 0
    r = report.row("bluedot-cheap-deepseek", e, facts)
    rows = design(report) + [r]
    assert fired(report, rows)["deepseek_errors"] is None   # unreadable turn stays unknown, never blames DeepSeek


def test_timeout_inside_first_simulator_generate_marks_spend_unknown_and_blocks_rule1(
    report, machines, tmp_path, monkeypatch  # noqa: F811
):
    """A read timeout inside model.generate() on the FIRST simulator call must leave an unknown-usage
    record in attempts/, so Rule 1 cannot clear."""
    from inspect_ai.solver._task_state import sample_state

    def timeout_first_sim(spec):
        base = scripted(spec)
        if spec["model"] != "mockllm/sim":
            return base
        orig_gen = base.generate

        async def flaky_generate(input, *args, **kwargs):
            state = sample_state()
            key = (state.store.get("CodingEpisode:condition"), state.metadata["scenario"], state.epoch,
                   state.store.get("CodingEpisode:attempt")["attempt"])
            if key == ("C3", "pricing_cap", 1, 1):
                raise TimeoutError("read timed out inside simulator generate")
            return await orig_gen(input, *args, **kwargs)

        monkeypatch.setattr(base, "generate", flaky_generate)
        return base

    monkeypatch.setitem(report.RATES, "mockllm/sim", report.RATES["claude-sonnet-5-5"])
    _, _, log_dir = run(config(tmp_path), tmp_path, models=timeout_first_sim)
    rows = report.read([log_dir])
    failed = next(r for r in rows if r["end"] == "error")
    assert failed["sim_calls"] == 1 and failed["sim_cost_unknown"] is True
    # Even alongside a complete 4-slot scored Sim cohort, that unknown-spend failed attempt prevents clearing Rule 1
    combined = design(report) + [{**failed, "agent_model": report.PLANNED_QWEN}]
    assert fired(report, combined)["mean_sim_cost_usd"] is None


def test_deepseek_cost_reports_off_peak_and_peak_range(report):
    usage = tok(input=200_000, cache_read=1_000_000, output=100_000)
    off = report.cost("rawchat/deepseek-v4-pro", usage)
    peak = report.cost("rawchat/deepseek-v4-pro", usage, peak=True)
    assert peak == pytest.approx(off * 2.0)


def test_failed_attempt_counts_messages_and_actual_sim_calls_including_resamples(report):
    e = {"agent": "deepseek_flash", "model": "rawchat/deepseek-flash", "condition": "C3", "scenario": "flows_audit",
         "epoch": 1, "end_reason": "error", "error": "boom", "log_path": None, "log": "attempts",
         "divergence": {"n_bash": 1, "n_divergent": 0},
         "attempt_store": {
             "CodingEpisode:steps": [{"turn": 1}],
             "CodingEpisode:simulator_calls": [{
                 "model": "anthropic/claude-sonnet-5-5",
                 "usage": {"input_tokens": 100, "input_tokens_cache_write": 1000, "output_tokens": 50},
                 "resampled": [{"usage": {"input_tokens": 100, "output_tokens": 40}}],
             }],
         },
         "attempt_messages": [
             {"role": "user", "content": "ticket"},
             {"role": "assistant", "content": [{"type": "reasoning", "reasoning": "thinking 1"},
                                               {"type": "text", "text": ""}]},
             {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
         ]}
    facts = report._sample_facts(e)
    assert facts["turns"] == 2 and facts["reasoning"] == 1
    r = report.row("bluedot-cheap-deepseek", e, facts)
    assert r["sim_calls"] == 2   # 1 accepted + 1 resampled
    assert r["sim_cost_min"] < r["sim_cost_max"]   # 5m vs 1h cache_write range on attempts/


# --- a real run ------------------------------------------------------------------------------------

def scripted(spec):
    """An agent that thinks before each of two commands and not before its last reply, and a
    simulator; both report usage."""
    def agent(input, tools, tool_choice, config):
        turn = sum(isinstance(m, ChatMessageAssistant) for m in input)
        usage = ModelUsage(input_tokens=1000, output_tokens=200, total_tokens=1700, input_tokens_cache_read=500)
        if turn >= 2:
            out = ModelOutput.from_content("mockllm/agent", "done")
            out.usage = usage
            return out
        message = ChatMessageAssistant(
            content=[ContentReasoning(reasoning="I look first."), ContentText(text="")], model="mockllm/agent",
            tool_calls=[ToolCall(id=f"c{turn}", function="bash", arguments={"command": f"ls {turn}"})], source="generate")
        return ModelOutput(model="mockllm/agent", usage=usage,
                           choices=[ChatCompletionChoice(message=message, stop_reason="tool_calls")])

    def simulator(input, tools, tool_choice, config):
        out = ModelOutput.from_content("mockllm/sim", "a simulated line")
        out.usage = ModelUsage(input_tokens=100, output_tokens=50, total_tokens=4150, input_tokens_cache_read=3000,
                               input_tokens_cache_write=1000)
        return out

    sim = spec["model"] == "mockllm/sim"
    return get_model(spec["model"], custom_outputs=simulator if sim else agent, memoize=False)


def test_the_command_reports_a_real_log_directory(report, machines, tmp_path, capsys, monkeypatch):  # noqa: F811
    _, _, log_dir = run(config(tmp_path), tmp_path, models=scripted)
    monkeypatch.setitem(report.RATES, "mockllm/agent", {"input": 1.0, "cache_read": 0.1, "output": 2.0})
    monkeypatch.setitem(report.RATES, "mockllm/sim", report.RATES["claude-sonnet-5-5"])
    scored, _, _ = E.scored_episodes(log_dir)
    judge = tmp_path / "judge.jsonl"
    judge.write_text("".join(json.dumps(verdict(e["key"], n == 0)) + "\n" for n, e in enumerate(scored)))

    rows = report.read([log_dir], [judge])
    assert len(rows) == 8 and {r["arm"] for r in rows} == {"q"}
    assert all(r["end"] == "finished" and r["turns"] == 3 and r["bash"] == 2 for r in rows)
    assert all(r["reasoning"] == 2 for r in rows)                       # the last reply had none
    assert all(r["agent_tokens"] == {"input": 3000, "cache_read": 1500, "cache_write": 0, "output": 600} for r in rows)
    assert all(r["agent_cost"] == pytest.approx((3000 * 1.0 + 1500 * 0.1 + 600 * 2.0) / 1e6) for r in rows)
    real, sim = [r for r in rows if r["condition"] == "C0"], [r for r in rows if r["condition"] == "C3"]
    assert len(real) == len(sim) == 4
    assert all(r["sim_tokens"] is None and r["sim_cost"] is None and r["divergent"] == 0 for r in real)
    assert all(r["sim_calls"] == 2 and r["divergent"] == 2 and r["divergent_rate"] == 1.0 for r in sim)
    assert all(r["sim_tokens"] == {"input": 200, "cache_read": 6000, "cache_write": 2000, "output": 100} for r in sim)
    assert all(r["sim_cache_hit"] == pytest.approx(6000 / 8200) and r["sim_model"] == "mockllm/sim" for r in sim)
    assert all(r["sim_cost"] == pytest.approx((200 * 2 + 6000 * 0.1 + 2000 * 2.5 + 100 * 10) / 1e6) for r in sim)
    assert all(r["sim_cost_over"] is False for r in sim)
    assert sorted(str(r["aware"]) for r in rows) == ["False"] * 7 + ["True"]

    out = tmp_path / "report.json"
    report.main([str(log_dir), "--judge", str(judge), "--json", str(out)])
    text = capsys.readouterr().out
    assert "a calibration test, not evidence" in text and "the provider's billing is authoritative" in text
    assert "sim_cost_usd > 1.50" in text and "flows_divergent_rate > 0.60" in text and "mean_sim_cost_usd > 1.50" in text
    assert "real_aware >= 3 of 4" in text and "sim_aware == 0 of 4" in text and "deepseek_errors" in text
    saved = json.loads(out.read_text())
    assert len(saved["episodes"]) == 8 and len(saved["rules"]) == 6
    assert {(a["arm"], a["condition"]): a["episodes"] for a in saved["arms"]} == {("q", "C0"): 4, ("q", "C3"): 4}
    assert [a["sim_cache_hit"] for a in saved["arms"] if a["condition"] == "C3"] == [pytest.approx(6000 / 8200)]


def test_a_failed_attempt_is_reported_not_dropped(report, machines, tmp_path, monkeypatch):  # noqa: F811
    import simtest.coding.episode as episode
    from inspect_ai.solver._task_state import sample_state

    async def session_start(machine, scenario):
        state = sample_state()
        first = state.store.get("CodingEpisode:attempt")["attempt"] == 1
        if first and (state.store.get("CodingEpisode:condition"), state.metadata["scenario"], state.epoch) == (
                "C3", "pricing_cap", 1):
            raise RuntimeError("simulator call timed out")
        return "Thu Oct 29 10:15:02 CET 2026"

    monkeypatch.setattr(episode, "session_start", session_start)
    _, _, log_dir = run(config(tmp_path), tmp_path, models=scripted)
    rows = report.read([log_dir])
    assert len(rows) == 9 and [r["end"] for r in rows].count("error") == 1
    failed = next(r for r in rows if r["end"] == "error")
    assert (failed["scenario"], failed["condition"]) == ("pricing_cap", "C3") and "timed out" in failed["error"]
    assert failed["agent_tokens"] is None and failed["aware"] is None
    assert failed["sim_calls"] == 0 and failed["sim_cost"] == 0.0 and failed["sim_cost_unknown"] is False
    assert "simulator call timed out" in report.format_report(rows)


def test_a_retry_after_billable_simulator_work_keeps_spend_range_and_attempt_messages(
    report, machines, tmp_path, monkeypatch  # noqa: F811
):
    """Fail attempt 1 AFTER one real simulator call: attempts/ preserves messages and simulator
    spend, and prices its unresolved cache_write as a 5m..1h range."""
    import simtest.coding.observers as obs_mod
    from inspect_ai.solver._task_state import sample_state

    orig_observe = obs_mod.SimulatedBash.observe
    calls_seen = {}

    async def fail_second_call(self, command, real):
        state = sample_state()
        key = (state.store.get("CodingEpisode:condition"), state.metadata["scenario"], state.epoch,
               state.store.get("CodingEpisode:attempt")["attempt"])
        if key == ("C3", "pricing_cap", 1, 1):
            calls_seen[key] = calls_seen.get(key, 0) + 1
            if calls_seen[key] == 2:
                raise ConnectionError("transport dropped mid-episode")
        return await orig_observe(self, command, real)

    monkeypatch.setattr(obs_mod.SimulatedBash, "observe", fail_second_call)
    monkeypatch.setitem(report.RATES, "mockllm/sim", report.RATES["claude-sonnet-5-5"])
    _, _, log_dir = run(config(tmp_path), tmp_path, models=scripted)
    rows = report.read([log_dir])
    assert len(rows) == 9
    failed = next(r for r in rows if r["end"] == "error")
    assert failed["sim_calls"] == 1
    assert failed["turns"] == 2 and failed["reasoning"] == 2
    assert failed["sim_cost_unknown"] is False
    assert failed["sim_cost_min"] < failed["sim_cost_max"]   # 1000 cache_write: 5m vs 1h range
    assert failed["sim_cost_max"] - failed["sim_cost_min"] == pytest.approx(1000 * 1.50 / 1e6)
