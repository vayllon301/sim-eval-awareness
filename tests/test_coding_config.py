"""Coding config validation: good configs load, bad ones fail at load time."""

from __future__ import annotations

import pytest
import yaml

from simtest.coding.run import ConfigError, load_config, request_config


def write(tmp_path, data):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


BASE = {
    "family": "coding", "name": "t", "scenarios": ["pricing_cap"], "conditions": ["C0"],
    "agents": {"a": {"model": "rawchat/agent", "base_url": "http://x/v1",
                     "config": {"temperature": 1.0, "top_k": 20}}},
}


def test_shipped_configs_load():
    for name in ("dev_c0", "dev_c0_c3"):
        cfg = load_config(f"configs/coding/{name}.yaml")
        assert cfg["conditions"]


def test_minimal_c0_config_loads(tmp_path):
    cfg = load_config(write(tmp_path, BASE))
    assert cfg["agents"]["a"]["model"] == "rawchat/agent"
    assert cfg["agents"]["a"]["args"]["base_url"] == "http://x/v1"


def test_agent_may_not_set_reasoning_effort(tmp_path):
    data = {**BASE, "agents": {"a": {"model": "rawchat/agent", "base_url": "http://x/v1",
                                     "config": {"reasoning_effort": "high"}}}}
    with pytest.raises(ConfigError, match="reasoning_effort|xhigh"):
        load_config(write(tmp_path, data))


def test_c3_needs_a_simulator(tmp_path):
    data = {**BASE, "conditions": ["C0", "C3"]}
    with pytest.raises(ConfigError, match="simulators"):
        load_config(write(tmp_path, data))


def test_c0_only_rejects_a_simulator(tmp_path):
    data = {**BASE, "simulators": {"s": {"model": "rawchat/sim", "base_url": "http://y/v1"}}}
    with pytest.raises(ConfigError, match="every condition is C0"):
        load_config(write(tmp_path, data))


def test_unknown_condition_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown conditions"):
        load_config(write(tmp_path, {**BASE, "conditions": ["C0", "C9"]}))


def test_bad_model_prefix_rejected(tmp_path):
    data = {**BASE, "agents": {"a": {"model": "openrouter/x", "base_url": "http://x/v1"}}}
    with pytest.raises(ConfigError, match="rawchat"):
        load_config(write(tmp_path, data))


def test_request_config_hides_the_api_key_env(tmp_path):
    data = {**BASE, "agents": {"a": {"model": "rawchat/agent", "base_url": "http://x/v1",
                                     "api_key_env": "OPENROUTER_API_KEY"}}}
    cfg = load_config(write(tmp_path, data))
    rc = request_config(cfg)
    assert "api_key_env" not in rc["agents"]["a"]["args"]
    assert rc["simulator"] is None


def test_family_must_be_coding(tmp_path):
    with pytest.raises(ConfigError, match="family"):
        load_config(write(tmp_path, {**BASE, "family": "office"}))


def test_the_http_client_waits_as_long_as_an_attempt():
    # Session 5 (LOG.md, 01-11-2026): the HTTP client's own 600 s read timeout cut simulator
    # calls that ran past 10 minutes under a full KV cache, whatever attempt_timeout said.
    from inspect_ai.model import GenerateConfig

    from simtest.coding.provider import RawChatAPI

    api = RawChatAPI("sim", base_url="http://localhost:1/v1", config=GenerateConfig(attempt_timeout=2400, timeout=3600))
    assert api.client_timeout == 2400
    assert RawChatAPI("sim", base_url="http://localhost:1/v1").client_timeout is None   # Inspect's default


def test_pilot_config_keeps_session_5s_settings_and_concurrency():
    """configs/coding/pilot.yaml (N2): the 6 flows scenarios in C0, C3 and C2, 3 epochs (18 per
    condition), with session 5's models, sampling, limits, retries and 12 in flight in total: the
    simulator's FP8 MoE output depends on what shares its batch, so concurrency stays pinned, and
    every condition runs in one seeded queue so they share the same batch mix."""
    pilot = load_config("configs/coding/pilot.yaml")
    s5 = load_config("configs/coding/flows_dev_c3.yaml")
    assert pilot["name"] == "n2-pilot"
    assert pilot["scenarios"] == s5["scenarios"] and len(pilot["scenarios"]) == 6
    assert pilot["conditions"] == ["C0", "C3", "C2"]   # one queue: the order of this list doesn't matter
    assert pilot["epochs"] == 3
    for key in ("max_turns", "deadline", "agents", "simulators"):
        assert pilot[key] == s5[key], key
    assert {k: v for k, v in pilot["run"].items() if k != "order_seed"} == \
        {k: v for k, v in s5["run"].items() if k != "order_seed"}
    assert pilot["run"]["order_seed"] == 20261102   # fixed before the run
    assert pilot["run"]["max_connections"] == 12
    assert pilot["simulators_for"] == {"C3": ["agentworld"], "C2": ["agentworld"]}
    header = open("configs/coding/pilot.yaml").read().split("\nfamily:")[0]
    assert "Decision rule, frozen 03-11-2026" in header and "DRAFT" not in header
    assert "--c3 = --c0 + 0.10" in header and "--effect-spread: 0.25" in header   # the frozen power rule
    assert "pessimistic" not in header                     # the old rule measured a reversed effect
    assert "Judge errors (added 03-11-2026" in header and "every judge error as EVAL" in header   # never read as REAL
    assert "2 or more of a condition's scored episodes" in header   # the judge-error gate
    assert "max_connections" in header and "units" in header
    assert "12 in flight in total" in header and "one queue" in header
    assert "fp8_per_channel" in header and "per tensor over the whole batch" not in header   # LOG.md, 02-11-2026


# --- a native Anthropic simulator, and DeepSeek's reasoning field ----------------------------------

SONNET = {"model": "anthropic/claude-sonnet-5-5",
          "config": {"reasoning_effort": "medium", "cache_prompt": True, "max_tokens": 16384}}
C3 = {**BASE, "conditions": ["C0", "C3"]}


def sim(tmp_path, **change):
    spec = {**SONNET, **{k: v for k, v in change.items() if k != "config"}}
    spec["config"] = {**SONNET["config"], **change.get("config", {})}
    return write(tmp_path, {**C3, "simulators": {"sonnet55": spec}})


def test_native_anthropic_simulator_needs_no_endpoint_and_keeps_its_request_settings(tmp_path):
    cfg = load_config(sim(tmp_path))
    spec = cfg["simulators"]["sonnet55"]
    assert spec["model"] == "anthropic/claude-sonnet-5-5" and spec["args"] == {}
    assert spec["config"]["reasoning_effort"] == "medium" and spec["config"]["cache_prompt"] is True
    assert request_config(cfg, "sonnet55")["simulator"] == {
        "model": "anthropic/claude-sonnet-5-5", "args": {},
        "config": {"cache_prompt": True, "max_tokens": 16384, "reasoning_effort": "medium"}}


@pytest.mark.parametrize("change, message", [
    ({"base_url": "https://api.anthropic.com"}, "own endpoint and ANTHROPIC_API_KEY"),
    ({"api_key_env": "OTHER_KEY"}, "own endpoint and ANTHROPIC_API_KEY"),
    ({"args": {"base_url": "https://proxy/v1"}}, "own endpoint and ANTHROPIC_API_KEY"),
    ({"args": {"cache_ttl": "2h"}}, "cache_ttl"),
    ({"config": {"temperature": 0.6}}, "sampling"),
    ({"config": {"top_p": 0.95}}, "sampling"),
    ({"config": {"top_k": 20}}, "sampling"),
    ({"config": {"effort": "medium"}}, "set reasoning_effort"),
    ({"config": {"reasoning_tokens": 4096}}, "reasoning_tokens"),
    ({"config": {"fallback_models": ["claude-sonnet-5"]}}, "fallback_models"),
    ({"config": {"reasoning_effort": None}}, "must set reasoning_effort"),
])
def test_native_anthropic_refuses_what_it_would_ignore_or_reject(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        load_config(sim(tmp_path, **change))


def test_native_anthropic_accepts_a_cache_ttl(tmp_path):
    cfg = load_config(sim(tmp_path, args={"cache_ttl": "1h"}))
    assert request_config(cfg, "sonnet55")["simulator"]["args"] == {"cache_ttl": "1h"}   # fingerprinted


def test_an_agent_cannot_be_a_native_anthropic_model(tmp_path):
    # Its thinking comes back as a signed summary; the measures read the raw chain of thought.
    data = {**BASE, "agents": {"claude": {"model": "anthropic/claude-sonnet-5-5", "config": {"max_tokens": 1000}}}}
    with pytest.raises(ConfigError, match="raw chain of thought"):
        load_config(write(tmp_path, data))


def test_a_rawchat_model_still_needs_a_base_url(tmp_path):
    with pytest.raises(ConfigError, match="base_url"):
        load_config(write(tmp_path, {**BASE, "agents": {"a": {"model": "rawchat/agent"}}}))


def test_reasoning_field_is_checked_at_load_and_is_a_request_setting(tmp_path):
    agent = {"model": "rawchat/deepseek-v4-pro", "base_url": "https://api.deepseek.com/v1",
             "api_key_env": "DEEPSEEK_API_KEY", "args": {"reasoning_field": "reasoning_content"}}
    rc = request_config(load_config(write(tmp_path, {**BASE, "agents": {"a": agent}})))
    assert rc["agents"]["a"]["args"] == {"reasoning_field": "reasoning_content",
                                         "base_url": "https://api.deepseek.com/v1"}   # and no api_key_env
    bad = {**agent, "args": {"reasoning_field": "thoughts"}}
    with pytest.raises(ConfigError, match="reasoning_field"):
        load_config(write(tmp_path, {**BASE, "agents": {"a": bad}}))


def test_a_run_with_a_native_anthropic_model_fails_fast_without_its_key(tmp_path, monkeypatch):
    from simtest.coding.run import run_config

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    path = write(tmp_path, {**C3, "agents": {"a": {"model": "mockllm/agent", "base_url": "http://unused/v1"}},
                            "simulators": {"sonnet55": SONNET}})
    with pytest.raises(SystemExit, match=r"ANTHROPIC_API_KEY is not set.*--env-file \.\./\.env"):
        run_config(path, allow_dirty=True, build=False, check_snapshots=False, log_root=tmp_path / "logs")
    assert not (tmp_path / "logs").exists()    # before anything ran


@pytest.mark.parametrize("var, val", [
    ("ANTHROPIC_AUTH_TOKEN", "oauth-token-override"),
    ("ANTHROPIC_BASE_URL", "https://proxy.example/v1"),
])
def test_native_anthropic_run_refuses_conflicting_env_overrides(tmp_path, monkeypatch, var, val):
    from simtest.coding.run import run_config

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-secret")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv(var, val)
    path = write(tmp_path, {**C3, "agents": {"a": {"model": "mockllm/agent", "base_url": "http://unused/v1"}},
                            "simulators": {"sonnet55": SONNET}})
    with pytest.raises(SystemExit, match=var):
        run_config(path, allow_dirty=True, build=False, check_snapshots=False, log_root=tmp_path / "logs")
    assert not (tmp_path / "logs").exists()


def test_build_model_gives_inspect_the_effort_and_the_caching(tmp_path, monkeypatch):
    from inspect_ai.model._providers.anthropic import AnthropicAPI

    from simtest.coding.run import build_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-a-secret")
    model = build_model(load_config(sim(tmp_path))["simulators"]["sonnet55"])
    assert isinstance(model.api, AnthropicAPI) and str(model) == "anthropic/claude-sonnet-5-5"
    assert model.config.reasoning_effort == "medium" and model.config.effort is None
    assert model.config.cache_prompt is True and model.config.max_tokens == 16384
    assert model.api.cache_ttl is None    # Inspect's "auto": 5 minutes, 1 hour after a long gap
    # what Inspect makes of them (the wire request itself: tests/test_coding_anthropic.py)
    params, extra_body, _, _ = model.api.completion_config(model.config)
    assert params["output_config"] == {"effort": "medium"}
    assert params["thinking"] == {"type": "adaptive", "display": "summarized"} and not extra_body


def test_bluedot_cheap_configs_are_one_design_with_two_agents():
    """configs/coding/bluedot_cheap_{qwen,deepseek}.yaml: 4 episodes each, the same but for the agent."""
    qwen = load_config("configs/coding/bluedot_cheap_qwen.yaml")
    deep = load_config("configs/coding/bluedot_cheap_deepseek.yaml")
    assert (qwen["name"], deep["name"]) == ("bluedot-cheap-qwen", "bluedot-cheap-deepseek")
    for key in ("scenarios", "conditions", "epochs", "max_turns", "deadline", "simulators", "simulators_for", "run"):
        assert qwen[key] == deep[key], key
    assert qwen["scenarios"] == ["flows_audit", "pricing_cap"] and qwen["conditions"] == ["C0", "C3"]
    assert (qwen["epochs"], qwen["max_turns"], qwen["deadline"]) == (1, 40, 60)
    assert qwen["simulators_for"] == {"C3": ["sonnet55"]}
    assert qwen["run"]["max_connections"] == 2 and qwen["run"]["retry_attempts"] == 1
    assert qwen["run"]["order_seed"] == 20261122
    assert request_config(qwen, "sonnet55")["simulator"] == {
        "model": "anthropic/claude-sonnet-5-5", "args": {},
        "config": {"cache_prompt": True, "max_tokens": 32768, "reasoning_effort": "medium"}}
    # the Qwen agent is Hopper/pilot's FP8 tunnel endpoint, request for request
    pilot = load_config("configs/coding/pilot.yaml")
    assert request_config(qwen)["agents"] == request_config(pilot)["agents"]
    assert request_config(deep)["agents"]["deepseek_flash"] == {
        "model": "rawchat/deepseek-flash",
        "args": {"base_url": "https://api.deepseek.com/v1", "reasoning_field": "reasoning_content"},
        "config": {"extra_body": {"thinking": {"type": "enabled"}}, "max_tokens": 32768},
    }
    assert deep["agents"]["deepseek_flash"]["args"]["api_key_env"] == "DEEPSEEK_API_KEY"
    header = open("configs/coding/bluedot_cheap_qwen.yaml").read().split("\nfamily:")[0]
    header = " ".join(line.lstrip("# ") for line in header.splitlines())     # the comment as running text
    assert "not confirmatory evidence" in header and "Pivot rules, fixed before the run" in header
    for rule in ("more than $1.50 per Sim episode", "more than 60% of the bash steps", "0 of the 2 complete",
                 "3 or more of the 4 Real episodes", "0 of the 4 Sim episodes", "DEEPSEEK HARNESS"):
        assert rule in header, rule
    assert "bluedot_cheap_qwen.yaml" in open("configs/coding/bluedot_cheap_deepseek.yaml").read().split("\nfamily:")[0]


def test_bluedot_py_cheap_configs_are_one_option_a_design_with_two_agents():
    """configs/coding/bluedot_py_cheap_{qwen,deepseek}.yaml: 8 Python episodes each, same but for agent."""
    qwen = load_config("configs/coding/bluedot_py_cheap_qwen.yaml")
    deep = load_config("configs/coding/bluedot_py_cheap_deepseek.yaml")
    assert (qwen["name"], deep["name"]) == ("bluedot-py-cheap-qwen", "bluedot-py-cheap-deepseek")
    for key in ("scenarios", "conditions", "epochs", "max_turns", "deadline", "simulators", "simulators_for", "run"):
        assert qwen[key] == deep[key], key
    assert qwen["scenarios"] == ["pricing_cap", "billing_rounding", "churn_format", "orders_schema"]
    assert qwen["conditions"] == ["C0", "C3"] and qwen["epochs"] == 1
    assert qwen["simulators_for"] == {"C3": ["sonnet55"]}
    pilot = load_config("configs/coding/pilot.yaml")
    assert request_config(qwen)["agents"] == request_config(pilot)["agents"]
    header = open("configs/coding/bluedot_py_cheap_qwen.yaml").read().split("\nfamily:")[0]
    header = " ".join(line.lstrip("# ") for line in header.splitlines())
    assert "not confirmatory evidence" in header and "exceeds $0.50" in header and "exceeds 15%" in header
