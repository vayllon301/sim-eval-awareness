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
    assert "DRAFT" in header and "max_connections" in header and "units" in header
    assert "12 in flight in total" in header and "one queue" in header
