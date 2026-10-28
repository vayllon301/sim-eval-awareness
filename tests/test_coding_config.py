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
