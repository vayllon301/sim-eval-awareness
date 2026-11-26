"""Validation of the independent Option C (latent-interaction) same-family vs cross-family configs."""

from __future__ import annotations

import pytest

from simtest.coding.run import load_config, request_config
from test_coding_queue_run import machines  # noqa: F401 (pytest fixture)

PYTHON_PAIRS = [
    "pricing_cap", "pricing_cap_benign",
    "billing_rounding", "billing_rounding_benign",
    "churn_format", "churn_format_benign",
    "orders_schema", "orders_schema_benign",
]

CONFIGS = (
    ("configs/coding/latent_qwen_hopper_only.yaml", "latent-qwen-hopper-only", ["qwen27b_sim", "agentworld"]),
    ("configs/coding/latent_qwen_hopper_v2.yaml", "latent-qwen-hopper-v2", ["qwen27b_sim", "agentworld"]),
    ("configs/coding/latent_qwen_same_family.yaml", "latent-qwen-same-family", ["qwen27b_sim", "sonnet55"]),
    ("configs/coding/latent_deepseek_same_family.yaml", "latent-deepseek-same-family", ["deepseek_sim", "sonnet55"]),
)


@pytest.mark.parametrize("path, name, sims", CONFIGS)
def test_latent_configs_load_and_share_the_four_python_pairs(path, name, sims):
    cfg = load_config(path)
    assert cfg["name"] == name
    assert cfg["scenarios"] == PYTHON_PAIRS
    assert cfg["conditions"] == ["C0", "C3"]
    assert cfg["epochs"] == 1 and cfg["max_turns"] == 40 and cfg["deadline"] == 60
    assert cfg["simulators_for"] == {"C3": sims}
    assert set(cfg["simulators"]) == set(sims)
    for sim_name in sims:
        rc = request_config(cfg, sim_name)
        assert "api_key_env" not in rc["simulator"]["args"]


def test_latent_qwen_configs_match_hopper_agent_and_same_family_endpoint():
    pilot = load_config("configs/coding/pilot.yaml")
    for path in ("configs/coding/latent_qwen_hopper_only.yaml", "configs/coding/latent_qwen_hopper_v2.yaml",
                 "configs/coding/latent_qwen_same_family.yaml"):
        cfg = load_config(path)
        assert request_config(cfg)["agents"] == request_config(pilot)["agents"]
        assert cfg["simulators"]["qwen27b_sim"]["model"] == cfg["agents"]["qwen27b"]["model"]
        assert cfg["simulators"]["qwen27b_sim"]["args"]["base_url"] == cfg["agents"]["qwen27b"]["args"]["base_url"]


def test_latent_deepseek_same_family_matches_direct_api_settings():
    cfg = load_config("configs/coding/latent_deepseek_same_family.yaml")
    assert cfg["agents"]["deepseek_flash"]["model"] == "rawchat/deepseek-flash"
    assert cfg["simulators"]["deepseek_sim"]["model"] == "rawchat/deepseek-flash"
    assert cfg["agents"]["deepseek_flash"]["args"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert cfg["simulators"]["deepseek_sim"]["args"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert cfg["simulators"]["deepseek_sim"]["args"]["reasoning_field"] == "reasoning_content"
    assert request_config(cfg, "sonnet55")["simulator"] == {
        "model": "anthropic/claude-sonnet-5-5",
        "args": {},
        "config": {"cache_prompt": True, "max_tokens": 32768, "reasoning_effort": "medium"},
    }



def test_preflight_runs_same_family_config_with_shared_served_model(tmp_path, machines):
    from simtest import preflight as P

    ok, problems, _, _ = P.preflight(
        "configs/coding/latent_qwen_hopper_only.yaml",
        root=tmp_path / "pf",
        machine="fake",
        build=False,
        check_snapshots=False,
    )
    assert ok and problems == []


def test_hopper_simulators_get_pilots_64k_output_cap_except_the_v1_record():
    """Job 7109: 13 calls stopped at 16,384 tokens (every qwen27b_sim episode, 5 of 8 agentworld
    ones). v2 and the not-yet-run same-family config use configs/coding/pilot.yaml's 65,536; v1
    stays as run."""
    pilot = load_config("configs/coding/pilot.yaml")
    cap = pilot["simulators"]["agentworld"]["config"]["max_tokens"]
    assert cap == 65536
    v2 = load_config("configs/coding/latent_qwen_hopper_v2.yaml")
    for name in ("qwen27b_sim", "agentworld"):
        assert v2["simulators"][name]["config"]["max_tokens"] == cap, name
    assert request_config(v2, "agentworld")["simulator"] == request_config(pilot, "agentworld")["simulator"]
    same = load_config("configs/coding/latent_qwen_same_family.yaml")
    assert same["simulators"]["qwen27b_sim"]["config"]["max_tokens"] == cap
    v1 = load_config("configs/coding/latent_qwen_hopper_only.yaml")
    assert {n: s["config"]["max_tokens"] for n, s in v1["simulators"].items()} == {"qwen27b_sim": 16384,
                                                                                  "agentworld": 16384}


def test_v2_differs_from_the_v1_record_only_in_name_and_simulator_caps():
    v1 = load_config("configs/coding/latent_qwen_hopper_only.yaml")
    v2 = load_config("configs/coding/latent_qwen_hopper_v2.yaml")
    for cfg in (v1, v2):
        cfg.pop("name")
        for spec in cfg["simulators"].values():
            spec["config"].pop("max_tokens")
    assert v1 == v2


V3_SCENARIOS = ["pricing_cap", "pricing_cap_benign", "billing_rounding", "billing_rounding_benign",
                "orders_schema", "orders_schema_benign"]


def test_v3_runs_c2_and_c3_on_the_six_scenarios_with_a_reference():
    from simtest.coding.scenario import load_scenario
    from simtest.preflight import expected

    v3 = load_config("configs/coding/latent_qwen_hopper_v3.yaml")
    v2 = load_config("configs/coding/latent_qwen_hopper_v2.yaml")
    assert v3["name"] == "latent-qwen-hopper-v3" and v3["scenarios"] == V3_SCENARIOS
    assert v3["conditions"] == ["C0", "C2", "C3"]
    assert v3["simulators_for"] == {"C2": ["qwen27b_sim", "agentworld"], "C3": ["qwen27b_sim", "agentworld"]}
    assert v3["run"] == {**v2["run"], "max_connections": 6, "retry_attempts": 1, "order_seed": 20261127}
    # agent and both simulators exactly as v2 (65,536-token simulator cap)
    assert v3["agents"] == v2["agents"] and v3["simulators"] == v2["simulators"]
    for name in ("qwen27b_sim", "agentworld"):
        assert request_config(v3, name) == request_config(v2, name)
    design = {"conditions": v3["conditions"], "simulators": v3["simulators_for"], "scenarios": v3["scenarios"],
              "epochs": v3["epochs"]}
    assert len(expected(design, None)) == 30
    for sid in V3_SCENARIOS:      # C2 needs every scenario's audited reference
        assert load_scenario(sid).reference_path.exists(), sid


def test_preflight_runs_c2_and_c3_sharing_a_same_family_simulator(tmp_path, machines):
    # C2 and C3 both use qwen27b_sim, whose served name is the agent's (rawchat/agent): every one
    # of the 30 slots must run with the right mock (agent vs simulator) and call its simulator
    from simtest import preflight as P

    ok, problems, log_dir, _ = P.preflight("configs/coding/latent_qwen_hopper_v3.yaml", root=tmp_path / "pf",
                                           machine="fake", build=False, check_snapshots=False)
    assert ok and problems == [], problems
    from simtest.coding.analysis import load_run

    episodes, _ = load_run(log_dir)
    got = {(e["condition"], e["simulator"]) for e in episodes}
    assert got == {("C0", "none"), ("C2", "qwen27b_sim"), ("C2", "agentworld"), ("C3", "qwen27b_sim"),
                   ("C3", "agentworld")}
    assert len({(e["condition"], e["simulator"], e["scenario"]) for e in episodes}) == 30
