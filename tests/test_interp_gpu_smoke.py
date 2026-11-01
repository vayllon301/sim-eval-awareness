"""The GPU smoke script (explore/interp/gpu_smoke.py) end to end, offline: the stock, lens and
report steps on the fake vLLM engine, and the hf step on a tiny random HF model whose files
the fake engine "loads", so a broken script doesn't cost a paid GPU session. The numbers
mean nothing here; the checks that hold on any model (alpha 0 token-identical, a steer
moves its layer by its delta) must pass."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from simtest.interp import vllm_backend as V
from simtest.interp.fake import FakeBackend, FakeTokenizer, fake_vllm_api

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke(monkeypatch):
    spec = importlib.util.spec_from_file_location("gpu_smoke", ROOT / "explore" / "interp" / "gpu_smoke.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for k in ("VLLM_LENS_DISABLE", "VLLM_LENS_STRICT_COMPATIBILITY", "VLLM_USE_V2_MODEL_RUNNER"):  # restored after
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(V, "_api", lambda: fake_vllm_api(FakeBackend()))  # every engine: the same fake weights
    small = {"layers": [1, 2, 3, 4], "steer": [1, 2, 3], "read": 4}  # the fake has 6 layers
    for name in ("agent", "simulator"):
        monkeypatch.setitem(m.PRESETS, name, {**m.PRESETS[name], **small})
    m.PROMPTS = [(msgs, kw, 12) for msgs, kw, _ in m.PROMPTS]
    m.SAMPLE_TOKENS = 8
    return m


def test_steered_and_read_layers_must_be_scanned(smoke, monkeypatch, tmp_path):
    monkeypatch.setitem(smoke.PRESETS, "agent", {**smoke.PRESETS["agent"], "read": 5})
    with pytest.raises(SystemExit, match="scanned layers"):
        run(smoke, monkeypatch, "lens", str(fake_dir(tmp_path / "agent")), "--name", "agent", "--out-dir", str(tmp_path))


def run(m, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["gpu_smoke.py", *argv])
    m.main()


def fake_dir(path):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"num_hidden_layers": 6, "hidden_size": 16}))
    (path / "model.safetensors").write_bytes(b"\1" * 64)
    return path


def test_stock_lens_and_report_on_the_fake_engine(smoke, monkeypatch, tmp_path):
    model, out = fake_dir(tmp_path / "agent"), tmp_path / "out"
    for mode in ("eager", "served"):
        run(smoke, monkeypatch, "stock", str(model), "--name", "agent", "--mode", mode, "--out-dir", str(out))
    run(smoke, monkeypatch, "lens", str(model), "--name", "agent", "--out-dir", str(out))
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        run(smoke, monkeypatch, "lens", str(model), "--name", "agent", "--out-dir", str(out))
    with pytest.raises(SystemExit):  # no hf step and no simulator: incomplete
        run(smoke, monkeypatch, "report", "--out-dir", str(out))
    rep = json.loads((out / "checks.json").read_text())
    c, n = rep["checks"]["agent"], rep["notes"]["agent"]
    assert c["alpha0"] and c["steer_applied"] and c["token_losses"] and "vectors" in c and "dose" in c
    assert n["served_equals_eager"] and n["stock_eager_repeat_equal"] and n["alpha0"]["lens_runs_agree"]
    assert n["steer_applied"]["rel_err"] < 1e-4  # float32: exact
    lens = json.loads((out / "agent" / "lens.json").read_text())
    assert lens["backend"]["lens"] == "enabled" and lens["vectors"]["chosen_layer"] in (1, 2, 3, 4)
    stock = json.loads((out / "agent" / "stock-eager.json").read_text())
    assert stock["backend"]["lens"] == "disabled" and stock["backend"]["engine_settings"]["enforce_eager"]
    # vllm-lens moves vLLM to its V1 model runner, so stock-eager must run there too; served keeps vLLM's default
    assert stock["backend"]["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    served = json.loads((out / "agent" / "stock-served.json").read_text())
    assert served["backend"]["engine_settings"]["enforce_eager"] is False
    assert served["backend"]["env"]["VLLM_USE_V2_MODEL_RUNNER"] is None
    assert len(list((out / "agent" / "specs").glob("*.yaml"))) == 4 and (out / "agent" / "lens-acts.npz").exists()
    assert set(rep["incomplete"]["simulator"]) == set(smoke.REQUIRED["simulator"]) and n["steps_run"] == ["stock-eager", "lens", "stock-served"]


def test_the_simulator_runs_with_vllms_fp8_and_has_no_hf_step(smoke, monkeypatch, tmp_path):
    model, out = fake_dir(tmp_path / "simulator"), tmp_path / "out"
    run(smoke, monkeypatch, "lens", str(model), "--name", "simulator", "--out-dir", str(out))
    lens = json.loads((out / "simulator" / "lens.json").read_text())
    assert lens["backend"]["fingerprint"].endswith("+vllm:fp8")
    assert lens["backend"]["engine_settings"]["trust_remote_code"] is True
    with pytest.raises(SystemExit, match="for the agent"):
        run(smoke, monkeypatch, "hf", str(model), "--name", "simulator", "--out-dir", str(out))


def test_hf_step_on_the_same_files(smoke, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    model, out = tmp_path / "agent", tmp_path / "out"
    Qwen3ForCausalLM(Qwen3Config(vocab_size=256, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
                                 num_attention_heads=2, num_key_value_heads=1, head_dim=8)).save_pretrained(model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: FakeTokenizer())
    run(smoke, monkeypatch, "lens", str(model), "--name", "agent", "--out-dir", str(out))
    run(smoke, monkeypatch, "hf", str(model), "--name", "agent", "--out-dir", str(out))
    with pytest.raises(SystemExit):  # no stock-eager and no simulator: incomplete
        run(smoke, monkeypatch, "report", "--out-dir", str(out))
    rep = json.loads((out / "checks.json").read_text())
    c, n = rep["checks"]["agent"], rep["notes"]["agent"]
    assert c["hf_alpha0"] and c["hf_steer_applied"]
    assert n["hf"]["vectors_fingerprint_equal"] and n["hf"]["prompt_ids_equal_lens"]
    assert set(n["hf"]["activations_vs_vllm"]) == {"2", "4"}  # the fake's and the tiny model's weights differ


def test_report_marks_missing_checks_incomplete_and_fails(smoke, monkeypatch, tmp_path):
    # A failed or budget-skipped step must not drop its checks from the report: without
    # stock-eager there is no alpha0, without hf no hf checks, and the report exits nonzero.
    model, out = fake_dir(tmp_path / "agent"), tmp_path / "out"
    run(smoke, monkeypatch, "lens", str(model), "--name", "agent", "--out-dir", str(out))
    with pytest.raises(SystemExit) as e:
        run(smoke, monkeypatch, "report", "--out-dir", str(out))
    assert e.value.code == 1
    rep = json.loads((out / "checks.json").read_text())
    c = rep["checks"]["agent"]
    assert c["alpha0"] is None and c["hf_alpha0"] is None and c["hf_steer_applied"] is None
    assert c["steer_applied"] is True
    assert set(rep["checks"]["simulator"]) == set(smoke.REQUIRED["simulator"])
    assert all(v is None for v in rep["checks"]["simulator"].values())
    assert rep["incomplete"]["agent"] == ["alpha0", "hf_alpha0", "hf_steer_applied"]
