"""The vLLM backend (vllm_backend.py) and its steering adapter, offline: no vLLM, vllm-lens or
torch. A fake engine on the fake numpy model plays vllm-lens's semantics (capture after
steering at resid_post, steering vectors added at every position), so these tests check our
plumbing: the spec-to-SteeringVector conversion, layer order, one request per call, the
engine settings, the fingerprint. Whether vllm-lens itself does this on vLLM 0.32 is the
GPU smoke test's job (explore/interp/gpu_smoke.py)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from simtest.interp import directions as D
from simtest.interp import vllm_backend as V
from simtest.interp.backend import Add, encode
from simtest.interp.fake import FakeBackend, FakeLensLLM, fake_vllm_api
from simtest.interp.steering import SteeringSpec, dose_curve, resolve
from simtest.interp.vectors import VectorSet

D_MODEL, N_LAYERS = 16, 6


@pytest.fixture(autouse=True)
def fake_api(monkeypatch):
    """vLLM's and vllm-lens's classes, as plain Python on the fake model."""
    api = fake_vllm_api()
    monkeypatch.setattr(V, "_api", lambda: api)
    monkeypatch.delenv("VLLM_LENS_DISABLE", raising=False)
    return api


def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": N_LAYERS, "hidden_size": D_MODEL}}))
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [0, 7]}))
    (tmp_path / "model.safetensors").write_bytes(b"\1" * 32)
    return tmp_path


def backend(tmp_path):
    fake = FakeBackend()
    llm = FakeLensLLM(fake)
    return V.VLLMBackend(model_dir(tmp_path), tokenizer=fake.tokenizer, llm=llm), fake, llm


def vectors(fingerprint, layers=(2, 3)):
    rng = np.random.default_rng(0)
    vs = VectorSet(meta={"model": {"fingerprint": fingerprint}, "resid_norm_mean": {str(L): 10.0 + L for L in layers}})
    for n in ("desperate", "topic:bakery"):
        for L in layers:
            vs.add(n, L, D.unit(rng.standard_normal(D_MODEL)))
    return vs


def test_lens_steering_stacks_the_resolved_deltas_in_layer_order():
    a, b = np.arange(D_MODEL, dtype=np.float32), np.ones(D_MODEL, np.float32)
    sv = V.lens_steering([Add(4, a), Add(1, b), Add(4, a)], N_LAYERS, D_MODEL)
    assert sv["layer_indices"] == [1, 4] and sv["activations"].dtype == np.float32
    assert np.array_equal(sv["activations"], np.stack([b, 2 * a]))
    assert sv["scale"] == 1.0 and sv["norm_match"] is False and sv["position_indices"] is None
    assert V.lens_steering([], N_LAYERS, D_MODEL) is None
    with pytest.raises(ValueError, match="not in 0..5"):
        V.lens_steering([Add(6, a)], N_LAYERS, D_MODEL)


def test_a_spec_becomes_one_steering_vector_with_alpha_times_norm_rows(tmp_path):
    b, _, _ = backend(tmp_path)
    vs = vectors(b.info["fingerprint"])
    spec = SteeringSpec(vectors="unused", direction="desperate", layers=(3, 2), alpha=0.05, sign=-1)
    sv = b._steering(resolve(spec, vs, fingerprint=b.info["fingerprint"]))["apply_steering_vectors"]
    assert len(sv) == 1 and sv[0].layer_indices == [2, 3]
    for i, L in enumerate([2, 3]):
        assert np.allclose(sv[0].activations[i], -0.05 * (10.0 + L) * vs.get("desperate", L), atol=1e-6)
    zero = resolve(dose_curve(spec, [0.0])[0], vs, fingerprint=b.info["fingerprint"])
    sv0 = b._steering(zero)["apply_steering_vectors"][0]  # alpha 0 still goes through the hooks
    assert sv0.layer_indices == [2, 3] and not sv0.activations.any()


def test_engine_settings_and_info(tmp_path, monkeypatch):
    b, _, _ = backend(tmp_path)
    assert b.engine["enforce_eager"] and b.engine["enable_prefix_caching"] is False and b.engine["language_model_only"]
    assert b.n_layers == N_LAYERS and b.d_model == D_MODEL and b.eos_token_ids == frozenset({0, 7})
    assert b.info["engine"] == "vllm" and b.info["lens"] == "enabled" and b.info["fingerprint"].startswith("fp1:")
    monkeypatch.setenv("VLLM_LENS_DISABLE", "1")
    sim = V.VLLMBackend(tmp_path, quantization="fp8")  # the engine and tokenizer from the (fake) API
    assert sim.info["fingerprint"] == b.info["fingerprint"] + "+vllm:fp8" and sim.info["lens"] == "disabled"
    assert sim.info["quantization"].startswith("vLLM fp8 at load") and sim.info["engine_settings"]["quantization"] == "fp8"


def test_activations_are_resid_post_with_additions_and_one_request_per_call(tmp_path):
    b, fake, llm = backend(tmp_path)
    ids = encode(b.tokenizer, "The deploy failed again and the release is at five.")
    acts = b.activations(ids, [3, 1])
    want = fake.activations(ids, [1, 3])
    assert sorted(acts) == [1, 3] and all(np.allclose(acts[L], want[L]) for L in (1, 3))
    delta = np.linspace(-1, 1, D_MODEL).astype(np.float32)
    steered = b.activations(ids, [1, 2], steer=[Add(2, delta)])
    clean = b.activations(ids, [1, 2])
    assert np.allclose(steered[1], clean[1]) and np.allclose(steered[2] - clean[2], delta, atol=1e-5)
    prompts, sp = llm.calls[0]
    assert len(prompts) == 1 and sp.max_tokens == 1 and sp.temperature == 0.0
    assert all(len(p) == 1 for p, _ in llm.calls)


def test_strength_zero_and_steered_generation(tmp_path):
    b, fake, llm = backend(tmp_path)
    ids = encode(b.tokenizer, "Name three rivers in Spain.")
    stock = b.generate(ids, max_tokens=12)
    assert stock == fake.generate(ids, max_tokens=12) and llm.calls[-1][1].extra_args is None
    zero = [Add(L, np.zeros(D_MODEL, np.float32)) for L in (2, 3)]
    assert b.generate(ids, max_tokens=12, steer=zero) == stock
    assert llm.calls[-1][1].extra_args["apply_steering_vectors"]  # zero additions are still sent
    assert llm.calls[-1][1].stop_token_ids == [0, 7]
    assert b.generate(ids, max_tokens=12, steer=[Add(2, 50 * D.unit(np.ones(D_MODEL)).astype(np.float32))]) != stock


def test_token_losses_from_prompt_logprobs(tmp_path):
    b, fake, _ = backend(tmp_path)
    ids = encode(b.tokenizer, "def add(a, b):\n    return a + b\n")
    assert np.allclose(b.token_losses(ids), fake.token_losses(ids))
    with pytest.raises(NotImplementedError, match="HF backend"):
        b.token_losses(ids, replace={2: lambda h, s: h})
    assert not hasattr(b, "continuation_logprobs")  # the logit monitor falls back to token_losses


def test_wrong_or_missing_activations_are_errors(tmp_path):
    b, _, llm = backend(tmp_path)
    ids = encode(b.tokenizer, "short text here")
    real = llm.generate

    def short(prompts, sp, use_tqdm=True):  # one position missing
        out = real(prompts, sp)
        out[0].activations["residual_stream"] = out[0].activations["residual_stream"][:, :-1]
        return out

    b.llm.generate = short
    with pytest.raises(RuntimeError, match="shape"):
        b.activations(ids, [1])
    b.llm.generate = lambda prompts, sp, use_tqdm=True: [SimpleNamespace(activations=None)]
    with pytest.raises(RuntimeError, match="no activations"):
        b.activations(ids, [1])


def test_reading_a_run_uses_the_served_context():
    """vllm_backend.ENGINE's 16K is the smoke test's. Session 5's agent prompts reach 80K tokens."""
    from simtest.interp.readrun import READ_ENGINE

    assert READ_ENGINE["max_model_len"] == 131072
