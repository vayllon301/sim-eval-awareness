"""Vector files, steering specs and steering checks on the fake backend: strength 0
leaves greedy output unchanged, the projection moves monotonically with strength, and
controls have the target's norm."""

import json

import numpy as np
import pytest

from simtest.interp import directions as D
from simtest.interp.backend import Add
from simtest.interp.fake import FakeBackend
from simtest.interp.meta import file_sha256
from simtest.interp.steering import SteeringSpec, controls, dose_curve, resolve
from simtest.interp.vectors import VectorSet

D_MODEL = 16
FP = FakeBackend().info["fingerprint"]  # the vector files below are built "on" the fake backend


def vector_file(tmp_path, layers=(2, 3), names=("desperate", "calm", "topic:bakery")):
    rng = np.random.default_rng(0)
    vs = VectorSet(meta={"model": {"fingerprint": "fake:0:6x16"}, "resid_norm_mean": {str(L): 10.0 + L for L in layers},
                         "offset": {n: {str(L): 0.5 for L in layers} for n in names}})
    for n in names:
        for L in layers:
            vs.add(n, L, D.unit(rng.standard_normal(D_MODEL)))
    path = tmp_path / "vecs.safetensors"
    vs.save(path)
    return path, vs


def test_vector_file_round_trip(tmp_path):
    path, vs = vector_file(tmp_path)
    back = VectorSet.load(path)
    assert back.names == ["desperate", "calm", "topic:bakery"] and back.layers == [2, 3]
    assert np.array_equal(back.get("calm", 3), vs.get("calm", 3))
    assert back.meta["format"] == "simtest-interp-vectors" and back.meta["format_version"] == 1
    assert back.meta["d_model"] == D_MODEL and back.resid_norm(2) == 12.0
    with pytest.raises(KeyError, match="no vector"):
        back.get("angry", 2)


def test_vector_file_rejects_bad_vectors(tmp_path):
    vs = VectorSet()
    with pytest.raises(ValueError, match="unit vectors"):
        vs.add("x", 1, np.ones(4))
    vs.add("x", 1, D.unit(np.ones(4)))
    with pytest.raises(ValueError, match="shape"):
        vs.add("y", 1, D.unit(np.ones(5)))
    with pytest.raises(ValueError, match="'@'"):
        vs.add("a@b", 1, D.unit(np.ones(4)))
        vs.save(tmp_path / "x.safetensors")


def test_vector_file_needs_its_metadata(tmp_path):
    from safetensors.numpy import save_file

    save_file({"x@1": np.ones(3, dtype=np.float32)}, str(tmp_path / "plain.safetensors"))
    with pytest.raises(ValueError, match="not a simtest interp vector file"):
        VectorSet.load(tmp_path / "plain.safetensors")


def spec_dict(path, **kw):
    return {"spec_version": 1, "vectors": str(path), "direction": "desperate", "layers": [2, 3], "alpha": 0.05,
            "sign": 1, "seed": None, **kw}


def test_spec_resolves_to_alpha_times_norm(tmp_path):
    path, vs = vector_file(tmp_path)
    adds = resolve(SteeringSpec.from_dict(spec_dict(path)), fingerprint=FP)
    assert [a.layer for a in adds] == [2, 3]
    for a in adds:
        assert np.isclose(np.linalg.norm(a.delta), 0.05 * (10.0 + a.layer), rtol=1e-5)
        assert np.allclose(a.delta / np.linalg.norm(a.delta), vs.get("desperate", a.layer), atol=1e-6)
    neg = resolve(SteeringSpec.from_dict(spec_dict(path, sign=-1)), fingerprint=FP)
    assert np.allclose(neg[0].delta, -adds[0].delta)


def test_resolve_refuses_another_models_backend(tmp_path):
    """Review B2: a spec only resolves for the model its vectors were built on."""
    path, vs = vector_file(tmp_path)
    spec = SteeringSpec.from_dict(spec_dict(path))
    with pytest.raises(ValueError, match="built on fake:0:6x16"):
        resolve(spec, vs, fingerprint="fp1:qwen3.5-27b")
    with pytest.raises(ValueError, match="built on fake:0:6x16"):
        resolve(spec, fingerprint="fp1:qwen3.5-27b")
    random = SteeringSpec.from_dict(spec_dict(path, direction="random", seed=0))
    with pytest.raises(ValueError, match="built on"):
        resolve(random, vs, fingerprint="fp1:qwen3.5-27b")  # its norm reference is model-specific too
    assert len(resolve(spec, vs, fingerprint=FP)) == 2


def test_spec_file_round_trip_and_relative_paths(tmp_path):
    path, _ = vector_file(tmp_path)
    spec = SteeringSpec.from_dict(spec_dict("vecs.safetensors", vectors_sha256=file_sha256(path)))
    spec.save(tmp_path / "specs" / "s.yaml")
    (tmp_path / "specs" / "vecs.safetensors").write_bytes(path.read_bytes())
    back = SteeringSpec.load(tmp_path / "specs" / "s.yaml")
    assert back.vectors == str((tmp_path / "specs" / "vecs.safetensors").resolve())
    assert back.layers == (2, 3) and back.label == "desperate+0.05@2,3"
    resolve(back, fingerprint=FP)
    (tmp_path / "s.json").write_text(json.dumps(spec_dict(path)))
    assert SteeringSpec.load(tmp_path / "s.json").direction == "desperate"


def test_spec_checks_the_vector_file_hash(tmp_path):
    path, _ = vector_file(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        resolve(SteeringSpec.from_dict(spec_dict(path, vectors_sha256="0" * 64)), fingerprint=FP)


@pytest.mark.parametrize("bad, match", [
    ({"sign": 2}, "sign"), ({"alpha": -0.1}, "alpha"), ({"alpha": float("nan")}, "alpha"),
    ({"layers": []}, "layers"), ({"layers": [2, 2]}, "layers"), ({"direction": "random"}, "seed"),
    ({"seed": 3}, "seed"), ({"spec_version": 2}, "spec_version"), ({"extra": 1}, "unknown"),
])
def test_spec_validation(tmp_path, bad, match):
    with pytest.raises(ValueError, match=match):
        SteeringSpec.from_dict(spec_dict(tmp_path / "v.safetensors", **bad))


def test_unknown_layer_or_direction_fails(tmp_path):
    path, _ = vector_file(tmp_path)
    with pytest.raises(KeyError):
        resolve(SteeringSpec.from_dict(spec_dict(path, layers=[4])), fingerprint=FP)
    with pytest.raises(KeyError):
        resolve(SteeringSpec.from_dict(spec_dict(path, direction="angry")), fingerprint=FP)


def test_controls_match_the_target_norm_and_cover_both_signs(tmp_path):
    path, vs = vector_file(tmp_path)
    target = SteeringSpec.from_dict(spec_dict(path))
    cs = controls(target, random_seeds=[0, 1], concepts=["topic:bakery"])
    assert len(cs) == 6 and {c.sign for c in cs} == {1, -1}
    want = np.linalg.norm(resolve(target, vs, fingerprint=FP)[0].delta)
    for c in cs:
        assert c.layers == target.layers and c.alpha == target.alpha
        assert np.isclose(np.linalg.norm(resolve(c, vs, fingerprint=FP)[0].delta), want, rtol=1e-5)
    r = [c for c in cs if c.direction == "random" and c.seed == 0 and c.sign == 1][0]
    assert np.allclose(resolve(r, vs, fingerprint=FP)[0].delta / want, D.random_unit(D_MODEL, 0, 2), atol=1e-6)


def test_dose_curve_signs(tmp_path):
    path, _ = vector_file(tmp_path)
    curve = dose_curve(SteeringSpec.from_dict(spec_dict(path)), [-0.1, 0, 0.1])
    assert [(c.alpha, c.sign) for c in curve] == [(0.1, -1), (0.0, 1), (0.1, 1)]


def test_strength_zero_leaves_greedy_output_unchanged():
    b = FakeBackend()
    ids = b.tokenizer("The deploy failed again and the release is at five.")["input_ids"]
    zero = [Add(2, np.zeros(D_MODEL, np.float32)), Add(3, np.zeros(D_MODEL, np.float32))]
    assert b.generate(ids, max_tokens=12, steer=zero) == b.generate(ids, max_tokens=12)
    steered = b.generate(ids, max_tokens=12, steer=[Add(2, 50 * D.unit(np.ones(D_MODEL)).astype(np.float32))])
    assert steered != b.generate(ids, max_tokens=12)


def test_projection_moves_monotonically_with_strength(tmp_path):
    path, vs = vector_file(tmp_path)
    b = FakeBackend()
    ids = b.tokenizer("A quiet afternoon; nothing needs doing until tomorrow.")["input_ids"]
    v2, v5 = vs.get("desperate", 2), vs.get("desperate", 2)
    means_at_2, means_at_5 = [], []
    for a in [-0.2, -0.1, 0, 0.1, 0.2]:
        spec = SteeringSpec.from_dict(spec_dict(path, alpha=abs(a), sign=1 if a >= 0 else -1, layers=[2]))
        acts = b.activations(ids, [2, 5], steer=resolve(spec, vs, fingerprint=FP))
        means_at_2.append(float((acts[2] @ v2).mean()))
        means_at_5.append(float((acts[5] @ v5).mean()))
    assert np.allclose(np.diff(means_at_2), 0.1 * 12.0, rtol=1e-4)  # exact at the steered layer
    assert all(np.diff(means_at_5) > 0)  # and still monotonic downstream
