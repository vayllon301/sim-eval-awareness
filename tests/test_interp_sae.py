"""TopK SAEs: the torch-free .pt reader (on a fixture written by torch 2.14), encode and
decode against dense formulas, FVU, and splicing a reconstruction into the fake model."""

import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pytest

from simtest.interp.fake import FakeBackend
from simtest.interp.sae import TopKSAE, fvu, load_qwen_scope, read_pt

FIX = Path(__file__).resolve().parent / "interp_fixtures"


def test_read_pt_matches_torch():
    w, want = read_pt(FIX / "tiny.sae.pt"), np.load(FIX / "tiny.sae.expected.npz")
    assert set(w) == set(want.files)
    for k in want.files:
        assert w[k].shape == want[k].shape and np.array_equal(np.asarray(w[k]), want[k]), k
    assert w["view"].tolist() == [3, 4, 5, 6, 7, 8]  # a storage offset
    assert w["t"].shape == (4, 3) and w["t"][1].tolist() == [1, 5, 9]  # transposed strides
    assert w["bf16"].dtype == np.float32 and w["bf16"].tolist() == [1.5, -2.25, 3.0]


class Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def test_read_pt_refuses_code(tmp_path):
    p = tmp_path / "evil.pt"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as z:
        z.writestr("evil/data.pkl", pickle.dumps({"W_enc": Evil()}, protocol=2))
    with pytest.raises(pickle.UnpicklingError, match="refusing"):
        read_pt(p)


def test_read_pt_wants_a_torch_zip(tmp_path):
    p = tmp_path / "x.pt"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("x/other.txt", "hi")
    with pytest.raises(ValueError, match="not a PyTorch zip"):
        read_pt(p)


def sae(seed=0, d=8, d_sae=40, k=5):
    rng = np.random.default_rng(seed)
    return TopKSAE(W_enc=rng.standard_normal((d_sae, d)).astype(np.float32), b_enc=rng.standard_normal(d_sae).astype(np.float32) * 0.1,
                   W_dec=rng.standard_normal((d, d_sae)).astype(np.float32), b_dec=rng.standard_normal(d).astype(np.float32) * 0.1,
                   k=k)


def dense_reference(s, x):
    pre = np.maximum(x @ s.W_enc.T + s.b_enc, 0)
    acts = np.zeros_like(pre)
    top = np.argsort(-pre, axis=1)[:, : s.k]
    np.put_along_axis(acts, top, np.take_along_axis(pre, top, 1), 1)
    return acts, acts @ s.W_dec.T + s.b_dec


def test_encode_and_decode_match_the_dense_formula():
    s = sae()
    x = np.random.default_rng(1).standard_normal((300, 8)).astype(np.float32)
    idx, val = s.encode(x, chunk=64)
    acts, rec = dense_reference(s, x)
    assert idx.shape == val.shape == (300, 5)
    assert (val[:, :-1] >= val[:, 1:]).all() and (val >= 0).all()
    assert np.allclose(np.take_along_axis(acts, idx.astype(np.int64), 1), val, atol=1e-5)
    assert np.allclose(s.decode(idx, val, chunk=7), rec, atol=1e-4)
    assert np.allclose(s.reconstruct(x), rec, atol=1e-4)


def test_shapes_are_checked():
    s = sae()
    with pytest.raises(ValueError, match="inconsistent"):
        TopKSAE(W_enc=s.W_enc, b_enc=s.b_enc, W_dec=s.W_dec.T, b_dec=s.b_dec, k=5)
    with pytest.raises(ValueError, match="k must"):
        TopKSAE(W_enc=s.W_enc, b_enc=s.b_enc, W_dec=s.W_dec, b_dec=s.b_dec, k=0)


def test_load_qwen_scope_layout(tmp_path):
    (tmp_path / "layer3.sae.pt").write_bytes((FIX / "tiny.sae.pt").read_bytes())
    with pytest.raises(ValueError, match="k is not given"):
        load_qwen_scope(tmp_path / "layer3.sae.pt")
    (tmp_path / "config.json").write_text('{"k": 3, "hf_repo_id": "Qwen/test", "hook_point": "resid_post"}')
    s = load_qwen_scope(tmp_path / "layer3.sae.pt")
    assert (s.d_model, s.d_sae, s.k) == (6, 10, 3) and s.info["repo"] == "Qwen/test"


def test_fvu():
    x = np.random.default_rng(2).standard_normal((50, 4))
    assert fvu(x, x) == 0
    assert np.isclose(fvu(x, np.broadcast_to(x.mean(0), x.shape)), 1)


def test_sae_prose_sample_is_mixed_and_records_its_rows():
    """Review B4: every held-out group contributes before truncation, and the rows are saved."""
    import importlib.util
    import json

    from simtest.interp.fake import FakeTokenizer

    root = FIX.parents[1]
    spec = importlib.util.spec_from_file_location("sae_transfer", root / "explore" / "interp" / "sae_transfer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = [json.loads(x) for x in (root / "explore" / "interp" / "data" / "texts-qwen3.8-27b.jsonl").read_text().splitlines()]
    ids, used = mod.prose_sample(rows, FakeTokenizer(), 12000)
    assert len(ids) == 12000
    by_id = {r["id"]: r for r in rows}
    assert all(by_id[i]["split"] == "test" for i in used)
    groups = [by_id[i]["emotion"] or "neutral" for i in used]
    assert set(groups[:7]) == {"desperate", "calm", "frustrated", "afraid", "angry", "happy", "neutral"}
    assert mod.prose_sample(rows, FakeTokenizer(), 12000) == (ids, used)  # fixed


def test_splicing_into_the_fake_model():
    b = FakeBackend()
    ids = b.tokenizer("A short text to splice a reconstruction into.")["input_ids"]
    clean = b.token_losses(ids)
    same = b.token_losses(ids, replace={3: lambda h, start: h})
    assert np.allclose(clean, same)
    zero = b.token_losses(ids, replace={3: lambda h, start: np.zeros_like(h)})
    assert not np.allclose(clean, zero)
    assert clean.shape == (len(ids) - 1,)
