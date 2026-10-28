"""Checks on a real local model (MLX). Skipped unless SIMTEST_INTERP_MODEL points at a model
directory, e.g.
    SIMTEST_INTERP_MODEL=~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit \\
        uv run --extra interp pytest -q -m interp_model
Loads the model once (about 30 GB for the 8-bit agent); free memory first."""

import os
from pathlib import Path

import numpy as np
import pytest

MODEL = os.environ.get("SIMTEST_INTERP_MODEL")
pytestmark = [pytest.mark.interp_model,
              pytest.mark.skipif(not MODEL, reason="set SIMTEST_INTERP_MODEL to a local MLX model directory")]
ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "tests" / "interp_fixtures" / "colab-probe-smoke.jsonl"


@pytest.fixture(scope="module")
def backend():
    pytest.importorskip("mlx_lm")
    from simtest.interp.mlx_backend import MLXBackend

    b = MLXBackend(Path(MODEL).expanduser())
    yield b
    b.close()


@pytest.fixture(scope="module")
def episode():
    from simtest.interp.reader import colab_probe_prompts, load_colab_probe

    return load_colab_probe(SMOKE, **colab_probe_prompts(ROOT / "explore"))


def test_render_with_the_real_template(backend, episode):
    from simtest.interp.reader import R, prompt_checks, render

    rd = render(episode, backend.tokenizer)
    assert "Reasoning effort is set to xhigh" in rd.text  # the template default the episodes ran with
    text = lambda role: backend.tokenizer.decode([int(i) for i, r in zip(rd.ids, rd.role) if r == R[role]])
    assert "Let's start by examining the repository" in text("reasoning")
    assert "PRICING-412" in text("user") and "total 44" in text("tool_output")
    checks = prompt_checks(episode, backend.tokenizer)
    assert [c["rendered"] for c in checks] == sorted(c["rendered"] for c in checks)


def test_hooks_are_removed_and_chunks_agree(backend):
    from simtest.interp.backend import encode

    layers_before = list(backend.layers)
    ids = encode(backend.tokenizer, "The release is at five and the tests are still red. " * 8)
    full = backend.activations(ids, [10, 20], chunk=4096)
    chunked = backend.activations(ids, [10, 20], chunk=16)
    assert list(backend.layers) == layers_before
    for L in (10, 20):
        cos = (full[L] * chunked[L]).sum(1) / (np.linalg.norm(full[L], axis=1) * np.linalg.norm(chunked[L], axis=1))
        assert cos.min() > 0.99


def test_strength_zero_is_identical_to_stock_generation(backend):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    from simtest.interp.backend import Add
    from simtest.interp.reader import chat_ids

    prompt = chat_ids(backend.tokenizer, [{"role": "user", "content": "Name three rivers in Spain."}], [],
                      add_generation_prompt=True, enable_thinking=False)
    stock = [r.token for r in stream_generate(backend.model, backend.tokenizer, prompt, max_tokens=24,
                                              sampler=make_sampler(temp=0.0))]
    zero = [Add(L, np.zeros(backend.d_model, np.float32)) for L in (30, 40)]
    assert backend.generate(prompt, max_tokens=24)[: len(stock)] == stock
    assert backend.generate(prompt, max_tokens=24, steer=zero)[: len(stock)] == stock


def test_losses_are_finite_and_splicing_identity_changes_nothing(backend):
    from simtest.interp.backend import encode

    ids = encode(backend.tokenizer, "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n" * 4)
    clean = backend.token_losses(ids, chunk=32)
    same = backend.token_losses(ids, replace={30: lambda h, start: h}, chunk=32)
    assert np.isfinite(clean).all() and clean.shape == (len(ids) - 1,)
    assert np.allclose(clean, same, atol=1e-3)
