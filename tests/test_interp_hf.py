"""The torch backend (hf_backend.py) on tiny random Hugging Face models on CPU, built in the
test (no downloads): a dense Qwen3 and a Qwen3.5-style hybrid (Gated DeltaNet linear
attention plus full attention, the agent's and the simulator's architecture). The contract
the MLX backend's model tests check: shapes, strength 0 token-identical to the library's own
greedy generation, additions land exactly at their layer, splicing identity, chunks agree,
hooks removed, and the fingerprint. Needs the `torch` extra:
    uv run --extra torch pytest -q tests/test_interp_hf.py"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from simtest.interp.backend import Add, encode  # noqa: E402
from simtest.interp.fake import FakeTokenizer  # noqa: E402
from simtest.interp.meta import weights_fingerprint  # noqa: E402

TEXT = "The release is at five and the tests are still red; nobody has looked at the logs yet. " * 2


def tiny(kind: str, path):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    common = dict(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=6, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=8, max_position_embeddings=1024)
    if kind == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**common))
    else:  # three linear-attention layers, then one full-attention layer, as in Qwen3.5
        types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**common, linear_key_head_dim=8, linear_value_head_dim=8,
                                                     linear_num_key_heads=2, linear_num_value_heads=4,
                                                     layer_types=types[:6]))
    model.save_pretrained(path)
    return path


@pytest.fixture(scope="module", params=["qwen3", "qwen3_5"])
def backend(request, tmp_path_factory):
    from simtest.interp.hf_backend import HFBackend

    path = tiny(request.param, tmp_path_factory.mktemp(request.param))
    b = HFBackend(path, tokenizer=FakeTokenizer(), dtype=torch.float32, device="cpu")
    yield b
    b.close()


def ids_of(b, text=TEXT):
    return encode(b.tokenizer, text)


def test_info_and_shapes(backend):
    b = backend
    assert b.info["engine"] == "hf" and b.info["fingerprint"] == weights_fingerprint(b.path)
    assert b.n_layers == 6 and b.d_model == 32 and b.eos_token_ids == frozenset({0})
    ids = ids_of(b)
    acts = b.activations(ids, [4, 1])
    assert sorted(acts) == [1, 4]
    assert all(a.shape == (len(ids), 32) and a.dtype == np.float32 for a in acts.values())
    assert b.token_losses(ids).shape == (len(ids) - 1,)


def test_layer_l_is_the_output_of_decoder_layer_l(backend):
    """resid_post: transformers' hidden_states[L + 1] (hidden_states[0] is the embeddings)."""
    b = backend
    ids = ids_of(b)
    with torch.inference_mode():
        hs = b.model(input_ids=torch.tensor([ids]), output_hidden_states=True).hidden_states
    acts = b.activations(ids, [0, 2, 4])
    for L in (0, 2, 4):
        assert np.allclose(acts[L], hs[L + 1][0].numpy(), atol=1e-5)


def test_strength_zero_is_identical_to_stock_generation(backend):
    b = backend
    ids = ids_of(b, "Name three rivers in Spain.")
    stock = b.model.generate(torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                             max_new_tokens=24, do_sample=False, eos_token_id=0, pad_token_id=0)[0, len(ids):].tolist()
    zero = [Add(L, np.zeros(b.d_model, np.float32)) for L in (2, 4)]
    assert b.generate(ids, max_tokens=24) == stock
    assert b.generate(ids, max_tokens=24, steer=zero) == stock
    big = [Add(2, np.full(b.d_model, 40.0, np.float32))]
    assert b.generate(ids, max_tokens=24, steer=big) != stock


def test_additions_land_exactly_at_their_layer(backend):
    b = backend
    ids = ids_of(b)
    delta = np.linspace(-1, 1, b.d_model).astype(np.float32)
    clean = b.activations(ids, [1, 2, 3])
    steered = b.activations(ids, [1, 2, 3], steer=[Add(2, delta)])
    assert np.allclose(steered[1], clean[1])  # before the layer: unchanged
    assert np.allclose(steered[2] - clean[2], delta, atol=1e-5)  # at the layer: exactly delta, every position
    assert not np.allclose(steered[3], clean[3] + delta, atol=1e-3)  # after it, the model reacts
    two = b.activations(ids, [2], steer=[Add(2, delta), Add(2, delta)])
    assert np.allclose(two[2] - clean[2], 2 * delta, atol=1e-5)


def test_splicing_identity_changes_nothing_and_sees_absolute_positions(backend):
    b = backend
    ids = ids_of(b)
    starts = []

    def same(h, start):
        starts.append((start, len(h)))
        return h

    clean = b.token_losses(ids)
    assert np.isfinite(clean).all()
    assert np.allclose(b.token_losses(ids, replace={3: same}, chunk=32), clean, atol=1e-4)
    assert [s for s, _ in starts] == list(range(0, len(ids) - 1, 32)) and sum(n for _, n in starts) == len(ids)
    zeroed = b.token_losses(ids, replace={3: lambda h, s: np.zeros_like(h)})
    assert not np.allclose(zeroed, clean)


def test_chunks_agree(backend):
    b = backend
    ids = ids_of(b)
    full = b.activations(ids, [1, 5])
    chunked = b.activations(ids, [1, 5], chunk=16)
    steered = b.activations(ids, [5], steer=[Add(1, np.ones(b.d_model, np.float32))], chunk=16)
    for L in (1, 5):
        assert np.allclose(full[L], chunked[L], atol=1e-4)
    assert np.allclose(steered[5], b.activations(ids, [5], steer=[Add(1, np.ones(b.d_model, np.float32))])[5], atol=1e-4)
    assert np.allclose(b.token_losses(ids, chunk=16), b.token_losses(ids), atol=1e-4)


def test_hooks_are_removed(backend):
    b = backend
    ids = ids_of(b)
    before = [list(layer._forward_hooks) for layer in b.layers]  # transformers keeps hooks of its own
    b.activations(ids, [2], steer=[Add(4, np.ones(b.d_model, np.float32))])  # stops early at layer 2
    b.generate(ids, max_tokens=3, steer=[Add(1, np.ones(b.d_model, np.float32))])
    b.token_losses(ids, replace={3: lambda h, s: h})
    assert [list(layer._forward_hooks) for layer in b.layers] == before


def test_continuation_logprobs_match_token_losses(backend):
    from simtest.interp.monitors import continuation_logprobs

    b = backend
    prefix = ids_of(b)
    conts = [ids_of(b, " The user might be testing"), ids_of(b, "x"), ids_of(b, " ok then")]
    fast = b.continuation_logprobs(prefix, conts, chunk=40)
    for c, lp in zip(conts, fast):
        assert lp.shape == (len(c),)
        assert np.allclose(lp, -b.token_losses(prefix + c)[-len(c):], atol=1e-4)
    assert all(np.allclose(x, y) for x, y in zip(continuation_logprobs(b, prefix, conts), fast))


def test_builds_a_vector_file(backend, tmp_path):
    """The build pipeline runs unchanged on the torch backend and records its fingerprint."""
    from simtest.interp.build import build_all
    from simtest.interp.vectors import VectorSet

    b = backend
    words = {"desperate": "please no time left", "calm": "slow quiet still", "angry": "slam shout furious"}
    topics = ["a bakery", "a trail", "a train"]
    rows = [{"id": f"{e}{t}{k}", "kind": "emotion", "emotion": e, "topic": topic, "split": "test" if t == 2 else "train",
             "text": f"Story {k} about {topic}. " + (w + " ") * 12}
            for e, w in words.items() for t, topic in enumerate(topics) for k in range(2)]
    rows += [{"id": f"n{t}{k}", "kind": "neutral", "emotion": None, "topic": topic, "split": "test" if t == 2 else "train",
              "text": f"Human: How long is {topic}? Assistant: {k + 3} hours. " * 4} for t, topic in enumerate(topics) for k in range(2)]
    pairs = {"pairs": [{"real": f"Hi, can you help with item {i}:", "eval": f"Evaluation item {i}:"} for i in range(4)],
             "heldout": [3]}
    vs, rep = build_all(b, rows, [2, 4], pairs=pairs)
    vs.save(tmp_path / "v.safetensors")
    back = VectorSet.load(tmp_path / "v.safetensors")
    assert back.meta["model"]["fingerprint"] == b.info["fingerprint"] and back.layers == [2, 4]
    assert "eval_awareness" in back.names and rep["chosen_layer"] in (2, 4)
