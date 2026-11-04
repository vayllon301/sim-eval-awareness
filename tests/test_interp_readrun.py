"""Reading a run: every exported episode once, on any backend; a run stopped half way resumes; a
reading says which episode and which vector file it came from; the server's own token IDs are checked
when they exist; a failure is listed, never dropped. On the fake tokenizer and backend."""

import json

import numpy as np
import pytest

from interp_helpers import sample
from simtest.interp import directions as D
from simtest.interp import readrun as RR
from simtest.interp.fake import FakeBackend, FakeTokenizer, render_fake_chat
from simtest.interp.meta import file_sha256
from simtest.interp.reader import Reading, completion_checks, render
from simtest.interp.sae import TopKSAE
from simtest.interp.vectors import VectorSet

LAYERS = [2, 4]
PHRASES = {"status": "draft", "source": "test", "groups": {"eval": ["This might be a test"], "control": ["Let me look"]}}


@pytest.fixture
def vectors(tmp_path):
    b = FakeBackend()
    rng = np.random.default_rng(0)
    vs = VectorSet(vectors={(n, L): D.unit(rng.standard_normal(b.d_model)) for n in ("eval_awareness", "train_awareness")
                            for L in LAYERS},
                   meta={"model": b.info, "layers": LAYERS, "d_model": b.d_model, "names": ["eval_awareness", "train_awareness"],
                         "offset": {"eval_awareness": {"2": 0.5, "4": 0.25}, "train_awareness": {"2": 0.0, "4": 0.0}},
                         "chosen_layer": 4})
    path = tmp_path / "vectors.safetensors"
    vs.save(path)
    return vs, path


def exported(tmp_path, uuids=("u1", "u2"), **sample_kw):
    from simtest.interp import episodes as E
    from test_interp_episodes import run_record

    tok = FakeTokenizer()

    def read(e):
        s = sample(uuid=e["uuid"], **sample_kw)
        for c in E.agent_calls(s):   # the server's counts, on the fake tokenizer's rendering
            rendered = E.from_inspect(s)
            msgs = rendered.messages[:len(c.call.request["messages"])]
            c.output.usage.input_tokens = len(tok.apply_chat_template(msgs, tools=rendered.tools, add_generation_prompt=True))
        return s

    records = [run_record(u, end_reason="no_tool_call") for u in uuids]
    design = {"conditions": ["C0"], "scenarios": ["flows_audit"], "epochs": 1}
    E.export_run(tmp_path / "logs" / "run", tmp_path / "eps", load=lambda d: (records, design), read=read)
    return tmp_path / "eps"


def run(tmp_path, vectors, parts=("proj",), backend=None, **kw):
    vs, path = vectors
    return RR.read_run(backend or FakeBackend(), tmp_path / "eps", tmp_path / "out", parts=parts, vectors=vs,
                       vectors_path=path, layers=LAYERS, **kw)


def test_every_episode_is_read_and_tied_to_its_vector_file(tmp_path, vectors):
    eps = exported(tmp_path)
    st = run(tmp_path, vectors)
    assert st["done"] == {"proj": 2} and st["failed"] == [] and st["complete"] is True
    r = Reading.load(tmp_path / "out" / "u1.npz")
    assert r.meta["episode"]["uuid"] == "u1" and r.meta["vectors_sha256"] == file_sha256(vectors[1])
    assert r.meta["episode_sha256"] == file_sha256(eps / "u1.json")
    assert r.names == ["eval_awareness", "train_awareness"] and r.layers == LAYERS
    assert r.proj.shape == (len(r.token_ids), 2, 2) and (r.prompt_end >= 0).all() and len(r.prompt_end) == 3
    assert not any(k.startswith("act") for k in np.load(tmp_path / "out" / "u1.npz").files)   # never raw activations


def test_prompt_end_and_turn_curves_are_in_the_summary_for_both_vectors(tmp_path, vectors):
    exported(tmp_path)
    run(tmp_path, vectors)
    s = json.loads((tmp_path / "out" / "u1.summary.json").read_text())
    r = Reading.load(tmp_path / "out" / "u1.npz")
    for name, off in (("eval_awareness", 0.25), ("train_awareness", 0.0)):
        rows = s["by_turn"][f"{name}@4"]
        assert [row["turn"] for row in rows] == [0, 1, 2]
        vi = r.names.index(name)
        for row in rows:
            at = r.prompt_end[row["turn"]]
            assert row["prompt_end"] == pytest.approx(float(r.proj[at, 1, vi]))
            assert row["prompt_end_centred"] == pytest.approx(row["prompt_end"] - off)
            assert row["agent_centred"] == pytest.approx(row["agent"] - off)


def test_a_stopped_run_resumes_where_it_stopped(tmp_path, vectors):
    exported(tmp_path, uuids=("u1", "u2", "u3"))
    b = FakeBackend()
    calls = []
    original = b.activations
    b.activations = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
    first = run(tmp_path, vectors, backend=b, limit=1)
    assert first["done"] == {"proj": 1} and first["complete"] is False and len(calls) == 1
    second = run(tmp_path, vectors, backend=b)
    assert second["done"] == {"proj": 3} and second["complete"] is True and len(calls) == 3


def test_the_budget_stops_before_an_episode_starts(tmp_path, vectors):
    exported(tmp_path, uuids=("u1", "u2", "u3"))
    t = [0.0]

    def clock():
        t[0] += 10
        return t[0]

    st = run(tmp_path, vectors, budget_s=25, clock=clock)
    assert st["complete"] is False and 0 < st["done"]["proj"] < 3


def test_resuming_with_other_settings_is_refused(tmp_path, vectors):
    exported(tmp_path)
    run(tmp_path, vectors, limit=1)
    vs, path = vectors
    with pytest.raises(RR.ReadError, match="other settings"):
        RR.read_run(FakeBackend(), tmp_path / "eps", tmp_path / "out", parts=("proj",), vectors=vs, vectors_path=path,
                    layers=[2])


def test_the_longest_episode_goes_first(tmp_path, vectors):
    """A memory failure then shows in the first minutes, not hours in."""
    exported(tmp_path)
    m = json.loads((tmp_path / "eps" / "manifest.json").read_text())
    m["episodes"][0]["tokens"], m["episodes"][1]["tokens"] = 10, 99
    (tmp_path / "eps" / "manifest.json").write_text(json.dumps(m))
    run(tmp_path, vectors, limit=1)
    assert [p.name for p in (tmp_path / "out").glob("*.npz")] == [f"{m['episodes'][1]['uuid']}.npz"]


def test_a_failed_episode_is_listed_and_the_rest_go_on(tmp_path, vectors):
    exported(tmp_path, uuids=("u1", "u2"))
    b = FakeBackend()
    original = b.activations

    def activations(ids, layers, **kw):
        if not hasattr(b, "failed"):
            b.failed = True
            raise RuntimeError("out of memory")
        return original(ids, layers, **kw)

    b.activations = activations
    st = run(tmp_path, vectors, backend=b)
    assert st["done"] == {"proj": 1} and st["complete"] is False
    assert len(st["failed"]) == 1 and "out of memory" in st["failed"][0]["error"]
    again = run(tmp_path, vectors)     # a failed episode is tried again, not skipped for ever
    assert again["done"] == {"proj": 2} and again["failed"] == []


def test_repeated_failures_stop_the_run(tmp_path, vectors):
    exported(tmp_path, uuids=tuple(f"u{i}" for i in range(6)))
    b = FakeBackend()
    b.activations = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("engine gone"))
    st = run(tmp_path, vectors, backend=b)
    assert st["stopped"] == "3 failures in a row" and len(st["failed"]) == 3


def test_the_logit_part_writes_its_own_file(tmp_path, vectors):
    exported(tmp_path)
    st = run(tmp_path, vectors, parts=("logit",), phrases=PHRASES, phrases_path="p.json")
    assert st["done"] == {"logit": 2}
    d = json.loads((tmp_path / "out" / "u1.logit.json").read_text())
    assert d["episode"]["uuid"] == "u1" and d["logit_monitor"]["status"] == "draft"
    assert len(d["logit_monitor"]["by_turn"]) == 3 and d["backend"]["engine"] == "fake"


def test_the_logit_part_is_refused_on_the_prompt_logprobs_path(tmp_path, vectors):
    """One pass over the prefix per turn and phrase: hundreds of millions of tokens on the pilot."""
    exported(tmp_path)
    with pytest.raises(RR.ReadError, match="scan"):
        run(tmp_path, vectors, parts=("logit",), phrases=PHRASES, phrases_path="p.json", require_scan=True)


def test_saes_keep_top_k_latents_only(tmp_path, vectors):
    exported(tmp_path, uuids=("u1",))
    b = FakeBackend()
    rng = np.random.default_rng(1)
    d, m = b.d_model, 40
    sae = TopKSAE(W_enc=rng.standard_normal((m, d)).astype(np.float32), b_enc=np.zeros(m, np.float32),
                  W_dec=rng.standard_normal((d, m)).astype(np.float32), b_dec=np.zeros(d, np.float32), k=5, info={"path": "s"})
    run(tmp_path, vectors, backend=b, saes={4: sae})
    r = Reading.load(tmp_path / "out" / "u1.npz")
    idx, val = r.sae[4]
    assert idx.shape == (len(r.token_ids), 5) and val.shape == idx.shape


# --- the IDs the server saw ------------------------------------------------------------------------

def test_the_servers_ids_are_checked_when_they_exist(tmp_path, vectors):
    eps = exported(tmp_path, uuids=("u1",))
    from simtest.interp.reader import load_episode, turn_prompts

    ep = load_episode(eps / "u1.json")
    good = turn_prompts(ep, FakeTokenizer())[-1]
    (eps / "u1.ids.json").write_text(json.dumps({"ids": good, "count": len(good)}))
    run(tmp_path, vectors)
    r = Reading.load(tmp_path / "out" / "u1.npz")
    assert r.meta["ids_check"] == {"source": "server", "n": len(good), "equal": True, "first_difference": None}


def test_without_them_the_reading_says_it_checked_lengths_only(tmp_path, vectors):
    exported(tmp_path, uuids=("u1",))
    run(tmp_path, vectors)
    r = Reading.load(tmp_path / "out" / "u1.npz")
    assert r.meta["ids_check"]["source"] == "lengths" and r.meta["ids_check"]["equal"] is True


def test_ids_that_differ_from_the_servers_are_refused(tmp_path, vectors):
    eps = exported(tmp_path, uuids=("u1",))
    from simtest.interp.reader import load_episode, turn_prompts

    good = turn_prompts(load_episode(eps / "u1.json"), FakeTokenizer())[-1]
    bad = good[:5] + [good[5] + 1] + good[6:]
    (eps / "u1.ids.json").write_text(json.dumps({"ids": bad, "count": len(bad)}))
    st = run(tmp_path, vectors)
    assert st["done"] == {"proj": 0} and "first differ at token 5" in st["failed"][0]["error"]


def test_lengths_that_differ_from_the_servers_are_kept_and_counted(tmp_path, vectors):
    eps = exported(tmp_path, uuids=("u1",))
    d = json.loads((eps / "u1.json").read_text())
    d["meta"]["prompt_tokens"][1] += 2
    (eps / "u1.json").write_text(json.dumps(d))
    st = run(tmp_path, vectors)
    assert st["done"] == {"proj": 1} and st["prompt_mismatches"] == {"u1": [1]}
    assert Reading.load(tmp_path / "out" / "u1.npz").meta["ids_check"]["equal"] is False


def test_server_ids_are_fetched_from_the_servers_own_renderer(tmp_path):
    eps = exported(tmp_path, uuids=("u1",))
    seen = []

    def post(url, body):
        seen.append((url, body))
        return {"tokens": [1, 2, 3], "count": 3, "max_model_len": 131072}

    st = RR.server_ids(eps, "http://localhost:18000/v1", post=post)
    assert st == {"fetched": 1, "kept": 0}
    url, body = seen[0]
    assert url == "http://localhost:18000/tokenize"      # the same renderer as chat completions (vLLM 0.32)
    assert body["model"] == "agent" and body["add_generation_prompt"] is True and body["tools"]
    assert body["messages"][-1]["role"] == "tool"        # the last request: every earlier prompt is its prefix
    assert json.loads((eps / "u1.ids.json").read_text())["ids"] == [1, 2, 3]
    assert RR.server_ids(eps, "http://localhost:18000", post=post) == {"fetched": 0, "kept": 1}


# --- generated tokens against the re-rendered turn ---------------------------------------------------

def test_completion_checks_compare_each_turn_with_what_was_generated():
    ids = [7, 7, 7, 1, 2, 9, 7, 7, 3, 9, 7]   # prompt ends at 2; turn 0 is 1, 2, EOS; prompt ends at 7; turn 1 is 3, EOS
    out = completion_checks(ids, np.array([2, 7]), [3, 3], frozenset({9}))
    assert out == {"n": 2, "equal": 1, "differences": {"-1": 1}}


def test_completion_checks_skip_what_they_cant_know():
    assert completion_checks([1, 2], np.array([0]), None, frozenset({9})) == {"n": 0, "equal": 0, "differences": {}}
    assert completion_checks([1, 2], np.array([-1]), [3], frozenset({2})) == {"n": 0, "equal": 0, "differences": {}}


# --- the sample of the SAE transfer check ------------------------------------------------------------

def test_the_transfer_sample_is_fixed_and_covers_each_condition():
    rows = [{"uuid": f"{c}{s}{e}", "condition": c, "scenario": s, "epoch": e, "scored": True, "status": "exported"}
            for c in ("C0", "C2", "C3") for s in ("a", "a_benign", "b") for e in (1, 2)]
    rows[0]["scored"] = False
    got = RR.transfer_sample(rows, 2)
    assert got == ["C0a_benign1", "C0b1", "C2a1", "C2a_benign1", "C3a1", "C3a_benign1"]
    assert RR.transfer_sample(list(reversed(rows)), 2) == got


# --- the first GPU run: do the reading paths agree on a real episode? --------------------------------

def noisy(backend, scale, seed=1):
    """The same model, with kernels that differ a little."""
    rng = np.random.default_rng(seed)
    real_acts, real_losses = backend.activations, backend.token_losses

    def activations(ids, layers, **kw):
        return {L: a + rng.normal(0, scale, a.shape).astype(np.float32) for L, a in real_acts(ids, layers, **kw).items()}

    def token_losses(ids, **kw):
        x = real_losses(ids, **kw)
        return x + rng.normal(0, scale, x.shape)

    backend.activations, backend.token_losses = activations, token_losses
    return backend


def checked(tmp_path, vectors, backend, name):
    eps = exported(tmp_path)
    from simtest.interp.reader import load_episode

    vs, _ = vectors
    return RR.path_check(backend, load_episode(eps / "u1.json"), vs, 4, PHRASES["groups"], label=name)


def test_a_path_check_says_what_it_took_and_read(tmp_path, vectors):
    a = checked(tmp_path, vectors, FakeBackend(), "a")
    assert a["label"] == "a" and a["engine"] == "fake" and a["tokens"] > 100 and a["path"] == "token_losses"
    assert a["seconds"]["pass"] >= 0 and a["seconds"]["phrases"] >= 0
    assert len(a["turns"]) == 3 and len(a["logprobs"]) == 3 * (20 + 11)   # both phrases, in characters, at each turn
    assert len(a["projection"]) == a["tokens"]


def test_two_identical_paths_agree_exactly(tmp_path, vectors):
    out = RR.compare_paths(checked(tmp_path, vectors, FakeBackend(), "a"), checked(tmp_path / "b" if (tmp_path / "b").mkdir() is None else None, vectors, FakeBackend(), "b"))
    assert out["projection_correlation"] == pytest.approx(1.0) and out["projection_rel_rmse"] == pytest.approx(0.0)
    assert out["logprob_rms"] == pytest.approx(0.0) and out["monitor_error_over_turn_sd"] == pytest.approx(0.0)


def test_noisy_kernels_show_in_the_comparison(tmp_path, vectors):
    a = checked(tmp_path, vectors, FakeBackend(), "a")
    b = checked(tmp_path / "b" if (tmp_path / "b").mkdir() is None else None, vectors, noisy(FakeBackend(), 0.5), "b")
    out = RR.compare_paths(a, b)
    assert 0 < out["projection_correlation"] < 1 and out["logprob_rms"] > 0.1


def test_paths_on_different_tokens_arent_compared(tmp_path, vectors):
    a = checked(tmp_path, vectors, FakeBackend(), "a")
    with pytest.raises(RR.ReadError, match="same tokens"):
        RR.compare_paths(a, {**a, "tokens": a["tokens"] + 1})
