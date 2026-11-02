"""explore/sim_quant.py: the pilot's first-hour measurement of fp8 against fp8_per_channel for the
simulator (docs/BACKENDS.md), offline. The speed sampler reads vLLM's /metrics text; the rule reads
the bake-off's scores, `simtest equiv` score files and the servers' own log lines. One test runs the
fidelity pass end to end against a local fake server."""

from __future__ import annotations

import http.server
import importlib.util
import json
import math
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from simtest import equiv as Q

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S = _load("sim_quant", ROOT / "explore" / "sim_quant.py")

METRICS = """# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="simulator"} 1200.0
vllm:prompt_tokens_total{engine="0",model_name="simulator"} 5.5e+04
vllm:num_requests_running{engine="0",model_name="simulator"} 12.0
vllm:num_requests_waiting{engine="0",model_name="simulator"} 0.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="simulator",reason="capacity"} 7.0
vllm:num_preemptions_total{engine="0",model_name="simulator"} 3.0
"""


# --- the speed sampler ----------------------------------------------------------------------------


def test_parse_metrics_sums_label_sets_and_keeps_names_apart():
    m = S.parse_metrics(METRICS + 'vllm:generation_tokens_total{engine="1",model_name="simulator"} 800.0\n')
    assert m["generation_tokens"] == 2000.0
    assert m["prompt_tokens"] == 55000.0
    assert m["running"] == 12.0
    assert m["waiting"] == 0.0            # not the by-reason gauge
    assert m["preemptions"] == 3.0


def _sample(t, gen, running=12, waiting=0, **extra):
    return {"t": t, "generation_tokens": gen, "running": running, "waiting": waiting, **extra}


def test_speed_uses_only_windows_with_12_running_and_none_waiting():
    samples = [_sample(0, 0, running=3), _sample(5, 100), _sample(10, 2100), _sample(15, 4600),
               _sample(20, 7100, waiting=2), _sample(25, 9000), _sample(30, 9100, running=4)]
    s = S.speed(samples, at=12, interval=5)
    assert s["windows"] == 2                     # 5-10 and 10-15; 15-20 and 20-25 have a queue
    assert s["median_tok_s"] == pytest.approx((400 + 500) / 2)
    assert s["overall_tok_s"] == pytest.approx(9100 / 30)


def test_speed_skips_gaps_restarts_and_errors():
    samples = [_sample(0, 0), _sample(5, 2000), _sample(100, 4000),        # a gap: not a window
               {"t": 101, "start": True}, _sample(105, 4000), _sample(110, 6000),
               {"t": 112, "error": "ConnectError"}, _sample(115, 100), _sample(120, 2600)]   # counter reset
    s = S.speed(samples, at=12, interval=5)
    assert s["windows"] == 3                     # 0-5, 105-110, 115-120
    assert s["median_tok_s"] == pytest.approx(400)


def test_speed_without_any_window_is_none():
    assert S.speed([_sample(0, 0, running=5), _sample(5, 100, running=5)], at=12, interval=5)["median_tok_s"] is None


# --- fidelity from the bake-off's scores ----------------------------------------------------------


def _scores(n=10, sim=0.9, nondet=(), truncated=(), errors=(), exact=None):
    rows = []
    for i in range(n):
        rows.append({"config": S.CONFIG, "mode": "forced", "seq": "s", "step": i, "command": f"c{i}",
                     "exact": (exact(i) if exact else sim == 1.0), "pytest": None,
                     "sim": 0.0 if i in truncated or i in errors else (sim(i) if callable(sim) else sim),
                     "len": 1.0, "tells": [], "nondet": i in nondet,
                     "finish": "length" if i in truncated else "stop", "error": "boom" if i in errors else None})
    return rows


def _keys(n):
    return [("s", i) for i in range(n)]


def test_fidelity_summary_scores_deterministic_steps_and_counts_failures():
    f = S.fidelity_summary(_scores(10, sim=lambda i: i / 10, nondet={9}, truncated={0}), _keys(10))
    assert f["complete"] and f["steps"] == 10 and f["deterministic"] == 9
    assert f["mean_sim"] == pytest.approx(sum(i / 10 for i in range(1, 9)) / 9)
    assert f["truncated"] == 1 and f["errors"] == 0


def test_fidelity_summary_is_incomplete_with_missing_or_errored_steps():
    assert not S.fidelity_summary(_scores(9), _keys(10))["complete"]
    assert not S.fidelity_summary(_scores(10, errors={3}), _keys(10))["complete"]
    assert not S.fidelity_summary(_scores(10), [("other", i) for i in range(10)])["complete"]   # not the same steps
    assert not S.fidelity_summary(_scores(12), _keys(10))["complete"]                          # steps it never had
    other = [dict(r, config="agentworld-think-vllm") for r in _scores(10)]   # another config's rows
    assert S.fidelity_summary(other, _keys(10))["steps"] == 0


def test_paired_difference_over_common_deterministic_steps():
    a = _scores(10, sim=0.8, nondet={0})
    b = _scores(10, sim=lambda i: 0.8 + (0.1 if i % 2 else -0.1), nondet={0})
    d = S.paired(a, b)
    assert d["n"] == 9
    assert d["mean"] == pytest.approx((5 * 0.1 - 4 * 0.1) / 9)
    assert d["se"] > 0


# --- the servers' own lines -----------------------------------------------------------------------


def _server(scheme, quant_flag, layers="340", backend="TRITON", stray=None):
    kinds = ["attn.qkv: 10", "linear_attn.in_proj_ba: 30", "linear_attn.in_proj_qkvz: 30", "mlp.experts: 40"]
    parts = [f"{k} (from {'moe' if 'experts' in k else 'linear'}: {stray if stray and i == 1 else scheme})"
             for i, k in enumerate(kinds)]
    return (f"INFO [base_loader.py:151] Quantizing {layers} layers of types: " + "; ".join(parts) + "\n"
            f"INFO [fp8.py:525] Using {backend} Fp8 MoE backend out of potential backends: ['AITER', 'TRITON']\n"
            "INFO [__init__.py:797] Selected CutlassFP8ScaledMMLinearKernel for Fp8PtpcOnlineLinearMethod\n"
            "vllm serve /content/models/simulator --served-model-name simulator --host 127.0.0.1 --port 8001 "
            f"--gpu-memory-utilization 0.52 --max-model-len 131072 --max-num-seqs 64 --language-model-only "
            f"--reasoning-parser qwen3 {quant_flag} --trust-remote-code\n")


PT = _server("fp8_per_tensor_static", "--quantization fp8")
PC = _server("fp8_per_channel_static", "--quantization fp8_per_channel --moe-backend triton")


def test_server_check_accepts_each_setting_as_served():
    assert S.server_check(PT, "fp8") == []
    assert S.server_check(PC, "fp8_per_channel") == []
    assert S.layer_count(PT) == 340


@pytest.mark.parametrize("text,setting,why", [
    (PT, "fp8_per_channel", "fp8_per_channel"),                                  # the wrong server
    (PC, "fp8", "fp8"),
    (_server("fp8_per_channel_static", "--quantization fp8_per_channel --moe-backend triton",
             stray="fp8_per_tensor_static"), "fp8_per_channel", "per_tensor"),  # one layer kind not per channel
    (_server("fp8_per_channel_static", "--quantization fp8_per_channel --moe-backend triton", backend="VLLM_CUTLASS"),
     "fp8_per_channel", "TRITON"),
    ("", "fp8", "Quantizing"),
])
def test_server_check_refuses_anything_else(text, setting, why):
    problems = S.server_check(text, setting)
    assert problems and any(why in p for p in problems), problems


# --- the noise floors from simtest equiv ----------------------------------------------------------


def _row(cid, model, n=50, drift=0.0, seed=0, sha="t", start=40):
    lp = [None] + [-1.0 - 0.01 * ((k * 7 + seed) % 13) + drift * math.sin(k + seed) for k in range(1, n)]
    top1 = [None] + [k % 5 for k in range(1, n)]
    return {"kind": "score", "id": cid, "model": model, "n_tokens": n, "tokens_sha256": sha + cid,
            "truncated": False, "response_start": start, "prompt_tokens_logged": start, "prompt_matches_log": True,
            "seconds": 0.1, "logprob": lp, "top1": top1}


def _write_equiv(out: Path, setting: str, drift: float, contexts=6, agent=True, contexts_sha="sha", conc12=12,
                 served="simulator", version="0.32.0", label=None, **kw):
    for conc, d in ((1, 0.0), (conc12, drift)):
        rows = [_row(f"s{i}", "simulator", drift=d, seed=i, **kw) for i in range(contexts)]
        if agent:
            rows += [_row(f"a{i}", "agent", drift=d / 10, seed=i) for i in range(2)]
        name = f"{setting}-c{12 if conc == conc12 else 1}.jsonl"
        servers = {"simulator": {"url": "http://localhost:18001/v1", "served_model": served, "models": [served],
                                 "max_model_len": 131072, "version": version}}
        Q.write_scores(out / "equiv" / name, {"label": label or f"G4, session 6, {setting}", "contexts_sha256": contexts_sha,
                                              "concurrency": conc, "top_k": 1, "max_tokens": None, "servers": servers},
                       rows, force=True)


def test_noise_floors_compare_the_simulator_of_both_settings(tmp_path):
    _write_equiv(tmp_path, "fp8", drift=0.05)
    _write_equiv(tmp_path, "fp8_per_channel", drift=0.01, agent=False)    # the agent isn't rescored
    n = S.noise(tmp_path)
    assert n["problems"] == []
    assert n["fp8"]["mean_abs_dlogprob"] > n["fp8_per_channel"]["mean_abs_dlogprob"] > 0
    lo, hi = n["bootstrap_pc_minus_pt"]["mean_abs_dlogprob"]
    assert lo <= hi < 0
    assert n["agent_fp8"]["contexts"] == 2       # reported for the record: the reference's agent floor


@pytest.mark.parametrize("break_it,why", [
    (lambda out: (out / "equiv" / "fp8_per_channel-c12.jsonl").unlink(), "missing"),
    (lambda out: _write_equiv(out, "fp8_per_channel", drift=0.01, contexts_sha="other"), "contexts"),
    (lambda out: _write_equiv(out, "fp8_per_channel", drift=0.01, contexts=5), "context"),
    (lambda out: _write_equiv(out, "fp8_per_channel", drift=0.01, conc12=8), "concurrency"),
])
def test_noise_floors_refuse_incomplete_or_mismatched_files(tmp_path, break_it, why):
    _write_equiv(tmp_path, "fp8", drift=0.05)
    _write_equiv(tmp_path, "fp8_per_channel", drift=0.01)
    break_it(tmp_path)
    problems = S.noise(tmp_path)["problems"]
    assert problems and any(why in p for p in problems), problems


@pytest.mark.parametrize("kw,why", [
    (dict(served="agent"), "served"),               # scored the agent's server as the simulator
    (dict(version="0.31.0"), "0.32.0"),            # another vLLM
    (dict(label="G4, session 6, fp8"), "label"),   # labelled as the other setting
])
def test_noise_floors_refuse_files_from_the_wrong_server_or_setting(tmp_path, kw, why):
    _write_equiv(tmp_path, "fp8", drift=0.05)
    _write_equiv(tmp_path, "fp8_per_channel", drift=0.01, **kw)
    problems = S.noise(tmp_path)["problems"]
    assert problems and any(why in p for p in problems), problems


def test_noise_floors_refuse_different_token_ids(tmp_path):
    _write_equiv(tmp_path, "fp8", drift=0.05)
    _write_equiv(tmp_path, "fp8_per_channel", drift=0.01, sha="T")       # same contexts file, other token IDs
    assert any("token" in p for p in S.noise(tmp_path)["problems"])


# --- the DRAFT rule -------------------------------------------------------------------------------


def _fid(mean_sim, truncated=0, complete=True):
    return {"complete": complete, "mean_sim": mean_sim, "truncated": truncated, "steps": 84, "errors": 0}


def _noise(pt=0.05, pc=0.01, boot=(-0.05, -0.03), top_pt=0.97, top_pc=0.99, problems=()):
    return {"problems": list(problems), "fp8": {"mean_abs_dlogprob": pt, "top1_agreement": top_pt},
            "fp8_per_channel": {"mean_abs_dlogprob": pc, "top1_agreement": top_pc},
            "bootstrap_pc_minus_pt": {"mean_abs_dlogprob": list(boot), "top1_agreement": [0.0, 0.03]}}


def _rule(fid_pt=None, fid_pc=None, noise=None, server=(), speed_ratio=1.0):
    return S.rule({"fp8": fid_pt or _fid(0.91), "fp8_per_channel": fid_pc or _fid(0.91)},
                  noise or _noise(), list(server), speed_ratio)


def test_rule_adopts_when_fidelity_holds_and_the_noise_is_clearly_lower():
    r = _rule()
    assert r["verdict"] == "ADOPT"
    assert "12" in r["pin"]
    assert not r["confirm_budget"]


@pytest.mark.parametrize("pc_sim,verdict", [(0.89, "ADOPT"), (0.8901, "ADOPT"), (0.8899, "KEEP"), (0.85, "KEEP"),
                                            (0.95, "ADOPT")])
def test_rule_fidelity_tolerance_is_002(pc_sim, verdict):
    assert _rule(fid_pc=_fid(pc_sim))["verdict"] == verdict


def test_rule_keeps_fp8_on_too_many_truncations():
    assert _rule(fid_pc=_fid(0.91, truncated=3))["verdict"] == "ADOPT"
    assert _rule(fid_pc=_fid(0.91, truncated=4))["verdict"] == "KEEP"


@pytest.mark.parametrize("noise", [
    _noise(pc=0.06, boot=(0.0, 0.02)),            # not lower
    _noise(pc=0.04, boot=(-0.02, 0.001)),         # lower, but the interval reaches zero
    _noise(top_pc=0.96),                          # top-1 agreement lower
])
def test_rule_keeps_fp8_unless_the_noise_is_clearly_lower(noise):
    assert _rule(noise=noise)["verdict"] == "KEEP"


@pytest.mark.parametrize("kw", [
    dict(fid_pc=_fid(0.91, complete=False)), dict(fid_pt=_fid(0.91, complete=False)),
    dict(noise=_noise(problems=["fp8_per_channel-c12: missing"])), dict(server=["fp8: no Quantizing line"]),
])
def test_rule_is_incomplete_and_keeps_fp8_without_complete_evidence(kw):
    r = _rule(**kw)
    assert r["verdict"] == "INCOMPLETE"
    assert "fp8" in r["action"]


def test_rule_asks_victor_when_per_channel_is_much_slower():
    assert _rule(speed_ratio=0.85)["confirm_budget"]
    assert not _rule(speed_ratio=0.95)["confirm_budget"]
    assert not _rule(fid_pc=_fid(0.5), speed_ratio=0.5)["confirm_budget"]   # kept: nothing to confirm


# --- the bake-off pieces it runs ------------------------------------------------------------------


def test_the_12_at_once_config_is_the_g4_bakeoff_config_at_12():
    bs = _load("bakeoff_simulate", ROOT / "explore" / "bakeoff_simulate.py")
    a, b = bs.CONFIGS["agentworld-think-vllm"], bs.CONFIGS[S.CONFIG]
    assert {**a, "parallel": 12} == b


def _bakeoff_dir(d: Path, steps=3):
    d.mkdir(parents=True)
    (d / "snapshot.txt").write_text("hostname: dev\n")
    real = [{"seq": "s1", "step": i, "command": f"echo {i}", "output": f"{i}\n", "output_b": f"{i}\n",
             "exit_code": 0, "exit_code_b": 0} for i in range(steps)]
    (d / "real.jsonl").write_text("".join(json.dumps(r) + "\n" for r in real))
    return real


def test_bakeoff_score_runs_on_vllm_rows_without_tok_s(tmp_path):
    """The speed table failed on vLLM rows (no tok_s; LOG.md, 29-10-2026), after the fidelity tables."""
    real = _bakeoff_dir(tmp_path / "b")
    rows = [{"config": S.CONFIG, "model": "simulator", "fmt": "common", "mode": "forced", "seq": "s1", "step": r["step"],
             "command": r["command"], "output": r["output"], "no_tag": False, "prompt_tokens": 10,
             "completion_tokens": 5, "reasoning_tokens": None, "ttft_s": None, "tok_s": None, "wall_s": 1.0,
             "finish": "stop", "error": None} for r in real]
    (tmp_path / "b" / f"sim-{S.CONFIG}-forced.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = subprocess.run([sys.executable, "-I", str(ROOT / "explore" / "bakeoff_score.py"), str(tmp_path / "b")],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    assert "## Speed" in p.stdout


class _Fake(http.server.BaseHTTPRequestHandler):
    """A simulator: echoes each step's real output (so every step is exact), and a /metrics page."""
    real: dict = {}
    calls = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404)
            return
        body = METRICS.replace("1200.0", str(1200.0 + 100 * _Fake.calls)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.calls += 1
        command = json.loads(req["messages"][-1]["content"].removeprefix("Tool call: "))["arguments"]["command"]
        out = {"choices": [{"message": {"content": _Fake.real[command], "reasoning_content": "thinking"},
                            "finish_reason": "stop"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        body = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


def _pin(monkeypatch, d: Path):
    import hashlib
    monkeypatch.setattr(S, "RECORDING", {n: hashlib.sha256((d / n).read_bytes()).hexdigest()
                                         for n in ("real.jsonl", "snapshot.txt")})


def test_the_pinned_recording_is_the_bakeoffs():
    assert set(S.RECORDING) == {"real.jsonl", "snapshot.txt"}
    assert S.RECORDING["real.jsonl"].startswith("c5fe28a1")    # logs/bakeoff/real.jsonl, 84 steps (26-10-2026)


def test_fidelity_pass_end_to_end_against_a_fake_server(tmp_path, monkeypatch):
    real = _bakeoff_dir(tmp_path / "bakeoff")
    _pin(monkeypatch, tmp_path / "bakeoff")
    _Fake.real = {r["command"]: r["output"] for r in real}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        out = tmp_path / "simquant"
        code = S.main(["fidelity", "fp8", "--out", str(out), "--bakeoff", str(tmp_path / "bakeoff"), "--url", url,
                       "--interval", "0.05"])
        assert code == 0
        d = out / "fp8"
        assert (d / "real.jsonl").read_bytes() == (tmp_path / "bakeoff" / "real.jsonl").read_bytes()
        sims = [json.loads(x) for x in (d / f"sim-{S.CONFIG}-forced.jsonl").read_text().splitlines()]
        assert len(sims) == 3 and all(r["error"] is None for r in sims)
        scores = [json.loads(x) for x in (d / "scores.jsonl").read_text().splitlines()]
        assert all(r["exact"] for r in scores)
        samples = [json.loads(x) for x in (d / "speed.jsonl").read_text().splitlines()]
        assert samples[0].get("start") and any("generation_tokens" in s for s in samples)
        # a second pass resumes: nothing is asked again, and another bake-off copy is refused
        before = _Fake.calls
        assert S.main(["fidelity", "fp8", "--out", str(out), "--bakeoff", str(tmp_path / "bakeoff"), "--url", url,
                       "--interval", "0.05"]) == 0
        assert _Fake.calls == before         # every step was done: no request is sent again
        (tmp_path / "bakeoff" / "real.jsonl").write_text("{}\n")
        with pytest.raises(SystemExit, match="real.jsonl"):
            S.main(["fidelity", "fp8", "--out", str(out), "--bakeoff", str(tmp_path / "bakeoff"), "--url", url])
    finally:
        server.shutdown()


def _decide_dirs(out: Path, monkeypatch, steps=10):
    recording = out.parent / "recording"
    _bakeoff_dir(recording, steps=steps)
    _pin(monkeypatch, recording)
    for setting, sim, text in (("fp8", 0.91, PT), ("fp8_per_channel", 0.905, PC)):
        d = out / setting
        d.mkdir(parents=True)
        for n in ("real.jsonl", "snapshot.txt"):
            (d / n).write_bytes((recording / n).read_bytes())
        rows = [dict(r, seq="s1") for r in _scores(steps, sim=sim)]
        (d / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (d / "speed.jsonl").write_text("".join(json.dumps(_sample(5 * i, 2000 * i)) + "\n" for i in range(5)))
        (d / "server.txt").write_text(text)
    _write_equiv(out, "fp8", drift=0.05)
    _write_equiv(out, "fp8_per_channel", drift=0.01, agent=False)


def test_decide_is_incomplete_unless_both_passes_predicted_the_pinned_recording(tmp_path, monkeypatch):
    out = tmp_path / "simquant"
    _decide_dirs(out, monkeypatch)
    (out / "fp8_per_channel" / "snapshot.txt").write_text("another machine\n")
    assert S.main(["decide", "--out", str(out)]) == 0
    r = json.loads((out / "decision.json").read_text())["rule"]
    assert r["verdict"] == "INCOMPLETE"
    assert any("snapshot.txt" in p for p in r["problems"])


def test_decide_is_incomplete_when_a_pass_scored_other_steps(tmp_path, monkeypatch):
    out = tmp_path / "simquant"
    _decide_dirs(out, monkeypatch)
    rows = [dict(r, seq="unrelated") for r in _scores(10, sim=0.95)]
    (out / "fp8_per_channel" / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert S.main(["decide", "--out", str(out)]) == 0
    assert json.loads((out / "decision.json").read_text())["rule"]["verdict"] == "INCOMPLETE"


def test_decide_end_to_end_writes_the_decision(tmp_path, monkeypatch):
    out = tmp_path / "simquant"
    _decide_dirs(out, monkeypatch)
    assert S.main(["decide", "--out", str(out)]) == 0
    d = json.loads((out / "decision.json").read_text())
    assert d["rule"]["verdict"] == "ADOPT"
    assert d["fidelity"]["fp8_per_channel"]["mean_sim"] == pytest.approx(0.905)
    assert d["speed"]["fp8"]["median_tok_s"] == pytest.approx(400)


def test_graded_fidelity_uses_the_grader_when_this_checkout_has_it(tmp_path, monkeypatch):
    """Reported only (not in the rule). With a stand-in grader: each forced step's context holds the
    snapshot and every earlier real output (forced mode gave the simulator those)."""
    import types

    seen_systems = []

    class Context:
        def __init__(self, system, condition):
            seen_systems.append(system)
            assert condition == "C3"

        def add(self, command, seen):
            assert seen == ""          # the simulator's own outputs are never "given"

        def for_step(self, command, real):
            return self

    fake = types.SimpleNamespace(Context=Context, FAITHFUL=frozenset({"exact"}),
                                 grade_step=lambda cmd, real, seen, ctx: {"grade": "exact" if seen == real else "x"})
    monkeypatch.setitem(sys.modules, "simtest.post.fidelity", fake)
    out = tmp_path / "simquant"
    for setting in S.SETTINGS:
        d = out / setting
        real = _bakeoff_dir(d, steps=3)
        sims = [{"config": S.CONFIG, "mode": "forced", "seq": "s1", "step": r["step"],
                 "output": r["output"] if setting == "fp8" or r["step"] else "wrong", "error": None, "finish": "stop"}
                for r in real]
        (d / f"sim-{S.CONFIG}-forced.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sims))
        (d / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _scores(3)))
    g = S.graded_fidelity(out)
    assert g["available"]
    assert g["fp8"]["faithful"] == 1.0 and g["fp8_per_channel"]["faithful"] == pytest.approx(2 / 3)
    assert "0\n" in seen_systems[2] and "1\n" in seen_systems[2]   # step 2 was given steps 0 and 1


def test_graded_fidelity_without_the_grader_is_reported_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "simtest.post.fidelity", None)    # import fails, as on this branch
    g = S.graded_fidelity(tmp_path)
    assert g == {"available": False, "why": g["why"]} and g["why"]
