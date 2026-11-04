"""explore/sim_quant.py: the simulator's setup check on fp8_per_channel (docs/BACKENDS.md), offline.
The speed sampler reads vLLM's /metrics text; the rule reads the bake-off's scores, `simtest equiv`
score files and the server's own lines; the gate is the last check before `simtest run` and writes
the scheme into the run's record. One test runs the forced pass end to end against a local fake
server."""

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


# --- the server's own lines -----------------------------------------------------------------------


GPU = "NVIDIA H100 80GB HBM3"


def _server(scheme, quant_flag, layers="340", backend="TRITON", stray=None, gpu=GPU):
    kinds = ["attn.qkv: 10", "linear_attn.in_proj_ba: 30", "linear_attn.in_proj_qkvz: 30", "mlp.experts: 40"]
    parts = [f"{k} (from {'moe' if 'experts' in k else 'linear'}: {stray if stray and i == 1 else scheme})"
             for i, k in enumerate(kinds)]
    return (f"gpu: {gpu}\n"
            "vllm serve /content/models/simulator --served-model-name simulator --host 127.0.0.1 --port 8001 "
            f"--gpu-memory-utilization 0.52 --max-model-len 131072 --max-num-seqs 64 --language-model-only "
            f"--reasoning-parser qwen3 {quant_flag} --trust-remote-code\n"
            f"INFO [base_loader.py:151] Quantizing {layers} layers of types: " + "; ".join(parts) + "\n"
            f"INFO [fp8.py:525] Using {backend} Fp8 MoE backend out of potential backends: ['AITER', 'TRITON']\n"
            "INFO [__init__.py:797] Selected CutlassFP8ScaledMMLinearKernel for Fp8PtpcOnlineLinearMethod\n")


PT = _server("fp8_per_tensor_static", "--quantization fp8")
PC = _server("fp8_per_channel_static", "--quantization fp8_per_channel --moe-backend triton")


def test_server_check_accepts_the_final_scheme_as_served():
    assert S.server_check(PC) == ([], [])
    assert S.layer_count(PC) == 340
    assert S.gpu_name(PC) == GPU


@pytest.mark.parametrize("text,why", [
    (PT, "fp8_per_channel"),                                                     # the old scheme
    (_server("fp8_per_channel_static", "--quantization fp8_per_channel --moe-backend triton",
             stray="fp8_per_tensor_static"), "per_tensor"),                      # one layer kind not per channel
    (_server("fp8_per_channel_static", "--quantization fp8_per_channel --moe-backend triton", backend="VLLM_CUTLASS"),
     "TRITON"),                                                                  # the H100's default kernel
    (_server("fp8_per_channel_static", "--quantization fp8_per_channel"), "triton"),   # the kernel not pinned
])
def test_lines_that_show_something_else_are_wrong(text, why):
    missing, wrong = S.server_check(text)
    assert wrong and any(why in p for p in wrong), wrong


@pytest.mark.parametrize("text,why", [
    ("", "command line"),
    ("", "Quantizing"),
    ("", "MoE"),
    (PC.replace("gpu: ", "nothing: "), "GPU"),
])
def test_lines_that_are_absent_are_missing_not_wrong(text, why):
    missing, wrong = S.server_check(text)
    assert not wrong and any(why in p for p in missing), missing


# --- the noise floors from simtest equiv ----------------------------------------------------------


def _row(cid, model, n=50, drift=0.0, seed=0, sha="t", start=40):
    lp = [None] + [-1.0 - 0.01 * ((k * 7 + seed) % 13) + drift * math.sin(k + seed) for k in range(1, n)]
    top1 = [None] + [k % 5 for k in range(1, n)]
    return {"kind": "score", "id": cid, "model": model, "n_tokens": n, "tokens_sha256": sha + cid,
            "truncated": False, "response_start": start, "prompt_tokens_logged": start, "prompt_matches_log": True,
            "seconds": 0.1, "logprob": lp, "top1": top1}


def _write_scores(out: Path, c: int, drift=0.0, concurrency=None, contexts=6, agent=True, contexts_sha="sha",
                  served="simulator", version="0.32.0", label="G4, session 6, fp8_per_channel", sha="t"):
    rows = [_row(f"s{i}", "simulator", drift=drift, seed=i, sha=sha) for i in range(contexts)]
    srv = {"url": "http://localhost:18001/v1", "served_model": served, "models": [served], "max_model_len": 131072,
           "version": version}
    servers = {"simulator": srv}
    if agent:
        rows += [_row(f"a{i}", "agent", drift=drift / 10, seed=i, sha=sha) for i in range(2)]
        servers["agent"] = {**srv, "url": "http://localhost:18000/v1", "served_model": "agent", "models": ["agent"]}
    Q.write_scores(out / f"equiv-c{c}.jsonl", {"label": label, "contexts_sha256": contexts_sha,
                                               "concurrency": concurrency or c, "top_k": 1, "max_tokens": None,
                                               "servers": servers}, rows, force=True)


def _write_equiv(out: Path, drift=0.02, **at12):
    _write_scores(out, 1)
    _write_scores(out, 12, drift=drift, **at12)


def test_noise_floors_cover_both_models(tmp_path):
    _write_equiv(tmp_path)
    n = S.noise(tmp_path)
    assert n["problems"] == []
    assert n["simulator"]["contexts"] == 6 and n["agent"]["contexts"] == 2
    assert n["simulator"]["mean_abs_dlogprob"] > n["agent"]["mean_abs_dlogprob"] > 0
    assert n["simulator"]["top1_agreement"] == 1.0


@pytest.mark.parametrize("kw,why", [
    (dict(contexts_sha="other"), "contexts"),            # another contexts file
    (dict(contexts=5), "missing"),                       # a context not scored at 12
    (dict(concurrency=8), "concurrency"),                # not 12 at once
    (dict(served="agent"), "served"),                    # scored the agent's server as the simulator
    (dict(version="0.31.0"), "0.32.0"),                  # another vLLM
    (dict(label="G4, session 6, fp8"), "label"),         # labelled as the old scheme
    (dict(sha="T"), "token"),                            # other token IDs
])
def test_noise_floors_refuse_incomplete_or_mismatched_files(tmp_path, kw, why):
    _write_equiv(tmp_path, **kw)
    problems = S.noise(tmp_path)["problems"]
    assert problems and any(why in p for p in problems), problems


def test_noise_floors_need_both_files(tmp_path):
    _write_equiv(tmp_path)
    (tmp_path / "equiv-c12.jsonl").unlink()
    assert any("missing" in p for p in S.noise(tmp_path)["problems"])


def test_noise_floors_need_both_models(tmp_path):
    _write_scores(tmp_path, 1, agent=False)
    _write_scores(tmp_path, 12, drift=0.02, agent=False)
    assert any("agent" in p for p in S.noise(tmp_path)["problems"])


# --- the rule ---------------------------------------------------------------------------------------


def _fid(mean_sim=0.905, truncated=0, complete=True):
    return {"complete": complete, "mean_sim": mean_sim, "truncated": truncated, "steps": 84, "real_steps": 84,
            "errors": 0}


def _noise(problems=(), mean=0.01):
    floor = {"mean_abs_dlogprob": mean, "top1_agreement": 0.99}
    return {"problems": list(problems), "agent": floor, "simulator": floor}


def _rule(fid=None, noise=None, missing=(), wrong=(), tok_s=500.0):
    return S.rule(fid or _fid(), noise or _noise(), list(missing), list(wrong), tok_s)


def test_the_bars_are_fixed_numbers_from_the_earlier_passes():
    """No fp8 pass runs in the session, so the bars come from the G4's 29-10 pass (fp8, 0.9124 on the 81
    deterministic steps, 0 at the cap) and the rates the 02-11 estimate used."""
    assert S.SIM_FLOOR == pytest.approx(0.912 - 0.02)
    assert S.MAX_TRUNCATED == 3
    assert S.REFERENCE_TOK_S * S.SPEED_CONFIRM == pytest.approx(405)


def test_rule_passes_when_every_bar_holds():
    r = _rule()
    assert r["verdict"] == "PASS" and not r["confirm_budget"]
    assert "12" in r["pin"]


@pytest.mark.parametrize("sim,verdict", [(0.892, "PASS"), (0.8921, "PASS"), (0.8919, "FAIL"), (0.80, "FAIL"), (0.95, "PASS")])
def test_rule_similarity_floor(sim, verdict):
    assert _rule(fid=_fid(sim))["verdict"] == verdict


def test_rule_truncations():
    assert _rule(fid=_fid(truncated=3))["verdict"] == "PASS"
    assert _rule(fid=_fid(truncated=4))["verdict"] == "FAIL"


def test_rule_has_no_bar_on_the_size_of_the_noise_floors():
    """There is no fp8 floor to compare with: the floors are recorded, and only their completeness counts."""
    assert _rule(noise=_noise(mean=5.0))["verdict"] == "PASS"


@pytest.mark.parametrize("kw", [dict(fid=_fid(complete=False)), dict(noise=_noise(problems=["equiv-c12.jsonl: missing"])),
                                dict(missing=["no 'Quantizing ... layers' line"])])
def test_rule_is_incomplete_without_complete_evidence(kw):
    r = _rule(**kw)
    assert r["verdict"] == "INCOMPLETE" and r["problems"] and not r["confirm_budget"]
    assert "no episode" in r["action"]


def test_a_wrong_server_fails_whatever_else_is_missing():
    r = _rule(fid=_fid(complete=False), noise=_noise(problems=["equiv-c1.jsonl: missing"]),
              missing=["no simulator contexts"], wrong=["layers quantized as ['fp8_per_tensor_static']"])
    assert r["verdict"] == "FAIL" and r["checks"]["scheme"] is False
    assert any("per_tensor" in p for p in r["problems"])


@pytest.mark.parametrize("kw", [dict(fid=_fid(0.80)), dict(fid=_fid(truncated=9)), dict(wrong=["not per channel"]),
                                dict(missing=["no lines"])])
def test_no_verdict_sends_the_session_back_to_fp8(kw):
    r = _rule(**kw)
    assert "no episode" in r["action"] and "SIM_QUANT=fp8" not in r["action"]


@pytest.mark.parametrize("tok_s,confirm", [(405.0, False), (404.9, True), (None, True), (900.0, False)])
def test_speed_never_decides_but_a_low_rate_asks_for_the_budget(tok_s, confirm):
    r = _rule(tok_s=tok_s)
    assert r["verdict"] == "PASS" and r["confirm_budget"] is confirm


def test_a_failed_check_has_no_budget_to_confirm():
    assert not _rule(fid=_fid(0.5), tok_s=100.0)["confirm_budget"]


# --- the gate: the last check before simtest run ------------------------------------------------------


HEAD = "0123456789abcdef"


def _passed(**kw):
    return {"verdict": "PASS", "commit": HEAD, "scheme": S.SCHEME, "gpu": GPU, "rule": {"confirm_budget": False},
            "fidelity": {"mean_sim": 0.905, "truncated": 1}, "speed": {"median_tok_s": 450.0}, **kw}


def test_gate_lets_a_passed_setup_through():
    assert S.gate_problems(_passed(), PC, HEAD, False) == []


@pytest.mark.parametrize("check,text,head,why", [
    (None, PC, HEAD, "no setup check"),
    (_passed(verdict="FAIL"), PC, HEAD, "FAIL"),
    (_passed(verdict="INCOMPLETE"), PC, HEAD, "INCOMPLETE"),
    (_passed(commit="fedcba"), PC, HEAD, "commit"),
    (_passed(), PC, None, "commit"),
    (_passed(scheme="fp8"), PC, HEAD, "scheme"),
    (_passed(), PT, HEAD, "fp8_per_channel"),                    # this job's server runs the old scheme
    (_passed(), PC.replace(GPU, "NVIDIA A100 80GB"), HEAD, "one backend"),   # another card than the check's
    (_passed(), "", HEAD, "Quantizing"),                         # this job's lines are gone
], ids=["no-check", "failed", "incomplete", "other-commit", "no-head", "other-scheme", "fp8-server", "other-gpu", "no-lines"])
def test_gate_refuses(check, text, head, why):
    problems = S.gate_problems(check, text, head, False)
    assert problems and any(why in p for p in problems), problems


def test_gate_holds_a_low_speed_until_victor_confirms():
    held = _passed(rule={"confirm_budget": True})
    problems = S.gate_problems(held, PC, HEAD, False)
    assert problems and any("SPEED_OK" in p for p in problems)
    assert S.gate_problems(held, PC, HEAD, True) == []


def test_a_confirmation_never_passes_anything_else():
    assert S.gate_problems(_passed(verdict="FAIL", rule={"confirm_budget": True}), PC, HEAD, True)


def _gate_files(tmp_path, check=None, text=PC):
    (tmp_path / "check.json").write_text(json.dumps(check or _passed()))
    (tmp_path / "server.txt").write_text(text)
    return ["gate", "--out", str(tmp_path), "--server", str(tmp_path / "server.txt"),
            "--run-log", str(tmp_path / "logs" / "n2-pilot"), "--where", "Hopper, job 1001 on gpu07"]


def test_gate_writes_the_scheme_in_the_runs_record_once_per_start(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_commit", lambda: HEAD)
    args = _gate_files(tmp_path)
    assert S.main(args) == 0
    assert S.main(args) == 0
    rows = [json.loads(x) for x in (tmp_path / "logs" / "n2-pilot" / "backend.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    r = rows[0]
    assert r["scheme"] == "fp8_per_channel" and r["gpu"] == GPU and r["commit"] == HEAD
    assert r["where"] == "Hopper, job 1001 on gpu07"
    assert "--quantization fp8_per_channel --moe-backend triton" in r["served"]
    assert r["setup_check"]["verdict"] == "PASS" and r["setup_check"]["mean_sim"] == 0.905


def test_a_refused_gate_returns_an_error_and_writes_no_record(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S, "_commit", lambda: HEAD)
    assert S.main(_gate_files(tmp_path, text=PT)) == 1
    assert not (tmp_path / "logs").exists()
    assert "fp8_per_channel" in capsys.readouterr().err


def test_the_confirmation_is_kept_in_the_record(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_commit", lambda: HEAD)
    args = _gate_files(tmp_path, check=_passed(rule={"confirm_budget": True})) + ["--speed-ok"]
    assert S.main(args) == 0
    row = json.loads((tmp_path / "logs" / "n2-pilot" / "backend.jsonl").read_text())
    assert row["setup_check"]["confirm_budget"] is True and row["setup_check"]["speed_ok"] is True


def test_lines_is_the_quick_look_before_any_gpu_time(tmp_path, capsys):
    f = tmp_path / "server.txt"
    f.write_text(PC)
    assert S.main(["lines", "--server", str(f)]) == 0
    f.write_text(PT)
    assert S.main(["lines", "--server", str(f)]) == 1
    assert "fp8_per_channel" in capsys.readouterr().err
    f.write_text("")
    assert S.main(["lines", "--server", str(f)]) == 1
    assert S.main(["lines", "--server", str(tmp_path / "none.txt")]) == 1


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


def test_recording_refuses_another_copy_before_any_vm(tmp_path, monkeypatch, capsys):
    """The session asks this before it starts a VM."""
    d = tmp_path / "bakeoff"
    _bakeoff_dir(d)
    _pin(monkeypatch, d)
    assert S.main(["recording", "--bakeoff", str(d)]) == 0
    (d / "snapshot.txt").write_text("another machine\n")
    assert S.main(["recording", "--bakeoff", str(d)]) == 1
    assert "snapshot.txt" in capsys.readouterr().out
    (d / "real.jsonl").unlink()
    assert S.main(["recording", "--bakeoff", str(d)]) == 1
    assert "real.jsonl is missing" in capsys.readouterr().out


def test_forced_pass_end_to_end_against_a_fake_server(tmp_path, monkeypatch):
    real = _bakeoff_dir(tmp_path / "bakeoff")
    _pin(monkeypatch, tmp_path / "bakeoff")
    _Fake.real = {r["command"]: r["output"] for r in real}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        out = tmp_path / "setup"
        args = ["fidelity", "--out", str(out), "--bakeoff", str(tmp_path / "bakeoff"), "--url", url]
        assert S.main([*args, "--interval", "0.05"]) == 0
        assert (out / "real.jsonl").read_bytes() == (tmp_path / "bakeoff" / "real.jsonl").read_bytes()
        sims = [json.loads(x) for x in (out / f"sim-{S.CONFIG}-forced.jsonl").read_text().splitlines()]
        assert len(sims) == 3 and all(r["error"] is None for r in sims)
        scores = [json.loads(x) for x in (out / "scores.jsonl").read_text().splitlines()]
        assert all(r["exact"] for r in scores)
        samples = [json.loads(x) for x in (out / "speed.jsonl").read_text().splitlines()]
        assert samples[0].get("start") and any("generation_tokens" in s for s in samples)
        # a second pass resumes: nothing is asked again, and another bake-off copy is refused
        before = _Fake.calls
        assert S.main([*args, "--interval", "0.05"]) == 0
        assert _Fake.calls == before         # every step was done: no request is sent again
        (tmp_path / "bakeoff" / "real.jsonl").write_text("{}\n")
        with pytest.raises(SystemExit, match="real.jsonl"):
            S.main(args)
    finally:
        server.shutdown()


def _check_dir(out: Path, monkeypatch, steps=10, sim=0.905, text=PC):
    recording = out.parent / "recording"
    _bakeoff_dir(recording, steps=steps)
    _pin(monkeypatch, recording)
    monkeypatch.setattr(S, "_commit", lambda: HEAD)
    out.mkdir(parents=True)
    for n in ("real.jsonl", "snapshot.txt"):
        (out / n).write_bytes((recording / n).read_bytes())
    rows = [dict(r, seq="s1") for r in _scores(steps, sim=sim)]
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "speed.jsonl").write_text("".join(json.dumps(_sample(5 * i, 2500 * i)) + "\n" for i in range(5)))
    (out / "server.txt").write_text(text)
    _write_equiv(out)


def _verdict(out):
    return json.loads((out / "check.json").read_text())


def test_check_end_to_end_passes(tmp_path, monkeypatch):
    out = tmp_path / "setup"
    _check_dir(out, monkeypatch)
    assert S.main(["check", "--out", str(out)]) == 0
    d = _verdict(out)
    assert d["verdict"] == "PASS" and d["commit"] == HEAD and d["scheme"] == "fp8_per_channel" and d["gpu"] == GPU
    assert d["fidelity"]["mean_sim"] == pytest.approx(0.905)
    assert d["speed"]["median_tok_s"] == pytest.approx(500)
    assert d["noise"]["agent"]["contexts"] == 2 and d["noise"]["simulator"]["contexts"] == 6
    assert d["constants"]["sim_floor"] == S.SIM_FLOOR
    assert d["layers"] == 340


def test_check_exit_says_fail(tmp_path, monkeypatch):
    out = tmp_path / "setup"
    _check_dir(out, monkeypatch, sim=0.85)
    assert S.main(["check", "--out", str(out)]) == 1
    assert _verdict(out)["verdict"] == "FAIL"


def test_check_on_a_server_with_the_old_scheme_fails(tmp_path, monkeypatch):
    out = tmp_path / "setup"
    _check_dir(out, monkeypatch, text=PT)
    assert S.main(["check", "--out", str(out)]) == 1
    assert _verdict(out)["verdict"] == "FAIL"


def _other_recording(out):
    (out / "snapshot.txt").write_text("another machine\n")


def _other_steps(out):
    rows = [dict(r, seq="unrelated") for r in _scores(10, sim=0.95)]
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("break_it", [_other_recording, _other_steps, lambda out: (out / "equiv-c1.jsonl").unlink(),
                                      lambda out: (out / "scores.jsonl").unlink(),
                                      lambda out: (out / "server.txt").write_text("")],
                         ids=["other-recording", "other-steps", "no-equiv", "no-pass", "no-lines"])
def test_check_is_incomplete_without_its_evidence(tmp_path, monkeypatch, break_it):
    out = tmp_path / "setup"
    _check_dir(out, monkeypatch)
    break_it(out)
    assert S.main(["check", "--out", str(out)]) == 2
    d = _verdict(out)
    assert d["verdict"] == "INCOMPLETE" and d["rule"]["problems"]


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
    d = tmp_path / "setup"
    real = _bakeoff_dir(d, steps=3)
    sims = [{"config": S.CONFIG, "mode": "forced", "seq": "s1", "step": r["step"],
             "output": r["output"] if r["step"] else "wrong", "error": None, "finish": "stop"} for r in real]
    (d / f"sim-{S.CONFIG}-forced.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sims))
    rows = [dict(r, seq="s1") for r in _scores(3)]
    (d / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    g = S.graded_fidelity(d)
    assert g["available"] and g["faithful"] == pytest.approx(2 / 3)
    assert "0\n" in seen_systems[2] and "1\n" in seen_systems[2]   # step 2 was given steps 0 and 1


def test_graded_fidelity_without_the_grader_is_reported_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "simtest.post.fidelity", None)    # import fails, as on this branch
    g = S.graded_fidelity(tmp_path)
    assert g == {"available": False, "why": g["why"]} and g["why"]
