"""Exploratory (N2): is the simulator served as fp8_per_channel no worse than as fp8 (per tensor)?

Not part of the testbed, and not evidence for any comparison. The measurement for the first hour
of the pilot session, before its episodes (docs/BACKENDS.md, "The simulator's quantization";
LOG.md, 02-11-2026). It runs on the laptop through the tunnel, one setting after the other on the
same GPU: serve.sh starts the simulator as fp8, and SIM_ONLY=1 SIM_QUANT=fp8_per_channel restarts it.

  fidelity SETTING   the bake-off's forced mode (explore/bakeoff_simulate.py, config
                     agentworld-think-vllm12): each of the 84 steps predicted from the real history,
                     12 requests at once, on the simulator now serving. Meanwhile it reads the
                     server's /metrics every few seconds, for the generation speed at 12 at once.
                     Then explore/bakeoff_score.py. Re-running resumes the missing or errored steps.
  decide             reads both settings' fidelity, speed, `simtest equiv` noise floors and the
                     servers' own lines, applies the DRAFT rule below, and writes decision.json.

    uv run python -I explore/sim_quant.py fidelity fp8 --out logs/simquant
    uv run python -I explore/sim_quant.py decide --out logs/simquant

Files under --out (logs/ is not in git):
  <setting>/real.jsonl, snapshot.txt   copies of logs/bakeoff/'s (the bake-off's recording, 26-10-2026)
  <setting>/sim-agentworld-think-vllm12-forced.jsonl, scores.jsonl, scores.md   the pass and its scores
  <setting>/speed.jsonl                /metrics samples
  <setting>/server.txt                 from the VM: the simulator's served command line (versions.txt)
                                       and its log's "Quantizing" and "Fp8 MoE backend" lines
  equiv/<setting>-c1.jsonl, equiv/<setting>-c12.jsonl   `simtest equiv score --concurrency 1 12`

DRAFT decision rule (02-11-2026, written before any data; frozen in LOG.md before the run).
fp8_per_channel is adopted for the pilot and N3 if all of these hold, and fp8 is kept otherwise:
1. Fidelity: on the 81 deterministic forced steps, its mean similarity is at least fp8's minus 0.02,
   and at most 3 of its 84 steps hit the 16K cap (the 29-10-2026 equivalence rule). 0.02 is about
   2 standard errors of the paired difference between two forced passes: 0.0097, from the MLX
   (26-10) and G4 (29-10) runs, an upper bound since those also differ in quantization and engine.
2. Batching noise: the simulator's `simtest equiv` noise floor (1 against 12 at once) has a lower
   mean |difference in log-prob| than fp8's, with the 95% paired bootstrap interval of the
   difference (per channel minus per tensor, over contexts) below zero, and its top-1 agreement
   is no lower.
3. Complete evidence: both passes predicted the pinned bake-off recording and scored exactly its
   steps, none with an error; the four score files come from the simulator server on vLLM 0.32.0,
   are labelled with their setting, and cover the same simulator contexts with the same token IDs;
   each server's lines show its scheme on every quantized layer and the Triton MoE kernel. If not,
   the verdict is INCOMPLETE and the pilot keeps fp8.
Speed never decides. If fp8_per_channel is adopted and makes fewer than 0.9 of fp8's tokens per
second at 12 at once (or its speed is missing), the rescaled budget is confirmed first.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = "agentworld-think-vllm12"     # explore/bakeoff_simulate.py: the G4 bake-off config at 12 at once
SETTINGS = ("fp8", "fp8_per_channel")  # serve.sh's SIM_QUANT
AT_ONCE = 12                           # the pilot's max_connections
VLLM_VERSION = "0.32.0"                # serve.sh's pin
# The bake-off's recording (logs/bakeoff/, 26-10-2026: 84 steps, 81 deterministic), which every
# forced pass so far predicted: both settings must predict exactly it.
RECORDING = {"real.jsonl": "c5fe28a1c815d83ca5515dd2bb15c1747bbb316bc0ceaaedbe24f83f422a6f46",
             "snapshot.txt": "2bfe511a49d10e4a1b4d52e897a94263662d1bb8a84c30fb2dfd95f53457e6b0"}
SIM_TOLERANCE = 0.02
MAX_TRUNCATED = 3
SPEED_CONFIRM = 0.90

# What each setting's server must show: its served line, the scheme of every quantized layer, the MoE kernel.
EXPECT = {
    "fp8": {"flag": re.compile(r"--quantization fp8(?=\s|$)"), "scheme": "fp8_per_tensor_static"},
    "fp8_per_channel": {"flag": re.compile(r"--quantization fp8_per_channel --moe-backend triton(?=\s|$)"),
                        "scheme": "fp8_per_channel_static"},
}
_QUANTIZING = re.compile(r"Quantizing (\d+) layers of types: (.*)")
_SCHEME = re.compile(r"\(from (?:linear|moe|targets): (?:[^,)]*, )?([A-Za-z0-9_]+)\)")


# --- the servers' own lines -----------------------------------------------------------------------


def layer_count(text: str) -> int | None:
    m = _QUANTIZING.search(text)
    return int(m.group(1)) if m else None


def server_check(text: str, setting: str) -> list[str]:
    """Problems with a setting's server.txt: the simulator's served line, the scheme vLLM reports
    for every quantized layer kind, and the MoE kernel it chose."""
    want, problems = EXPECT[setting], []
    served = [x for x in text.splitlines() if "--served-model-name simulator" in x]
    if not served:
        problems.append(f"{setting}: no simulator command line in server.txt")
    elif not want["flag"].search(served[-1]):
        problems.append(f"{setting}: the served simulator line is not {want['flag'].pattern.split('(?=')[0]}")
    m = _QUANTIZING.search(text)
    if not m:
        problems.append(f"{setting}: no 'Quantizing ... layers' line from the simulator's log")
    else:
        schemes = _SCHEME.findall(m.group(2))
        if not schemes or set(schemes) != {want["scheme"]}:
            problems.append(f"{setting}: layers quantized as {sorted(set(schemes))}, not only {want['scheme']}")
    if "Using TRITON Fp8 MoE backend" not in text:
        problems.append(f"{setting}: the simulator's MoE kernel is not TRITON")
    return problems


# --- speed (vLLM's /metrics) -----------------------------------------------------------------------

_METRICS = {"vllm:generation_tokens_total": "generation_tokens", "vllm:prompt_tokens_total": "prompt_tokens",
            "vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting",
            "vllm:num_preemptions_total": "preemptions"}
_SERIES = re.compile(r"^([A-Za-z_:][\w:]*)(?:\{[^}]*\})?\s+(\S+)")


def parse_metrics(text: str) -> dict[str, float]:
    """The few series we read from Prometheus text, summed over label sets."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _SERIES.match(line)
        if line.startswith("#") or not m or m.group(1) not in _METRICS:
            continue
        key = _METRICS[m.group(1)]
        out[key] = out.get(key, 0.0) + float(m.group(2))
    return out


def speed(samples: list[dict], at: int = AT_ONCE, interval: float = 5.0) -> dict:
    """Generation tokens per second between consecutive samples: the median over windows with
    `at` requests running at both ends and none waiting, and the overall rate while any ran.
    A window across a gap, a new pass, an error or a counter reset is skipped."""
    rates, total_tokens, total_time = [], 0.0, 0.0
    for a, b in zip(samples, samples[1:]):
        if any("error" in s or "start" in s or "generation_tokens" not in s for s in (a, b)):
            continue
        dt, dg = b["t"] - a["t"], b["generation_tokens"] - a["generation_tokens"]
        if dt <= 0 or dt > 3 * interval or dg < 0:
            continue
        if a.get("running", 0) > 0 or b.get("running", 0) > 0:
            total_tokens, total_time = total_tokens + dg, total_time + dt
        if a.get("running") == b.get("running") == at and a.get("waiting") == b.get("waiting") == 0:
            rates.append(dg / dt)
    return {"at": at, "windows": len(rates), "median_tok_s": statistics.median(rates) if rates else None,
            "mean_tok_s": statistics.fmean(rates) if rates else None,
            "overall_tok_s": total_tokens / total_time if total_time else None}


def _poll(url: str, path: Path, interval: float, stop: threading.Event, setting: str) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"t": time.time(), "start": True, "setting": setting, "url": url}) + "\n")
        while True:
            try:
                with urllib.request.urlopen(f"{url.rstrip('/')}/metrics", timeout=10) as r:
                    row = {"t": time.time(), **parse_metrics(r.read().decode())}
            except Exception as exc:   # kept, never fatal: a window touching it is skipped
                row = {"t": time.time(), "error": f"{type(exc).__name__}: {exc}"[:200]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if stop.wait(interval):
                return


# --- fidelity (the bake-off's scores) ----------------------------------------------------------------


def _ours(rows: list[dict]) -> dict:
    return {(r["seq"], r["step"]): r for r in rows if r.get("config") == CONFIG and r.get("mode") == "forced"}


def recording_check(d: Path, setting: str) -> list[str]:
    """Problems if a setting's copy of the recording isn't the pinned bake-off recording."""
    problems = []
    for name, sha in RECORDING.items():
        path = d / name
        if not path.exists():
            problems.append(f"{setting}: {name} is missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            problems.append(f"{setting}: {name} is not the bake-off's recording (sha256 {sha[:12]}...)")
    return problems


def step_keys(real: list[dict]) -> list[tuple]:
    return [(r["seq"], r["step"]) for r in real]


def fidelity_summary(rows: list[dict], real_keys: list[tuple]) -> dict:
    """From bakeoff_score.py's scores.jsonl: mean similarity and exact share on the deterministic
    steps, and the steps that errored or hit the cap (they score 0, as the bake-off scores them).
    Complete only if it scored exactly the recording's steps, none with an error."""
    by_key = _ours(rows)
    mine = list(by_key.values())
    det = [r for r in mine if not r.get("nondet")]
    errors = sum(bool(r.get("error")) for r in mine)
    tested = [r for r in mine if r.get("pytest") is not None]
    return {"steps": len(mine), "real_steps": len(real_keys), "deterministic": len(det), "errors": errors,
            "truncated": sum(r.get("finish") == "length" for r in mine),
            "complete": bool(real_keys) and set(by_key) == set(real_keys) and errors == 0,
            "mean_sim": statistics.fmean(r["sim"] for r in det) if det else None,
            "exact": statistics.fmean(bool(r["exact"]) for r in det) if det else None,
            "pytest_right": f"{sum(bool(r['pytest']) for r in tested)}/{len(tested)}",
            "steps_with_tells": sum(bool(r.get("tells")) for r in mine)}


def paired(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """b minus a, per deterministic step both scored: mean and standard error."""
    a, b = _ours(rows_a), _ours(rows_b)
    d = [b[k]["sim"] - a[k]["sim"] for k in a if k in b and not a[k].get("nondet")]
    return {"n": len(d), "mean": statistics.fmean(d) if d else None,
            "se": statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else None}


def graded_fidelity(out: Path) -> dict:
    """The graded per-step fidelity (simtest.post.fidelity, branch n2-fidelity) on both passes, if
    this checkout has it. Reported, not part of the rule. In forced mode the simulator was given the
    snapshot and every earlier real output, so those count as given."""
    try:
        fid = importlib.import_module("simtest.post.fidelity")
        sys.path.insert(0, str(HERE))
        import bakeoff_simulate as bs
        result = {"available": True}
        for s in SETTINGS:
            d = out / s
            real = _lines(d / "real.jsonl")
            sims = _ours(_lines(d / f"sim-{CONFIG}-forced.jsonl"))       # the last attempt of a step, as scored
            nondet = {k for k, r in _ours(_lines(d / "scores.jsonl")).items() if r.get("nondet")}
            system = bs.system_prompt("common", (d / "snapshot.txt").read_text())
            by_seq: dict[str, list[dict]] = {}
            for r in real:
                by_seq.setdefault(r["seq"], []).append(r)
            grades = []
            for seq, steps in by_seq.items():
                for i, r in enumerate(steps):
                    key = (seq, r["step"])
                    if key in nondet or key not in sims:
                        continue
                    sim = sims[key]
                    seen = "" if sim.get("error") or sim.get("finish") == "length" else sim["output"]
                    ctx = fid.Context(system + "".join("\n" + e["output"] for e in steps[:i]), "C3")
                    for e in steps[:i]:
                        ctx.add(e["command"], "")
                    grades.append(fid.grade_step(r["command"], r["output"], seen,
                                                 ctx.for_step(r["command"], r["output"]))["grade"])
            result[s] = {"n": len(grades), "counts": dict(Counter(grades)),
                         "faithful": sum(g in fid.FAITHFUL for g in grades) / len(grades) if grades else None}
        return result
    except Exception as exc:    # not on this branch, or its interface changed: reported as unavailable
        return {"available": False, "why": f"{type(exc).__name__}: {exc}"[:300]}


# --- the noise floors (simtest equiv) ---------------------------------------------------------------


def _floor(g: dict | None) -> dict | None:
    if g is None:
        return None
    keys = ("contexts", "positions", "mean_abs_dlogprob", "median_abs_dlogprob", "p99_abs_dlogprob",
            "max_abs_dlogprob", "top1_agreement")
    return {**{k: g[k] for k in keys}, "reply_mean_abs_dlogprob": g["response"]["mean_abs_dlogprob"]}


def noise(out: Path) -> dict:
    """The simulator's noise floor (1 against 12 at once) under each setting, from the four
    `simtest equiv score` files; the gap between the settings at 1 at once; fp8's agent floor."""
    from simtest import equiv as Q

    paths = {(s, c): out / "equiv" / f"{s}-c{c}.jsonl" for s in SETTINGS for c in (1, AT_ONCE)}
    missing = [f"{p.name}: missing" for p in paths.values() if not p.exists()]
    if missing:
        return {"problems": missing}
    files = {k: Q.read_scores(p) for k, p in paths.items()}
    problems = []
    for key in ("contexts_sha256", "top_k", "max_tokens"):
        if len({json.dumps(h.get(key)) for h, _ in files.values()}) > 1:
            problems.append("the files were scored on different contexts files" if key == "contexts_sha256"
                            else f"the files were scored with different {key}")
    for (s, c), (h, _) in files.items():
        name, srv = paths[(s, c)].name, (h.get("servers") or {}).get("simulator") or {}
        if h.get("concurrency") != c:
            problems.append(f"{name}: scored at concurrency {h.get('concurrency')}, not {c}")
        if srv.get("served_model") != "simulator" or "simulator" not in (srv.get("models") or []):
            problems.append(f"{name}: the server scored as the simulator served {srv.get('models')}")
        if srv.get("version") != VLLM_VERSION:
            problems.append(f"{name}: scored on vLLM {srv.get('version')}, not {VLLM_VERSION}")
        label = h.get("label") or ""
        if label != s and not label.endswith(f", {s}"):
            problems.append(f"{name}: its label {label!r} doesn't end with the setting ({s})")
    sim = {k: [r for r in rows if r.get("model") == "simulator"] for k, (_, rows) in files.items()}
    if len({frozenset(r["id"] for r in rows) for rows in sim.values()}) > 1:
        problems.append("the files score different simulator contexts")
    gaps = {"fp8": Q.gap(sim[("fp8", 1)], sim[("fp8", AT_ONCE)]).get("simulator"),
            "fp8_per_channel": Q.gap(sim[("fp8_per_channel", 1)], sim[("fp8_per_channel", AT_ONCE)]).get("simulator"),
            "cross_c1": Q.gap(sim[("fp8", 1)], sim[("fp8_per_channel", 1)]).get("simulator")}
    for name, g in gaps.items():
        if g is None:
            problems.append(f"{name}: no simulator contexts compared")
            continue
        if g["missing"]:
            problems.append(f"{name}: {len(g['missing'])} simulator context(s) missing or errored")
        if g["token_mismatch"]:
            problems.append(f"{name}: different token IDs in {len(g['token_mismatch'])} context(s)")
        if not g["contexts"] and not g["missing"] and not g["token_mismatch"]:
            problems.append(f"{name}: no simulator contexts compared")
    agent = [r for r in files[("fp8", 1)][1] if r.get("model") == "agent"]
    agent12 = [r for r in files[("fp8", AT_ONCE)][1] if r.get("model") == "agent"]
    ok = gaps["fp8"] and gaps["fp8_per_channel"]
    return {"problems": problems, **{k: _floor(g) for k, g in gaps.items()},
            "agent_fp8": _floor(Q.gap(agent, agent12).get("agent")) if agent else None,
            "bootstrap_pc_minus_pt": (Q._bootstrap(gaps["fp8_per_channel"]["per_context"], gaps["fp8"]["per_context"])
                                      if ok else None)}


# --- the rule -----------------------------------------------------------------------------------------

PIN_KEEP = ("max_connections stays 12 for every condition and session: under fp8 the MoE's activation "
            "scale is the batch's maximum, so a bigger batch coarsens every token's quantization.")
PIN_ADOPT = ("max_connections stays 12 in the pilot and N3: batch dependence remains (Triton MoE tiles chosen "
             "by the batch's size, attention splits, prefill chunk boundaries), and vLLM's batch-invariant mode "
             "refuses the GDN layers. Another number at once (Hopper) needs a simtest equiv check recorded in "
             "LOG.md first: the noise floor at 1 against N no larger than at 1 against 12, for both models.")


def rule(fid: dict, noise_: dict, server_problems: list[str], speed_ratio: float | None) -> dict:
    missing = [f"{s}: the fidelity pass is incomplete ({fid[s]['steps']} of {fid[s].get('real_steps', '?')} "
               f"steps, {fid[s]['errors']} errors)" for s in SETTINGS if not fid[s]["complete"]]
    missing += list(noise_.get("problems") or []) + list(server_problems)
    if missing:
        return {"verdict": "INCOMPLETE", "problems": missing, "checks": None, "confirm_budget": False,
                "action": "keep SIM_QUANT=fp8 for this pilot (the status quo); report what is missing", "pin": PIN_KEEP}
    pt, pc = noise_["fp8"], noise_["fp8_per_channel"]
    hi = noise_["bootstrap_pc_minus_pt"]["mean_abs_dlogprob"][1]
    diff = fid["fp8_per_channel"]["mean_sim"] - fid["fp8"]["mean_sim"]
    checks = {"fidelity": round(diff, 6) >= -SIM_TOLERANCE,
              "truncations": fid["fp8_per_channel"]["truncated"] <= MAX_TRUNCATED,
              "noise_lower": pc["mean_abs_dlogprob"] < pt["mean_abs_dlogprob"] and hi < 0,
              "top1_not_lower": pc["top1_agreement"] >= pt["top1_agreement"]}
    adopt = all(checks.values())
    return {"verdict": "ADOPT" if adopt else "KEEP", "problems": [], "checks": checks,
            "fidelity_difference": diff,
            "confirm_budget": adopt and (speed_ratio is None or speed_ratio < SPEED_CONFIRM),
            "action": ("serve the simulator with SIM_QUANT=fp8_per_channel in the pilot and N3, in every condition "
                       "and phase (episodes, vectors, readings)") if adopt else
                      "keep SIM_QUANT=fp8: restart the simulator with SIM_ONLY=1 (fp8) before the episodes",
            "pin": PIN_ADOPT if adopt else PIN_KEEP}


# --- command line -------------------------------------------------------------------------------------


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def main_fidelity(args) -> int:
    d = Path(args.out) / args.setting
    d.mkdir(parents=True, exist_ok=True)
    for name, sha in RECORDING.items():
        src, dst = Path(args.bakeoff) / name, d / name
        if hashlib.sha256(src.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{src} is not the bake-off's recording (sha256 {sha[:12]}...)")
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise SystemExit(f"{dst} differs from {src}: the passes must predict the same recording")
        if not dst.exists():
            shutil.copyfile(src, dst)
    stop = threading.Event()
    poller = threading.Thread(target=_poll, args=(args.url, d / "speed.jsonl", args.interval, stop, args.setting),
                              daemon=True)
    poller.start()
    env = {**os.environ, "CHAT_URL": f"{args.url.rstrip('/')}/v1/chat/completions"}
    try:
        run = subprocess.run([sys.executable, "-I", str(HERE / "bakeoff_simulate.py"), CONFIG, "forced", str(d)], env=env)
    finally:
        stop.set()
        poller.join(timeout=30)
    score = subprocess.run([sys.executable, "-I", str(HERE / "bakeoff_score.py"), str(d)], capture_output=True, text=True)
    (d / "scores.md").write_text(score.stdout)
    if score.returncode:
        print(score.stderr, file=sys.stderr)
        return score.returncode
    f = fidelity_summary(_lines(d / "scores.jsonl"), step_keys(_lines(d / "real.jsonl")))
    s = speed(_lines(d / "speed.jsonl"), interval=args.interval)
    print(f"{args.setting}: {f['steps']} of {f['real_steps']} steps, {f['errors']} errors, {f['truncated']} at the cap; "
          f"mean similarity {_f(f['mean_sim'])} on {f['deterministic']} deterministic steps; "
          f"{_f(s['median_tok_s'], 0)} tokens/s at {AT_ONCE} at once ({s['windows']} windows)")
    if not f["complete"]:
        print("incomplete: run the same command again to redo the missing or errored steps", file=sys.stderr)
    return run.returncode


def _f(x, nd=4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def main_decide(args) -> int:
    out = Path(args.out)
    rows = {s: _lines(out / s / "scores.jsonl") for s in SETTINGS}
    fid = {s: fidelity_summary(rows[s], step_keys(_lines(out / s / "real.jsonl"))) for s in SETTINGS}
    spd = {s: speed(_lines(out / s / "speed.jsonl"), interval=args.interval) for s in SETTINGS}
    texts = {s: (out / s / "server.txt").read_text() if (out / s / "server.txt").exists() else "" for s in SETTINGS}
    server_problems = [p for s in SETTINGS for p in recording_check(out / s, s) + server_check(texts[s], s)]
    counts = {s: layer_count(texts[s]) for s in SETTINGS}
    if None not in counts.values() and len(set(counts.values())) > 1:
        server_problems.append(f"the settings quantized different numbers of layers: {counts}")
    nz = noise(out)
    a, b = spd["fp8"]["median_tok_s"], spd["fp8_per_channel"]["median_tok_s"]
    ratio = b / a if a and b is not None else None
    r = rule(fid, nz, server_problems, ratio)
    decision = {"rule": r, "fidelity": fid, "paired_sim": paired(rows["fp8"], rows["fp8_per_channel"]),
                "graded": graded_fidelity(out), "speed": spd, "speed_ratio": ratio, "noise": nz,
                "layers": counts, "constants": {"sim_tolerance": SIM_TOLERANCE, "max_truncated": MAX_TRUNCATED,
                                                "speed_confirm": SPEED_CONFIRM, "at_once": AT_ONCE}}
    (out / "decision.json").write_text(json.dumps(decision, indent=1, default=str) + "\n")
    for s in SETTINGS:
        f, sp, n = fid[s], spd[s], nz.get(s) or {}
        print(f"{s}: fidelity {_f(f['mean_sim'])} (exact {_f(f['exact'], 2)}, pytest {f['pytest_right']}, "
              f"{f['truncated']} at the cap, {f['errors']} errors, {f['steps']}/{f['real_steps']} steps); "
              f"noise floor |d logprob| {_f(n.get('mean_abs_dlogprob'))}, top-1 {_f(n.get('top1_agreement'))}; "
              f"{_f(sp['median_tok_s'], 0)} tokens/s at {AT_ONCE}")
    p = decision["paired_sim"]
    print(f"per channel minus per tensor, similarity: {_f(p['mean'])} (SE {_f(p['se'])}, {p['n']} steps); "
          f"noise bootstrap {nz.get('bootstrap_pc_minus_pt')}; speed ratio {_f(ratio, 2)}")
    g = decision["graded"]
    print("graded fidelity: " + (", ".join(f"{s} faithful {_f(g[s]['faithful'], 2)} {g[s]['counts']}" for s in SETTINGS)
                                 if g.get("available") else f"unavailable ({g.get('why')})"))
    print(f"\nDRAFT rule -> {r['verdict']}: {r['action']}")
    for x in r["problems"]:
        print(f"  missing: {x}")
    if r["checks"]:
        print("  " + ", ".join(f"{k} {'ok' if v else 'NO'}" for k, v in r["checks"].items()))
    if r["confirm_budget"]:
        print("  per channel is slower than 0.9 of per tensor (or its speed is missing): confirm the budget")
    print(f"  pin: {r['pin']}")
    print(f"wrote {out / 'decision.json'}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fidelity", help="the bake-off's forced pass at 12 at once, with /metrics")
    f.add_argument("setting", choices=SETTINGS, help="what the simulator is serving now (serve.sh's SIM_QUANT)")
    f.add_argument("--out", default="logs/simquant")
    f.add_argument("--bakeoff", default="logs/bakeoff", help="the bake-off's recording (real.jsonl, snapshot.txt)")
    f.add_argument("--url", default="http://localhost:18001", help="the simulator server, through the tunnel")
    f.add_argument("--interval", type=float, default=5.0, help="seconds between /metrics samples")
    d = sub.add_parser("decide", help="the DRAFT rule on both settings' files")
    d.add_argument("--out", default="logs/simquant")
    d.add_argument("--interval", type=float, default=5.0, help="the fidelity passes' sampling interval")
    args = p.parse_args(argv)
    return main_fidelity(args) if args.cmd == "fidelity" else main_decide(args)


if __name__ == "__main__":
    sys.exit(main())
