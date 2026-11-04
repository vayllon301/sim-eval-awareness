"""Exploratory (N2): the simulator's setup check, on fp8_per_channel, before a run's first episode.

Not part of the testbed, and not evidence for any comparison. The simulator runs fp8_per_channel in
every condition and phase from the pilot on (LOG.md, 02-11-2026), and this checks that the session
really serves it and that it works: docs/BACKENDS.md, "The setup check". session.sh and run_chunks.sh
run it on the laptop, through the tunnel, before any episode:

  lines              the server's scheme lines, before any GPU time goes on the rest.
  fidelity           the bake-off's forced mode (explore/bakeoff_simulate.py, config
                     agentworld-think-vllm12): each of the 84 steps predicted from the real history,
                     12 requests at once. Meanwhile it reads the server's /metrics every few seconds,
                     for the generation speed at 12 at once. Then explore/bakeoff_score.py. Re-running
                     resumes the missing or errored steps.
  check              applies the rule below to the server's lines, the pass, the speed and the
                     `simtest equiv` noise floors, and writes check.json. Exit: 0 PASS, 1 FAIL,
                     2 INCOMPLETE.
  gate               the last check before each `simtest run`: a PASS on this commit and these
                     lines on this very server. Then it appends the scheme to the run's record,
                     logs/<name>/backend.jsonl. Refused: exit 1, and no episode starts.
  recording          whether the bake-off's recording is the pinned one (before a VM exists).

    uv run python -I explore/sim_quant.py fidelity --out logs/setup/<dir>
    uv run python -I explore/sim_quant.py check --out logs/setup/<dir>

Files under --out, one commit's check on one backend (logs/ is not in git):
  server.txt                     serve.sh's lines: the GPU, the simulator's served line, what vLLM
                                 says it quantized and the MoE kernel it chose
  equiv-c1.jsonl, equiv-c12.jsonl    `simtest equiv score --concurrency 1 12`, both models
  real.jsonl, snapshot.txt       copies of logs/bakeoff/'s (the bake-off's recording, 26-10-2026)
  sim-agentworld-think-vllm12-forced.jsonl, scores.jsonl, scores.md   the pass and its scores
  speed.jsonl                    /metrics samples
  check.json                     the verdict, its numbers and the commit

The rule (LOG.md, frozen before the pilot; the same text is in docs/BACKENDS.md). No fp8 pass runs in
the session, so the bars are fixed numbers from earlier sessions. PASS needs all of these:
1. Scheme: the served line says --quantization fp8_per_channel --moe-backend triton, every quantized
   layer kind says fp8_per_channel_static, and the MoE kernel is TRITON.
2. Fidelity: on the 81 deterministic forced steps, the mean similarity is at least 0.892: 0.912 (the
   G4's 29-10-2026 pass, on fp8) minus 0.02, about 2 standard errors of the paired difference between
   two forced passes (0.0097). That pair also differed in backend, so it is an upper bound.
3. Truncation: at most 3 of the 84 steps hit the 16K cap (the 29-10-2026 rule; that pass had 0).
4. Complete evidence: the pass predicted the pinned bake-off recording and scored exactly its steps,
   none with an error; both `equiv` files are from vLLM 0.32.0 on the right servers, labelled with the
   scheme, on the same contexts file, at 1 and 12 at once, for both models, every context on identical
   token IDs; the server's lines are all there.
FAIL: a line shows another scheme or kernel, or bar 2 or 3 is missed. INCOMPLETE: 4 is missed.
Either way no episode runs, and the simulator never falls back to fp8 by itself.
The noise floors have no bar on their size: no fp8 floor was ever measured, so there is nothing to
hold them against. They are recorded as this backend's reference.
Speed never decides. Under 405 tokens per second at 12 at once (0.9 of 450, the middle of the rates
behind the 02-11 estimate), or with no speed at all, the episodes wait until Victor confirms the
rescaled budget (SPEED_OK=1).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
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
SCHEME = "fp8_per_channel"             # serve.sh's default: the only one episodes run on, from the pilot on
AT_ONCE = 12                           # the pilot's max_connections
VLLM_VERSION = "0.32.0"                # serve.sh's pin
MODELS = ("agent", "simulator")
# The bake-off's recording (logs/bakeoff/, 26-10-2026: 84 steps, 81 deterministic), which every
# forced pass so far predicted.
RECORDING = {"real.jsonl": "c5fe28a1c815d83ca5515dd2bb15c1747bbb316bc0ceaaedbe24f83f422a6f46",
             "snapshot.txt": "2bfe511a49d10e4a1b4d52e897a94263662d1bb8a84c30fb2dfd95f53457e6b0"}
# The bars. logs/bakeoff-g4/ (29-10-2026; fp8, 8 at once): mean similarity 0.9124 on the 81
# deterministic steps, 0 steps at the cap, 0 errors.
REFERENCE_SIM = 0.912
SIM_TOLERANCE = 0.02
SIM_FLOOR = round(REFERENCE_SIM - SIM_TOLERANCE, 6)
MAX_TRUNCATED = 3
REFERENCE_TOK_S = 450.0                # the middle of the 400-500 the 02-11 estimate used for this pass
SPEED_CONFIRM = 0.90

# What the server must show: its served line, the scheme of every quantized layer, the MoE kernel.
_FLAG = re.compile(r"--quantization fp8_per_channel --moe-backend triton(?=\s|$)")
_LAYER_SCHEME = "fp8_per_channel_static"
_QUANTIZING = re.compile(r"Quantizing (\d+) layers of types: (.*)")
# One layer kind, as vLLM prints it: "mlp.experts: 40 (from moe: fp8_per_tensor_static)".
_KIND = re.compile(r"[\w.]+: (\d+) \(from [^():]+: (?:[^,)]*, )?([A-Za-z0-9_]+)\)")
_MOE = re.compile(r"Using (\w+) Fp8 MoE backend")
_GPU = re.compile(r"^gpu: (.+)$", re.M)


# --- the server's own lines -----------------------------------------------------------------------


def layer_count(text: str) -> int | None:
    m = _QUANTIZING.search(text)
    return int(m.group(1)) if m else None


def gpu_name(text: str) -> str | None:
    m = _GPU.search(text)
    return m.group(1).strip() if m else None


def served_line(text: str) -> str | None:
    served = [x for x in text.splitlines() if "--served-model-name simulator" in x]
    return served[-1] if served else None


def server_check(text: str) -> tuple[list[str], list[str]]:
    """(missing, wrong) from server.txt: the simulator's served line, the scheme vLLM reports for
    every quantized layer kind, the MoE kernel it chose, and the GPU. Missing is a line that isn't
    there, a layer kind that can't be read, or kinds that don't add up to the layers quantized; wrong
    is a line that shows something else."""
    missing, wrong = [], []
    served = served_line(text)
    if served is None:
        missing.append("no simulator command line in server.txt")
    elif not _FLAG.search(served):
        wrong.append(f"the served simulator line is not --quantization {SCHEME} --moe-backend triton")
    lines = list(_QUANTIZING.finditer(text))
    if not lines:
        missing.append("no 'Quantizing ... layers' line from the simulator's log")
    for m in lines:
        kinds = [k.strip() for k in m.group(2).split(";")]
        read = [hit for k in kinds if (hit := _KIND.fullmatch(k))]
        unread = [k for k in kinds if not _KIND.fullmatch(k)]
        schemes = {hit.group(2) for hit in read}
        total = sum(int(hit.group(1)) for hit in read)
        if unread:
            missing.append(f"layer kinds this check can't read: {unread}")
        elif total != int(m.group(1)):
            missing.append(f"the layer kinds add up to {total}, not the {m.group(1)} quantized: the line is cut short")
        if schemes - {_LAYER_SCHEME}:
            wrong.append(f"layers quantized as {sorted(schemes)}, not only {_LAYER_SCHEME}")
    moe = _MOE.findall(text)
    if not moe:
        missing.append("no MoE backend line from the simulator's log")
    elif set(moe) != {"TRITON"}:
        wrong.append(f"the simulator's MoE kernel is {sorted(set(moe))}, not TRITON")
    if gpu_name(text) is None:
        missing.append("no GPU line in server.txt")
    return missing, wrong


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


def _poll(url: str, path: Path, interval: float, stop: threading.Event, scheme: str) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"t": time.time(), "start": True, "scheme": scheme, "url": url}) + "\n")
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


def recording_check(d: Path) -> list[str]:
    """Problems if a directory's copy of the recording isn't the pinned bake-off recording."""
    problems = []
    for name, sha in RECORDING.items():
        path = d / name
        if not path.exists():
            problems.append(f"{path}: {name} is missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            problems.append(f"{path}: {name} is not the bake-off's recording (sha256 {sha[:12]}...)")
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


def graded_fidelity(d: Path) -> dict:
    """The graded per-step fidelity (simtest.post.fidelity) of the pass. Reported, not part of the
    rule. In forced mode the simulator was given the snapshot and every earlier real output, so
    those count as given."""
    try:
        fid = importlib.import_module("simtest.post.fidelity")
        sys.path.insert(0, str(HERE))
        import bakeoff_simulate as bs
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
        return {"available": True, "n": len(grades), "counts": dict(Counter(grades)),
                "faithful": sum(g in fid.FAITHFUL for g in grades) / len(grades) if grades else None}
    except Exception as exc:    # its interface changed, or nothing to grade: reported as unavailable
        return {"available": False, "why": f"{type(exc).__name__}: {exc}"[:300]}


# --- the noise floors (simtest equiv) ---------------------------------------------------------------


def _floor(g: dict | None) -> dict | None:
    if g is None:
        return None
    keys = ("contexts", "positions", "mean_abs_dlogprob", "median_abs_dlogprob", "p99_abs_dlogprob",
            "max_abs_dlogprob", "top1_agreement")
    return {**{k: g[k] for k in keys}, "reply_mean_abs_dlogprob": g["response"]["mean_abs_dlogprob"]}


def noise(out: Path) -> dict:
    """Each model's noise floor (1 against 12 at once), from the two `simtest equiv score` files."""
    from simtest import equiv as Q

    paths = {c: out / f"equiv-c{c}.jsonl" for c in (1, AT_ONCE)}
    missing = [f"{p.name}: missing" for p in paths.values() if not p.exists()]
    if missing:
        return {"problems": missing}
    files = {c: Q.read_scores(p) for c, p in paths.items()}
    problems = []
    for key in ("contexts_sha256", "top_k", "max_tokens"):
        if len({json.dumps(h.get(key)) for h, _ in files.values()}) > 1:
            problems.append("the files were scored on different contexts files" if key == "contexts_sha256"
                            else f"the files were scored with different {key}")
    for c, (h, _) in files.items():
        name = paths[c].name
        if h.get("concurrency") != c:
            problems.append(f"{name}: scored at concurrency {h.get('concurrency')}, not {c}")
        label = h.get("label") or ""
        if label != SCHEME and not label.endswith(f", {SCHEME}"):
            problems.append(f"{name}: its label {label!r} doesn't end with the scheme ({SCHEME})")
        for m in MODELS:
            srv = (h.get("servers") or {}).get(m) or {}
            if srv.get("served_model") != m or m not in (srv.get("models") or []):
                problems.append(f"{name}: the server scored as the {m} served {srv.get('models')}")
            if srv.get("version") != VLLM_VERSION:
                problems.append(f"{name}: the {m} was scored on vLLM {srv.get('version')}, not {VLLM_VERSION}")
    gaps = Q.gap(files[1][1], files[AT_ONCE][1])
    for m in MODELS:
        g = gaps.get(m)
        if g is None:
            problems.append(f"{m}: no contexts compared")
            continue
        if g["missing"]:
            problems.append(f"{m}: {len(g['missing'])} context(s) missing or errored")
        if g["token_mismatch"]:
            problems.append(f"{m}: different token IDs in {len(g['token_mismatch'])} context(s)")
        if not g["contexts"] and not g["missing"] and not g["token_mismatch"]:
            problems.append(f"{m}: no contexts compared")
    return {"problems": problems, **{m: _floor(gaps.get(m)) for m in MODELS}}


# --- the rule -----------------------------------------------------------------------------------------

PIN = ("max_connections stays 12 in the pilot and N3, on every backend: batch dependence remains (Triton MoE "
       "tiles chosen by the batch's size, attention splits, prefill chunk boundaries), and vLLM's batch-invariant "
       "mode refuses the GDN layers.")
STOP = ("no episode runs and the session stops. It never falls back to fp8: that would be a new decision, in "
        "LOG.md. ")
ON_FAIL = STOP + "Victor decides what comes next; a new check, after a fix, runs in a new directory and both are reported."
ON_INCOMPLETE = STOP + "Fix what is missing and run the same command again: the forced pass keeps its finished steps."


def rule(fid: dict, noise_: dict, missing: list[str], wrong: list[str], tok_s: float | None) -> dict:
    """The verdict from the pass, the floors and the server's lines. `missing` and `wrong` as
    server_check's, with the recording's problems in `missing`."""
    base = {"checks": None, "confirm_budget": False, "pin": PIN}
    if wrong:
        return {**base, "verdict": "FAIL", "problems": wrong, "checks": {"scheme": False}, "action": ON_FAIL}
    problems = [] if fid["complete"] else [f"the forced pass is incomplete ({fid['steps']} of "
                                           f"{fid.get('real_steps', '?')} steps, {fid['errors']} errors)"]
    problems += list(noise_.get("problems") or []) + list(missing)
    if problems:
        return {**base, "verdict": "INCOMPLETE", "problems": problems, "action": ON_INCOMPLETE}
    checks = {"scheme": True, "fidelity": round(fid["mean_sim"], 6) >= SIM_FLOOR,
              "truncations": fid["truncated"] <= MAX_TRUNCATED}
    ok = all(checks.values())
    return {**base, "verdict": "PASS" if ok else "FAIL", "problems": [], "checks": checks,
            "confirm_budget": ok and (tok_s is None or round(tok_s, 6) < REFERENCE_TOK_S * SPEED_CONFIRM),
            "action": f"the episodes may start on {SCHEME}" if ok else ON_FAIL}


def gate_problems(check: dict | None, text: str, head: str | None, speed_ok: bool) -> list[str]:
    """Why a `simtest run` must not start on the server whose lines are `text`."""
    if check is None:
        return ["no setup check (check.json): the setup step runs before any episode"]
    missing, wrong = server_check(text)
    problems = missing + wrong
    if check.get("verdict") != "PASS":
        problems.append(f"the setup check says {check.get('verdict')}, not PASS")
    if check.get("scheme") != SCHEME:
        problems.append(f"the setup check ran on the scheme {check.get('scheme')}, not {SCHEME}")
    if not head or check.get("commit") != head:
        problems.append(f"the setup check ran on commit {check.get('commit')}, not on HEAD {head}")
    gpu = gpu_name(text)
    if gpu is not None and gpu != check.get("gpu"):
        problems.append(f"this server runs on {gpu}, and the setup check on {check.get('gpu')}: one backend per experiment")
    if (check.get("rule") or {}).get("confirm_budget") and not speed_ok:
        problems.append(f"the speed is under {REFERENCE_TOK_S * SPEED_CONFIRM:.0f} tokens/s, or missing: Victor confirms "
                        "the rescaled budget first, then SPEED_OK=1")
    return problems


# --- command line -------------------------------------------------------------------------------------


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def _text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def main_fidelity(args) -> int:
    d = Path(args.out)
    d.mkdir(parents=True, exist_ok=True)
    for name, sha in RECORDING.items():
        src, dst = Path(args.bakeoff) / name, d / name
        if hashlib.sha256(src.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"{src} is not the bake-off's recording (sha256 {sha[:12]}...)")
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise SystemExit(f"{dst} differs from {src}: the pass must predict the pinned recording")
        if not dst.exists():
            shutil.copyfile(src, dst)
    stop = threading.Event()
    poller = threading.Thread(target=_poll, args=(args.url, d / "speed.jsonl", args.interval, stop, SCHEME), daemon=True)
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
    print(f"{SCHEME}: {f['steps']} of {f['real_steps']} steps, {f['errors']} errors, {f['truncated']} at the cap; "
          f"mean similarity {_f(f['mean_sim'])} on {f['deterministic']} deterministic steps; "
          f"{_f(s['median_tok_s'], 0)} tokens/s at {AT_ONCE} at once ({s['windows']} windows)")
    if not f["complete"]:
        print("incomplete: run the same command again to redo the missing or errored steps", file=sys.stderr)
    return run.returncode


def _f(x, nd=4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def main_check(args) -> int:
    out = Path(args.out)
    fid = fidelity_summary(_lines(out / "scores.jsonl"), step_keys(_lines(out / "real.jsonl")))
    spd = speed(_lines(out / "speed.jsonl"), interval=args.interval)
    text = _text(out / "server.txt")
    missing, wrong = server_check(text)
    nz = noise(out)
    r = rule(fid, nz, recording_check(out) + missing, wrong, spd["median_tok_s"])
    check = {"verdict": r["verdict"], "commit": _commit(), "scheme": SCHEME, "gpu": gpu_name(text), "rule": r,
             "fidelity": fid, "graded": graded_fidelity(out), "speed": spd, "noise": nz, "layers": layer_count(text),
             "served": served_line(text),
             "constants": {"sim_floor": SIM_FLOOR, "max_truncated": MAX_TRUNCATED, "reference_tok_s": REFERENCE_TOK_S,
                           "speed_confirm": SPEED_CONFIRM, "at_once": AT_ONCE}}
    (out / "check.json").write_text(json.dumps(check, indent=1, default=str) + "\n")
    print(f"forced pass: {fid['steps']} of {fid['real_steps']} steps, {fid['errors']} errors, {fid['truncated']} at the "
          f"cap (at most {MAX_TRUNCATED}); mean similarity {_f(fid['mean_sim'])} (at least {SIM_FLOOR}), exact "
          f"{_f(fid['exact'], 2)}, pytest {fid['pytest_right']}")
    print(f"speed at {AT_ONCE} at once: {_f(spd['median_tok_s'], 0)} tokens/s, {spd['windows']} windows "
          f"(under {REFERENCE_TOK_S * SPEED_CONFIRM:.0f}: Victor confirms the budget)")
    for m in MODELS:
        n = nz.get(m) or {}
        print(f"noise floor, {m}: |d logprob| {_f(n.get('mean_abs_dlogprob'))}, top-1 {_f(n.get('top1_agreement'))}, "
              f"{n.get('contexts', 0)} contexts")
    g = check["graded"]
    print("graded fidelity: " + (f"faithful {_f(g['faithful'], 2)} {g['counts']}" if g.get("available")
                                 else f"unavailable ({g.get('why')})"))
    print(f"\nsetup check on {check['gpu']}: {r['verdict']}: {r['action']}")
    for x in r["problems"]:
        print(f"  problem: {x}")
    if r["checks"]:
        print("  " + ", ".join(f"{k} {'ok' if v else 'NO'}" for k, v in r["checks"].items()))
    if r["confirm_budget"]:
        print("  the speed is low or missing: the episodes wait until Victor confirms the budget (SPEED_OK=1)")
    print(f"  pin: {r['pin']}")
    print(f"wrote {out / 'check.json'}")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[r["verdict"]]


def main_gate(args) -> int:
    out, server = Path(args.out), Path(args.server)
    text = _text(server)
    check = json.loads((out / "check.json").read_text()) if (out / "check.json").exists() else None
    head = _commit()
    problems = gate_problems(check, text, head, args.speed_ok)
    if problems:
        for x in problems:
            print(f"gate: refused: {x}", file=sys.stderr)
        return 1
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "where": args.where, "commit": head, "scheme": SCHEME,
           "gpu": gpu_name(text), "served": served_line(text),
           "setup_check": {"dir": str(out), "verdict": check["verdict"], "gpu": check.get("gpu"),
                           "mean_sim": (check.get("fidelity") or {}).get("mean_sim"),
                           "truncated": (check.get("fidelity") or {}).get("truncated"),
                           "median_tok_s": (check.get("speed") or {}).get("median_tok_s"),
                           "confirm_budget": (check.get("rule") or {}).get("confirm_budget"),
                           "speed_ok": args.speed_ok}}
    log = Path(args.run_log)
    log.mkdir(parents=True, exist_ok=True)
    with (log / "backend.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"gate: {SCHEME} on {row['gpu']}, setup check PASS; recorded in {log / 'backend.jsonl'}")
    return 0


def main_lines(args) -> int:
    missing, wrong = server_check(_text(Path(args.server)))
    for x in missing + wrong:
        print(f"lines: {x}", file=sys.stderr)
    return 1 if missing or wrong else 0


def main_recording(args) -> int:
    problems = recording_check(Path(args.bakeoff))
    for x in problems:
        print(x)
    return 1 if problems else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fidelity", help="the bake-off's forced pass at 12 at once, with /metrics")
    f.add_argument("--out", required=True)
    f.add_argument("--bakeoff", default="logs/bakeoff", help="the bake-off's recording (real.jsonl, snapshot.txt)")
    f.add_argument("--url", default="http://localhost:18001", help="the simulator server, through the tunnel")
    f.add_argument("--interval", type=float, default=5.0, help="seconds between /metrics samples")
    c = sub.add_parser("check", help="the rule on the directory's files; exit 0 PASS, 1 FAIL, 2 INCOMPLETE")
    c.add_argument("--out", required=True)
    c.add_argument("--interval", type=float, default=5.0, help="the forced pass's sampling interval")
    g = sub.add_parser("gate", help="the last check before simtest run; on success, the run's record")
    g.add_argument("--out", required=True, help="the setup check's directory")
    g.add_argument("--server", required=True, help="server.txt of the servers the run will use")
    g.add_argument("--run-log", required=True, help="the run's log directory (backend.jsonl goes there)")
    g.add_argument("--where", required=True, help="which session or job, for the record")
    g.add_argument("--speed-ok", action="store_true", help="Victor confirmed the budget (SPEED_OK=1)")
    n = sub.add_parser("lines", help="the server's scheme lines alone")
    n.add_argument("--server", required=True)
    r = sub.add_parser("recording", help="is the bake-off's recording the pinned one")
    r.add_argument("--bakeoff", default="logs/bakeoff")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return {"fidelity": main_fidelity, "check": main_check, "gate": main_gate, "lines": main_lines,
            "recording": main_recording}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
