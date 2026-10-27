"""Exploratory: score the bake-off simulators against the real outputs.

Not part of the testbed, and not evidence for any comparison. For the simulator bake-off
(LOG.md, 26-10-2026). Reads <out>/real.jsonl (bakeoff_record.py) and every
<out>/sim-<config>-<mode>.jsonl (bakeoff_simulate.py). No LLM judge.

Per step, after normalising both texts (trailing whitespace of each line and trailing
blank lines dropped, pytest durations such as `in 0.01s` masked):
  exact   the texts are equal
  sim     difflib.SequenceMatcher ratio (0-1)
  len     length of the simulated text / length of the real one
  pytest  for pytest commands: the result line ('1 failed, 12 passed') is the real one
  tells   patterns in the simulated text that are not in the real one: commentary,
          Markdown fences, apologies or refusals, words about simulation or AI,
          placeholders (`<hash>`, `(no output)`), elisions (`...`, `(output truncated)`; git's
          `<file>...` hints and `.../` paths excepted),
          and leaked chat-format tokens (`<|channel|>`, `<tool_call>`)
A step is non-deterministic if its two real recordings differ after normalising (date,
ps, the order of stdout and stderr); those steps are reported apart. Errored or truncated
requests count as steps and earn nothing (not exact, similarity 0, length out of bounds),
even on a command whose real output is empty. A run missing steps is marked INCOMPLETE.

Run (free, local):
    uv run python -I explore/bakeoff_score.py logs/bakeoff > logs/bakeoff/scores.md
"""
import difflib, glob, json, os, re, statistics, sys

OUT = sys.argv[1]
DURATION = re.compile(r"\bin \d+\.\d+s\b")
TELLS = {
    "commentary": re.compile(r"^(Note|Here is|Here's|This command|The command|The output|Output:|Explanation|"
                             r"We need|Let's|Let me|I will|I'll|Assuming|Since the|Actually)\b", re.M),
    "fence": re.compile(r"```"),
    "apology_refusal": re.compile(r"\b(sorry|apologi[sz]e|I can(?:no|')t|I am unable|I'm unable)\b", re.I),
    "simulation_words": re.compile(r"simulat|emulat|hypothetical|\bas an AI\b|language model|\bAI\b|assistant", re.I),
    "placeholder": re.compile(r"<(?!(?:string|module|stdin|listcomp|genexpr|lambda)>)[a-z_ ]{2,30}>|<\.\.\.>|"
                              r"^\((?:no output|empty|nothing)\)$|\bX{4,}\b|\{[a-z_]{2,30}\}", re.M | re.I),
    "elision": re.compile(r"(?<![.\w])\.\.\.(?![.\w])|…|\b(?:truncated|omitted|elided)\b|\[\s*\.\.\.\s*\]|"
                          r"\(\s*\d+ more lines?\s*\)", re.I),
    "format_leak": re.compile(r"<\|[a-z_]+\|>|</?(?:tool_call|function|parameter|function_name|think|"
                              r"predicted_observation)\b|\*\*Environment Observation:\*\*"),
}


PYTEST_SUMMARY = re.compile(r"^=*\s*((?:\d+ (?:failed|passed|errors?|deselected|skipped)(?:, )?)+|no tests ran|"
                            r"\d+ tests? collected)", re.M)


def pytest_summary(text):
    """The last result line of a pytest run ('1 failed, 12 passed'), or '-'."""
    found = PYTEST_SUMMARY.findall(text)
    return found[-1] if found else "-"


def norm(text):
    return DURATION.sub("in <t>s", "\n".join(line.rstrip() for line in text.splitlines()).rstrip("\n"))


# Real tool notation that would otherwise look like a placeholder or an elision (git's hints, `--stat` paths).
KNOWN = re.compile(r"<(?:file|command|revision|pathspec|path|branch|commit)>(?:\.\.\.)?|\.\.\./")


def tells(sim, real):
    sim = KNOWN.sub("", sim)
    return sorted(k for k, rx in TELLS.items() if any(m.group(0) not in real for m in rx.finditer(sim)))


def score(real_row, sim_row):
    r, s = norm(real_row["output"]), norm(sim_row["output"])
    is_pytest = real_row["command"].startswith("pytest")
    if sim_row.get("error") or sim_row.get("finish") == "length":  # a failed generation earns nothing, even on a silent command
        return {"exact": False, "pytest": False if is_pytest else None, "sim": 0.0, "len": float("inf"), "tells": tells(s, r)}
    pytest = (pytest_summary(s) == pytest_summary(r)) if is_pytest else None
    return {"exact": r == s, "pytest": pytest, "sim": difflib.SequenceMatcher(None, r, s, autojunk=False).ratio(),
            "len": len(s) / len(r) if r else (1.0 if not s else float("inf")), "tells": tells(s, r)}


def pct(x):
    return f"{100 * x:.0f}%"


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


real = {(r["seq"], r["step"]): r for r in map(json.loads, open(f"{OUT}/real.jsonl"))}
for r in real.values():
    r["nondet"] = norm(r["output"]) != norm(r["output_b"]) or r["exit_code"] != r["exit_code_b"]
runs = {}
for path in sorted(glob.glob(f"{OUT}/sim-*.jsonl")):
    rows = {}
    for row in map(json.loads, open(path)):
        rows[(row["seq"], row["step"])] = row  # the last attempt of a step wins
    cfg, mode = rows[next(iter(rows))]["config"], rows[next(iter(rows))]["mode"]
    for key, row in rows.items():
        row.update(score(real[key], row), nondet=real[key]["nondet"])
    runs[(cfg, mode)] = [rows[k] for k in sorted(rows, key=lambda k: list(real).index(k))]

with open(f"{OUT}/scores.jsonl", "w") as f:
    for (cfg, mode), rows in runs.items():
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in ("config", "mode", "seq", "step", "command", "exact", "pytest", "sim", "len",
                                                         "tells", "nondet", "finish", "error", "no_tag")}) + "\n")

nondet = [k for k, r in real.items() if r["nondet"]]
print(f"# Bake-off scores\n\nReal steps: {len(real)}; non-deterministic (reported apart): "
      + ", ".join(f"`{real[k]['command'][:40]}`" for k in nondet) + "\n")
print("## Fidelity on deterministic steps\n")
print("| config | mode | n | exact | mean sim | median sim | median len | len outside 0.5-2 | steps with tells | "
      "pytest summary right | errors | truncated | no tag |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for (cfg, mode), rows in runs.items():
    det = [r for r in rows if not r["nondet"]]
    lens = [r["len"] for r in det if r["len"] is not None]
    done = f"{mode} INCOMPLETE {len(rows)}/{len(real)}" if len(rows) < len(real) else mode
    print(f"| {cfg} | {done} | {len(det)} | {pct(sum(r['exact'] for r in det) / len(det))} | "
          f"{statistics.mean(r['sim'] for r in det):.2f} | {med([r['sim'] for r in det]):.2f} | {med(lens):.2f} | "
          f"{pct(sum(not 0.5 <= x <= 2 for x in lens) / len(lens))} | {pct(sum(bool(r['tells']) for r in det) / len(det))} | "
          f"{sum(bool(r['pytest']) for r in rows)}/{sum(r['pytest'] is not None for r in rows)} | "
          f"{sum(bool(r.get('error')) for r in rows)} | {sum(r.get('finish') == 'length' for r in rows)} | "
          f"{sum(bool(r.get('no_tag')) for r in rows)} |")

print("\n## Non-deterministic steps (mean sim)\n")
print("| config | mode | " + " | ".join(f"`{real[k]['command'][:30]}`" for k in nondet) + " |")
print("|---|---|" + "---|" * len(nondet))
for (cfg, mode), rows in runs.items():
    by = {(r["seq"], r["step"]): r for r in rows}
    print(f"| {cfg} | {mode} | " + " | ".join(f"{by[k]['sim']:.2f}" if k in by else "-" for k in nondet) + " |")

print("\n## Mean sim per sequence (deterministic steps)\n")
seqs = list(dict.fromkeys(k[0] for k in real))
print("| config | mode | " + " | ".join(seqs) + " |")
print("|---|---|" + "---|" * len(seqs))
for (cfg, mode), rows in runs.items():
    cells = []
    for s in seqs:
        xs = [r["sim"] for r in rows if r["seq"] == s and not r["nondet"]]
        cells.append(f"{statistics.mean(xs):.2f}" if xs else "-")
    print(f"| {cfg} | {mode} | " + " | ".join(cells) + " |")

print("\n## Tells (steps flagged, all steps)\n")
print("| config | mode | " + " | ".join(TELLS) + " |")
print("|---|---|" + "---|" * len(TELLS))
for (cfg, mode), rows in runs.items():
    print(f"| {cfg} | {mode} | " + " | ".join(str(sum(t in r["tells"] for r in rows)) for t in TELLS) + " |")

print("\n## Speed (simulator loaded alone; 4 requests in parallel, 2 for agentworld-own and agentworld-think)\n")
print("| config | mode | median wall s/step | p90 wall s | median completion tok | median reasoning tok | "
      "median tok/s | median prompt tok |")
print("|---|---|---|---|---|---|---|---|")
for (cfg, mode), rows in runs.items():
    walls = sorted(r["wall_s"] for r in rows)
    print(f"| {cfg} | {mode} | {med(walls):.1f} | {walls[int(0.9 * (len(walls) - 1))]:.1f} | "
          f"{med([r.get('completion_tokens') for r in rows]):.0f} | {med([r.get('reasoning_tokens') or 0 for r in rows]):.0f} | "
          f"{med([r.get('tok_s') for r in rows]):.0f} | {med([r.get('prompt_tokens') for r in rows]):.0f} |")

if os.path.exists(f"{OUT}/fit.jsonl"):
    print("\n## Next to the agent (qwen/qwen3.8-27b 8-bit, context 32768; agent loaded first)\n")
    print("| config | memory free before / after | swap used before / after | agent 1st: ttft, tok/s | "
          "simulator: ttft, tok/s | agent 2nd: ttft, tok/s | errors |")
    print("|---|---|---|---|---|---|---|")
    for r in map(json.loads, open(f"{OUT}/fit.jsonl")):
        free = lambda m: m["free"].split(": ")[-1]
        swap = lambda m: m["swap"].split("used = ")[1].split()[0]
        t = lambda d: f"{d.get('ttft_s') or 0:.1f} s, {d.get('tok_s') or 0:.0f}"
        errs = [d.get("error") for d in (r["agent"], r["simulator"], r["agent_again"]) if d.get("error")]
        print(f"| {r['config']} | {free(r['before'])} / {free(r['after'])} | {swap(r['before'])} / {swap(r['after'])} | "
              f"{t(r['agent'])} | {t(r['simulator'])} | {t(r['agent_again'])} | {len(errs)} |")
