"""`simtest preflight CONFIG`: run a coding config end to end offline before a GPU session.

Everything a GPU session runs, except the models: the config is loaded and validated, the
Docker images are built, the frozen snapshots are checked against fresh containers, and every
episode of the config (or of `--epochs`) runs through the same queue, the same number in flight,
the real machines, the observers, the shadow execution, the probe, the scorer and the analysis.
The agent and the simulator are scripted mock models (mockllm): the agent runs a few harmless
commands in its box, then replies without a tool call; the simulator answers each command with a
fixed line. No network, no API key, no GPU.

It passes only if every episode is scored with a valid end, none errors, every simulated episode
called its simulator, and the queue never ran more than the config's total at once. On a clean
tree it then writes `logs/preflight/<name>.ok` holding the commit, which
`explore/colab_probe/session.sh` checks before it starts a VM. The logs go to
`logs/preflight/<name>/`, replaced on each preflight; they are never evidence.

It lives outside `coding/`, so changing it never changes a run's fingerprint.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.tool import ToolCall

from . import runqueue
from .coding.analysis import load_run
from .coding.episode import VALID_END_REASONS
from .coding.run import REPO_DIR, git_dirty, load_config, run_config
from .task import NO_SIMULATOR

PREFLIGHT_ROOT = Path("logs") / "preflight"
COMMANDS = ["pwd && ls -la", "git status --short && git log --oneline -3"]
AGENT_DONE = "Preflight: nothing to change."
SIMULATED = "preflight: a simulated output line"


def _agent(input, tools, tool_choice, config):
    turn = sum(isinstance(m, ChatMessageAssistant) for m in input)
    if turn >= len(COMMANDS):
        return ModelOutput.from_content("mockllm/agent", AGENT_DONE)
    call = ToolCall(id=f"pf{turn}", function="bash", arguments={"command": COMMANDS[turn]})
    msg = ChatMessageAssistant(content="", tool_calls=[call], model="mockllm/agent", source="generate")
    return ModelOutput(model="mockllm/agent", choices=[ChatCompletionChoice(message=msg, stop_reason="tool_calls")])


def _simulator(input, tools, tool_choice, config):
    return ModelOutput.from_content("mockllm/simulator", SIMULATED)


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def expected(design, epochs) -> list[tuple]:
    return [(c, s, sc, e) for c in design["conditions"]
            for s in ([NO_SIMULATOR] if c == "C0" else design["simulators"][c])
            for sc in design["scenarios"] for e in range(1, design["epochs"] + 1)
            if epochs is None or e in epochs]


def check(log_dir: Path, in_flight: int, epochs: set[int] | None) -> list[str]:
    """What is wrong with a preflight's logs (empty when it passed)."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    episodes, design = load_run(log_dir)
    if design is None:
        return ["no episode logs"]
    problems = [f"{e['condition']}/{e['simulator']} {e['scenario']} epoch {e['epoch']}: {e['end_reason']} "
                f"{(e.get('error') or '')[:200]}".rstrip()
                for e in episodes if e["end_reason"] not in VALID_END_REASONS]
    got = {(e["condition"], e["simulator"], e["scenario"], e["epoch"]) for e in episodes
           if e["end_reason"] in VALID_END_REASONS}
    problems += [f"missing: {'/'.join(map(str, k))}" for k in expected(design, epochs) if k not in got]
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        if log.eval.task_args.get("condition") == "C0":
            continue
        for s in log.samples or []:
            if s.error is None and not (s.store or {}).get("CodingEpisode:simulator_calls"):
                problems.append(f"{log.eval.task_args['condition']} {s.id} epoch {s.epoch}: no simulator call")
    queue_log = log_dir / "queue.jsonl"
    events = [json.loads(line) for line in queue_log.read_text().splitlines()] if queue_log.exists() else []
    admits = [e for e in events if e["event"] == "admit"]
    if not admits:
        problems.append("the queue admitted nothing")
    elif (peak := max(sum(e["running"].values()) for e in admits)) > in_flight:
        problems.append(f"the queue ran {peak} at once, above {in_flight}")
    return problems


def preflight(path, epochs=None, root=PREFLIGHT_ROOT, machine="docker", build=True,
              check_snapshots=True) -> tuple[bool, list[str], Path, bool]:
    """Returns (passed, problems, log dir, recorded)."""
    cfg = load_config(path)
    root = Path(root)
    log_dir = root / cfg["name"]
    marker = root / f"{cfg['name']}.ok"
    marker.unlink(missing_ok=True)
    if log_dir.exists():
        shutil.rmtree(log_dir)
    sims = {s["model"] for s in cfg["simulators"].values()}
    if sims & {s["model"] for s in cfg["agents"].values()}:
        raise SystemExit("an agent and a simulator share a served name: the preflight can't tell them apart")
    allowed = runqueue.parse_epochs(epochs, cfg["epochs"])
    dirty = git_dirty()

    def models(spec):
        sim = spec["model"] in sims
        return get_model(f"mockllm/{'simulator' if sim else 'agent'}",
                         custom_outputs=_simulator if sim else _agent, memoize=False)

    run_config(path, allow_dirty=True, log_root=root, display="none", build=build, check_snapshots=check_snapshots,
               epochs=epochs, machine=machine, models=models)
    problems = check(log_dir, cfg["run"]["max_connections"], allowed)
    recorded = False
    if not problems and not dirty and (commit := _commit()):
        marker.write_text(commit + "\n")
        recorded = True
    return not problems, problems, log_dir, recorded


def add_parser(sub) -> None:
    p = sub.add_parser("preflight", help="coding: run a config end to end offline (mock models, real Docker)")
    p.add_argument("config", type=Path)
    p.add_argument("--epochs", default=None, help="N or A-B: only these epochs (default: all)")
    p.add_argument("--no-build", action="store_true", help="don't build the images first")


def main(args) -> int:
    ok, problems, log_dir, recorded = preflight(args.config, epochs=args.epochs, build=not args.no_build)
    if ok:
        print(f"PREFLIGHT PASSED: {log_dir}" + ("; recorded for this commit" if recorded
                                                 else " (dirty tree: not recorded; commit and re-run)"))
        return 0
    print(f"PREFLIGHT FAILED ({len(problems)} problems), logs in {log_dir}:")
    for p in problems[:50]:
        print("  " + p)
    return 1
