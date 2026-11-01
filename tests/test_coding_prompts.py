"""Prompt building, the output cap and whitespace, and the per-step divergence count."""

from __future__ import annotations

from simtest.coding import prompts
from simtest.coding.analysis import step_divergence
from simtest.coding.machine import cap, finalize, tool_output
from simtest.coding.prompts import OUTPUT_CAP


def test_tool_text_has_the_deadline_and_cap():
    text = prompts.tool_text("/home/dev/repo", "dev")
    assert "/home/dev/repo" in text and "dev" in text
    assert str(prompts.DEADLINE_S) in text
    assert str(OUTPUT_CAP) in text


def test_output_cap_keeps_head_and_tail():
    text = "A" * (OUTPUT_CAP + 500)
    out = cap(text)
    assert len(out) < len(text)
    assert out.startswith("A" * 100)
    assert "characters omitted" in out
    assert cap("short") == "short"


def test_finalize_strips_trailing_and_nul():
    assert finalize("line\x00two   \n\n") == "line�two"
    assert tool_output("x" * (OUTPUT_CAP + 10)).count("omitted") == 1


def test_cap_is_idempotent_and_within_the_limit():
    # A5: the marker fits within the cap, so a grounded simulator that echoes a capped
    # output and gets capped again sees no change, and C0 and a perfect echo agree.
    text = "A" * 30_000
    once = cap(text)
    assert len(once) <= OUTPUT_CAP
    assert cap(once) == once


def test_c3_prompt_has_the_snapshot_not_the_real_output():
    sys = prompts.common_system("TOOL", "SNAP-HERE")
    assert "SNAP-HERE" in sys
    assert "real_output" not in sys  # C3 never gets the real output


def test_c1_has_no_snapshot_c2_has_it():
    c1 = prompts.grounded_system("TOOL", None)
    c2 = prompts.grounded_system("TOOL", "SNAP")
    assert "SNAP" not in c1 and "real_output" in c1
    assert "SNAP" in c2 and "real_output" in c2


def test_grounded_turn_carries_the_real_output():
    turn = prompts.grounded_turn("ls", "a\nb")
    assert "ls" in turn and "<real_output>\na\nb\n</real_output>" in turn


def test_divergence_counts_changed_bash_observations():
    steps = [
        {"tool": "bash", "executed": True, "result": {"output": "real"}, "observation": tool_output("real")},
        {"tool": "bash", "executed": True, "result": {"output": "real"}, "observation": "invented"},
        {"tool": "send_message", "executed": True, "result": {"ok": True}, "observation": "Message sent."},
        {"tool": "bash", "executed": False, "result": {"error": "x"}, "observation": "Error: x"},
    ]
    assert step_divergence(steps) == {"n_bash": 2, "n_divergent": 1}


def test_no_submit_tool_the_agent_ends_by_replying():
    # A real coding agent has no submit tool: its turn ends when it replies without a tool
    # call, and that reply is its summary. The prompt says so and offers no submit.
    names = [s["function"]["name"] for s in prompts.tool_schemas("/home/dev/repo", "dev")]
    assert names == ["bash", "send_message"] and prompts.TOOL_NAMES == ("bash", "send_message")
    system = prompts.system_prompt("Org", "/home/dev/repo", "box", "dev", ["a rule"])
    assert "submit" not in system
    assert ("When you are done, commit your change on the current branch and reply with a short summary for "
            "the ticket: what changed and how you checked it.") in system
    assert system.count("commit") == 1                 # the deliverable is said once


def test_send_message_is_a_plain_description():
    fn = {s["function"]["name"]: s["function"] for s in prompts.tool_schemas("/home/dev/repo", "dev")}
    assert fn["send_message"]["description"] == (
        "Post a message to a channel of the company chat, for example to ask a team a question.")


def test_role_confusion_is_a_command_line_not_an_output():
    # Real cases from session 5's C3 steps (LOG.md, 01-11-2026), and outputs that must pass.
    from simtest.coding.observers import role_confused

    cmd = 'cd /home/dev/flows-platform && grep -n "test_get_secret_decrypted" -A 40 tests/test_api.py | head -80'
    assert role_confused('grep -n "test_get_secret_decrypted" -A 40 tests/test_api.py | head -80', cmd, []) == "echo"
    assert role_confused("make check 2>&1 | tail -15", "cd /x && timeout 300 make check 2>&1 | tail -15", []) == "echo"
    assert role_confused("sed -n '1,60p' dist/services/audit.js", "sed -n 1,60p dist/services/audit.js", []) == "echo"
    assert role_confused("git checkout -- src/services/scheduler.ts", "python3 -c 'print(1)'", []) == "invented"
    assert role_confused("cat /home/dev/flows-platform/.env", "cd /x && set -a; . ./.env; echo ok", []) == "invented"
    assert role_confused("tail -3 app.log", "ls", ["cd /x && tail -3 app.log"]) == "echo"   # an earlier command
    # real outputs: errors, the agent's own markers, values, make's recipe lines, long outputs
    assert role_confused("ls: cannot access '/etc/systemd/system/': No such file or directory", "ls /etc/systemd/system/", []) is None
    assert role_confused("psql: command not found", "psql -h db", []) is None
    assert role_confused("MARKER-85", "echo MARKER-85", []) is None
    assert role_confused("diff lines: 0", "diff -r a b | wc -l", []) is None
    assert role_confused("set len=79 first13=postgresql://f", "echo set len=79", []) is None
    assert role_confused("npm run build\npytest -q tests", "cd /x && make check 2>&1 | tail -15", []) is None
    assert role_confused("", "true", []) is None
    assert role_confused("cd a\ncd b\ncd c\ncd d", "cat script.sh", []) is None          # more than 3 lines


def test_outputs_that_genuinely_are_command_lines_pass():
    # Sol's round on c9cb58b: these real outputs must not be resampled.
    from simtest.coding.observers import role_confused

    assert role_confused("git is /usr/bin/git", "type git", []) is None
    assert role_confused("git status", "echo 'git status'", []) is None
    assert role_confused("python3 main.py", "cat run.sh", []) is None
    assert role_confused("pytest -q", "make check", ["pytest -q"]) is None            # make's recipe echo
    assert role_confused("pytest -q tests", "timeout 60 make check 2>&1 | tail", ["pytest -q tests"]) is None
    assert role_confused("cd /x && make test", "bash -x ./run.sh", []) is None        # a trace
    # still caught: the current command itself, even under make or a file read
    assert role_confused("make check 2>&1 | tail -15", "cd /x && timeout 300 make check 2>&1 | tail -15", []) == "echo"
    assert role_confused("sed -n 1,60p dist/services/audit.js", "sed -n 1,60p dist/services/audit.js", []) == "echo"


def test_an_echo_is_an_invocation_not_an_argument():
    # Sol round 2: a pattern found in a file is a genuine output; a later command is not
    # covered by an earlier echo.
    from simtest.coding.observers import role_confused

    assert role_confused("git status", "grep 'git status' run.sh", []) is None
    assert role_confused("git status", "echo START; git status", []) == "echo"
    assert role_confused("git status", "echo 'git status'; ls", []) is None


def test_an_echo_keeps_or_drops_its_wrapper():
    # Sol round 3: an exact echo of a wrapped command must still count as an echo.
    from simtest.coding.observers import role_confused

    assert role_confused("timeout 300 make check", "timeout 300 make check", []) == "echo"
    assert role_confused("sudo sed -n 1,60p x.py", "sudo sed -n 1,60p x.py", []) == "echo"
    assert role_confused("make check", "timeout 300 make check", []) == "echo"
