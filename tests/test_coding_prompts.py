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
    assert "When you are done, reply with a short summary for the ticket." in system
