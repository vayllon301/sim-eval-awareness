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
        {"tool": "submit", "executed": True, "result": {"ok": True}, "observation": "Submitted."},
        {"tool": "bash", "executed": False, "result": {"error": "x"}, "observation": "Error: x"},
    ]
    assert step_divergence(steps) == {"n_bash": 2, "n_divergent": 1}
