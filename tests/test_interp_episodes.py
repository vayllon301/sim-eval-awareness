"""The Inspect adapter and the export of a run's log directory: an episode is rebuilt from the
requests its server received, every check on that rebuild is kept, and every episode of a run is
either exported or listed with the reason. Fabricated samples only (tests/interp_helpers.py); the
same code was run on the real session logs (docs/INTERP.md, "Inspect episodes")."""

import json
from types import SimpleNamespace as NS

import pytest

from interp_helpers import SYSTEM, TASK, TOOLS, sample
from simtest.interp import episodes as E
from simtest.interp.fake import FakeTokenizer
from simtest.interp.reader import load_episode, prompt_checks, render, save_episode, turn_prompts


def test_the_episode_is_the_last_request_plus_the_last_reply():
    ep = E.from_inspect(sample())
    assert [m["role"] for m in ep.messages] == ["system", "user", "assistant", "tool", "assistant", "tool", "tool",
                                                "assistant"]
    assert ep.turns == [-1, -1, 0, 0, 1, 1, 1, 2]
    assert ep.tools == TOOLS
    assert ep.messages[0] == {"role": "system", "content": SYSTEM} and ep.messages[1]["content"] == TASK
    a = ep.messages[2]
    assert a["reasoning_content"] == "Let me look." and a["content"] == ""
    assert a["tool_calls"] == [{"type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}]
    assert "reasoning" not in a
    assert ep.messages[-1] == {"role": "assistant", "content": "The export is fixed.", "reasoning_content": "Done now."}


def test_the_servers_counts_and_the_episode_are_kept():
    m = E.from_inspect(sample(uuid="abc"), source="x.eval").meta
    assert m["adapter"] == "inspect" and m["uuid"] == "abc" and m["source"] == "x.eval"
    assert m["sample_id"] == "flows_audit" and m["epoch"] == 1
    assert m["prompt_tokens"] == [100, 150, 200] and m["completion_tokens"] == [20, 21, 22]
    assert m["n_turns"] == 3 and m["failed_agent_calls"] == 0 and m["unread_outputs_dropped"] == 0
    assert m["requests_are_prefixes"] is True and m["replay_differs"] == [] and m["partial_last_turn"] is False


def test_outputs_no_turn_read_are_dropped_and_counted():
    turns = [("r", "", [("bash", {"command": "ls"}, "a")]), ("r2", "", [("bash", {"command": "ls"}, "b")])]
    ep = E.from_inspect(sample(turns, end="max_turns", unread_output=True))
    assert [m["role"] for m in ep.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert ep.meta["unread_outputs_dropped"] == 1


def test_a_failed_agent_call_is_counted_not_read():
    s = sample()
    s.events.insert(2, NS(event="model", role=None, error="Connection error.", call=None, output=NS(usage=None)))
    ep = E.from_inspect(s)
    assert ep.meta["failed_agent_calls"] == 1 and ep.meta["n_turns"] == 3


def test_a_history_that_changed_between_requests_is_recorded():
    s = sample()
    s.events[1].call.request["messages"][1] = {"role": "user", "content": "another task"}
    assert E.from_inspect(s).meta["requests_are_prefixes"] is False


def test_a_reply_replayed_differently_is_recorded():
    s = sample()
    calls = [e for e in s.events if e.event == "model" and e.role is None]
    calls[-1].call.request["messages"][2]["reasoning"] = "not what it said"
    assert E.from_inspect(s).meta["replay_differs"] == [0]


def test_a_reply_with_a_cut_stop_reason_is_a_partial_last_turn():
    s = sample()
    [e for e in s.events if e.event == "model" and e.role is None][-1].output.stop_reason = "max_tokens"
    assert E.from_inspect(s).meta["partial_last_turn"] is True


def test_a_sample_without_requests_is_refused():
    s = sample()
    for e in s.events:
        if e.event == "model":
            e.call = None
    with pytest.raises(E.AdapterError, match="no agent call"):
        E.from_inspect(s)


def test_unresolved_attachments_are_refused():
    s = sample()
    E.agent_calls(s)[-1].call.request["messages"][1]["content"] = "attachment://abc"
    with pytest.raises(E.AdapterError, match="attachment"):
        E.from_inspect(s)


def test_the_adapter_output_renders_and_lines_up_with_its_prompts():
    tok = FakeTokenizer()
    ep = E.from_inspect(sample())
    rd = render(ep, tok)
    prompts = turn_prompts(ep, tok)
    assert len(prompts) == 3 and rd.text.startswith("<|im_start|>system")
    assert all(rd.ids[:len(p)].tolist() == p for p in prompts)
    assert [c["recorded"] for c in prompt_checks(ep, tok, prompts)] == [100, 150, 200]


def test_save_and_load_keep_the_episode(tmp_path):
    ep = E.from_inspect(sample())
    p = tmp_path / "e.json"
    save_episode(ep, p, wire={"messages": []})
    back = load_episode(p)
    assert back.messages == ep.messages and back.turns == ep.turns and back.tools == ep.tools
    assert back.meta == ep.meta and back.template_kwargs == ep.template_kwargs
    assert json.loads(p.read_text())["wire"] == {"messages": []}


def test_save_keeps_the_order_of_keys_the_template_renders(tmp_path):
    """The template prints tools with `tojson` and arguments in their order, so a sorted file renders
    other tokens (2 fewer on every prompt of the real logs)."""
    ep = E.from_inspect(sample())
    ep.tools = [{"type": "function", "function": {"name": "bash", "description": "d", "parameters": {}}}]
    ep.messages[2]["tool_calls"][0]["function"]["arguments"] = {"z": "1", "a": "2"}
    p = tmp_path / "e.json"
    save_episode(ep, p)
    back = load_episode(p)
    assert list(back.tools[0]) == ["type", "function"] and list(back.tools[0]["function"])[0] == "name"
    assert list(back.messages[2]["tool_calls"][0]["function"]["arguments"]) == ["z", "a"]


# --- a run ---------------------------------------------------------------------------------------

def run_record(uuid, condition="C0", scenario="flows_audit", epoch=1, end_reason="no_tool_call", log_path="x.eval", **over):
    return {"uuid": uuid, "condition": condition, "scenario": scenario, "variant": "tempting", "epoch": epoch,
            "simulator": "none", "end_reason": end_reason, "error": None, "log": "x.eval", "log_path": log_path,
            "sample_id": scenario, "metrics": {}, "notes": [], **over}


def export(tmp_path, records, read=None):
    design = {"conditions": ["C0"], "scenarios": ["flows_audit"], "epochs": 1}
    return E.export_run(tmp_path / "logs" / "n2-pilot", tmp_path / "out", load=lambda d: (records, design),
                        read=read or (lambda e: sample(uuid=e["uuid"])))


def test_every_episode_is_exported_or_listed_with_its_reason(tmp_path):
    def read(e):
        if e["uuid"] == "bad":
            raise E.AdapterError("no agent call")
        return sample(uuid=e["uuid"])

    records = [run_record("good"), run_record("flagged", end_reason="simulator_truncated"), run_record("bad"),
               run_record(None, end_reason="error", log_path=None, attempt={"id": "a1"}, notes=["rerun_in_queue"])]
    m = export(tmp_path, records, read)
    by = {r["uuid"]: r for r in m["episodes"]}
    assert by["good"]["status"] == "exported" and by["good"]["scored"] is True
    assert by["flagged"]["status"] == "exported" and by["flagged"]["scored"] is False
    assert by["bad"]["status"] == "adapter_error" and "no agent call" in by["bad"]["reason"]
    assert by[None]["status"] == "not_read" and by[None]["end_reason"] == "error"
    assert sorted(p.name for p in (tmp_path / "out").glob("*.json")) == ["flagged.json", "good.json", "manifest.json"]
    assert len(by["good"]["sha256"]) == 64 and by["good"]["tokens"] == 222
    assert load_episode(tmp_path / "out" / "good.json").meta["uuid"] == "good"
    assert json.loads((tmp_path / "out" / "manifest.json").read_text())["run"] == "n2-pilot"
    assert m["counts"] == {"adapter_error": 1, "exported": 2, "not_read": 1}


def test_the_wire_request_is_kept_for_the_servers_tokenizer(tmp_path):
    export(tmp_path, [run_record("good")])
    w = json.loads((tmp_path / "out" / "good.json").read_text())["wire"]
    assert w["model"] == "agent" and w["tools"] == TOOLS
    assert w["messages"][-1]["role"] == "tool"    # the last request: what the last turn read


def test_export_refuses_a_directory_of_another_run(tmp_path):
    export(tmp_path, [run_record("good")])
    (tmp_path / "out" / "manifest.json").write_text(json.dumps({"run": "other"}))
    with pytest.raises(E.AdapterError, match="another run"):
        export(tmp_path, [run_record("good")])
