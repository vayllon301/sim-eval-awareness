"""The backend equivalence check (src/simtest/equiv.py, docs/BACKENDS.md): contexts from saved
episodes, teacher-forced on OpenAI-compatible vLLM servers (mocked here), and the comparison
with its pass rule. No network: every server is an httpx.MockTransport."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import httpx
import pytest

from simtest import equiv as Q

# --- a fake vLLM server --------------------------------------------------------------------


def render(messages, tools=None, add_generation_prompt=False) -> str:
    """A toy chat template: each message opens with <role>, an assistant's reasoning and tool
    calls are kept, and the generation prompt is the assistant's opening."""
    out = "<tools>" + json.dumps(tools) if tools else ""
    for m in messages:
        out += f"<{m['role']}>" + (m.get("reasoning") or "") + "|" + (m.get("content") or "")
        out += json.dumps(m.get("tool_calls")) if m.get("tool_calls") else ""
    return out + ("<assistant>" if add_generation_prompt else "")


def toks(text: str) -> list[int]:
    return [ord(c) % 1000 for c in text]


class FakeVLLM:
    """/v1/models, /version, /tokenize and /v1/completions with prompt_logprobs. Log-probs are a
    fixed function of the tokens, plus `offset` (another backend), plus a drift that grows with
    the number of scoring requests in flight (batch dependence, like the simulator's FP8 MoE)."""

    def __init__(self, name="agent", offset=0.0, batch_noise=0.0, version="0.32.0", flip_every=0):
        self.name, self.offset, self.batch_noise, self.version, self.flip_every = name, offset, batch_noise, version, flip_every
        self.inflight = self.peak = 0
        self.requests: list[tuple[str, dict]] = []

    def logprobs(self, tokens, inflight):
        out = [None]
        for i in range(1, len(tokens)):
            lp = -((tokens[i] * 31 + i) % 97) / 10 - 0.01 + self.offset + self.batch_noise * (inflight - 1) * math.sin(i)
            top = tokens[i] if i % 3 == 0 else (tokens[i - 1] * 7 + 1) % 1000
            if self.flip_every and i % self.flip_every == 0:
                top = (top + 1) % 1000
            if top == tokens[i]:
                out.append({str(tokens[i]): {"logprob": lp, "rank": 1, "decoded_token": "x"}})
            else:
                out.append({str(tokens[i]): {"logprob": lp, "rank": 2, "decoded_token": "x"},
                            str(top): {"logprob": lp + 0.5, "rank": 1, "decoded_token": "y"}})
        return out

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        path = request.url.path
        self.requests.append((path, body))
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": self.name, "max_model_len": 131072, "root": "/m"}]})
        if path == "/version":
            return httpx.Response(200, json={"version": self.version})
        if path == "/tokenize":
            text = render(body["messages"], body.get("tools"), body.get("add_generation_prompt", True))
            t = toks(text)
            return httpx.Response(200, json={"tokens": t, "count": len(t), "max_model_len": 131072})
        if path == "/v1/completions":
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
            try:
                await asyncio.sleep(0.02)      # let concurrent requests overlap
                lps = self.logprobs(body["prompt"], self.inflight)
            finally:
                self.inflight -= 1
            return httpx.Response(200, json={"choices": [{"index": 0, "text": "", "prompt_logprobs": lps}]})
        return httpx.Response(404, json={"error": path})


def transport(**servers: FakeVLLM) -> httpx.MockTransport:
    """One transport for several fake servers, by port: agent 18000, simulator 18001."""
    ports = {"agent": 18000, "simulator": 18001}
    by_port = {ports[k]: v for k, v in servers.items()}

    async def handler(request):
        return await by_port[request.url.port](request)

    return httpx.MockTransport(handler)


URLS = {"agent": Q.Server("http://localhost:18000/v1", "agent"),
        "simulator": Q.Server("http://localhost:18001/v1", "simulator")}


def ctx(i, model="agent", n=40, prompt_tokens=None):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": f"ticket {i} " + "y" * n},
            {"role": "assistant", "content": f"done {i}", "reasoning": "think " * (i + 1)}]
    c = {"id": f"log.eval::s{i}::1::{model}::0", "model": model, "messages": msgs, "tools": None,
         "prompt_tokens": prompt_tokens}
    if prompt_tokens == "auto":
        c["prompt_tokens"] = len(toks(render(msgs[:-1], None, True)))
    return c


def score(contexts, concurrency=1, **servers):
    servers = servers or {"agent": FakeVLLM("agent")}
    rows = asyncio.run(Q.score(contexts, {k: URLS[k] for k in servers}, concurrency=concurrency,
                               transport=transport(**servers)))
    return rows, servers


# --- parsing vLLM's prompt log-probs ----------------------------------------------------------


def test_prompt_logprobs_give_the_forced_token_and_the_top_1():
    tokens = [5, 7, 9]
    entries = [None,
               {"7": {"logprob": -0.1, "rank": 1, "decoded_token": "a"}},
               {"9": {"logprob": -2.0, "rank": 3}, "4": {"logprob": -0.3, "rank": 1}}]
    lp, top1 = Q.parse_prompt_logprobs(entries, tokens)
    assert lp == [None, -0.1, -2.0]
    assert top1 == [None, 7, 4]


def test_prompt_logprobs_of_the_wrong_length_are_refused():
    with pytest.raises(Q.EquivError, match="positions"):
        Q.parse_prompt_logprobs([None, {"7": {"logprob": -0.1, "rank": 1}}], [5, 7, 9])


def test_a_missing_forced_token_is_refused():
    with pytest.raises(Q.EquivError, match="token 7"):
        Q.parse_prompt_logprobs([None, {"4": {"logprob": -0.1, "rank": 1}}], [5, 7])


# --- contexts from saved episodes -------------------------------------------------------------


class Ev:
    """The fields of an Inspect ModelEvent the extractor reads."""

    def __init__(self, role=None, request=None, response=None, error=None, model="rawchat/agent"):
        self.event, self.role, self.error, self.model = "model", role, error, model
        self.call = None if request is None else type("Call", (), {"request": request, "response": response})()


AGENT_REQ = {"model": "agent", "tools": [{"type": "function", "function": {"name": "bash"}}],
             "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                          {"role": "assistant", "content": "", "reasoning": "r1",
                           "tool_calls": [{"id": "c1", "type": "function",
                                           "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]},
                          {"role": "tool", "content": "a.txt", "tool_call_id": "c1"}]}
AGENT_RESP = {"choices": [{"message": {"role": "assistant", "content": "Done.", "reasoning": "r2", "tool_calls": None},
                           "finish_reason": "stop"}], "usage": {"prompt_tokens": 123, "completion_tokens": 9}}


def test_a_context_is_the_request_plus_the_reply_as_the_next_turn_would_send_it():
    c = Q.context_from_event(Ev(request=AGENT_REQ, response=AGENT_RESP), "agent")
    assert c["messages"][:-1] == AGENT_REQ["messages"]
    assert c["messages"][-1] == {"role": "assistant", "content": "Done.", "reasoning": "r2"}
    assert c["tools"] == AGENT_REQ["tools"]
    assert c["prompt_tokens"] == 123


def test_a_reply_with_tool_calls_keeps_them_and_reasoning_content_counts_as_reasoning():
    resp = {"choices": [{"message": {"content": None, "reasoning_content": "rc", "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"pwd\"}"},
         "index": 0}]}, "finish_reason": "tool_calls"}]}
    last = Q.context_from_event(Ev(request=AGENT_REQ, response=resp), "agent")["messages"][-1]
    assert last == {"role": "assistant", "content": "", "reasoning": "rc", "tool_calls": [
        {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"pwd\"}"}}]}


def test_an_empty_tool_list_is_sent_as_none():
    req = {**AGENT_REQ, "tools": []}
    assert Q.context_from_event(Ev(request=req, response=AGENT_RESP), "simulator")["tools"] is None


@pytest.mark.parametrize("ev", [
    Ev(),                                                        # no recorded call
    Ev(request=AGENT_REQ, response=AGENT_RESP, error="boom"),    # a failed call
    Ev(request=AGENT_REQ, response={"choices": []}),             # no reply
    Ev(request=AGENT_REQ, response={"choices": [{"message": {"content": "half"}, "finish_reason": "length"}]}),
])
def test_unusable_events_give_no_context(ev):
    assert Q.context_from_event(ev, "agent") is None


def test_the_simulator_is_told_apart_by_its_role():
    assert Q.event_model(Ev(role="simulator", model="rawchat/simulator")) == "simulator"
    assert Q.event_model(Ev(role=None, model="rawchat/agent")) == "agent"


@pytest.mark.parametrize("n,k,want", [(10, 3, [0, 4, 9]), (3, 5, [0, 1, 2]), (5, 1, [4]), (0, 2, [])])
def test_spaced_picks_evenly_and_keeps_the_last(n, k, want):
    assert Q.spaced(list(range(n)), k) == want


def _sample(sid, epoch, n_sim, uuid=None, error=None):
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import EvalSample
    from inspect_ai.model import GenerateConfig, ModelCall, ModelOutput

    events = []
    for i in range(n_sim + 1):
        events.append(ModelEvent(model="rawchat/agent", input=[], tools=[], tool_choice="auto",
                                 config=GenerateConfig(), output=ModelOutput.from_content("rawchat/agent", f"a{i}"),
                                 call=ModelCall(request={**AGENT_REQ, "messages": AGENT_REQ["messages"] + [
                                     {"role": "user", "content": f"turn {i}"}]}, response=AGENT_RESP)))
        if i < n_sim:
            events.append(ModelEvent(model="rawchat/simulator", role="simulator", input=[], tools=[], tool_choice="none",
                                     config=GenerateConfig(), output=ModelOutput.from_content("rawchat/simulator", "o"),
                                     call=ModelCall(request={"model": "simulator", "tools": [], "messages": [
                                         {"role": "system", "content": "sim"}, {"role": "user", "content": f"cmd {i}"}]},
                                         response={"choices": [{"message": {"content": f"out {i}", "reasoning": "t"},
                                                                "finish_reason": "stop"}],
                                                   "usage": {"prompt_tokens": 20}})))
    return EvalSample(id=sid, epoch=epoch, input="x", target="", events=events, uuid=uuid, error=error)


def _write_log(tmp_path, name, samples):
    """A real .eval file: a template log from an offline mock run, with these samples."""
    from inspect_ai import Task, eval
    from inspect_ai.dataset import Sample
    from inspect_ai.log import write_eval_log

    log = eval(Task(dataset=[Sample(input="x")]), model="mockllm/model", display="none",
               log_dir=str(tmp_path / "tmpl"))[0]
    log.samples = samples
    log.eval.task_args = {"condition": "C3"}
    path = tmp_path / "logs" / name
    path.parent.mkdir(exist_ok=True)
    write_eval_log(log, str(path))
    return path


def test_contexts_from_a_log_directory(tmp_path):
    from inspect_ai.log import EvalError

    err = EvalError(message="boom", traceback="", traceback_ansi="")
    _write_log(tmp_path, "2026-11-01T00-00-00_a.eval", [
        _sample("audit", 1, n_sim=5, uuid="u1"), _sample("cron", 1, n_sim=2, uuid="u2"),
        _sample("gitsync", 1, n_sim=1, uuid="u3", error=err)])
    # a later attempt of the same sample (same uuid) replaces the earlier one
    _write_log(tmp_path, "2026-11-01T01-00-00_b.eval", [_sample("cron", 1, n_sim=3, uuid="u2")])
    contexts, notes = Q.extract_contexts([tmp_path / "logs"], episodes=12, sim_calls=2)
    by = {}
    for c in contexts:
        by.setdefault((c["sample_id"], c["model"]), []).append(c)
    assert sorted(by) == [("audit", "agent"), ("audit", "simulator"), ("cron", "agent"), ("cron", "simulator")]
    # the agent's last call holds the whole episode; the simulator's calls are spread, the last kept
    assert [c["event"] for c in by[("audit", "agent")]] == [5]
    assert by[("audit", "agent")][0]["messages"][-2] == {"role": "user", "content": "turn 5"}
    assert [c["event"] for c in by[("audit", "simulator")]] == [0, 4]
    assert [c["event"] for c in by[("cron", "simulator")]] == [0, 2]          # from the later attempt
    assert all(c["log"].endswith("_b.eval") for c in by[("cron", "agent")])
    assert all(c["condition"] == "C3" for c in contexts)
    assert notes["errored_samples"] == 1
    assert len({c["id"] for c in contexts}) == len(contexts)


def test_contexts_pick_episodes_evenly(tmp_path):
    _write_log(tmp_path, "l.eval", [_sample(f"s{i}", 1, n_sim=1, uuid=f"u{i}") for i in range(7)])
    contexts, _ = Q.extract_contexts([tmp_path / "logs" / "l.eval"], episodes=3, sim_calls=1)
    assert sorted({c["sample_id"] for c in contexts}) == ["s0", "s3", "s6"]


# --- scoring on a server ----------------------------------------------------------------------


def test_scoring_teacher_forces_the_server_tokens_without_sampling():
    rows, servers = score([ctx(0, prompt_tokens="auto")])
    (row,) = rows
    sent = [b for p, b in servers["agent"].requests if p == "/v1/completions"]
    assert len(sent) == 1
    body = sent[0]
    tokens = toks(render(ctx(0)["messages"]))
    assert body["prompt"] == tokens
    assert body["max_tokens"] == 1 and body["temperature"] == 0 and body["prompt_logprobs"] == 1
    assert body["model"] == "agent"
    assert row["n_tokens"] == len(tokens) and len(row["logprob"]) == len(tokens) and row["logprob"][0] is None
    assert row["tokens_sha256"] == Q.sha256_tokens(tokens)
    # where the reply starts, and the prompt length matches what the episode's server counted
    assert row["response_start"] == len(toks(render(ctx(0)["messages"][:-1], None, True)))
    assert row["prompt_matches_log"] is True


def test_a_prompt_that_differs_from_the_logged_count_is_flagged():
    rows, _ = score([ctx(0, prompt_tokens=7)])
    assert rows[0]["prompt_matches_log"] is False


def test_max_tokens_cuts_every_context_the_same_way():
    rows = asyncio.run(Q.score([ctx(0, n=200)], {"agent": URLS["agent"]}, concurrency=1, max_tokens=50,
                               transport=transport(agent=FakeVLLM())))
    assert rows[0]["n_tokens"] == 50 and rows[0]["truncated"] is True


def test_concurrency_batches_requests_and_a_server_error_is_recorded_not_raised():
    srv = FakeVLLM()
    bad = ctx(9)
    bad["messages"] = "not a list"
    rows, _ = score([ctx(i) for i in range(6)] + [bad], concurrency=4, agent=srv)
    assert srv.peak == 4
    assert [r["kind"] for r in rows].count("score") == 6
    (err,) = [r for r in rows if r["kind"] == "error"]
    assert err["id"] == bad["id"]


def test_only_models_with_a_server_are_scored():
    rows, _ = score([ctx(0), ctx(1, model="simulator")])
    assert [r["model"] for r in rows] == ["agent"]


# --- comparing two score files ----------------------------------------------------------------


def test_identical_servers_have_no_gap():
    a, _ = score([ctx(i) for i in range(4)])
    b, _ = score([ctx(i) for i in range(4)])
    g = Q.gap(a, b)["agent"]
    assert g["mean_abs_dlogprob"] == 0 and g["top1_agreement"] == 1.0 and g["contexts"] == 4
    assert g["positions"] == sum(r["n_tokens"] - 1 for r in a)


def test_the_noise_floor_shows_batch_dependence():
    contexts = [ctx(i) for i in range(6)]
    c1, _ = score(contexts, concurrency=1, agent=FakeVLLM(batch_noise=0.05))
    c1b, _ = score(contexts, concurrency=1, agent=FakeVLLM(batch_noise=0.05))
    c6, _ = score(contexts, concurrency=6, agent=FakeVLLM(batch_noise=0.05))
    assert Q.gap(c1, c1b)["agent"]["mean_abs_dlogprob"] == 0       # one at a time: repeatable
    assert Q.gap(c1, c6)["agent"]["mean_abs_dlogprob"] > 0.01      # batched: not


def test_the_gap_counts_logprob_differences_and_top1_flips():
    contexts = [ctx(i) for i in range(3)]
    a, _ = score(contexts)
    b, _ = score(contexts, agent=FakeVLLM(offset=0.2, flip_every=5))
    g = Q.gap(a, b)["agent"]
    assert g["mean_abs_dlogprob"] == pytest.approx(0.2)
    assert g["max_abs_dlogprob"] == pytest.approx(0.2)
    assert 0.7 < g["top1_agreement"] < 0.9
    assert g["response"]["positions"] > 0     # the reply's own tokens, reported apart


def test_token_mismatch_and_missing_contexts_are_reported():
    a, _ = score([ctx(i) for i in range(3)])
    b, _ = score([ctx(i) for i in range(2)])
    b[1] = {**b[1], "tokens_sha256": "other"}
    g = Q.gap(a, b)["agent"]
    assert g["contexts"] == 1
    assert g["token_mismatch"] == [a[1]["id"]]
    assert g["missing"] == [a[2]["id"]]


def _gate(cross_offset, noise, flip_cross=0, flip_noise=0):
    contexts = [ctx(i) for i in range(6)]
    ref, _ = score(contexts)
    noisy, _ = score(contexts, agent=FakeVLLM(offset=noise, flip_every=flip_noise))
    cand, _ = score(contexts, agent=FakeVLLM(offset=cross_offset, flip_every=flip_cross))
    return Q.verdict(Q.gap(ref, cand), Q.gap(ref, noisy))


def test_pass_when_the_cross_backend_gap_is_within_the_batching_noise():
    v = _gate(cross_offset=0.01, noise=0.02)
    assert v["status"] == "PASS" and v["models"]["agent"]["pass"] is True


def test_fail_when_the_gap_is_bigger_than_the_noise():
    assert _gate(cross_offset=0.05, noise=0.02)["status"] == "FAIL"


def test_fail_when_top1_agrees_less_than_under_batching():
    v = _gate(cross_offset=0.01, noise=0.02, flip_cross=4, flip_noise=0)
    assert v["status"] == "FAIL"
    assert v["models"]["agent"]["checks"]["top1_agreement"] is False


def test_a_token_mismatch_fails_and_a_missing_context_is_incomplete():
    contexts = [ctx(i) for i in range(3)]
    ref, _ = score(contexts)
    noisy, _ = score(contexts, agent=FakeVLLM(offset=0.02))
    cand, _ = score(contexts, agent=FakeVLLM(offset=0.01))
    v = Q.verdict(Q.gap(ref, cand[:2]), Q.gap(ref, noisy))
    assert v["status"] == "INCOMPLETE"
    cand[0] = {**cand[0], "tokens_sha256": "x"}
    assert Q.verdict(Q.gap(ref, cand), Q.gap(ref, noisy))["status"] == "FAIL"


def test_a_model_with_no_contexts_is_incomplete_never_pass():
    v = Q.verdict({}, {})
    assert v["status"] == "INCOMPLETE"


# --- files and the command line ---------------------------------------------------------------


def test_score_files_round_trip_and_refuse_to_be_overwritten(tmp_path):
    rows, _ = score([ctx(0)])
    p = tmp_path / "s.jsonl"
    Q.write_scores(p, {"concurrency": 1}, rows)
    header, back = Q.read_scores(p)
    assert header["concurrency"] == 1 and back == rows
    with pytest.raises(Q.EquivError, match="exists"):
        Q.write_scores(p, {}, rows)


def test_compare_refuses_files_scored_on_different_contexts(tmp_path):
    rows, _ = score([ctx(0)])
    Q.write_scores(tmp_path / "a.jsonl", {"contexts_sha256": "1", "top_k": 1, "max_tokens": None}, rows)
    Q.write_scores(tmp_path / "b.jsonl", {"contexts_sha256": "2", "top_k": 1, "max_tokens": None}, rows)
    with pytest.raises(SystemExit, match="different contexts"):
        Q.main_compare(tmp_path / "a.jsonl", tmp_path / "b.jsonl", None)


def test_cli_contexts_score_and_compare(tmp_path, monkeypatch, capsys):
    from simtest.cli import main

    _write_log(tmp_path, "l.eval", [_sample(f"s{i}", 1, n_sim=2, uuid=f"u{i}") for i in range(3)])
    cfile = tmp_path / "contexts.jsonl"
    main(["equiv", "contexts", str(tmp_path / "logs"), "--out", str(cfile), "--episodes", "3", "--sim-calls", "1"])
    assert len(cfile.read_text().splitlines()) == 6

    servers = {"g4": dict(agent=FakeVLLM("agent", batch_noise=0.05), simulator=FakeVLLM("simulator", batch_noise=0.05)),
               "hopper": dict(agent=FakeVLLM("agent", offset=0.001), simulator=FakeVLLM("simulator", offset=0.001))}
    current = {}
    monkeypatch.setattr(Q, "_transport", lambda: transport(**current["s"]))
    common = ["--agent-url", "http://localhost:18000/v1", "--simulator-url", "http://localhost:18001/v1"]
    current["s"] = servers["g4"]
    main(["equiv", "score", str(cfile), *common, "--concurrency", "1", "3", "--out", str(tmp_path / "g4")])
    current["s"] = servers["hopper"]
    main(["equiv", "score", str(cfile), *common, "--out", str(tmp_path / "hopper")])
    assert (tmp_path / "g4-c1.jsonl").exists() and (tmp_path / "g4-c3.jsonl").exists()
    header, _ = Q.read_scores(tmp_path / "hopper-c1.jsonl")
    assert header["servers"]["agent"]["version"] == "0.32.0" and header["concurrency"] == 1
    capsys.readouterr()
    main(["equiv", "compare", str(tmp_path / "g4-c1.jsonl"), str(tmp_path / "hopper-c1.jsonl"),
          "--noise", str(tmp_path / "g4-c3.jsonl")])
    out = capsys.readouterr().out
    assert "PASS" in out and "agent" in out and "simulator" in out
