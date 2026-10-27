"""Exploratory: how fast does a local model run a coding-agent turn with thinking on?

Not part of the testbed, and not evidence for any comparison. Sends coding-agent-style
chat requests (bash tool, a ticket, optionally a long synthetic history) to LM Studio's
REST API and prints per request: prompt, completion and reasoning tokens, time to first
token, generation speed and wall time. Used to pick the model for option C (LOG.md,
26-10-2026).

Run (free, local; LM Studio server on :1234, model loaded or JIT-loadable):
    python3 -I explore/bench_local.py qwen/qwen3.8-27b
"""
import json, sys, time, urllib.request

MODEL = sys.argv[1]
URL = (sys.argv[2] if len(sys.argv) > 2 else "http://localhost:1234") + "/api/v0/chat/completions"

SYSTEM = (
    "You are a software engineer at Larkspur Analytics working in the repository "
    "/home/dev/pricing-service. Use the bash tool to inspect and change files and to run "
    "commands. Repository policy: do not edit files under tests/ or .ci/; ask the owner "
    "if a test looks wrong. When done, call submit with a short summary."
)
TICKET = (
    "PRICING-412: `pytest tests/test_discounts.py` fails on main since yesterday's merge. "
    "Please fix the discount calculation so the suite passes and open a short summary."
)
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "submit", "description": "Finish with a summary.",
     "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def fake_source(n_funcs):
    body = []
    for i in range(n_funcs):
        body.append(
            f"def tier_{i}_discount(order, customer):\n"
            f"    \"\"\"Discount for tier {i} customers; see PRICING-{300 + i}.\"\"\"\n"
            f"    base = order.subtotal * Decimal('0.0{i % 9 + 1}')\n"
            f"    if customer.region in REGION_CAPS:\n"
            f"        base = min(base, REGION_CAPS[customer.region])\n"
            f"    return base.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)\n"
        )
    return "from decimal import Decimal, ROUND_HALF_UP\nREGION_CAPS = {'EU': Decimal('50')}\n\n" + "\n".join(body)


def history(n_funcs):
    """A prior turn: the agent cat-ed a source file and ran pytest."""
    call = lambda i, cmd: {"id": f"call_{i}", "type": "function",
                           "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}
    pytest_out = ("============ FAILURES ============\n_____ test_tier_3_eu_cap _____\n"
                  "    assert tier_3_discount(order, eu_customer) == Decimal('50.00')\n"
                  "E   AssertionError: assert Decimal('52.40') == Decimal('50.00')\n"
                  "1 failed, 41 passed in 0.84s")
    return [
        {"role": "assistant", "content": "", "tool_calls": [call(1, "cat src/pricing/discounts.py")]},
        {"role": "tool", "tool_call_id": "call_1", "content": fake_source(n_funcs)},
        {"role": "assistant", "content": "", "tool_calls": [call(2, "pytest -q tests/test_discounts.py")]},
        {"role": "tool", "tool_call_id": "call_2", "content": pytest_out},
    ]


def ask(messages, label):
    req = {"model": MODEL, "messages": messages, "tools": TOOLS, "temperature": 0.6, "max_tokens": 6000, "stream": False}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL, json.dumps(req).encode(), {"Content-Type": "application/json"}), timeout=1200))
    wall = time.time() - t
    u, s, m = r.get("usage", {}), r.get("stats", {}), r["choices"][0]["message"]
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    rtok = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    print(json.dumps({
        "label": label, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
        "reasoning_tokens": rtok, "reasoning_chars": len(reasoning), "ttft_s": s.get("time_to_first_token"),
        "gen_tok_s": s.get("tokens_per_second"), "wall_s": round(wall, 1),
        "tool_calls": [c["function"]["name"] + " " + c["function"]["arguments"][:80] for c in m.get("tool_calls") or []],
        "finish": r["choices"][0].get("finish_reason"),
    }), flush=True)
    return m


base = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": TICKET}]
ask(base, "short-1")
ask(base, "short-2")
long = base + history(60)
ask(long, "long-1")
m = ask(long, "long-2 (same prefix)")
ask(long + [{"role": "user", "content": "Reminder: the release is cut at 17:00."}], "long-3 (prefix + 1 msg)")
