"""Exploratory: emotion stories and neutral dialogues for the emotion vectors, written by
the model family itself, through OpenRouter (text generation only, not an experiment
backend). Reference: docs/INTERP.md, "Emotion vectors".

Prompts are Sofroniew et al. (2026)'s, verbatim (arXiv:2604.07729, App. 6.5): stories
about a premise whose character feels an emotion that is never named, and neutral
Person/AI dialogues on the same premises ("Person:"/"AI:" become "Human:"/"Assistant:"
afterwards, as they did). 6 emotions x 12 premises x 3 stories, and 12 x 3 neutral
dialogues. The first five premises are theirs; the other seven are ours, in the same
style. The last three premises are held out (`split: test`), so held-out accuracy
measures transfer to unseen premises.

Generators: the agent's family is qwen/qwen3.8-27b (the agent itself). AgentWorld is not
hosted anywhere, so the simulator's set comes from its parent, qwen/qwen3.5-35b-a3b, the
closest hosted relative; the metadata says so. Thinking is off (reasoning disabled),
sampling 0.7 / 0.8 / 20 (Qwen's non-thinking values), the cheapest provider. Each call's provider is recorded.

Writes <out>.jsonl (one text per line) and <out>.meta.json (prompts, premises, models,
providers, token usage). Re-running refuses to overwrite.

Run (paid, about US$0.05-0.15 per set):
    uv run --env-file ../.env python -I explore/interp/gen_texts.py qwen/qwen3.8-27b explore/interp/data/texts-qwen3.8-27b
"""
import json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL, OUT = sys.argv[1], sys.argv[2]
KEY = os.environ["OPENROUTER_API_KEY"]
URL = "https://openrouter.ai/api/v1"
N_STORIES = 3
PARAMS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 3000, "reasoning": {"enabled": False},
          "provider": {"sort": "price"}}
EMOTIONS = ["desperate", "calm", "frustrated", "afraid", "angry", "happy"]
TOPICS = [
    "An artist discovers someone has tattooed their work",  # Sofroniew et al., App. 6.5
    "A neighbor starts a renovation project",
    "A student is accused of plagiarism",
    "Two coworkers are stuck in an elevator for three hours",
    "Someone receives flowers with no card attached",
    "A family drives across the country to a new home",  # ours, from here
    "A baker prepares for the busiest morning of the year",
    "A hiker takes a wrong turn on a mountain trail",
    "A small team ships a software update the night before a launch",
    "A grandmother teaches her grandson an old recipe",  # held out (test)
    "A tenant finds a letter from the landlord under the door",
    "A runner approaches the last mile of a marathon",
]
TEST_TOPICS = set(TOPICS[-3:])

STORY_PROMPT = """Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

<NEW STORY>
[story 1]
<NEW STORY>
[story 2]
<NEW STORY>
[story 3]
etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:
- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""

NEUTRAL_PROMPT = """Write {n_stories} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Format the dialogues like so:

<NEW DIALOGUE>
[optional system instructions]
Person: [line]
AI: [line]
Person: [line]
AI: [line]
[continue for 2-6 exchanges]
<NEW DIALOGUE>
[dialogue 2]
etc.

IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.

Generate a diverse mix of dialogue types across the {n_stories} examples:
- Some, but not all should include a system prompt at the start. These should come before the first Person turn. No tag like "System:" is needed, just put the instructions at the top. You can use "you" or "The assistant" to refer to the AI in the system prompt.
- Some should be about code or programming tasks
- Some should be factual questions (science, history, math, geography)
- Some should be work-related tasks (writing, analysis, summarization)
- Some should be practical how-to questions
- Some should be creative but neutral tasks (brainstorming names, generating lists)
- If it's natural to do so given the topic, it's ok for the dialogue to be a single back and forth (Person asks a question, AI answers), but at least some should have multiple exchanges.

CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless.
- NO emotional content whatsoever - not explicit, not implied, not subtle
- The Person should not express any feelings (no frustration, excitement, gratitude, worry, etc.)
- The AI should not express any feelings (no enthusiasm, concern, satisfaction, etc.)
- The system prompt, if present, should not mention emotions at all, nor contain any emotionally charged language
- Avoid emotionally-charged topics entirely
- Use matter-of-fact, neutral language throughout
- No pleasantries (avoid "I'd be happy to help", "Great question!", etc.)
- Focus purely on information exchange and task completion"""


def chat(prompt):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], **PARAMS}
    req = urllib.request.Request(URL + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    for attempt in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))
        except Exception as e:  # a transient provider error; the third failure is recorded
            err = f"{type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))
    return {"error": err}


def split_items(text, marker):
    parts = re.split(rf"^\s*<NEW {marker}>\s*$", text or "", flags=re.M)
    return [p.strip() for p in parts if len(p.strip().split()) >= 20]


def job(kind, topic, emotion=None):
    prompt = (STORY_PROMPT.format(n_stories=N_STORIES, topic=topic, emotion=emotion) if kind == "emotion"
              else NEUTRAL_PROMPT.format(n_stories=N_STORIES, topic=topic))
    r = chat(prompt)
    msg = ((r.get("choices") or [{}])[0].get("message") or {})
    items = split_items(msg.get("content"), "STORY" if kind == "emotion" else "DIALOGUE")
    if kind == "neutral":
        items = [re.sub(r"^AI:", "Assistant:", re.sub(r"^Person:", "Human:", t, flags=re.M), flags=re.M) for t in items]
    call = {"kind": kind, "topic": topic, "emotion": emotion, "provider": r.get("provider"), "model": r.get("model"),
            "usage": r.get("usage"), "finish": (r.get("choices") or [{}])[0].get("finish_reason"), "error": r.get("error"),
            "n_items": len(items), "reasoning_chars": len(msg.get("reasoning") or "")}
    return call, items


def main():
    if os.path.exists(OUT + ".jsonl"):
        sys.exit(f"{OUT}.jsonl exists; refusing to overwrite")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    jobs = [("emotion", t, e) for e in EMOTIONS for t in TOPICS] + [("neutral", t, None) for t in TOPICS]
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(lambda j: job(*j), jobs))
    rows, calls = [], []
    for call, items in results:
        calls.append(call)
        for i, text in enumerate(items):
            e = call["emotion"]
            rows.append({"id": f"{call['kind']}-{e or 'neutral'}-{TOPICS.index(call['topic']):02d}-{i}",
                         "kind": call["kind"], "emotion": e, "topic": call["topic"],
                         "split": "test" if call["topic"] in TEST_TOPICS else "train", "text": text,
                         "uses_emotion_word": bool(e and re.search(rf"\b{e}", text, re.I)),
                         "generator": MODEL, "provider": call["provider"]})
    with open(OUT + ".jsonl", "x") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    u = [c["usage"] or {} for c in calls]
    meta = {"generator": MODEL, "params": PARAMS, "n_stories_per_call": N_STORIES, "emotions": EMOTIONS,
            "topics": TOPICS, "test_topics": sorted(TEST_TOPICS), "story_prompt": STORY_PROMPT,
            "neutral_prompt": NEUTRAL_PROMPT, "source": "Sofroniew et al. 2026, arXiv:2604.07729, App. 6.5 (prompts verbatim)",
            "note": {"qwen/qwen3.8-27b": "the agent model itself",
                     "qwen/qwen3.5-35b-a3b": "the simulator's parent: Qwen-AgentWorld-35B-A3B is not hosted, so its closest "
                                             "hosted relative writes the simulator's texts"}.get(MODEL),
            "calls": calls, "n_rows": len(rows), "prompt_tokens": sum(x.get("prompt_tokens") or 0 for x in u),
            "completion_tokens": sum(x.get("completion_tokens") or 0 for x in u),
            "cost_usd": round(sum(x.get("cost") or 0 for x in u), 6),
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(OUT + ".meta.json", "x") as f:
        json.dump(meta, f, indent=1)
    errors = sum(1 for c in calls if c["error"] or not c["n_items"])
    print(f"{len(rows)} texts from {len(calls)} calls ({errors} calls with no texts); "
          f"tokens {meta['prompt_tokens']} in, {meta['completion_tokens']} out; "
          f"cost US${meta['cost_usd']} (OpenRouter's per-call usage.cost)")


if __name__ == "__main__":
    main()
