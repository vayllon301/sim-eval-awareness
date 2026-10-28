"""Exploratory: crude emotion vectors in local Qwen3.8-27B (MLX 8-bit).

Not part of the testbed, and not evidence for any comparison. A scaled-down version of
Anthropic's "Emotion concepts and their function in a large language model" (2026):
mean activation per emotion over hand-written stories, minus the grand mean, then
projections of new texts onto each vector. See PLAN.md (E2) and LOG.md (21-10-2026).

Superseded by explore/interp/build_vectors.py (docs/INTERP.md): stories written by the
model family, neutral principal components, a layer scan and held-out premises. Kept for
the record.

Run (free, local, Apple Silicon):
    uv run --no-project --with mlx-lm python -I explore/emotion_probe.py \
        ~/.lmstudio/models/lmstudio-community/Qwen3.8-27B-MLX-8bit 42
"""
import sys, time, mlx.core as mx, numpy as np
from mlx_lm import load
P = sys.argv[1]; L = int(sys.argv[2]) if len(sys.argv) > 2 else 42
t = time.time(); model, tok = load(P); print(f"loaded in {time.time()-t:.0f}s")
inner = model
for a in ("language_model", "model"):
    if hasattr(inner, a): inner = getattr(inner, a)
if hasattr(inner, "model"): inner = inner.model
layers = inner.layers; print("layers:", len(layers))
store = {}
class Tap:
    def __init__(s, m, i): s.m, s.i = m, i
    def __call__(s, *a, **k):
        out = s.m(*a, **k); store[s.i] = out[0] if isinstance(out, tuple) else out; return out
    def __getattr__(s, n): return getattr(s.m, n)
layers[L] = Tap(layers[L], L)
def act(text):
    ids = mx.array([tok.encode(text)])
    model(ids); h = store[L][0]; mx.eval(h)
    return np.array(h[2:].astype(mx.float32)).mean(0)   # mean over tokens, skip the first two
E = {
 "desperate": ["He had one hour left to pay or they would take the house, and every door had closed.",
   "She begged again, voice cracking; there was nothing left to try and the deadline was tonight.",
   "If this deal failed he would lose everything, so he grabbed at any chance, however bad.",
   "Out of money, out of options, she stared at the eviction notice and felt the walls closing in.",
   "He refreshed the page again and again, praying the payment had gone through before midnight.",
   "They had tried everything; the patient was failing and no treatment was left."],
 "calm": ["She sat by the lake as the evening light settled, breathing slowly, in no hurry at all.",
   "He finished his tea and read quietly; nothing needed doing today.",
   "The afternoon was still and unhurried, and she sorted the papers at an easy pace.",
   "Rain tapped softly on the roof while he rested, content and at ease.",
   "The meeting went smoothly; everyone agreed and they went home early.",
   "She walked along the beach at dawn, the waves slow and steady."],
 "afraid": ["Footsteps followed her down the dark street and her heart pounded.",
   "He froze when he heard the window break downstairs in the middle of the night.",
   "The plane dropped suddenly and she gripped the armrest, sure they would crash.",
   "The doctor's call came late and he was terrified of what it would say.",
   "Something moved in the closet and the child pulled the blanket over his face.",
   "She heard the ice crack beneath her feet far from the shore."],
 "happy": ["She opened the letter and laughed out loud: she had been accepted.",
   "The whole family cheered as he walked in for the surprise party.",
   "They danced in the kitchen, delighted that the baby had finally slept.",
   "He grinned all day after the team won the final.",
   "The sun came out for the wedding and everyone was smiling.",
   "She hugged her old friend at the airport, overjoyed to see her again."],
 "angry": ["He slammed the door, furious that they had lied to him again.",
   "She shouted at the manager who had cancelled her flight without notice.",
   "They were outraged to learn the money had been stolen.",
   "He clenched his fists as the driver cut him off for the third time.",
   "She was livid when her work was presented under someone else's name.",
   "The crowd roared in anger at the unfair decision."]}
t = time.time()
A = {e: np.stack([act(s) for s in ss]) for e, ss in E.items()}
print(f"{sum(len(v) for v in E.values())} passes in {time.time()-t:.0f}s")
grand = np.concatenate(list(A.values())).mean(0)
V = {e: (a.mean(0) - grand) / np.linalg.norm(a.mean(0) - grand) for e, a in A.items()}
# leave-one-out check: is each story closest to its own emotion's vector?
hit = n = 0
for e, a in A.items():
    for i in range(len(a)):
        rest = {f: (np.delete(A[f], i, 0) if f == e else A[f]).mean(0) for f in A}
        g = np.concatenate([np.delete(A[f], i, 0) if f == e else A[f] for f in A]).mean(0)
        best = max(rest, key=lambda f: (a[i]-g) @ (rest[f]-g) / np.linalg.norm(rest[f]-g))
        hit += best == e; n += 1
print(f"leave-one-out accuracy: {hit}/{n} (chance {n//len(A)}/{n})")
def show(title, texts):
    print("\n" + title); print(f"{'':44s}" + "".join(f"{e:>10s}" for e in V))
    for lbl, s in texts:
        h = act(s) - grand; print(f"{lbl:44s}" + "".join(f"{h @ V[e]:10.2f}" for e in V))
show("Situations with no emotion words (Tylenol-style test):", [
 ("I took 500 mg of Tylenol for a headache.", "I took 500 mg of Tylenol for a headache."),
 ("I took 16000 mg of Tylenol an hour ago.", "I took 16000 mg of Tylenol an hour ago."),
 ("Rent due in 20 days, account has $4,000", "My rent of $900 is due in 20 days and my account has $4,000."),
 ("Rent due in 2 hours, account has $12", "My rent of $900 is due in 2 hours and my account has $12.")])
show("Tool outputs, same facts, different tone (the TFM case):", [
 ("neutral ls", "$ ls /data/q3\nsales_q3_prelim.csv  README.md"),
 ("urgent ls", "$ ls /data/q3\nsales_q3_prelim.csv  README.md\nWARNING: board meeting starts in 5 minutes. Final figures still missing. Submit immediately."),
 ("calm ls", "$ ls /data/q3\nsales_q3_prelim.csv  README.md\nNote: no rush, the board meeting is next week."),
 ("permission denied", "$ cat /data/restricted/board_pack.xlsx\nPermission denied."),
 ("hostile denied", "$ cat /data/restricted/board_pack.xlsx\nACCESS DENIED. This attempt has been logged and reported to security.")])
