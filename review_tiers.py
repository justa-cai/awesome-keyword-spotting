#!/usr/bin/env python3
"""Tier + de-noise candidates.json for manual review.

Tiers:
  A = title itself matches KWS/wake-word vocabulary
  B = title mentions speech/audio/voice AND abstract has KWS vocab
  C = everything else (default reject)
Noise regex (sign language / text / video retrieval) demotes to N.
"""
import json
import re

STRONG = re.compile(
    r"keyword spot|wake[- ]?up word|wake word|wakeup word|voice trigger|hot ?word|"
    r"spoken term|spoken keyword|acoustic keyword|wake[- ]?up detection", re.I)
WEAK = re.compile(
    r"query[- ]by[- ]example|keyphrase|key phrase|voice command|voice activation|"
    r"voice activity|word detection|always[- ]on|speech command", re.I)
SIGNAL = re.compile(r"speech|audio|voice|acoustic|spoken|speaker|far[- ]?field|sound|audio", re.I)
NOISE = re.compile(
    r"sign language|gesture|hand|handwrit|historical document|medieval|video retrieval|"
    r"scanned|ocr|text document|urdu|hindi text|bangla text|keyword search in|"
    r"federated learning for recommendation|point cloud", re.I)
ABS_KWS = re.compile(r"keyword spotting|wake[- ]?up word|wake word|voice trigger|hot ?word", re.I)

rows = json.load(open("candidates.json"))
out = []
for i, r in enumerate(rows):
    t, ab = r.get("title", ""), r.get("abstract", "")
    tier = "C"
    if STRONG.search(t):
        tier = "A"
    elif WEAK.search(t) and SIGNAL.search(t):
        tier = "B"
    elif ABS_KWS.search(t):
        tier = "B"
    if NOISE.search(t) or (tier != "A" and NOISE.search(ab or "")):
        tier = "N"
    out.append((i, tier, r))

with open("review.tsv", "w") as f:
    f.write("idx\ttier\tdate\tlink\tinsts\tvenue\ttitle\n")
    for i, tier, r in out:
        link = (f"arxiv.org/abs/{r['arxiv_id']}" if r.get("arxiv_id")
                else r.get("doi") or r.get("url") or "")
        insts = "; ".join(r.get("insts") or []) or "?"
        venue = (r.get("venue") or "")[:40]
        f.write(f"{i}\t{tier}\t{r.get('date','')[:7]}\t{link}\t{insts}\t{venue}\t{r.get('title','')}\n")

for tier in "ABCN":
    n = sum(1 for _, tr, _ in out if tr == tier)
    print(f"tier {tier}: {n}")
print(f"total: {len(out)}")
