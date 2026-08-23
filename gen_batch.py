#!/usr/bin/env python3
"""Emit batch.json: papers with PDF on disk but no .md note yet.

Each item: pdf, md, title, authors, insts, date, url
(authors/insts from candidates.json when title matches; else empty)
"""
import json
import re
import subprocess


def tkey(t):
    return "".join(c for c in t.lower() if c.isalnum())


def fname_for(title):
    t = title.replace("&amp;", "&")
    t = t.replace("μ", "u").replace("–", "-").replace("‐", "-").replace("’", "'")
    words = []
    for w in t.split():
        w = re.sub(r"[^0-9A-Za-z\-']+", "", w)
        if not w:
            continue
        if w[0].islower():
            w = w[0].upper() + w[1:]
        words.append(w)
    return "_".join(words)[:200]


# README entries (year, title, url)
readme = open("README.md").read()
sec = readme.split("## Publications", 1)[1].split("## OpenSource Code", 1)[0]
meta = {}
year = None
for line in sec.split("\n"):
    m = re.match(r"^### (\d{4})$", line)
    if m:
        year = m.group(1)
        continue
    m = re.match(r"^\* \[(.*?)\]\((https?://[^)]+)\)", line)
    if m and year:
        meta[tkey(m.group(1))] = {"title": m.group(1), "url": m.group(2), "year": year}

# candidates for authors/insts
cands = {tkey(r.get("title", "")): r for r in json.load(open("candidates.json"))}

import glob
import os

batch = []
for pdf in sorted(glob.glob("papers/*/*.pdf")):
    y = pdf.split("/")[1]
    if not y.isdigit() or int(y) < 2022:
        continue
    md = pdf[:-4] + ".md"
    if os.path.exists(md):
        continue
    base = os.path.basename(pdf)[:-4]
    # reverse-lookup: match normalized filename to a README title
    hit = None
    for k, v in meta.items():
        if fname_for(v["title"]) == base and v["year"] == y:
            hit = (k, v)
            break
    if not hit:
        for k, v in meta.items():
            if fname_for(v["title"])[:120] == base[:120] and v["year"] == y:
                hit = (k, v)
                break
    if not hit:
        print("!! no metadata for", pdf)
        continue
    k, v = hit
    c = cands.get(k, {})
    batch.append({
        "pdf": pdf, "md": md, "year": y,
        "title": v["title"], "url": v["url"], "date": c.get("date", "")[:7],
        "authors": c.get("authors", []), "insts": c.get("insts", []),
    })

json.dump(batch, open("batch.json", "w"), ensure_ascii=False, indent=1)
from collections import Counter
print("batch size:", len(batch), Counter(b["year"] for b in batch))
