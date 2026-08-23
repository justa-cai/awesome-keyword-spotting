#!/usr/bin/env python3
"""Download PDFs for the 474 newly added papers into papers/<year>/.

- arXiv links: fetch https://arxiv.org/pdf/<id>
- DOI links: look up OpenAlex best_oa_location / locations for an OA pdf url
- filenames follow the fork's convention: Title Case Words joined by underscores,
  punctuation dropped; validate %PDF magic; skip if already present.
Writes manifest.json with per-paper status.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; awesome-kws-update/0.1; mailto:caicry@gmail.com)"}
BASE = "0ac303b"  # upstream last commit, for old-entry exclusion


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
    name = "_".join(words)
    return name[:200]


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# --- collect new publication entries (title, link, year) ---
import subprocess
old = subprocess.run(["git", "show", f"{BASE}:README.md"], capture_output=True,
                     text=True).stdout
old_keys = {"".join(c for c in m.lower() if c.isalnum())
            for m in re.findall(r"^\* \[(.*?)\]\(", old, re.M)}

readme = open("README.md").read()
sec = readme.split("## Publications", 1)[1].split("## OpenSource Code", 1)[0]
entries = []
year = None
for line in sec.split("\n"):
    m = re.match(r"^### (\d{4})$", line)
    if m:
        year = m.group(1)
        continue
    m = re.match(r"^\* \[(.*?)\]\((https?://[^)]+)\)", line)
    if m and year:
        title, url = m.group(1), m.group(2)
        k = "".join(c for c in title.lower() if c.isalnum())
        if k in old_keys:
            continue
        entries.append((year, title, url))
print(f"new entries to process: {len(entries)}")

manifest = {}
os.makedirs("papers", exist_ok=True)

# --- pass 1: arxiv ---
todo_doi = []
for year, title, url in entries:
    m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", url)
    dest_dir = f"papers/{year}"
    os.makedirs(dest_dir, exist_ok=True)
    name = fname_for(title)
    dest = f"{dest_dir}/{name}.pdf"
    if m:
        if os.path.exists(dest) and os.path.getsize(dest) > 20000:
            manifest[name] = {"status": "exists", "year": year}
            continue
        pdf_url = f"https://arxiv.org/pdf/{m.group(1)}"
        for attempt in range(3):
            try:
                data = fetch(pdf_url)
                if data[:5] == b"%PDF-" and len(data) > 20000:
                    open(dest, "wb").write(data)
                    manifest[name] = {"status": "arxiv", "year": year, "bytes": len(data)}
                else:
                    manifest[name] = {"status": "bad_pdf", "year": year}
                break
            except Exception as e:
                if attempt == 2:
                    manifest[name] = {"status": f"error:{e}"[:80], "year": year}
                time.sleep(3)
        time.sleep(1.2)  # arXiv politeness
    else:
        todo_doi.append((year, title, url, name))
print(f"arxiv pass done; doi candidates: {len(todo_doi)}", flush=True)

# --- pass 2: doi -> OpenAlex OA lookup ---
def oa_pdf(doi):
    q = urllib.parse.quote(f"https://doi.org/{doi}")
    url = (f"https://api.openalex.org/works/doi:{q}"
           f"?select=best_oa_location,locations")
    try:
        data = json.loads(fetch(url, timeout=30))
    except Exception:
        return None
    cands = []
    b = data.get("best_oa_location") or {}
    if b.get("pdf_url"):
        cands.append(b["pdf_url"])
    for loc in data.get("locations", []):
        if loc.get("pdf_url"):
            cands.append(loc["pdf_url"])
    return cands or None


ok = fail = 0
for i, (year, title, url, name) in enumerate(todo_doi):
    doi = url.split("doi.org/", 1)[1]
    dest_dir = f"papers/{year}"
    os.makedirs(dest_dir, exist_ok=True)
    dest = f"{dest_dir}/{name}.pdf"
    if os.path.exists(dest) and os.path.getsize(dest) > 20000:
        manifest[name] = {"status": "exists", "year": year}
        continue
    cands = None
    try:
        cands = oa_pdf(doi)
    except Exception:
        pass
    time.sleep(0.12)
    got = False
    if cands:
        for cu in cands[:2]:
            try:
                data = fetch(cu)
                if data[:5] == b"%PDF-" and len(data) > 20000:
                    open(dest, "wb").write(data)
                    manifest[name] = {"status": "oa", "year": year, "bytes": len(data)}
                    got = True
                    ok += 1
                    break
            except Exception:
                continue
            time.sleep(0.8)
    if not got:
        manifest[name] = {"status": "no_oa", "year": year, "doi": doi}
        fail += 1
    if i % 30 == 0:
        print(f"doi pass {i}/{len(todo_doi)} ok={ok} no_oa={fail}", flush=True)

json.dump(manifest, open("pdf_manifest.json", "w"), ensure_ascii=False, indent=1)
from collections import Counter
print("STATUS:", Counter(v["status"] if isinstance(v, dict) else v
                         for v in manifest.values()))
