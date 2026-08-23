#!/usr/bin/env python3
"""Second-pass PDF retrieval for no_oa DOIs: Unpaywall + known OA publisher patterns."""
import json
import os
import re
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (compatible; awesome-kws-update/0.1; mailto:caicry@gmail.com)"}
EMAIL = "caicry@gmail.com"


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def try_save(data, dest):
    if data[:5] == b"%PDF-" and len(data) > 20000:
        open(dest, "wb").write(data)
        return True
    return False


# rebuild title->info from README for dest paths (reuse fname logic)
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


import subprocess
old = subprocess.run(["git", "show", "0ac303b:README.md"], capture_output=True, text=True).stdout
old_keys = {"".join(c for c in m.lower() if c.isalnum()) for m in re.findall(r"^\* \[(.*?)\]\(", old, re.M)}
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
        k = "".join(c for c in m.group(1).lower() if c.isalnum())
        if k in old_keys:
            continue
        meta[k] = {"title": m.group(1), "url": m.group(2), "year": year}

manifest = json.load(open("pdf_manifest.json"))
# map name -> manifest key
def name_of(key):
    return key

# publisher-pattern pdf urls by doi prefix
def pattern_pdf(doi):
    if doi.startswith("10.21437/"):  # ISCA archive
        return f"https://www.isca-speech.org/archive/{doi.split('/', 1)[1]}.pdf"
    if doi.startswith("10.3390/"):  # MDPI
        return f"https://www.mdpi.com/{doi.split('10.3390/', 1)[1]}/pdf"
    if doi.startswith("10.1186/"):  # SpringerOpen
        return f"https://link.springer.com/content/pdf/{doi}.pdf"
    if doi.startswith("10.1109/"):  # IEEE — try open instances via doi.org only rarely; skip
        return None
    return None


todo = []
for k, v in manifest.items():
    if isinstance(v, dict) and (v.get("status") == "no_oa" or str(v.get("status", "")).startswith("error")):
        doi = v.get("doi")
        if doi:
            title_year = None
            # find meta by fname match
            fn = k
            for mk, mv in meta.items():
                if fname_for(mv["title"]) == fn and mv["year"] == v.get("year"):
                    title_year = mv
                    break
            if not title_year:
                for mk, mv in meta.items():
                    if fname_for(mv["title"])[:100] == fn[:100] and mv["year"] == v.get("year"):
                        title_year = mv
                        break
            if title_year:
                todo.append((k, doi, title_year["year"], title_year["title"]))
print(f"second-pass candidates: {len(todo)}", flush=True)

got = fail = 0
for i, (k, doi, yr, title) in enumerate(todo):
    dest_dir = f"papers/{yr}"
    os.makedirs(dest_dir, exist_ok=True)
    dest = f"{dest_dir}/{fname_for(title)}.pdf"
    if os.path.exists(dest) and os.path.getsize(dest) > 20000:
        got += 1
        continue
    ok = False
    # 1) unpaywall
    try:
        data = json.loads(fetch(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}", 30))
        loc = data.get("best_oa_location") or {}
        for cand in [loc.get("url_for_pdf"), loc.get("url")]:
            if not cand:
                continue
            try:
                if try_save(fetch(cand), dest):
                    ok = True
                    break
            except Exception:
                continue
            time.sleep(0.5)
    except Exception:
        pass
    # 2) publisher pattern
    if not ok:
        p = pattern_pdf(doi)
        if p:
            try:
                if try_save(fetch(p), dest):
                    ok = True
            except Exception:
                pass
            time.sleep(0.5)
    if ok:
        manifest[k]["status"] = "oa2"
        got += 1
    else:
        fail += 1
    if i % 25 == 0:
        print(f"{i}/{len(todo)} got={got}", flush=True)
    time.sleep(0.3)

json.dump(manifest, open("pdf_manifest.json", "w"), ensure_ascii=False, indent=1)
from collections import Counter
print("FINAL:", Counter(v["status"] for v in manifest.values() if isinstance(v, dict)))
