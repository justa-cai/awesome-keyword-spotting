#!/usr/bin/env python3
"""Build README entries from candidates.json per original author's format.

Format: `* [Title](url), Inst A & Inst B, YYYY.MM`
- year sections descending, month descending inside a year
- arXiv abs link preferred, else DOI link
- date = arXiv v1 month when known (author's convention), else venue date
- dedupe against existing README titles
"""
import difflib
import json
import re
import time
import urllib.parse
import urllib.request

REJECT = {
    # ASR hotword biasing (different task)
    21, 253, 374, 423, 514, 606, 899,
    # document/image/text KWS
    200, 213, 215, 516, 879, 910, 920, 1209,
    # theses / reports / preprint-server deposits / zenodo dumps
    11, 35, 41, 85, 107, 148, 158, 162, 177, 198, 207, 217, 415-415 if False else 0,
    453, 513, 552, 779, 797, 798, 801, 826, 1076, 1079,
    # student / junk / app-project papers
    6, 7, 50, 87, 93, 293, 316, 587, 632, 654, 662, 708, 733, 814-814 if False else 0,
    827, 829, 896, 913, 916, 1028, 1041, 1230, 233, 469, 524, 587, 653, 910,
    # off-topic (VAD / diarization / SE-eval / anonymization / underwater / neural)
    195, 785, 968, 1030, 1183, 94,
    # duplicates of same work (keep one version each)
    552, 653, 864, 887, 943, 1034, 1142, 1143, 1201,
}
# clean the accidental inline-if zeros above
REJECT.discard(0)
REJECT = {i for i in REJECT if i}

def tkey(t):
    return "".join(ch for ch in t.lower() if ch.isalnum())

def clean_title(t):
    t = t.replace("&amp;", "&").replace("&quot;", '"')
    t = t.replace("\\mu", "μ").replace("\\mathrm", "")
    t = t.replace("$", "").replace("{", "").replace("}", "")
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r'\s+', " ", t).strip().strip('"')
    return t

def clean_inst(i):
    i = re.sub(r'\s*\([^)]*\)\s*$', "", i).strip()   # drop "(China)" etc.
    return i

rows = json.load(open("candidates.json"))
tiers = {}
for line in open("review.tsv").readlines()[1:]:
    f = line.rstrip("\n").split("\t")
    tiers[int(f[0])] = f[1]

kept = [r for i, r in enumerate(rows)
        if tiers.get(i) in ("A", "B") and i not in REJECT]

# existing README titles/arxiv ids for dedupe
readme = open("README.md").read()
existing_titles = {tkey(m) for m in re.findall(r"^\* \[(.*?)\]\(", readme, re.M)}
existing_ids = set(re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", readme))

# fallback affiliation fetch for rows without insts (OpenAlex title search, cached)
try:
    cache = json.load(open("affs_cache.json"))
except FileNotFoundError:
    cache = {}

missing = [r for r in kept if not r.get("insts")]
print(f"kept={len(kept)}  missing_insts={len(missing)}")
if missing:
    todo = [r for r in missing if tkey(r["title"]) not in cache]
    print(f"fetching affiliations for {len(todo)} rows ...")
    for n, r in enumerate(todo):
        k = tkey(r["title"])
        q = urllib.parse.quote(f'"{clean_title(r["title"])[:120]}"')
        url = (f"https://api.openalex.org/works?filter=title.search:{q}"
               f"&per-page=5&select=title,authorships")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "awesome-kws-update/0.1 (mailto:caicry@gmail.com)"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            best, ratio = None, 0.0
            for w in data.get("results", []):
                rr = difflib.SequenceMatcher(None, k, tkey(w.get("title") or "")).ratio()
                if rr > ratio:
                    best, ratio = w, rr
            if best and ratio > 0.82:
                insts = []
                for a in best.get("authorships", []):
                    for inst in a.get("institutions", []) or []:
                        nm = inst.get("display_name")
                        if nm and nm not in insts:
                            insts.append(nm)
                cache[k] = insts[:4]
            else:
                cache[k] = []
        except Exception as e:
            cache[k] = []
        if n % 25 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
        time.sleep(0.12)
    json.dump(cache, open("affs_cache.json", "w"), ensure_ascii=False)

# build entries
entries = []
seen_titles = set(existing_titles)
dropped_dups = []
for r in kept:
    title = clean_title(r["title"])
    k = tkey(title)
    if k in seen_titles:
        dropped_dups.append(title)
        continue
    ax = r.get("arxiv_id")
    if ax:
        base = re.sub(r"v\d+$", "", ax)
        if base in existing_ids:
            dropped_dups.append(title)
            continue
        url = f"https://arxiv.org/abs/{base}"
        date = r.get("arxiv_date") or r.get("date")
    else:
        doi = r.get("doi")
        url = f"https://doi.org/{doi}" if doi else (r.get("url") or "")
        date = r.get("date")
    if not url or not date:
        continue
    insts = r.get("insts") or cache.get(tkey(r["title"])) or []
    insts = [clean_inst(i) for i in insts if i]
    insts = [i for i in insts if i]
    inst_str = " & ".join(insts[:2])
    ym = f"{date[:4]}.{date[5:7]}"
    entries.append((ym, f"* [{title}]({url}), {inst_str}, {ym}" if inst_str
                    else f"* [{title}]({url}), {ym}"))
    seen_titles.add(k)

# also dedupe among ourselves by exact line title
print(f"entries={len(entries)}  dropped_vs_readme={len(dropped_dups)}")
for d in dropped_dups:
    print(f"  dup-skip: {d[:90]}")

# group: year desc, month desc
entries.sort(key=lambda e: (e[0],), reverse=True)
by_year = {}
for ym, line in entries:
    by_year.setdefault(ym[:4], []).append(line)

out_sections = []
for year in sorted(by_year, reverse=True):
    out_sections.append(f"### {year}\n" + "\n".join(by_year[year]))

# splice into README
new_readme = readme
toc_old = "    - [2022](#2022)"
toc_new = "".join(f"    - [{y}](#{y})\n" for y in sorted(by_year, reverse=True) if int(y) > 2022) + toc_old
assert toc_old in new_readme
new_readme = new_readme.replace(toc_old, toc_new, 1)

marker = "### 2022\n"
assert marker in new_readme
# 1) new year sections (>2022) go above the existing 2022 section
sec_years = [y for y in sorted(by_year, reverse=True) if y != "2022"]
if sec_years:
    block = "\n\n".join(f"### {y}\n" + "\n".join(by_year[y]) for y in sec_years)
    new_readme = new_readme.replace(marker, block + "\n\n" + marker, 1)
# 2) 2022.06+ entries go at the top of the existing 2022 section (month desc already)
if "2022" in by_year:
    new_readme = new_readme.replace(marker, marker + "\n".join(by_year["2022"]) + "\n", 1)

open("README.md", "w").write(new_readme)

n22 = len(by_year.get("2022", []))
print("\nper-year:", {y: len(v) for y, v in sorted(by_year.items(), reverse=True)})
print(f"2022-section additions (2022.06+): {n22}")
print("missing insts in final:",
      sum(1 for _, l in entries if l.count(",") == 1))
