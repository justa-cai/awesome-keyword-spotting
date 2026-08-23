#!/usr/bin/env python3
"""Pull candidate KWS papers 2022.06-2026.08 from arXiv + OpenAlex into candidates.json.

Queries (each run separately, merged):
- arXiv: all:"keyword spotting" / "wake word" / "voice trigger" / "hotword" / "wake-up word" / "word detection"(scoped later)
- OpenAlex: title/abstract search same phrases, plus "spoken term detection"
"""
import json
import time
import urllib.parse
import urllib.request

FROM, TO = "202206010000", "202608232359"
OA_FROM, OA_TO = "2022-06-01", "2026-08-23"
UA = {"User-Agent": "awesome-kws-update/0.1 (mailto:caicry@gmail.com)"}


def http_json(url):
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            print(f"  retry {attempt}: {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def arxiv_fetch(phrase):
    q = urllib.parse.quote(f'all:"{phrase}"')
    out, start = [], 0
    while True:
        url = (f"https://export.arxiv.org/api/query?search_query={q}"
               f"+AND+submittedDate:[{FROM}+TO+{TO}]&start={start}&max_results=100"
               f"&sortBy=submittedDate&sortOrder=descending")
        req = urllib.request.Request(url, headers=UA)
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    xml = r.read().decode()
                break
            except Exception as e:
                print(f"  retry {attempt}: {e}", flush=True)
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(url)
        import re
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
        if start == 0:
            total = int(re.search(r"totalResults>(\d+)<", xml).group(1))
            print(f"arxiv '{phrase}': {total} results", flush=True)
        if not entries:
            break
        for e in entries:
            aid = re.search(r"<id>http://arxiv.org/abs/(.*?)</id>", e).group(1)
            title = re.sub(r"\s+", " ", re.search(r"<title>(.*?)</title>", e, re.S).group(1)).strip()
            pub = re.search(r"<published>(\d{4}-\d{2}-\d{2})", e).group(1)
            abs_m = re.search(r"<summary>(.*?)</summary>", e, re.S)
            abstract = re.sub(r"\s+", " ", abs_m.group(1)).strip() if abs_m else ""
            authors = re.findall(r"<name>(.*?)</name>", e)
            out.append({"id": aid, "title": title, "date": pub, "source": "arxiv",
                        "abstract": abstract, "authors": authors[:6]})
        start += 100
        if start >= total:
            break
        time.sleep(3.1)  # arXiv rate limit: 1 req / 3 s
    return out


def openalex_fetch(phrase):
    q = urllib.parse.quote(f'"{phrase}"')
    out, cursor = [], "*"
    while True:
        url = (f"https://api.openalex.org/works?filter=title_and_abstract.search:{q},"
               f"from_publication_date:{OA_FROM},to_publication_date:{OA_TO}"
               f"&per-page=200&cursor={cursor}&select=id,doi,title,publication_date,primary_location,authorships")
        data = http_json(url)
        results = data.get("results", [])
        if cursor == "*":
            print(f"openalex '{phrase}': {data['meta']['count']} results", flush=True)
        for w in results:
            # institution list (dedup, keep order)
            insts = []
            for a in w.get("authorships", []):
                for i in a.get("institutions", []) or []:
                    n = i.get("display_name")
                    if n and n not in insts:
                        insts.append(n)
            loc = w.get("primary_location") or {}
            src = ((loc.get("source") or {}).get("display_name")) or ""
            landing = (loc.get("landing_page_url") or "") or ""
            pdf = (loc.get("pdf_url") or "") or ""
            out.append({
                "id": w["id"].rsplit("/", 1)[-1],
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "title": w.get("title") or "",
                "date": w.get("publication_date") or "",
                "source": "openalex",
                "venue": src,
                "url": landing,
                "pdf": pdf,
                "insts": insts[:4],
                "authors": [a["author"]["display_name"] for a in w.get("authorships", [])][:6],
            })
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor or not results:
            break
        time.sleep(0.15)
    return out


ARXIV_PHRASES = ["keyword spotting", "wake word", "wake-up word", "voice trigger", "hotword"]
OA_PHRASES = ["keyword spotting", "wake word", "wake-up word", "voice trigger detection",
              "hotword", "spoken term detection", "query-by-example"]

all_rows = []
for p in ARXIV_PHRASES:
    print(f"== arxiv: {p}", flush=True)
    all_rows += arxiv_fetch(p)
    time.sleep(3.1)
for p in OA_PHRASES:
    print(f"== openalex: {p}", flush=True)
    all_rows += openalex_fetch(p)

# merge by fuzzy title (lowercase alnum)
def tkey(t):
    return "".join(ch for ch in t.lower() if ch.isalnum())


merged = {}
for r in all_rows:
    k = tkey(r["title"])
    if not k:
        continue
    if k in merged:
        m = merged[k]
        # prefer arxiv metadata, merge in openalex insts/venue
        if r["source"] == "openalex":
            m.setdefault("insts", r.get("insts"))
            m.setdefault("venue", r.get("venue"))
            m.setdefault("doi", r.get("doi"))
            m.setdefault("oa_url", r.get("url"))
        else:
            m["arxiv"] = True
            # arxiv row arrives possibly after openalex row; enrich fields
            m["abstract"] = r.get("abstract", "")
            m["arxiv_id"] = r["id"]
            m["arxiv_date"] = r["date"]
            if not m.get("authors"):
                m["authors"] = r.get("authors")
    else:
        row = dict(r)
        if row["source"] == "arxiv":
            row["arxiv"] = True
            row["arxiv_id"] = row["id"]
        merged[k] = row

rows = sorted(merged.values(), key=lambda r: r.get("date") or "", reverse=True)
with open("candidates.json", "w") as f:
    json.dump(rows, f, ensure_ascii=False, indent=1)
print(f"TOTAL merged unique: {len(rows)}")
