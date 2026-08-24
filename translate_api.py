#!/usr/bin/env python3
"""Batch-translate Chinese deep-reading notes to English via local vLLM (Qwen3-30B-A3B).
Reads translate_queue.json, skips items whose dst already exists.
Usage: python3 translate_api.py [limit] [start_offset]
"""
import json
import os
import re
import sys
import time
import concurrent.futures
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
API = "http://192.168.13.228:8000/v1/chat/completions"
KEY = "123"
MODEL = "Qwen3-30B-A3B"
CONC = int(os.environ.get("CONC", "6"))
MAX_TOKENS = 16000

PROMPT = """You are a professional academic translator. Translate the following Chinese deep-reading note about a keyword spotting (KWS) speech paper into COMPLETE English.

STRICT RULES:
1. The H1 title line must become exactly: # {en_title}
2. Metadata labels: 作者/机构 -> **Authors/Affiliations**, 发表日期 -> **Date**, 链接 -> **Link**, 关键词 -> **Keywords**. Keep person/institution names in English (translate Chinese institution names to their official English names). Translate keyword terms to English.
3. Section headings: ## 问题陈述 -> ## Problem Statement, ## 方法论 -> ## Methodology, ## 主要贡献 -> ## Main Contributions, ## 实验结果 -> ## Experimental Results, ## 局限性与展望 -> ## Limitations and Future Work. Translate ALL sub-headings (### ...) faithfully too. KEEP the original section order as-is in the source.
4. Keep all LaTeX ($...$, $$...$$), Table/Fig numbers, numeric values, URLs, and citation markers EXACTLY as-is. Markdown table structure must be preserved.
5. Terminology: use standard speech-processing terms (senone, triphone, Viterbi, false alarm rate, etc.). Do not invent terms.
6. Translate the FULL text — no omission, no summarizing, every paragraph and bullet.
7. Output ONLY the translated markdown. No preamble, no code fences, no explanations.

--- SOURCE NOTE (Chinese) ---
{src}
--- END SOURCE ---

Translated English markdown:"""


def api_call(src_text, en_title, retries=3):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": PROMPT.replace("{src}", src_text).replace("{en_title}", en_title)}
        ],
        "temperature": 0.2,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.loads(r.read())
            text = data["choices"][0]["message"]["content"].strip()
            # strip potential fences
            if text.startswith("```"):
                text = re.sub(r"^```(markdown)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            return text
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def validate(text, en_title):
    if not text or len(text) < 1500:
        return "too short"
    if text.count("# ") < 6:  # h1 + 5 sections
        return "missing sections"
    for sec in ["Problem Statement", "Methodology", "Experimental Results", "Main Contributions", "Limitations"]:
        if sec not in text:
            return "missing " + sec
    if re.search(r"[一-鿿]{5,}", text):
        return "chinese residue"
    return None


def one(item):
    src_abs = os.path.join(ROOT, item["src"])
    dst_abs = os.path.join(ROOT, item["dst"])
    if os.path.exists(dst_abs) and os.path.getsize(dst_abs) > 2000:
        return ("skip", item["src"], "")
    src_text = open(src_text_path(src_abs), encoding="utf-8").read()
    try:
        out = api_call(src_text, item["en_title"])
        err = validate(out, item["en_title"])
        if err:
            return ("bad", item["src"], err)
        os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
        with open(dst_abs, "w", encoding="utf-8") as f:
            f.write(out + "\n")
        return ("ok", item["dst"], str(len(out)))
    except Exception as e:
        return ("fail", item["src"], repr(e)[:200])


def src_text_path(p):
    return p


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    q = json.load(open(os.path.join(ROOT, "translate_queue.json")))
    todo = [x for x in q if not os.path.exists(os.path.join(ROOT, x["dst"]))]
    if offset:
        todo = todo[offset:]
    if limit:
        todo = todo[:limit]
    print(f"translating {len(todo)} notes, concurrency {CONC}", flush=True)
    stats = {"ok": 0, "skip": 0, "bad": 0, "fail": 0}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONC) as ex:
        for status, path, info in ex.map(one, todo):
            stats[status] += 1
            if status in ("bad", "fail"):
                print(f"[{status}] {path}: {info}", flush=True)
            elif stats["ok"] % 10 == 0:
                print(f"progress {sum(stats.values())}/{len(todo)} ok={stats['ok']} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
    print("FINAL:", stats, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
