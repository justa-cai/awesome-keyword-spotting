# Awesome Keyword Spotting — Paper Library

> Chinese & English bilingual deep-reading library for keyword spotting (wake-word detection) papers.
> 305 papers · 305 deep-reading notes · PDFs included.
> 中文版：[首页](/) · 切换语言：[🇨🇳 中文](/)

This repo extends [zycv/awesome-keyword-spotting](https://github.com/zycv/awesome-keyword-spotting) (last updated 2022-05) with **474 additional papers from 2022.06 to 2026.08**, each with a full deep-reading note (problem → method → experiments → contributions → limitations, all numbers traced back to the original tables).

- Browse by year in the sidebar; click any paper for its deep-reading note and the original PDF.
- Full-text search supports both Chinese and English.
- Notes are bilingual: `#/en/...` routes serve the English translation, falling back to the Chinese original where a translation is not yet available.

## Publication index

See the [Chinese index](/) for the full paper list (titles are in English), or browse by year:

- [2026](#/en/papers/2026/) · [2025](#/en/papers/2025/) · [2024](#/en/papers/2024/) · [2023](#/en/papers/2023/) · [2022](#/en/papers/2022/)
- 2018-2021 classics and pre-2018 seminal works in [Others](#/en/papers/others/)

## How the library is maintained

An automated pipeline (in this repo) refreshes the index from arXiv + OpenAlex, downloads OA PDFs, and batches deep-reading notes via AI agents. See `fetch_papers.py`, `download_pdfs.py`, `gen_batch.py` at the repo root.
