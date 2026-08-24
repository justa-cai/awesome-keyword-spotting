# 英文版翻译续跑手册（给新会话的 Claude）

目标：把 papers/*/*.md（中文深读笔记）全量英译到 en/papers/<年份>/<同名>.md，队列在 `translate_queue.json`（305 项，含 src/dst/en_title）。

## 状态
- 已完成：见 `en/papers/*/*.md` 文件存在与否（spawn 前先 `os.path.exists(dst)` 过滤）
- 每波 18 个 subagent，并发上限 20；翻译很快（~1.5 分钟/篇）

## Agent 提示词模板（每篇一个 agent，替换 {src}/{dst}/{en_title}）
```
你在 KWS 论文库 /home/justa/work/DAILY.Video/kws/awesome-keyword-spotting 工作。把一篇中文深读笔记完整翻译成英文。

【铁律】不要 Read 任何 PDF（会崩）。只读中文笔记（长则分 offset/limit 读完）。
源：{src 绝对路径}
目标：Write 到 {dst 绝对路径}

要求：H1 用原英文标题「{en_title}」；元信息标签译为 **Authors/Affiliations**/**Date**/**Link**/**Keywords**（人名机构保持英文）；五段式结构原样（## Problem Statement / ## Methodology / ## Experimental Results / ## Main Contributions / ## Limitations and Future Work 及全部小节）；LaTeX、Table/Fig 编号、数字原样；术语准确不造词；全文完整翻译不省略；写完即结束，回复仅一行：路径+词数。
```

## 取下一波清单
```python
import json, os
q = json.load(open('translate_queue.json'))
todo = [x for x in q if not os.path.exists(x['dst'])]
for x in todo[:18]: print('|'.join([x['src'], x['dst'], x['en_title']]))
```

## 全部完成后的收尾（每次阶段性提交也适用）
```bash
python3 -c "import glob,json; json.dump(sorted(p[3:-3] for p in glob.glob('en/papers/*/*.md')), open('en/manifest.json','w'))"
git add en/ && git commit -m "Add English translations." && git push origin master
```
（manifest 驱动侧边栏 EN 链接改写与搜索索引；en/_sidebar.md 无需重生成，链接按 manifest 动态改写）
