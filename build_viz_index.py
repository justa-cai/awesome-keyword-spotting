# -*- coding: utf-8 -*-
"""从 305 个可视化精读 HTML 提取索引数据 → viz_index.json（供导览页使用）"""
import re, json, glob, os, html as H

TAG_RULES = [
    ('省算力', ['量化', '压缩', '剪枝', '低秩', 'SVD', '二值', 'TinyML', 'MCU', '微控制器', '低功耗', '芯片', '加速', 'FPGA', '能效', '省电', '轻量', '小足迹', '小足迹', 'footprint', '参数量', 'INT8', '稀疏', '乘法']),
    ('端到端/CTC', ['CTC', '端到端', 'Transducer', 'RNN-T', '流式', '解码', '序列到一', 'Seq2Seq', 'seq2seq']),
    ('自定义/零样本', ['零样本', '自定义', '少样本', '开放词汇', '开放词表', '注册', 'QbE', '示例查询', 'query-by-example', '任意词', '文本注册', ' enrollment', 'Enrollment']),
    ('多通道/远场', ['多通道', '远场', '麦克风阵列', '波束', '回声', 'AEC', '混响', '声学前端', 'PCEN', '特征提取', 'Sinc', '前端']),
    ('脉冲/新范式', ['脉冲', 'SNN', '神经形态', 'Loihi', '类神经', '事件驱动', 'Mamba', '状态空间', 'Transformer', '注意力', 'attention', '图神经网络', 'GCN']),
    ('训练策略', ['蒸馏', '对抗训练', '数据增强', '增强', '多任务', '迁移学习', '联邦', '持续学习', '增量学习', '课程', '正则', '损失', '训练策略', 'sMBR', '判别训练']),
    ('鲁棒与安全', ['鲁棒', '噪声', '对抗攻击', '攻击', '后门', '安全', '隐私', '抗噪', '域偏移', 'OOD', '泛化', '测试时']),
    ('架构搜索', ['NAS', '架构搜索', 'DARTS', '自动', '硬件感知']),
    ('基准设施', ['基准', '数据集', 'benchmark', 'Speech Commands', 'GSC', '挑战赛', '评估', '排行榜', '复测']),
    ('工业实战', ['生产', '工业', '车载', '量产', '落地', '实测', '真机', '部署', '产品', '音箱', '手机', 'Alexa', 'Siri', 'Cortana', '小米', '百度', '腾讯', 'Amazon', 'Google', 'Apple', 'Arm', 'NIO']),
    ('多模态', ['多模态', '视觉', '唇', 'audio-visual', '音视频', '跨模态', '文本嵌入', '音频-文本']),
    ('预训练/SSL', ['预训练', '自监督', 'SSL', 'wav2vec', 'HuBERT', 'WavLM', '表征学习', '嵌入']),
    ('低资源/多语言', ['低资源', '零资源', '多语言', '跨语言', '小语种', '人道主义', '声纹桥', '语言']),
]

def strip_tags(s):
    s = re.sub(r'<br\s*/?>', ' ', s)
    s = re.sub(r'<[^>]+>', '', s)
    return H.unescape(s).strip()

def clean_ws(s):
    return re.sub(r'\s+', ' ', s).strip()

items = []
for f in sorted(glob.glob('papers/2*/*.html') + glob.glob('papers/others/*.html')):
    if not os.path.exists(f[:-5] + '.md'):
        continue
    c = open(f, encoding='utf-8').read()
    # 标题：双模板兼容（<h1 class="hero"> 与 裸 <h1>）
    m = re.search(r'<h1[^>]*>(.*?)</h1>', c, re.S)
    title = clean_ws(strip_tags(m.group(1))) if m else os.path.basename(f)
    # 一句话：新模板 .meta 内「一句话：」；旧模板取第一个 tldr li
    onel = ''
    m = re.search(r'一句话：?(.*?)(?:<br|</div>|</p>)', c, re.S)
    if m:
        onel = clean_ws(strip_tags(m.group(1)))
    if not onel:
        m = re.search(r'<div class="tldr">.*?<li>(.*?)</li>', c, re.S)
        if m:
            onel = clean_ws(strip_tags(m.group(1)))
    if len(onel) > 130:
        onel = onel[:130].rstrip() + '……'
    # chips
    chips = []
    for cm in re.finditer(r'<span class="chip[^"]*">([^<]+)</span>', c):
        t = cm.group(1).strip()
        if t and t not in chips:
            chips.append(t)
        if len(chips) >= 5:
            break
    # kicker（年份 №序号 · 系列）—— 旧模板无则留空
    no, series = '', ''
    km = re.search(r'<div class="kicker">([^<]*)</div>', c)
    if km:
        nm = re.search(r'№\s*(\d+)', km.group(1))
        if nm: no = nm.group(1)
        sm = re.search(r'№\s*\d+\s*·\s*(.+?)\s*$', km.group(1))
        if sm: series = sm.group(1).strip()
    # 标签：标题+chips+一句话 命中关键词
    hay = title + ' ' + ' '.join(chips) + ' ' + onel
    tags = [tag for tag, kws in TAG_RULES if any(k.lower() in hay.lower() for k in kws)]
    if not tags:
        tags = ['其他']
    year = f.split('/')[1]
    items.append({'p': f, 'y': year, 't': title, 's': onel, 'c': chips, 'tags': tags, 'no': no, 'series': series})

json.dump(items, open('viz_index.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('items:', len(items))
from collections import Counter
print('by year:', dict(Counter(i['y'] for i in items)))
print('tag coverage:', {t: sum(1 for i in items if t in i['tags']) for t, _ in TAG_RULES}, '| 其他:', sum(1 for i in items if i['tags'] == ['其他']))
print('sample:', items[0]['t'][:40], '|', items[0]['tags'])
