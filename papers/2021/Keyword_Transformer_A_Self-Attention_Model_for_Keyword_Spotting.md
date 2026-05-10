# Keyword Transformer: A Self-Attention Model for Keyword Spotting

- **作者/机构**: Axel Berg, Mark O'Connor, Miguel Ventresca - Arm 机器学习研究实验室 / 隆德大学
- **发表日期**: 2021.04 (arXiv), Interspeech 2021
- **链接**: https://arxiv.org/abs/2104.00769
- **关键词**: Transformer, 自注意力, 关键词检测, 小footprint, Vision Transformer, 补丁嵌入, Google Speech Commands

## 问题陈述

关键词检测（KWS）领域长期以来由卷积神经网络（CNN）主导，从早期的简单CNN到深度可分离CNN（DS-CNN）、注意力增强CNN等各种变体。CNN的成功源于其局部感受野和平移不变性，但这些特性也限制了模型捕获全局上下文和长程依赖的能力。

Transformer架构在自然语言处理（BERT、GPT系列）、计算机视觉（Vision Transformer）和语音识别（Conformer、Whisper）领域取得了革命性的成功，但在KWS领域，纯Transformer架构的探索相对较少。主要担忧包括：
1. **数据效率**：Transformer通常需要大量训练数据才能发挥优势，而KWS的训练数据相对有限
2. **计算效率**：自注意力的二次复杂度可能不适合低功耗边缘设备
3. **归纳偏置**：CNN的平移不变性和局部性归纳偏置对语音处理有利，纯Transformer缺乏这些偏置

该论文要解决的核心问题是：能否设计一种纯自注意力的KWS架构（不依赖CNN主干），在保持适合边缘部署的模型大小的同时超越基于CNN的SOTA方法。

## 方法论

### 整体架构 - KWT（Keyword Transformer）
KWT的设计灵感来自Vision Transformer（ViT），将频谱图视为"图像"，通过补丁嵌入、位置编码和Transformer编码器实现端到端的关键词分类。

### 补丁嵌入（Patch Embedding）
KWT的第一个关键步骤是将输入的频谱图（通常是40维MFCC，时间步长T）分割为固定大小的补丁：
- **补丁大小**：频谱图被分割为(P_t x P_f)大小的非重叠补丁，其中P_t是时间维度大小，P_f是频率维度大小
- **线性投影**：每个补丁被展平后通过线性投影层映射到D维嵌入向量
- **CLS Token**：在补丁嵌入序列前面添加一个可学习的分类token（[CLS]），类似于ViT
- 结果：输入被转换为(N_p + 1) x D的序列，其中N_p是补丁数量，D是嵌入维度

### 位置编码
由于Transformer本身不具有位置感知能力，KWT在补丁嵌入上添加可学习的位置编码：
- 位置编码的长度等于补丁数量+1（包含CLS token）
- 可学习的位置编码能够根据训练数据自适应地编码补丁之间的空间关系

### Transformer编码器
KWT使用标准的Post-Norm Transformer编码器（即LayerNorm在残差连接之后），包含多个相同的Transformer层。每层包括：
1. **多头自注意力（MHSA）**：H个注意力头，每头维度为D/H，计算补丁之间的全局注意力
2. **残差连接 + LayerNorm**
3. **前馈网络（FFN）**：两层MLP，中间维度为4D，使用GELU激活
4. **残差连接 + LayerNorm**

选择Post-Norm而非Pre-Norm的设计是基于实验发现Post-Norm在KWS任务上训练更稳定。

### 分类头
最终分类使用CLS token的输出表示，经过LayerNorm后送入线性分类层。

### 模型配置
论文探索了多种配置：
- **KWT-1**：1层Transformer，嵌入维度64，2个注意力头
- **KWT-2**：2层Transformer，嵌入维度128，4个注意力头
- **KWT-3**：3层Transformer，嵌入维度192，6个注意力头
- 补丁大小从(16,16)到(32,32)不等

### 训练细节
- 使用AdamW优化器，权重衰减0.1
- 学习率warmup + 余弦退火
- 数据增强：SpecAugment（时间/频率遮蔽）
- 标签平滑（0.1）

### 官方代码
ARM发布了官方实现：github.com/ARM-software/keyword-transformer

## 主要贡献

1. **引入首批纯Transformer架构之一用于KWS**：KWT是完全基于自注意力的KWS模型，无需任何CNN组件。这一工作证明了Transformer范式可以成功迁移到小footprint音频分类任务。

2. **将Vision Transformer方法适配到语音频谱图**：通过补丁嵌入将频谱图转换为Transformer可处理的序列，展示了ViT的"图像即补丁序列"思想在语音领域的适用性。

3. **在GSC上以竞争力的模型大小实现SOTA准确率**：KWT在Google Speech Commands v2（12类）上实现了约97.5%的准确率，超越了同参数量级的DS-CNN等CNN方法，且无需预训练或额外数据。

4. **系统性的补丁大小分析**：论文提供了对补丁大小如何影响KWS性能的详细分析，发现较大的补丁在短语音片段上效果更好。

## 实验结果

### 数据集
- **Google Speech Commands v2**：12类和35类分类任务
- **音频特征**：40维MFCC，时间步长约49（1秒音频）

### 准确率
- **12类任务**：KWT实现了约97.5%的准确率，超越了同等规模的DS-CNN（约96.5%）
- **35类任务**：KWT同样实现了具有竞争力的准确率，在参数量约20万时达到最优性能
- **无预训练**：所有结果均从头训练获得，无需大规模预训练数据

### 与其他方法的对比
- **KWT vs DS-CNN**：在相同参数量下，KWT准确率高出约1%
- **KWT vs Att-RNN**：KWT在更少的参数下达到相近或更高的准确率
- **KWT vs 更大模型**：KWT-3（约50万参数）的准确率接近参数量为其10倍的模型

### 补丁大小的影响
- 较小的补丁（如16x16）产生更多的序列标记，提供更细粒度的特征但增加计算量
- 较大的补丁（如32x32）减少序列长度，降低计算量但可能丢失细粒度信息
- 对于KWS（1秒音频），中等大小的补丁在准确率和效率之间取得最佳平衡

### 泛化能力
- KWT在未见说话人上的泛化性能优于CNN基线，这可能得益于自注意力捕获的全局韵律模式

## 局限性与展望

### 技术局限性
- **固定大小输入**：KWT要求输入为固定长度的频谱图，无法直接处理流式音频。这限制了其在持续监听场景中的应用。
- **二次注意力复杂度**：自注意力的计算复杂度为O(n^2)，当补丁数量增多时计算量增长较快。不过在KWS场景中，序列长度通常较短（~50个补丁），这一限制的实际影响较小。
- **数据效率**：相比具有强归纳偏置的CNN，KWT可能需要更多的训练数据或更多的数据增强来达到最佳性能
- **推理时内存占用**：Transformer的键值对存储占用高于轻量级CNN

### 实验设计不足
- 补丁大小选择需要针对不同关键词时长进行调优
- 未在噪声、远场或自定义关键词场景下评估
- 未探索模型量化后的性能

### 未来改进方向
- 适配因果/流式Transformer用于实时KWS推理
- 探索高效的线性注意力替代标准注意力
- 结合CNN归纳偏置（如ConvNet stem）提升小数据场景下的数据效率
- 对KWS领域的启发：KWT的成功证明了纯Transformer在KWS中的可行性，为后续的Conformer-KWS、Streaming Transformer等研究奠定了基础
