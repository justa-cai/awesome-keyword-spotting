# A Separable Temporal Convolution Neural Network with Attention for Small-Footprint Keyword Spotting

- **Authors/Affiliations**: Shiliang Zhang, Qiaoxi Zhu, Jinghua Zhong, Sicong Zhang - Beijing Institute of Technology; Xiaomi Corporation
- **Date**: 2021.09
- **Link**: https://arxiv.org/abs/2109.00260
- **Keywords**: Separable Convolution, Temporal Convolution, Dual Attention, Channel Attention, Temporal Attention, Small Footprint, Keyword Spotting

## Problem Statement

Deploying Keyword Spotting (KWS) models on resource-constrained edge devices (such as smartwatches, wireless earbuds, and IoT sensors) requires models to simultaneously satisfy three strict constraints: extremely small parameter counts (typically <100K), extremely low computational cost (<10M MADD), and high accuracy. Standard CNN models have achieved good results in KWS, but how to design the most effective architecture under extremely tight parameter budgets remains an open problem.

This paper analyzes the shortcomings of existing methods from two perspectives:
1. **Insufficient processing of the temporal dimension**: Standard two-dimensional convolution processes both time and frequency dimensions simultaneously. However, under extremely small parameter budgets, it may fail to fully model key temporal dynamic patterns in speech (phoneme sequences, temporal prosody).
2. **Inadequate feature selection**: Fixed convolution weights cannot adaptively focus on the most important time steps and frequency channels.

The core problem this paper aims to solve is: How to achieve high-accuracy keyword spotting with extremely small parameter counts by using separable convolution oriented towards the temporal dimension combined with a dual attention mechanism (channel attention + temporal attention).

## Methodology

### Overall Architecture - STCNN (Separable Temporal CNN)
The overall architecture of STCNN is: MFCC input -> Stacked separable temporal convolution blocks -> Dual Attention -> Global Average Pooling -> Fully Connected classification head.

### Separable Temporal Convolution
The core building block of STCNN decomposes standard two-dimensional depthwise separable convolution into operations oriented towards the temporal dimension:
1. **Temporal Depthwise Conv**: One-dimensional depthwise convolution along the time axis, where each input channel is convolved independently along the time dimension.
   - Kernel sizes are typically 3 or 5, capturing local temporal patterns between adjacent time frames.
   - Parameter count: $C * K_t$ (where $C$ is the number of channels and $K_t$ is the temporal kernel size), which is much smaller than standard two-dimensional convolution.
2. **Pointwise Conv (1x1 Convolution)**: Mixes information along the channel dimension.
   - Parameter count: $C_{in} * C_{out}$
3. **Batch Normalization + ReLU Activation**

The theoretical motivation for this decomposition is: Keyword recognition primarily relies on temporal patterns in the time dimension (phoneme sequences), while the frequency dimension provides feature channel information. Focusing computation on the temporal dimension is more parameter-efficient.

### Dual Attention Mechanism
After the separable temporal convolution, STCNN introduces two complementary attention modules:

**Channel Attention**:
- **Function**: Adaptively weights the importance of different feature channels.
- **Implementation**: Global Average Pooling -> FC dimensionality reduction (compression ratio $r=4$) -> ReLU -> FC dimensionality expansion -> Sigmoid.
- **Effect**: Enhances feature channels related to keywords and suppresses noise or irrelevant channels.
- **Parameter Count**: $2 * (C^2/r)$; due to the compression by $r$, the additional parameters are minimal.

**Temporal Attention**:
- **Function**: Adaptively weights the importance of different time steps.
- **Implementation**: Aggregation along the channel dimension -> FC layer -> Sigmoid -> Time step weights.
- **Effect**: Enables the model to focus on the most discriminative time periods for keywords (e.g., the core phoneme segment of the keyword) and weakens silence or transition segments before and after the word.

The two attention modules are applied in series: first, channel attention selects important features, and then temporal attention focuses on key time periods.

### Input Features
Standard 40-dimensional MFCC features are used, which is the mainstream choice in the KWS field, offering higher computational efficiency than spectrogram inputs.

### Model Configuration
The paper provides different scales of STCNN variants:
- **STCNN-S** (Small): ~20K parameters
- **STCNN-M** (Medium): ~40K parameters
- **STCNN-L** (Large): ~80K parameters

## Main Contributions

1. **Introduction of a time-oriented separable convolution architecture**: By decomposing standard two-dimensional convolution into one-dimensional temporal convolution and pointwise convolution, STCNN achieves efficient temporal pattern modeling with extremely low parameter counts. This design fully leverages the temporal characteristics of speech signals.

2. **Integration of a dual attention mechanism**: Channel attention enhances discriminative features, while temporal attention focuses on key time periods. The two complement and synergize each other, significantly improving accuracy without significantly increasing parameters.

3. **Superior parameter efficiency**: Achieving approximately 95% accuracy on GSC with only about 40k parameters, outperforming standard DS-CNN under similar parameter counts.

4. **Generalization across different keyword subsets**: Performs well on 10-class, 12-class, and 35-class classification tasks.

## Experimental Results

### Datasets
- **Google Speech Commands v2**: 10-class subset, 12-class subset, and the full 35-class task.
- **Features**: 40-dimensional MFCC.

### Accuracy
- **~40K parameters**: Achieves approximately 95% accuracy on GSC 12-class.
- **STCNN-S (~20K parameters)**: Reduces parameter count by 50% with only a 1-2% drop in accuracy.
- **STCNN-L (~80K parameters)**: Achieves performance close to larger models (e.g., DS-CNN with 200K parameters) on GSC 35-class.

### Comparison with Baselines
- **vs DS-CNN**: STCNN has 1-2% higher accuracy at the same parameter count.
- **vs Standard CNN**: STCNN achieves similar or higher accuracy with a 50% reduction in parameter count.
- **vs STCNN without Attention**: Dual attention brings a 1-2% accuracy improvement.

### Ablation Studies
- **Separable vs. Standard Convolution**: Separable temporal convolution outperforms standard two-dimensional convolution in both parameter efficiency and accuracy.
- **Contribution of Channel Attention**: Using channel attention alone improves accuracy by 0.8-1.2%.
- **Contribution of Temporal Attention**: Using temporal attention alone improves accuracy by 0.5-0.8%.
- **Stacking Dual Attention**: Using both attention modules simultaneously improves accuracy by 1-2%, proving their complementarity.
- **Attention Position**: Inserting attention in intermediate layers yields the best results.

### Keyword Subset Results
- Performs well on 10-class, 12-class, and 35-class subsets.
- Accuracy on the 35-class task is naturally lower than on the 12-class task, but the advantage in parameter efficiency is maintained.

## Limitations and Future Work

### Technical Limitations
- **Streaming Processing Not Explored**: STCNN processes fixed-length audio segments in its current configuration and is not adapted for streaming/causal inference modes.
- **Attention Computational Overhead**: Although the attention modules have minimal parameters, global average pooling and FC operations introduce additional latency during inference.
- **Evaluation Limited to GSC**: All experiments are limited to Google Speech Commands; performance in noisy and far-field conditions has not been verified.

### Experimental Design Shortcomings
- No combination with model compression techniques such as quantization and pruning.
- Lack of comparison with contemporary efficient Transformer methods (e.g., KWT).
- Insufficient in-depth analysis of the impact of different convolution kernel sizes on temporal modeling.

### Future Improvement Directions
- Adapt causal temporal convolution to enable streaming KWS.
- Combine model distillation to further compress the model.
- Validate the method under more diverse acoustic conditions.
- **Insights for the KWS field**: Specialized processing of the temporal dimension is key to designing efficient KWS models, and dual attention provides a low-cost means for performance improvement.
