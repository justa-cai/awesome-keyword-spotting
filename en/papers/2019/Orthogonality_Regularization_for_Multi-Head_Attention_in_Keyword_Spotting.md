# Orthogonality Regularization for Multi-Head Attention in Keyword Spotting

- **Authors/Affiliations**: Mingu Lee, Jinkyu Lee, Hye Jin Jang, Byeonggeun Kim, Wonil Chang, Kyuwoong Hwang (Qualcomm AI Research)
- **Date**: October 2019 (ASRU 2019)
- **Link**: https://arxiv.org/abs/1910.04500
- **Keywords**: Keyword Spotting, Multi-Head Attention, Orthogonal Regularization, Attention Mechanism, Wake Word Detection, Attention Redundancy

## Problem Statement

The Multi-Head Attention (MHA) mechanism has been shown to improve model performance by capturing different aspects of the input through multiple attention heads. However, in resource-constrained applications like Keyword Spotting (KWS), multi-head attention faces a key challenge:

1. **Attention Head Redundancy**: Without appropriate constraints, multiple attention heads may learn **highly similar (redundant)** representations. This means that multiple attention heads are effectively performing the same task, wasting valuable model capacity. In KWS applications, where model size is strictly limited, every parameter should be utilized efficiently.
2. **Lack of Representation Diversity**: An ideal MHA design expects each attention head to focus on different subspaces of the input (e.g., different time-frequency patterns, different acoustic features), but the standard training process lacks any mechanism to guarantee this diversity.
3. **Specific Needs in Resource-Constrained Scenarios**: Unlike large-scale NLP models (such as BERT or GPT), KWS models typically have few attention heads (2–4), making the impact of redundancy in each head on overall performance more significant.

Therefore, the core challenge is: How to encourage different attention heads in multi-head attention to learn **diverse and complementary** representations without increasing model size or inference cost?

## Methodology

This paper introduces **Orthogonality Regularization** to constrain the diversity between the representations learned by different attention heads in multi-head attention.

### 1. Multi-Head Attention Basics

Standard multi-head attention maps the input to multiple subspaces:
- Each attention head $h_i$ computes independently: $head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
- The outputs of all heads are concatenated and linearly transformed: $\text{MHA} = \text{Concat}(head_1, ..., head_H)W^O$

### 2. Orthogonality Regularization Term

The core idea of orthogonality regularization is to encourage the output representations of different attention heads to be **mutually orthogonal**—that is, they should point in different directions in the feature space, covering different information subspaces.

The orthogonality regularization term can be defined as:

$$\mathcal{L}_{orth} = \|HH^T - I\|_F^2$$

where $H$ is the matrix formed by concatenating the outputs of all attention heads, and $I$ is the identity matrix. When $HH^T$ approaches the identity matrix, the representations of different heads become close to orthogonal (i.e., uncorrelated).

### 3. Total Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{KWS} + \lambda \cdot \mathcal{L}_{orth}$$

where $\mathcal{L}_{KWS}$ is the standard KWS classification loss (cross-entropy), and $\lambda$ is a hyperparameter controlling the strength of the orthogonality constraint.

### 4. Key Characteristics

- **Zero Inference Overhead**: Orthogonality regularization serves only as an additional loss term during training and does not alter the model's inference architecture. During inference, the model size and computational cost are identical to those of standard MHA.
- **Encourages Representation Diversity**: By minimizing the correlation between the outputs of different attention heads, it forces each head to learn unique, complementary feature patterns.

### 5. Application Scenario

This method was validated on Qualcomm’s commercial wake word **"Hey Snapdragon"** detection task. This is a real-world, product-deployment-oriented KWS application.

## Main Contributions

1. **Orthogonality Regularization Method**: Proposes, for the first time, an orthogonality regularization method specifically designed for multi-head attention in KWS. It enforces diverse feature representations across different attention heads through mathematical constraints (orthogonality), eliminating redundancy.

2. **Zero Inference Cost Improvement**: Orthogonality regularization affects only the training process and does not increase model size or inference cost. This is particularly important for resource-constrained KWS deployments—achieving performance improvements without adding latency or memory footprint.

3. **Commercial Product Validation**: Validates the effectiveness of the method on Qualcomm’s real commercial wake word "Hey Snapdragon," demonstrating its practical value in product-level KWS systems.

4. **Published at ASRU 2019**: Represents an important contribution by Qualcomm to attention optimization in KWS.

## Experimental Results

- On the "Hey Snapdragon" wake word detection task, the multi-head attention model with orthogonality regularization showed higher detection accuracy compared to the standard multi-head attention baseline.
- The model footprint (parameter count) remained unchanged, with no inference overhead.
- The orthogonality constraint effectively reduced representation redundancy between attention heads.

## Limitations and Future Work

### Technical Limitations
- **Limited Evaluation Scope**: The experiments focused on a single commercial keyword ("Hey Snapdragon"), and the generalization ability to other keywords, multi-keyword scenarios, or public benchmarks (such as Google Speech Commands) was not thoroughly tested.
- **Hyperparameter Sensitivity**: The strength of the orthogonality regularization term $\lambda$ requires careful tuning—constraints that are too strong may limit the model's representational capacity, while those that are too weak fail to effectively reduce redundancy.
- **Lack of In-Depth Analysis of Head Learning Content**: There is limited analysis of what specific features different attention heads learn after orthogonality regularization (e.g., which heads focus on spectral patterns and which focus on temporal patterns).

### Future Directions
- Evaluate the generalizability of orthogonality regularization on public KWS benchmarks and diverse keyword sets.
- Visualize and analyze changes in the attention patterns of each attention head before and after orthogonality regularization to understand the specific forms of representation diversity.
- Explore the effectiveness of other forms of diversity constraints (such as mutual information-based constraints or contrastive learning objectives) in KWS attention mechanisms.
- Combine orthogonality regularization with other KWS optimization techniques (such as knowledge distillation and quantization) to explore complementary improvement paths.
- Investigate the differences in the effectiveness of orthogonality constraints across attention models of different scales—from small KWS models to large ASR models.
