# AutoKWS: Keyword Spotting with Differentiable Architecture Search

**Authors/Affiliations**: Hao Li, Xue Jiang, Zishen Huang, Ji Li, Yu Hu, Kai Li (Xiaomi AI Lab)

**Date**: September 2020 (arXiv:2009.03658)

**Link**: https://arxiv.org/abs/2009.03658

**Keywords**: Keyword Spotting, Differentiable Architecture Search, NAS, AutoML, Architecture Discovery

## Problem Statement

The design of KWS neural network architectures typically requires deep domain expertise and extensive trial-and-error experimentation. Hand-designed architectures may suffer from designer bias and fail to achieve optimal performance. Different KWS application scenarios (different keyword sets, different noise conditions, different hardware platforms) may require different optimal architectures, making it impractical to manually design architectures for each scenario.

AutoML technology offers the possibility of automatically discovering optimal architectures, but existing NAS methods face the following challenges when applied to KWS:
- The definition of the search space lacks guidance from KWS domain knowledge
- Low search efficiency, requiring significant computational resources
- Discovered architectures may be unsuitable for actual deployment (excessive parameter count, high latency)

## Methodology

### DARTS Framework Adaptation

AutoKWS is based on the DARTS (Differentiable Architecture Search) framework and has been specifically adapted for the KWS task:

**Continuous Relaxation**:
- Transforms discrete architecture selection (choosing which operation) into a continuous optimization problem
- Each node uses a softmax-weighted mixed operation: $o\_bar(x) = \sum(\text{softmax}(\alpha_i) * o_i(x))$
- $\alpha$ represents the architecture parameters, determining the weight of each operation

**Bi-level Optimization**:
- Architecture parameters ($\alpha$) and model weights ($w$) are optimized alternately
- The training set updates model weights, while the validation set updates architecture parameters
- Ensures that the discovered architecture possesses good generalization capabilities

### KWS-Specific Search Space

A search space designed for small-footprint KWS models:
- **Convolutional Operations**: 3x3, 5x5 standard convolutions and depthwise separable convolutions
- **Pooling Operations**: 3x3 average pooling, 3x3 max pooling
- **Special Operations**: Skip Connect, None (zero operation)
- **Constraints**: All operations have a stride of 1, using the same padding to maintain feature map dimensions

### Search and Evaluation Process

1. Define a super-network containing multiple Cells, where each Cell contains N nodes
2. Search on the Google Speech Commands training set, evaluating architecture parameters on the validation set
3. After the search, select the optimal operation for each node based on the architecture parameters $\alpha$
4. Retrain and evaluate using the discovered architecture

## Main Contributions

1. **Differentiable NAS in the KWS Domain**: Successfully adapts the DARTS framework to the KWS task, demonstrating the effectiveness of differentiable NAS in speech classification tasks. This is a practical implementation from the industry (Xiaomi), offering high practical value.

2. **Small-Footprint Model Search Space**: Designs a search space suitable for small-footprint KWS models, ensuring that the discovered architectures meet the requirements for actual deployment in terms of parameter count and computational cost.

3. **Automated Architecture Design**: Provides an automated method that does not require manual architecture design expertise, lowering the technical barrier for KWS system development.

4. **Architecture Pattern Discovery**: The architecture patterns discovered by AutoKWS differ from common hand-designed ones, providing new design inspiration and directions for KWS architecture design.

## Experimental Results

### Experimental Setup
- Google Speech Commands dataset
- Search space: Each Cell contains 4 nodes, with 7 candidate operations
- Search performed using a single GPU

### Main Results
- The architecture discovered by AutoKWS achieves competitive accuracy on Google Speech Commands
- Compared to hand-designed baseline models (such as DS-CNN), it achieves higher accuracy with the same parameter count
- The search process is faster than non-differentiable NAS methods (such as reinforcement learning, evolutionary algorithms)
- The discovered architectures exhibit unique design patterns: more frequent use of depthwise separable convolutions and asymmetric connections

### Architecture Visualization Analysis
- Different Cells exhibit different architectural patterns, indicating that NAS can automatically adapt to the functional requirements of different layers
- Shallow layers tend to use larger convolutional kernels, while deeper layers use smaller convolutional kernels and skip connections more frequently
- Depthwise separable convolutions are selected in most positions, validating their efficiency in KWS

## Limitations and Future Work

### Method Limitations
- **GPU Resource Requirements**: The search process still requires GPU resources, which may pose a barrier for small teams
- **Search Space Impact**: The quality of the final architecture is influenced by the definition of the search space; an unreasonable search space may limit the discovery of better architectures
- **Classification-Only Approach**: The method is only applicable to classification-based KWS and does not cover sequence-to-sequence KWS methods (such as CTC, RNN-T)
- **Unverified Transferability**: The transfer performance of the discovered architectures across different datasets and keyword sets has not been fully validated

### Future Directions
- Introduce hardware-constraint-aware search, incorporating latency and memory usage into the search objectives
- Expand the search space to include modern architecture components such as attention mechanisms and Transformers
- Research weight-sharing NAS methods to further reduce search costs
- Explore one-shot search to discover universal architectures applicable to various KWS scenarios
- Combine with quantization-aware search to directly discover architectures that are friendly to quantization
