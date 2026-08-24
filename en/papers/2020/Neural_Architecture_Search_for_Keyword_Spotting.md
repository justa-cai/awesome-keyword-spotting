# Neural Architecture Search for Keyword Spotting

**Authors/Affiliations**: Ryuga Sugimura, Mingkun Zhang, Yichao Zhou, Zhiru Zhang, Naiyan Wang, Zhaozhong Yin, C.-C. Jay Kuo (University of Alberta, Huawei Technologies)

**Date**: September 2020 (arXiv:2009.00165)

**Link**: https://arxiv.org/abs/2009.00165

**Keywords**: Neural Architecture Search, Keyword Spotting, Efficient Model Design, DARTS, Differentiable Search

## Problem Statement

Designing high-performance KWS neural network architectures requires extensive domain expertise and significant experimental tuning. Hand-designed architectures may suffer from the following issues:
- **Design Bias**: Researchers' prior preferences may restrict the architectural exploration space.
- **Sub-optimality**: It is difficult to find the global optimum in the vast architectural space through manual design.
- **Task Specificity**: General-purpose architectures may not be optimal for the KWS task.

Neural Architecture Search (NAS) can automate this process by discovering optimal architectures within a given search space. However, applying NAS to the KWS task requires addressing two key issues: (1) how to define a search space suitable for KWS; and (2) how to discover architectures that surpass hand-designed ones while ensuring search efficiency.

## Methodology

### Differentiable NAS (DARTS)

This paper adopts the DARTS (Differentiable Architecture Search) method:
- Relaxing the discrete architecture search problem into a continuous optimization problem.
- The selection of operations at each node is replaced by a softmax-weighted sum of all candidate operations.
- Architecture parameters ($\alpha$) and model weights ($w$) are updated alternately through Bi-level Optimization.

### KWS Search Space Design

The search space defines a set of candidate operations effective for KWS:
- **Convolutional Operations**: Standard convolutions and depthwise separable convolutions with different kernel sizes (3x3, 5x5, 7x7).
- **Pooling Operations**: Average pooling, Max pooling.
- **Skip Connections**: Identity mapping, Zero operation (no connection).
- **Special Operations**: Dilated Convolution.

### Search Process

1. Construct a super-network (Cell) containing all candidate operations.
2. Train the super-network on the Google Speech Commands dataset.
3. Optimize architecture parameters and model weights simultaneously via gradient descent.
4. After the search, select the operation with the highest weight at each node as the final architecture.
5. Re-train the discovered architecture and evaluate its performance.

### Architecture Analysis

Visual analysis was performed on the architectures discovered by NAS:
- Analyzed the distribution of effective operation types.
- Investigated architectural patterns across different layers.
- Correlated the discovered architectural patterns with the characteristics of the KWS task.

## Main Contributions

1. **Application of DARTS to KWS**: Successfully applied differentiable NAS to the keyword spotting task, validating the effectiveness of NAS in speech tasks and providing an automated tool for KWS architecture design.

2. **Performance Beyond Hand-Designed Models**: The architectures discovered by NAS achieved accuracy equal to or exceeding that of hand-designed models at similar parameter scales, demonstrating the value of automatic architecture search in the KWS domain.

3. **KWS Architecture Pattern Analysis**: By analyzing architectures discovered by NAS, this work reveals effective network structural patterns for the KWS task. These findings can guide future manual architecture design.

4. **Parameter Efficiency**: The discovered models are more parameter-efficient than hand-designed models at the same accuracy level, which is significant for embedded deployment.

## Experimental Results

### Dataset
Google Speech Commands dataset

### Main Results
- The architectures discovered by NAS achieved competitive accuracy on the Google Speech Commands dataset.
- Compared to hand-designed models, they exhibited higher accuracy at the same parameter count or fewer parameters at the same accuracy.
- The discovered architectures exhibited structural patterns distinct from hand-designed ones.

### Architecture Analysis Findings
- NAS tends to use larger convolutional kernels in shallow layers (to capture low-level acoustic features).
- Deeper layers more frequently use small convolutional kernels and skip connections (for high-level feature combination).
- Depthwise separable convolutions were selected in most layers, validating their efficiency in KWS.
- Dilated convolutions were selected in certain layers, potentially helping to expand the receptive field without increasing parameters.

## Limitations and Future Work

### Methodological Limitations
- **High Search Cost**: Training the super-network requires substantial GPU resources. Although more efficient than Reinforcement Learning-based NAS, it is still not lightweight.
- **Search Space Constraints**: Defining the search space requires domain knowledge, and different search spaces may lead to different optimal architectures.
- **Focus on Accuracy Only**: The search objective primarily targets classification accuracy, without incorporating hardware metrics such as latency and energy consumption into the optimization goals.
- **Limited Comparison Scope**: The comparison with hand-designed architectures is not extensive, lacking comparisons with contemporary NAS methods.

### Future Directions
- Introduce hardware-aware NAS, incorporating latency and energy consumption as search constraints.
- Explore larger search spaces, including attention mechanisms and Transformer components.
- Investigate One-shot NAS methods to further reduce search costs.
- Combine NAS with Quantization-Aware Training to directly search for quantization-friendly architectures.
- Extend to streaming KWS scenarios, considering causality and real-time constraints.
