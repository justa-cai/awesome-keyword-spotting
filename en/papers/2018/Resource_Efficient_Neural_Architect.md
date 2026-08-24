# Resource-Efficient Neural Architect

- **Authors/Affiliations**: Yanqi Zhou, Siavash Ebrahimi, Haonan Yu, Hairong Liu, Sercan O. Arik, Greg Diamos (Baidu Research, Silicon Valley AI Lab)
- **Date**: 2018.06 (arXiv:1806.07912)
- **Link**: https://arxiv.org/abs/1806.07912
- **Keywords**: neural architecture search, resource constraints, reinforcement learning, keyword spotting, CIFAR-10, NAS

## Problem Statement

The goal of Neural Architecture Search (NAS) is to automatically discover optimal neural network architectures. However, traditional NAS methods (such as search based on reinforcement learning or evolutionary algorithms) typically use accuracy as the sole objective, without considering computational resource constraints. In actual deployment, different hardware platforms (smartphones, drones, smart speakers, IoT devices) have drastically different resource limitations:

**Domain Pain Points**
- Different deployment targets require different models: server-side deployments pursue the highest accuracy, while edge devices pursue the best accuracy-efficiency trade-off
- Manually designing models that satisfy specific resource constraints requires substantial expert experience and repeated trial and error
- Architectures found by existing NAS methods may fail to meet latency, memory, or power requirements on real hardware
- There are complex trade-offs among three metrics: model size, computational complexity (FLOPs), and computational intensity (FLOPs/byte)

**Shortcomings of Existing NAS Methods**
- Although the NAS of Zoph et al. (2017) and the evolution-based search of Real et al. can find high-accuracy architectures, the models they search for are often computationally enormous
- Resource constraints are not directly optimized during the search; instead, architectures that fail to satisfy the constraints are manually filtered out after the search
- The design of the search space does not consider resource-efficiency-related operations (such as depthwise separable convolutions and grouped convolutions)

**Key Challenges This Paper Aims to Solve**
- How to directly incorporate multiple resource constraints (model size, computational complexity, computational intensity) into the NAS process
- How to efficiently find constraint-satisfying, high-accuracy architectures in a large search space
- How to validate the effectiveness of the method simultaneously in two domains: image classification and speech keyword spotting

## Methodology

### Overall Framework: RENA (Resource-Efficient Neural Architect)

RENA uses Reinforcement Learning (RL) and Network Embedding to progressively improve an existing architecture while satisfying user-specified resource constraints.

**Search Space Design**
The search space consists of the following building blocks:
- Standard convolutions (various kernel sizes: 1x1, 3x3, 5x5, 7x7)
- Depthwise separable convolution
- Grouped convolution
- Pooling operations (max pooling, average pooling)
- Skip connections

The choice of each block is decided by the policy network.

**Network Embedding**

One of RENA's core innovations is the use of a network embedding to represent the current architecture state:
1. The current neural network architecture is encoded as a fixed-length vector
2. The encoding includes: each layer's type, number of parameters, input/output channel counts, receptive field size, and so on
3. The network embedding allows the policy network to "understand" the global structure of the current architecture

**Policy Network**

The policy network is an RNN (typically an LSTM), whose input is the network embedding of the current architecture and whose output is the next architecture modification decision:
1. It receives the network embedding of the current architecture as input
2. It outputs modification actions: adding a layer, removing a layer, modifying layer parameters, and so on
3. Each modification action corresponds to a concrete architectural change

**Reward Function**

The reward function is RENA's key design, considering both accuracy and resource constraints:
$$R = \text{Accuracy} - \lambda_1 \cdot \text{Penalty}(\text{ModelSize}) - \lambda_2 \cdot \text{Penalty}(\text{FLOPs}) - \lambda_3 \cdot \text{Penalty}(\text{Intensity})$$

The three resource metrics:
- **Model size**: total number of parameters (in bytes), directly affecting storage and memory requirements
- **Computational complexity**: FLOPs (floating-point operations), directly affecting inference latency and power consumption
- **Computational intensity**: FLOPs/byte, reflecting the ratio of computation to memory access. Architectures with high computational intensity are better suited to parallel computing platforms such as GPUs/TPUs

The penalty function uses a step function or a soft constraint:
$$\text{Penalty}(x) = \max(0, x - x_{target})$$

**Progressive Architecture Adaptation**

Rather than searching from scratch, RENA starts from a known good architecture and progressively modifies it via the policy network:
1. Start from a manually designed baseline architecture
2. The policy network proposes modifications
3. Evaluate the modified architecture (fast evaluation on a validation set, or using a proxy model)
4. Update the policy network (REINFORCE algorithm)
5. Repeat until the convergence condition is met

Advantages of progressive search:
- The starting point of the search is already a reasonable architecture rather than a random initialization
- Each modification is small in magnitude, reducing the discrete jumps in the search space
- The trained baseline model can be leveraged to accelerate the evaluation of new architectures

### Application to KWS

RENA performs KWS architecture search on the Google Speech Commands dataset:
1. The search space is adapted to the speech task: 1D convolutions (along the temporal dimension) replace 2D convolutions
2. Input feature: log-mel spectrogram
3. Output: 12-class classification (10 target words + unknown + silence)
4. Resource constraints are set according to the actual limitations of embedded devices

## Main Contributions

1. **Resource-constrained NAS**: For the first time, three interpretable resource metrics (model size, computational complexity, computational intensity) are directly optimized during the NAS process, rather than only filtering after the search. This makes the search process more efficient — no time is wasted exploring architectures that clearly violate the constraints.

2. **Progressive architecture adaptation**: Using network embedding and a progressive modification strategy is more efficient than searching from scratch. The progressive approach leverages the knowledge of existing architectures and reduces the effective size of the search space.

3. **Cross-domain validation**: The effectiveness of RENA was validated simultaneously in two domains — CIFAR-10 image classification and Google Speech Commands keyword spotting — demonstrating the generality of the method.

4. **Unconstrained SOTA**: Under no resource constraints, RENA achieved the highest accuracy at the time on Google Speech Commands.

5. **Superiority under constraints**: Under strict resource constraints, the architectures discovered by RENA outperform manually optimized architectures.

## Experimental Results

### CIFAR-10 Image Classification

| Constraint | Test Error Rate | Notes |
|---------|----------|------|
| Computational intensity > 100 FLOPs/byte | 2.95% | High computational efficiency |
| Parameters < 3M | 3.87% | Small model |

### Google Speech Commands Keyword Spotting

- **Unconstrained**: achieved the state-of-the-art accuracy at the time
- **Under strict resource constraints**: outperformed manually optimized architectures
- The novel architectures discovered by RENA combine depthwise separable convolutions with skip connections, a non-intuitive combination within the search space

### Search Efficiency
- Progressive search converges faster than search from scratch
- The network embedding provides an effective representation of the architecture state
- The policy network can discover high-quality architectures after only a few hundred iterations

## Limitations and Future Work

### Technical Limitations of the Method
- **Computational cost of RL search**: although progressive search reduces the cost, RL-based NAS still requires a large number of evaluation iterations, and the total computational cost is high (search times on the order of GPU-days)
- **Search space limitation**: the building blocks in the search space are predefined and do not include some emerging modules (such as attention mechanisms and Transformer layers)
- **Local optima**: progressive modifications may cause the search to fall into local optima, failing to discover completely different architecture patterns
- **Proxy evaluation error**: to accelerate the search, proxy models may be used to estimate accuracy, and proxy error may mislead the search direction

### Shortcomings of the Experimental Design
- Actual latency and power consumption were not measured on real embedded hardware
- No comparison was made with later, more efficient NAS methods such as DARTS and ProxylessNAS
- Limited disclosure of the specific accuracy numbers for the KWS results

### Possible Future Improvement Directions
- Combine with differentiable NAS (such as DARTS) to drastically reduce search cost
- Extend the search space to include attention mechanisms and Transformer modules
- Introduce hardware-aware latency models to directly optimize actual inference latency during the search
- Extend RENA to multi-task search (simultaneously searching for shared architectures for KWS and ASR)

### Implications for the KWS Field
- Automated architecture search is an important tool for KWS model design and can replace time-consuming manual tuning
- Resource-constrained NAS enables models to be customized for specific hardware platforms
- Baidu's RENA is one of the early works applying NAS to KWS and inspired a large amount of subsequent research
- The introduction of computational intensity as an optimization objective is inspiring for understanding the actual efficiency of neural networks on different hardware
- The "progressive improvement" search paradigm is highly actionable in engineering practice
