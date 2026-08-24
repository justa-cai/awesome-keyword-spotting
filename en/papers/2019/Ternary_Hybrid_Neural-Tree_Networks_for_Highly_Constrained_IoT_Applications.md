# Ternary Hybrid Neural-Tree Networks for Highly Constrained IoT Applications

- **Authors/Affiliations**: Dibakar Gope, Ganesh Dasika, Matthew Mattina (Arm ML Research Lab)
- **Date**: March 2019 (SysML 2019)
- **Link**: https://arxiv.org/abs/1903.01531
- **Keywords**: Keyword Spotting, Ternary Quantization, Neural Tree, IoT, Model Compression, Hybrid Architecture, Decision Tree

## Problem Statement

The power and storage constraints of Internet of Things (IoT) devices are extremely strict—many IoT devices use low-power microcontrollers (MCUs) with only tens to hundreds of KB of SRAM and less than 1 MB of Flash storage. Deploying neural network-based Keyword Spotting (KWS) on these devices faces severe challenges:

1. **Extreme Resource Constraints**: SRAM on IoT devices typically ranges from 64-256 KB, and Flash storage ranges from 256 KB to 1 MB. Standard KWS models (such as DS-CNN, ResNet) may exceed these limits even after quantization.
2. **Individual Limitations of Existing Compression Techniques**:
   - **Binarization (1-bit)**: Highest compression ratio but largest accuracy loss.
   - **Quantization (4-bit/8-bit)**: Better accuracy-compression trade-off, but limited compression ratio.
   - **Pruning**: Reduces the number of non-zero weights, but requires hardware support for sparse computation.
   - **Knowledge Distillation**: Requires training a complex teacher model.
3. **Computational Patterns of IoT Hardware**: IoT processors such as the Arm Cortex-M series lack dedicated matrix multiplication accelerators, making standard neural network operations inefficient. However, these processors typically offer good support for bitwise operations and simple conditional branches.
4. **Power Constraints**: KWS, as an "always-on" feature, needs to run continuously, making battery-powered IoT devices extremely sensitive to power consumption.

Therefore, the core challenge is: to design a hybrid architecture that combines multiple compression techniques, achieving extreme model compression while maintaining KWS accuracy, and optimizing for the computational characteristics of IoT hardware.

## Methodology

This paper proposes a **Ternary Hybrid Neural-Tree Network**, which combines neural networks with decision trees and compresses them via ternary quantization.

### 1. Neural Tree Architecture

The neural tree is a hybrid architecture that fuses neural networks and decision trees:

#### 1.1 Combination of Tree Structure and Neural Network Nodes

- **Internal Nodes**: The internal nodes of the decision tree are replaced by small neural networks that perform feature transformation and routing decisions.
- **Leaf Nodes**: Leaf nodes output classification results.
- **Routing Mechanism**: The output of the neural network at each internal node determines which branch of the tree the input sample continues to propagate along.
- **Soft Routing**: Differentiable routing decisions are implemented using softmax or sigmoid, allowing the entire tree structure to be trained end-to-end.

#### 1.2 Replacement of Certain Neural Network Layers

Replace standard fully connected or convolutional layers with tree-based routing:
- The tree structure provides the capability of **conditional computation**—different inputs follow different paths.
- Input samples only need to traverse a portion of the tree's path, rather than all layers of the entire network.
- This reduces the average computational cost per sample.

### 2. Ternary Quantization

Apply ternary quantization to the weights of the neural network:
- **Weights restricted to {-1, 0, +1}**: Each weight is represented using only 2 bits (including the zero state).
- **2-bit Representation**: Achieves a theoretical compression ratio of 16x compared to full precision (32-bit float).
- **Computational Optimization**:
  - Multiplication by +1 and -1 becomes simple addition and subtraction.
  - Zero weights mean the connection is skipped, further reducing computation.
  - Ternary weights can utilize bitwise operations to implement efficient convolution and matrix multiplication.

$$w_{ternary} = \begin{cases} +1 & \text{if } w > \Delta \\ 0 & \text{if } |w| \leq \Delta \\ -1 & \text{if } w < -\Delta \end{cases}$$

where $\Delta$ is the quantization threshold.

### 3. Synergy of the Hybrid Architecture

The combination of neural trees and ternary quantization produces a synergistic effect:
- **Conditional computation of neural trees** reduces the average computation during inference.
- **Ternary quantization** reduces the overhead per weight and per operation.
- **The combination of both** achieves "double compression"—reducing both the length of the computational path and the cost of each computation.

### 4. Optimization for IoT Hardware

The design of the hybrid architecture considers the specific computational patterns of IoT hardware:
- Ternary operations can be accelerated using bitwise operations.
- The conditional branches of the tree structure are suitable for the processor's branch prediction mechanisms.
- The smaller model footprint fits within limited SRAM and Flash.

## Main Contributions

1. **Hybrid Architecture of Neural Tree + Ternary Quantization**: First proposed a hybrid architecture that combines neural networks with decision trees and enhances them via ternary quantization. This design achieves the synergy of multiple compression strategies within a single framework—conditional computation (tree routing) + weight compression (ternary quantization).

2. **Significant Comprehensive Compression Effects**:
   - Computation reduced by **11.1%**
   - Model size reduced by **52.2%**
   - Overall memory footprint reduced by **30.6%**
   These compressions were achieved with negligible accuracy loss.

3. **IoT Hardware-Friendly Design**: The hybrid architecture is specifically optimized for the computational characteristics of IoT processors such as the Arm Cortex-M series—leveraging bitwise operations to accelerate ternary computations and conditional branches to optimize tree routing.

4. **Detailed Analysis of the Design Space**: Provides a detailed analysis of the model design space and compression trade-offs, including the impact of hyperparameters such as tree depth, branching factor, and ternary quantization thresholds on compression effects and accuracy.

5. **Published at SysML 2019**, representing important work by Arm in IoT edge intelligence.

## Experimental Results

### Compression Effects

| Metric | Reduction Ratio |
|------|---------|
| Computation | **11.1%** |
| Model Size | **52.2%** |
| Overall Memory Footprint | **30.6%** |
| Accuracy Loss | Negligible |

### Key Results
- On the Google Speech Commands dataset, the hybrid model maintained accuracy close to that of the original model after aggressive compression.
- Ternary quantization was the main contributor to the 52.2% reduction in model size, as each weight was compressed from 32-bit to approximately 2-bit.
- The conditional computation of the neural tree contributed to an 11.1% reduction in computation.
- The total compression effect of the combination was superior to using either technique alone.

## Limitations and Future Work

### Technical Limitations
- **Need for Hardware-Specific Optimization**: To fully leverage the speed advantages of ternary operations, hardware-specific optimizations (such as custom bitwise operation kernels) are required. On general-purpose processors lacking specialized optimization, ternary operations may not be as efficient as expected.
- **Irregular Memory Access**: Tree-based routing introduces irregular memory access patterns—different samples follow different paths, leading to decreased cache hit rates. On IoT processors with limited cache, this may affect actual inference speed.
- **Cost of Design Space Exploration**: The hybrid architecture involves hyperparameters in multiple dimensions (tree depth, branching factor, ternary threshold, whether to use tree routing per layer, etc.), and the computational cost of exploring the design space may be high.

### Future Directions
- Measure real inference latency and power consumption on actual IoT hardware (such as Arm Cortex-M4/M33), rather than only reporting theoretical compression ratios.
- Explore automated neural tree architecture search methods to reduce the cost of manual design space exploration.
- Investigate the scalability of the hybrid architecture on more complex KWS tasks (multi-keyword, continuous detection).
- Combine other compression techniques (such as structured pruning, knowledge distillation) to further compress the model.
- Develop specialized IoT processor microarchitecture optimizations to fully leverage the advantages of ternary operations and conditional computation.
