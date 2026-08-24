# Very Fast Keyword Spotting System with Real Time Factor Below 0.01

**Authors/Affiliations**: Jan Siler, Jan Cernocky (Technical University of Liberec, Brno University of Technology)

**Date**: July 2020 (arXiv:2007.10706)

**Link**: https://arxiv.org/abs/2007.10706

**Keywords**: Keyword Spotting, Real-Time Factor, Fast Inference, Neural Networks, Low Latency

## Problem Statement

Keyword Spotting (KWS) systems require extremely high computational efficiency when running on resource-constrained devices. The Real-Time Factor (RTF) is a key metric for measuring inference efficiency, defined as the ratio of processing time to audio duration:
- RTF = 1.0: Processing speed is equal to audio playback speed
- RTF = 0.1: Processing speed is 10 times faster than real-time
- RTF < 0.01: Processing speed is more than 100 times faster than real-time

Most existing KWS systems have an RTF significantly higher than 0.01, meaning that on the most resource-constrained devices (such as low-power MCUs), they may fail to meet real-time processing requirements or interfere with other tasks during KWS processing. The technical challenges in achieving an RTF below 0.01 include:
- The processing time budget per audio frame is extremely low (on the order of microseconds)
- The number of floating-point operations (FLOPs) is strictly limited
- The model must be small enough to fit within extremely low memory budgets

## Methodology

### Ultra-Lightweight Architecture Design

The system design follows the principle of extreme efficiency:

**Feature Extraction**:
- Uses Mel-frequency cepstral coefficients (MFCC) or Mel filter banks as input
- Feature dimensions are compressed as much as possible to reduce the input size for subsequent network layers

**Compact Neural Network**:
- Uses a minimal number of network layers and neurons
- Aggressive parameter reduction: each layer contains only the necessary minimum amount of parameters
- Avoids computationally intensive operations (such as large matrix multiplications)

**Optimization Strategies**:
- Reduces the number of floating-point operations (FLOPs) per frame
- Optimizes memory access patterns to reduce cache misses
- Customizes model structures for specific keywords

### Performance-Efficiency Trade-off Analysis

Systematically analyzes the accuracy-efficiency trade-off under extreme compression conditions:
- Measures RTF and accuracy for different model sizes
- Determines the optimal model configuration under the constraint of RTF < 0.01
- Analyzes the sources of accuracy loss and mitigation methods

## Main Contributions

1. **KWS System with RTF < 0.01**: For the first time, a KWS system with an RTF below 0.01 was achieved, processing more than 100 times faster than real-time on standard hardware. This breakthrough enables the deployment of KWS on devices with the most limited computational resources.

2. **Accuracy-Efficiency Analysis under Extreme Compression**: Systematically analyzes model accuracy performance under the constraint of RTF < 0.01, providing a quantitative reference for the design of ultra-efficient KWS systems.

3. **Balance between High Efficiency and Reasonable Accuracy**: Demonstrates that reasonable KWS accuracy can still be maintained even under extreme computational constraints.

4. **Practical Reference**: Provides performance benchmarks and design guidelines for the industry to design ultra-low-power KWS systems.

## Experimental Results

### Experimental Setup
- Limited vocabulary KWS task
- Standard CPU hardware used to measure RTF
- Comparison of RTF and accuracy under different model configurations

### Main Results
- **RTF**: Achieved an RTF below 0.01, with processing speed far exceeding real-time requirements
- **Accuracy**: Maintained acceptable detection accuracy even under extreme compression
- **Efficiency Improvement**: More than an order of magnitude faster than existing KWS systems
- **Resource Usage**: Extremely low memory and computational resource usage, suitable for the most constrained embedded platforms

### Performance-Efficiency Curve
- Model size is approximately linearly related to RTF
- Accuracy decreases slowly as the model is compressed, then drops sharply after a turning point
- Models with RTF < 0.01 are located in the region of slow accuracy decline, sacrificing limited accuracy

## Limitations and Future Work

### Methodological Limitations
- **Accuracy Sacrifice**: The achievement of extreme efficiency comes at the cost of some accuracy loss
- **Vocabulary Limitations**: Evaluation was conducted only on a limited vocabulary; scalability to large vocabulary scenarios has not been verified
- **Architectural Scalability**: The ultra-lightweight architecture may not scale well to larger keyword sets
- **Hardware-Specific Optimization**: Some optimizations may rely on specific hardware characteristics, limiting portability
- **Robustness**: Very small models may have poor robustness under noisy and far-field conditions

### Future Directions
- Explore the limits of model compression: Is it feasible under the condition of RTF < 0.001?
- Study binarized/trinarized networks to further reduce computational requirements
- Combine more efficient feature representations (e.g., learned features) to reduce front-end computation
- Explore the design of dedicated hardware accelerators to achieve even greater efficiency
- Study dynamic computation allocation: use small models for simple samples and large models for complex samples
