# Hardware Aware Training for Efficient Keyword Spotting on General Purpose and Specialized Hardware

**Authors/Affiliations**: Peter O'Connor, Mike Hopkins, Greg Kiar, Chris Eliasmith (Applied Brain Research Inc.)

**Date**: September 2020 (arXiv:2009.04465)

**Link**: https://arxiv.org/abs/2009.04465

**Keywords**: Hardware-Aware Training, Keyword Spotting, Quantization, Neuromorphic Computing, Efficient Inference

## Problem Statement

Neural network models trained on general-purpose hardware (GPUs) face severe accuracy degradation when deployed to specialized or resource-constrained hardware. The root causes of this "training-deployment hardware mismatch" include:

1. **Quantization Error**: Training uses 32-bit floating-point arithmetic, whereas deployment hardware may only support low-precision arithmetic (8-bit integers, or even binary).
2. **Nonlinearity Differences**: Numerical differences may exist in the implementation of activation functions across different hardware platforms.
3. **Computational Order Differences**: Differences in the accumulation order during parallel computation lead to variations in floating-point precision.
4. **Rounding Mode Differences**: Different rounding strategies on various hardware platforms can lead to the accumulation of systematic errors.

In particular, when using aggressive low-precision quantization, simple Post-Training Quantization (PTQ) can result in significant accuracy degradation, potentially rendering the model completely ineffective.

## Methodology

### Core Idea of Hardware-Aware Training

Explicitly simulate the constraints of the target hardware during the training process, allowing the model to "adapt" to the characteristics of the deployment hardware during the training phase:

**Quantization Simulation**:
- Insert pseudo-quantization nodes during the forward pass.
- Simulate low-precision fixed-point arithmetic: quantize floating-point weights and activations to a specified bit-width, then dequantize them back to floating-point.
- Quantization function: $Q(x) = \text{clip}(\text{round}(x/\text{scale}) + \text{zero\_point}) * \text{scale}$
- Handle the non-differentiability of the quantization operation using the Straight-Through Estimator (STE).

**Fixed-Point Arithmetic Simulation**:
- Simulate the overflow and truncation behavior of fixed-point numbers.
- Control the integer and fractional bit-widths of fixed-point numbers.
- Ensure that numerical behavior during training is consistent with the deployment hardware.

**Hardware-Specific Nonlinearity Simulation**:
- Simulate approximate implementations of activation functions on the target hardware.
- Includes fixed-point approximations of functions such as ReLU and Sigmoid.

### Support for Neuromorphic Computing

This paper specifically focuses on deployment on neuromorphic chips:
- Neuromorphic chips use the Spiking Neural Network (SNN) computational paradigm.
- Converting traditional Artificial Neural Networks (ANNs) to SNNs requires specialized training strategies.
- Simulate the membrane potential accumulation and threshold firing mechanisms of spiking neurons.

### Training Process

1. Pre-train the model at standard floating-point precision.
2. Insert hardware simulation layers into the training loop.
3. Fine-tune using simulated hardware constraints.
4. The model gradually adapts to the numerical characteristics of the target hardware.

## Main Contributions

1. **Hardware-Aware Training Methodology**: Proposes a complete hardware-aware KWS training methodology that systematically addresses the training-deployment hardware mismatch. This method is not limited to specific hardware and can serve as a general framework adaptable to different target platforms.
2. **Quantization Accuracy Recovery**: Demonstrates that hardware-aware training can recover most of the accuracy loss caused by naive quantization. This has direct engineering value for practical deployment.
3. **Multi-Hardware Platform Support**: The method is applicable to both general-purpose processors and specialized hardware (such as neuromorphic chips), demonstrating good platform generality.
4. **End-to-End Training-Deployment Pipeline**: Provides a complete technical path from training to deployment, reducing the iterative adjustments between training and deployment in traditional workflows.

## Experimental Results

### Experimental Setup
- Google Speech Commands dataset.
- Comparison between standard training and hardware-aware training.
- Evaluation of accuracy changes under different quantization bit-widths.

### Main Results
- **8-bit Quantization**: Hardware-aware training almost completely eliminates quantization accuracy loss.
- **4-bit Quantization**: Hardware-aware training significantly mitigates accuracy degradation, with the model maintaining usable accuracy.
- **Lower Precision**: Even under aggressive quantization conditions, hardware-aware training still provides meaningful accuracy recovery.
- **Cross-Platform**: The method demonstrates effectiveness on both general-purpose processors and simulated neuromorphic hardware.

### Comparative Analysis
- **Naive Post-Training Quantization (PTQ)**: Accuracy drops sharply as the degree of quantization increases.
- **Hardware-Aware Training (QAT)**: The accuracy curve is significantly flatter and more robust to quantization.
- **Training Overhead**: Hardware simulation increases training time by approximately 10-20%, but training is performed only once.

## Limitations and Future Work

### Method Limitations
- **Hardware Characteristic Knowledge**: Requires knowledge of the specific characteristics of the target hardware (bit-width, rounding mode, etc.) during training, which can be difficult when hardware information is not fully public.
- **Training Overhead**: Hardware simulation increases computational and memory overhead during training.
- **Limited Validation on Actual Hardware**: Many experiments are conducted in simulated environments, with insufficient validation on actual hardware.
- **Limited Scope of Quantization**: Focuses primarily on quantization effects, without fully considering other hardware constraints (such as memory bandwidth and cache size).

### Future Directions
- Conduct more comprehensive validation on actual specialized hardware (rather than simulators).
- Research automated methods for extracting hardware characteristics to reduce reliance on hardware specification manuals.
- Extend to more types of hardware constraints (such as irregular computation patterns after sparsification and structured pruning).
- Combine with Neural Architecture Search (NAS) to consider hardware constraints during the architecture search phase.
- Research cross-hardware transfer learning to enable a single model to adapt to multiple deployment platforms.
