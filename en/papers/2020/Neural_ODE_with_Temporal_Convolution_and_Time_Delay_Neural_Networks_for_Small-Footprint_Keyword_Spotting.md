# Neural ODE with Temporal Convolution and Time Delay Neural Networks for Small-Footprint Keyword Spotting

**Authors/Affiliations**: Hiroshi Fuketa, Yukinori Morita (National Institute of Advanced Industrial Science and Technology - AIST)

**Date**: August 2020 (arXiv:2008.00209)

**Link**: https://arxiv.org/abs/2008.00209

**Keywords**: Neural Ordinary Differential Equations, Keyword Spotting, Temporal Convolution, Time Delay Neural Networks, Small-Footprint Models

## Problem Statement

Traditional KWS neural network models (such as ResNet, TCNN, TDNN) typically consist of 5-15 stacked neural network layers, resulting in a large number of model parameters and high memory usage. Deploying these models on extremely resource-constrained embedded devices faces severe challenges.

From a mathematical perspective, the stacking of layers in neural networks can be understood as a discretized approximation of continuous transformations. Neural ODE (Neural Ordinary Differential Equation) provides an elegant alternative: it treats the depth of the network as a continuous dimension and replaces discrete layer stacking with an ODE solver. This brings the following potential advantages:
- Continuous transformations described by ODE solvers can achieve equivalent expressive power with fewer "layers"
- Parameter sharing mechanisms can significantly reduce the number of model parameters
- The trade-off between computational precision and efficiency can be flexibly controlled by adjusting the precision of the ODE solver

However, applying Neural ODE to KWS tasks presents technical challenges: the compatibility issue between Batch Normalization and ODE networks, and the computational efficiency issues during the ODE solving process.

## Methodology

### Neural ODE Basics

Neural ODE models the forward propagation of a neural network as an initial value problem for an ordinary differential equation:
- dz/dt = f(z(t), t, theta), where z(t) is the hidden state and f is the parameterized dynamics function
- Given the initial state z(t0)=x, z(t1) is calculated as the output via an ODE solver
- The Adjoint Method is used for efficient gradient computation, with memory consumption independent of network depth

### Integration of NODE with KWS

This paper is the first to apply Neural ODE to the KWS task:
- The stacked layers in traditional KWS networks are replaced with ODE blocks
- The transformation function f within the ODE block combines Temporal Convolution and Time Delay Neural Network (TDNN) components
- The entire network requires only 3 "layers": an input layer, an ODE block, and an output layer

### Batch Normalization Compatibility Handling

Batch Normalization (BN) is used in standard neural networks to accelerate training and stabilize convergence. However, in NODE:
- The statistics of BN (mean and variance) need to remain consistent during the ODE solving process
- The traditional application of BN between discrete layers is not directly applicable to the continuous solving process of ODEs

This paper proposes a technical solution to make BN compatible with NODE:
- Pre-compute BN statistics before the ODE solving process
- Use fixed BN parameters during the ODE solving process to ensure consistency in gradient computation

### Inference Computation Optimization

Neural ODE may require a large number of ODE solving steps during inference:
- Methods to reduce inference computation are proposed
- Solver accuracy is controlled by adjusting the tolerance parameter of the ODE solver
- Inference computation is significantly reduced while keeping precision loss controllable

## Main Contributions

1. **First Application of Neural ODE to KWS**: Pioneers the introduction of ODE methods into the field of keyword detection, providing a completely new technical route for KWS model compression. Previously, Neural ODE was mainly applied to simple tasks such as image classification.

2. **BN Compatibility Technology**: Proposes a technical solution to make Batch Normalization work normally in NODE networks, solving the convergence issue in ODE training.

3. **Significant Parameter Reduction**: The number of model parameters is reduced by 68% compared to traditional KWS models, achieving competitive performance with only 3 layers.

4. **Inference Computation Optimization**: Proposes methods to reduce inference computation within the NODE framework, making the model more suitable for deployment in resource-constrained environments.

## Experimental Results

### Experimental Setup
- Google Speech Commands dataset
- Comparison models: Traditional KWS models such as ResNet, TCNN, TDNN
- Evaluation metrics: Classification accuracy, number of parameters

### Main Results
- **Parameter Reduction**: The number of model parameters is reduced by approximately 68% compared to traditional methods
- **Layer Reduction**: Reduced from the traditional 5-15 layers to only 3 layers
- **Accuracy**: Maintains competitive classification accuracy on Google Speech Commands
- **ODE Solving Efficiency**: By adjusting solver parameters, a flexible trade-off between accuracy and computation can be achieved

### Comparison with Traditional Methods
- Under the same number of parameters, the NODE model has stronger expressive power (benefiting from the continuous transformation of ODE)
- ODE solving during training may increase computation time (requiring multiple function evaluations)
- Inference computation can be controlled by lowering solving accuracy

## Limitations and Future Work

### Method Limitations
- **High Cost of Single-Layer Computation**: Although the number of layers is reduced, the computation per step within the ODE solver may be more complex than that of standard layers
- **Large Training Computation**: NODE training is typically slower than standard networks (ODE solving requires multiple function evaluations)
- **Limited ODE Solver Selection**: Insufficient systematic exploration of different ODE solvers (Euler, RK45, etc.)
- **Accuracy Gap**: On benchmarks for deep model optimization, accuracy may still be lower than carefully tuned deep networks
- **Hardware Support**: The dynamic computation graph of ODE solving is not conducive to efficient execution on fixed-function hardware

### Future Directions
- Research ODE solving strategies more suitable for embedded deployment (e.g., fixed-step solvers)
- Combine NODE with other model compression techniques (quantization, pruning) to further reduce model size
- Explore the application of Neural SDE (Stochastic Differential Equation) in KWS
- Research Discrete NODE methods to improve hardware compatibility while maintaining parameter efficiency
- Extend the NODE concept to streaming KWS scenarios to achieve online ODE solving
