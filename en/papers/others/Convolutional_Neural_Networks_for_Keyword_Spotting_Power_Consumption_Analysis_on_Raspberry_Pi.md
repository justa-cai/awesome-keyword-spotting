# Convolutional Neural Networks for Keyword Spotting: Power Consumption Analysis on Raspberry Pi

- **Authors/Affiliations**: Raphael Tang, Weijie Wang, Zhucheng Tu, Jimmy Lin (University of Waterloo, David R. Cheriton School of Computer Science)
- **Date**: 2018
- **Link**: https://arxiv.org/abs/1712.01961
- **Keywords**: Keyword Spotting, CNN, Power Consumption Analysis, Raspberry Pi, Edge Computing, Energy Efficiency

## Problem Statement

Although Convolutional Neural Networks (CNNs) have demonstrated excellent accuracy in keyword spotting tasks, their deployment on edge devices such as the Raspberry Pi raises concerns regarding actual power consumption. For always-on Keyword Spotting (KWS) applications, the energy efficiency of neural network models is as important as their accuracy. However, most existing research focuses solely on model accuracy and parameter count, lacking empirical studies on the actual power consumption of different CNN architectures on real edge hardware.

The core problem addressed by the paper: How do different CNN architectures perform in terms of power consumption on real edge hardware? What is the trade-off relationship between accuracy and power consumption? Which architectural factors (depth, width, kernel size) have the greatest impact on power consumption? The paper conducts a systematic empirical measurement of power consumption for various CNN configurations on the Raspberry Pi 3, filling the gap in empirical power consumption analysis in KWS research.

## Methodology

### Hardware Platform

- **Device**: Raspberry Pi 3 Model B
- **Processor**: Broadcom BCM2837, Quad-core ARM Cortex-A53 @ 1.2GHz
- **Memory**: 1GB LPDDR2
- **Power Measurement**: Real-time power consumption during inference is measured directly using external hardware power monitoring devices (such as a Monsoon power meter or USB power meter)
- **Measurement Method**: Record the power consumption curve during inference and calculate the average power consumption and energy consumption

### CNN Architecture Evaluation

The paper systematically evaluates various CNN architecture configurations:

1. **Depth Variation**: Different network depths ranging from shallow (2-3 layers) to deep (10+ layers)
2. **Width Variation**: Different numbers of convolutional channels (32, 64, 128, 256, etc.)
3. **Kernel Size**: Different convolutional kernel sizes (3×3, 5×5, 7×7)
4. **Pooling Strategy**: Max pooling vs. Average pooling, different pooling window sizes
5. **Fully Connected Layers**: Configurations of fully connected layers with different sizes

### Evaluation Framework

1. **Accuracy Evaluation**: Evaluate classification accuracy on the Google Speech Commands dataset (35 words)
2. **Power Measurement**: Run inference on the Raspberry Pi and measure actual power consumption using an external power meter
3. **Inference Latency**: Measure the latency of a single inference (in milliseconds)
4. **Energy Efficiency Metrics**: Calculate the energy consumption per inference (Joules/inference) and the energy efficiency ratio (Accuracy/Watt)

### Experimental Configuration
- **Dataset**: Google Speech Commands dataset (35 word classes)
- **Training Environment**: Trained on a GPU server, deployed for inference on the Raspberry Pi
- **Inference Engine**: Uses an optimized CPU inference framework (such as TensorFlow Lite or custom optimized implementation)
- **Batch Size**: Batch size is 1 during inference, simulating the actual usage pattern of always-on scenarios

## Main Contributions

1. **First Systematic Empirical Analysis of KWS Power Consumption**: The paper provides the first comprehensive empirical analysis of the power consumption of CNN-KWS models on real edge hardware (Raspberry Pi). This fills an important gap in the literature—most KWS studies only report model accuracy and parameter count, whereas actual power consumption is a more critical limiting factor for always-on deployments.

2. **Analysis of the Impact of Architectural Factors on Power Consumption**: Identifies key architectural factors affecting CNN power consumption:
   - **Depth Impact**: The impact of network depth on power consumption is more significant than that of width
   - **Width Impact**: Increasing the number of channels linearly increases power consumption
   - **Kernel Size**: Larger convolutional kernels significantly increase computational load and power consumption

3. **Accuracy-Power Pareto Frontier**: Establishes the accuracy-power Pareto frontier for CNN-KWS on the Raspberry Pi, identifying the optimal architecture configuration that achieves the highest accuracy given a specific power budget. This has direct guiding value for the design of practical edge KWS systems.

4. **Validation of the Effectiveness of Compact Models**: Demonstrates that carefully designed small CNN models can significantly reduce power consumption (compared to large models) while incurring only a 1-2% loss in accuracy. This validates the practicality of compact models in edge KWS deployments.

5. **Power Measurement Methodology**: Establishes a reproducible methodology for measuring the power consumption of neural network models, providing a reference for subsequent research on edge AI energy efficiency.

## Experimental Results

### Power Consumption Measurement Results
- The power consumption of CNN-KWS models on the Raspberry Pi 3 is suitable for always-on applications
- The inference power consumption of small CNN models is approximately 1-2 Watts (within the overall power consumption range of the Raspberry Pi 3)
- The power consumption of large CNN models is significantly higher, but the marginal gain in accuracy diminishes
- The impact of network depth on power consumption is non-linear—deeper networks not only increase computational load but also increase memory access次数 (number of memory accesses)

### Accuracy-Power Trade-off
- Carefully designed compact CNNs achieve a 1-2% drop in accuracy while reducing power consumption by 50-60%
- There is a clear Pareto optimum: beyond this point, the accuracy gain brought by increased power consumption decreases sharply
- Depth has a greater impact on power consumption than width: under the same parameter count, shallow and wide networks consume less power than deep and narrow networks
- Convolutional layers are the main contributors to power consumption, while the proportion of power consumption from fully connected layers varies with model design

### Key Findings
- **Optimal Depth**: 4-6 convolutional layers provide the best balance between accuracy and power consumption
- **Number of Channels**: 64-128 channels are a reasonable choice for most layers, with further increases yielding limited accuracy improvements
- **Inference Latency**: Small models have an inference time of 5-15ms on the Raspberry Pi, meeting the requirements for real-time KWS
- **Memory Bandwidth**: Inference power consumption depends not only on computational load (FLOPs) but also on memory access patterns—reading weight matrices is a significant source of power consumption

## Limitations and Future Work

### Limitations

1. **Hardware Platform Limitations**: The study is limited to the Raspberry Pi 3 platform, and the results cannot be directly generalized to other edge devices (microcontrollers, DSPs, dedicated AI accelerators). Different hardware platforms have vastly different power characteristics, computing units, and memory hierarchies.

2. **Incomplete Architecture Coverage**: Only CNN architectures were evaluated, excluding power consumption comparisons for RNNs (LSTM/GRU), Transformers, or hybrid architectures (such as CRNN). Different architectures may have significantly different computational efficiency and power consumption characteristics on CPUs.

3. **Measurement Environment Limitations**: Power consumption measurements are based on ARM CPU inference on the Raspberry Pi and do not reflect scenarios involving GPU acceleration or dedicated Neural Processing Units (NPUs). Modern edge devices are often equipped with dedicated AI accelerators, whose energy efficiency characteristics are completely different from general-purpose CPUs.

4. **Dataset Limitations**: The Google Speech Commands dataset (35 words) was used, which may not fully represent the scenario of actual wake-word detection—wake-word detection typically focuses on a single target word and has stricter requirements for false alarm rates.

5. **Software Stack Impact**: Different inference frameworks (e.g., TFLite vs. ONNX Runtime vs. custom implementations) may have significantly different power consumption for the same model and hardware. The paper does not fully discuss the impact of this factor.

### Future Work

1. **Cross-Platform Power Modeling**: Develop analytical models capable of predicting the power consumption of CNN-KWS on different edge hardware, reducing the need for actual measurements.
2. **Neural Architecture Search (NAS)**: Incorporate power consumption as one of the optimization objectives in NAS to automatically search for architectures with optimal accuracy under specific power budgets.
3. **Impact of Quantization on Power Consumption**: Systematically evaluate the impact of INT8/INT4 quantization on the power consumption of edge devices—quantization not only reduces model size but may also significantly reduce power consumption by lowering memory bandwidth requirements.
4. **Evaluation on Dedicated Accelerators**: Evaluate the power consumption of CNN-KWS on dedicated AI accelerators (such as Google Edge TPU, NVIDIA Jetson Nano) and compare the results with those from general-purpose CPUs.
5. **Dynamic Power Management**: Develop strategies to dynamically adjust model complexity based on device battery level and application requirements, using large models for high-accuracy detection when battery is sufficient, and switching to small models for low-power mode when battery is low.
