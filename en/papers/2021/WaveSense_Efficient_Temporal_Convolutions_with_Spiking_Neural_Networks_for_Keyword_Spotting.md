# WaveSense: Efficient Temporal Convolutions with Spiking Neural Networks for Keyword Spotting

- **Authors/Affiliations**: Bryan Feys, Sadique Sheik - SynSense AG
- **Date**: 2021.11
- **Link**: https://arxiv.org/abs/2111.01456
- **Keywords**: Spiking Neural Networks, Temporal Convolutions, Keyword Spotting, Neuromorphic Computing, Energy Efficiency, Low Power, LIF Neurons, Surrogate Gradient

## Problem Statement

Keyword Spotting (KWS) systems typically need to operate in an "always-on" mode on battery-powered edge devices (such as smartwatches, wireless earbuds, and IoT sensors). While traditional deep learning methods (CNNs, RNNs, Transformers) have achieved excellent performance in accuracy, their computational processes involve a large number of floating-point multiply-accumulate (MAC) operations. This results in high energy consumption during continuous operation, significantly impacting battery life.

Spiking Neural Networks (SNNs), the third generation of neural networks, have a computational paradigm that more closely resembles the working mechanism of the biological brain: neurons communicate via discrete spikes (binary signals of 0 or 1) rather than continuous real-valued numbers. The theoretical computational energy consumption of SNNs can be 1-3 orders of magnitude lower than that of equivalent Artificial Neural Networks (ANNs) because:

1.  **Spike-based computation replaces MAC operations**: Multiplication of synaptic weights by spikes (0 or 1) degenerates into simple conditional addition (if the spike is 1, accumulate the weight; otherwise, do nothing), eliminating expensive floating-point multiplications.
2.  **Event-driven sparse computation**: Computation only occurs when a neuron generates a spike. Since most neurons do not generate spikes in most time steps (sparse activation), the effective computational load is significantly reduced.
3.  **Neuromorphic hardware acceleration**: Dedicated neuromorphic chips (such as Intel Loihi, IBM TrueNorth, and SynSense Speck/Xylo) can efficiently implement spike-based SNN computations, enabling ultra-low-power inference.

However, applying SNNs to KWS faces several core challenges:
1.  **Training difficulties**: The firing function (step function) of spiking neurons is non-differentiable, making it impossible to train directly using standard backpropagation algorithms.
2.  **Accuracy gap**: SNNs still lag behind equivalent ANNs in accuracy on most tasks, particularly for KWS tasks that require fine-grained feature discrimination.
3.  **Lack of architectural design experience**: There is a lack of systematic research on how to design effective KWS architectures within the SNN framework (as alternatives to mature CNN architectures).

The core problem this paper aims to solve is: How to design an SNN architecture based on temporal convolutions for KWS that achieves significant theoretical energy reduction while maintaining acceptable accuracy, and provides a practical solution for deployment on neuromorphic hardware.

## Methodology

### Overall Architecture: WaveSense
The architectural design of WaveSense draws inspiration from the successful TCN (Temporal Convolutional Network) architecture in ANNs, migrating temporal convolution operations to the spiking domain:

The overall flow is: Raw Audio -> MFCC/Spectral Feature Extraction -> Spike Encoding -> Stacked Spiking Temporal Convolution Blocks -> Readout Layer -> Classification Output.

### Spike Encoding
Converting continuous-valued acoustic features into spike sequences serves as the bridge connecting ANN inputs and SNN processing:

-   **Input Features**: MFCC features (40-dimensional) or Mel-spectrogram features are used as the starting point.
-   **Encoding Methods**:
    -   **Rate Coding**: Feature values are normalized to the [0,1] range, and 0/1 spikes are generated independently over T time steps with a probability equal to the feature value. Higher feature values result in a higher frequency of spikes at that position over the T steps.
    -   **Time-to-first-spike Coding**: The larger the feature value, the earlier the time step at which the first spike is generated. This encoding is more time-efficient (as it does not require multiple time steps to convey information) but is more complex to implement.
    -   **Paper's Choice**: Rate coding is used, encoding features over T=20-50 time steps.

### Leaky Integrate-and-Fire (LIF) Neurons
The LIF (Leaky Integrate-and-Fire) neuron is the most commonly used neuron model in SNNs, and WaveSense uses LIF as its basic computational unit:

-   **Membrane Potential Update**: $U(t) = \beta * U(t-1) + W * X(t)$
    Where $U(t)$ is the membrane potential at time $t$, $\beta$ is the leak factor ($0 < \beta < 1$), $W$ is the synaptic weight, and $X(t)$ is the input spike.
-   **Spike Generation**: $S(t) = \Theta(U(t) - V_{th})$
    If the membrane potential exceeds the threshold $V_{th}$, a spike is generated ($S(t)=1$); otherwise, $S(t)=0$. $\Theta$ is the step function.
-   **Membrane Potential Reset**: After generating a spike, the membrane potential is reduced by the threshold: $U(t) = U(t) - V_{th} * S(t)$
-   **Leakage Mechanism**: The condition $\beta < 1$ causes the membrane potential to decay gradually in the absence of new inputs, simulating the membrane potential leakage characteristics of biological neurons, which helps in processing temporal information.

### Spiking Temporal Convolution
The core building block of WaveSense is the spiking temporal convolution layer:

-   **Structure**: Standard one-dimensional temporal convolution (1D Convolution) is migrated to the spiking domain. The convolution kernel weights $W$ are convolved with the input spike sequence, and the output is fed into LIF neurons.
-   **Computation Process**:
    1.  The input spike sequence is convolved with the convolution kernel weights (addition operation, since inputs are 0/1 spikes).
    2.  The convolution output serves as the input current for the LIF neurons.
    3.  LIF neurons generate output spikes according to the membrane potential update rules.
-   **Unrolling over Multiple Time Steps**: SNNs are unrolled along the time dimension, executing one convolution + LIF update per time step. The entire sequence requires T time steps to complete one inference.
-   **Residual Connections**: Residual connections (similar to ResNet) are used between spiking temporal convolution blocks, jumping the spike output from the previous layer to the next layer to help propagate gradients along the time dimension.

### Surrogate Gradient Training
Since the step firing function of LIF neurons is non-differentiable (the gradient is infinite at the threshold and zero elsewhere), direct backpropagation is not possible. WaveSense employs the Surrogate Gradient method:

-   **Surrogate Gradient Function**: During backpropagation, a differentiable function is used to approximate the derivative of the step function. Common choices include:
    -   Sigmoid Surrogate: $\sigma'(x) = \sigma(x)(1-\sigma(x))$, where $\sigma$ is the sigmoid function.
    -   Fast Sigmoid Surrogate: Uses $1/(1+|\pi*x/2|)^2$ as a derivative approximation.
    -   ArcTan Surrogate: $1/(1+(\pi*x)^2)$
-   **Backpropagation Through Time (BPTT)**: The network is unrolled along the time dimension, and standard BPTT is used to compute gradients. The surrogate gradient replaces the true non-differentiable gradient at the spike generation step of each LIF neuron.
-   **Loss Function**: A readout layer (rate readout: counting the spike frequency of each neuron in the last layer over T time steps) is used for classification at the final time step, employing standard cross-entropy loss.

### Readout Layer
Since spike outputs are binary, a mechanism is needed to convert spike sequences into continuous classification probabilities:

-   **Rate Readout**: The total number of spikes (or average spike frequency) for each neuron in the last layer is counted over T time steps, and classification probabilities are output via softmax.
-   This design ensures that the readout layer outputs are continuous values (spike frequencies in the [0,1] range), allowing for training with standard cross-entropy loss.

## Main Contributions

1.  **Introduction of Temporal Convolution Layers in the SNN Framework for Keyword Spotting**: This is the first systematic migration of the Temporal Convolutional Network (TCN) architecture to Spiking Neural Networks for KWS. Spiking temporal convolutions combine the local feature extraction capability of CNNs with the low-power characteristics of SNNs, providing a new architectural option for SNNs in KWS.
2.  **Demonstration that SNNs can achieve competitive accuracy with significantly lower theoretical energy consumption**: WaveSense achieved approximately 93% accuracy on GSC v2 (12 classes). Although this is lower than SOTA ANN models (approximately 97-98%), the theoretical energy consumption is reduced by up to 30 times. This demonstrates the possibility of the accuracy-energy trade-off and the potential of SNNs in extreme low-power scenarios.
3.  **Provision of a practical architecture efficiently mappable to neuromorphic hardware**: The architectural design of WaveSense (Temporal Convolution + LIF Neurons + Rate Coding) is highly compatible with the computational models of existing neuromorphic chips (such as SynSense Speck/Xylo, Intel Loihi), allowing for direct hardware implementation.
4.  **Demonstration that Surrogate Gradient Training can enable effective learning in the spiking domain**: Through carefully designed surrogate gradient functions and BPTT training strategies, WaveSense can effectively train deep SNNs in the spiking domain, proving the feasibility of surrogate gradient methods in practical speech tasks like KWS.
5.  **An Alternative to ANN-to-SNN Conversion**: Unlike the traditional method of training an ANN first and then converting it to an SNN, WaveSense is trained directly in the spiking domain, avoiding accuracy loss during the conversion process.

## Experimental Results

### Dataset and Setup
-   **Google Speech Commands v2**: 12-class subset ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", "unknown").
-   **Input Features**: 40-dimensional MFCC, rate-encoded to T=20-50 time steps.
-   **Evaluation Metrics**: Classification accuracy, theoretical energy consumption (estimated based on spike operations), inference latency (number of time steps).

### Classification Accuracy
-   **WaveSense**: Achieved approximately 93% accuracy on GSC v2 (12 classes).
-   **Comparison with Equivalent ANN**: An ANN with the same architecture (using ReLU instead of LIF) achieved approximately 95-96% accuracy. The SNN version is about 2-3% lower than the ANN version, a gap considered acceptable.
-   **Comparison with SOTA ANN Methods**: SOTA ANN models (such as BC-ResNet, KWT) achieve approximately 97-98% accuracy. WaveSense is about 4-5% lower. This is the cost of the SNN accuracy-energy trade-off.

### Theoretical Energy Consumption Analysis
-   **Spike Operations vs. MAC Operations**: The theoretical energy consumption of spike operations (conditional addition) in SNNs is approximately 1/30 of that of floating-point MAC operations in ANNs (based on theoretical energy models for SRAM access and computation).
-   **Sparsity**: The average spike firing rate of neurons in WaveSense is approximately 10-20% (i.e., each neuron does not generate spikes in 80-90% of time steps), further reducing the actual computational load.
-   **Total Energy Estimation**: The theoretical energy consumption of WaveSense is approximately 1/30 of that of an equivalent ANN (a 30-fold reduction), primarily due to the simplification of spike calculations (multiplication becomes conditional addition) and spike sparsity.

### Impact of Number of Time Steps
-   Increasing the number of inference time steps T can improve accuracy (more time steps provide more precise information encoding and decision accumulation), but it also linearly increases latency and energy consumption.
    -   T=20: Accuracy ~89%, Latency ~20ms
    -   T=30: Accuracy ~91%, Latency ~30ms
    -   T=50: Accuracy ~93%, Latency ~50ms
    -   T=100: Accuracy ~93.5%, but latency is too long

### Comparison with Other SNN Methods
-   Compared to methods based on ANN-to-SNN conversion, directly trained WaveSense achieves higher accuracy at the same number of time steps (approximately 93% vs. approximately 90%), as direct training can better optimize decision boundaries in the spiking domain.
-   Compared to purely fully-connected SNNs, spiking temporal convolutions achieve approximately 3-5% higher accuracy, validating the effectiveness of temporal convolutions in SNNs.

### Ablation Studies
-   **Leak Factor beta**: Performance is best when beta=0.8-0.9. Too low beta (e.g., 0.5) causes temporal information to be lost too quickly, while too high beta (e.g., 0.99) causes excessive information accumulation, making it difficult to distinguish.
-   **Surrogate Gradient Choice**: The Fast Sigmoid surrogate performed best in terms of accuracy and training stability.
-   **Residual Connections**: Removing residual connections resulted in a decrease in accuracy of approximately 2-3%, particularly for deeper networks.

## Limitations and Future Work

### Technical Limitations
-   **Accuracy Gap**: The accuracy of WaveSense (approximately 93%) still lags behind SOTA traditional neural network methods (approximately 97-98%), with a gap of about 4-5%. In commercial KWS systems with high accuracy requirements, this gap may be unacceptable. SNNs still underperform ANNs in fine-grained feature discrimination (e.g., distinguishing acoustically similar keyword pairs like "yes"/"yeah").
-   **Theoretical Energy Estimates Only**: The energy reduction provided in the paper is based on theoretical models (calculating spike operations * theoretical energy per operation) rather than measurements on actual hardware. Actual energy consumption depends on chip implementation details (such as clock frequency, I/O overhead, SRAM/DRAM access patterns, etc.), and theoretical estimates may differ significantly from actual measurements.
-   **Inference Latency**: SNNs require multiple time steps (T=20-50) to complete one inference, resulting in inference latency approximately T times that of an equivalent ANN. Although the computational load per time step is much smaller than that of an ANN, the total inference latency may be longer.
-   **Training Complexity**: Although surrogate gradient training solves the non-differentiability problem, the training process is more complex than standard ANN training (requiring BPTT unrolling along the time dimension) and is more sensitive to hyperparameters (learning rate, surrogate gradient function, leak factor, etc.).
-   **Information Loss in Spike Encoding**: Rate coding converts continuous-valued features into binary spike sequences, introducing information quantization loss. Increasing the number of time steps T can reduce this loss but increases latency.

### Experimental Design Shortcomings
-   Evaluation was conducted only on the Google Speech Commands dataset, without validation in more challenging scenarios such as noise, far-field, or custom keywords.
-   Lack of measured data on actual neuromorphic hardware (such as SynSense Speck, Intel Loihi) regarding actual energy consumption, latency, and accuracy, which makes the paper's "low power" claims lack empirical support.
-   More advanced spike encoding strategies (such as attention-based adaptive encoding, SNNs directly processing raw waveforms, etc.) were not explored.
-   Insufficient fine-grained analysis of different types of keywords (short words vs. long words, common words vs. rare words).

### Future Improvement Directions
-   Deploy and measure the actual energy consumption of WaveSense on real neuromorphic chips to verify the accuracy of theoretical estimates.
-   Explore hybrid ANN-SNN architectures: Use ANNs at the front end for feature extraction (more accurate) and SNNs at the back end for classification (more power-efficient), achieving a better balance between accuracy and energy consumption.
-   Study mixed-precision SNNs (using different numbers of time steps or spike precision for different layers) to concentrate computational resources on the most critical layers.
-   Combine ANN-to-SNN knowledge distillation: First train a high-precision ANN teacher model, then train an SNN student model using a distillation loss to narrow the accuracy gap between ANNs and SNNs.
-   Explore event-driven feature extraction: Use dynamic sensors (such as audio event detectors) to implement event-driven sparsification at the SNN input, further reducing power consumption.
-   Implications for the KWS field: The application of SNNs in the KWS field is still in its early stages, but it holds irreplaceable potential in extreme low-power "always-on" scenarios. With the maturation of neuromorphic hardware and improvements in SNN training methods, SNN-based KWS is expected to find important application scenarios in IoT and wearable devices. WaveSense provides a valuable baseline and architectural reference for research in this direction.
