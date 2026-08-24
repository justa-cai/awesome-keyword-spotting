# Efficient Keyword Spotting Using Dilated Convolutions and Gating

- **Authors/Affiliations**: Alice Coucke, Mohammed Chlieh, Thibault Gisselbrecht, David Leroy, Mathieu Poumeyrol, Thibaut Lavril (Snips, Paris, France)
- **Date**: November 2018 / February 2019 (ICASSP 2019)
- **Link**: https://arxiv.org/abs/1811.07684
- **Keywords**: Keyword Spotting, Dilated Convolutions, Gating Mechanism, WaveNet, Stateless Temporal Modeling, Wake-word Detection, End-to-End

## Problem Statement

Wake-word Detection is a critical KWS application in speech interaction systems, used to detect predefined wake words in continuous audio streams to initiate speech interaction. Practical deployment faces threefold challenges: (a) constrained device resources—continuous listening is required under conditions of low memory and low computational cost; (b) real-time requirements—low-latency response is needed; (c) high accuracy—low false rejection rate (FRR) and low false alarm rate (FA) must be maintained simultaneously.

Existing temporal modeling methods have key deficiencies:

1. **State Management Issues in RNN/LSTM**: Although LSTMs can model long-range temporal dependencies through gating mechanisms and internal states, they are prone to **state saturation** when facing continuous input streams, requiring periodic resetting of internal states. This state management not only increases implementation complexity but may also lose useful information during the reset process.
2. **Limited Receptive Field of Standard CNNs**: Traditional CNNs leverage local dependencies and perform well in terms of inference speed and computational cost, but they cannot capture sufficiently long temporal patterns with a reasonably sized model—the duration of wake words is typically 1-2 seconds, corresponding to hundreds of frames of acoustic features.
3. **Complexity of Multi-stage Pipelines**: Traditional HMM methods require precise phoneme alignment for keywords and background, making system design complex and inflexible.

Therefore, the core need is: to design a **stateless** temporal modeling method that can efficiently capture long-range dependencies without using internal recursive states, while maintaining model smallness and low computational cost.

## Methodology

This paper proposes an end-to-end stateless KWS system inspired by WaveNet, based on Dilated Convolutions and Gated Activations.

### 1. Acoustic Feature Extraction

The input features use **20-dimensional Log-Mel Filterbank Energies (LFBEs)**, with one frame extracted every 10ms. Compared to MFCCs, LFBE features retain more spectral details and have lower computational costs, making them suitable for real-time systems.

### 2. Core Network Architecture: Stacked Dilated Convolutions Inspired by WaveNet

#### 2.1 Dilated/Atrous Convolutions

Dilated convolutions are the core temporal modeling tool in this paper. Unlike standard convolutions, dilated convolutions introduce gaps (i.e., dilation rate) between convolution kernel elements, causing the receptive field to grow **exponentially** without increasing the number of parameters or computational cost.

Specifically, the dilation rate for layer $l$ is typically set to $2^{l-1}$ (where $l$ starts from 1). For example, a 4-layer dilated convolution with dilation rate sequence $\{1, 2, 4, 8\}$ has a total receptive field of $1 + 2 + 4 + 8 + 4 = 19$ positions (assuming a kernel size of 2), which is much larger than the receptive field of 5 for the same 4-layer standard convolution.

The complete model in this paper uses **24 layers of dilated convolutions**, with the dilation rate cycling through the pattern $\{1, 2, 4, 8\}$ (i.e., repeating 6 blocks), achieving a total receptive field of **182 frames**, corresponding to approximately **1.83 seconds** of audio duration, which is sufficient to cover the duration of typical wake words.

#### 2.2 Gated Activations

Drawing on the design of WaveNet, the activation function of each dilated convolution layer adopts a gating mechanism:

$$z = \tanh(W_{f,k} * x) \odot \sigma(W_{g,k} * x)$$

where $W_{f,k}$ and $W_{g,k}$ are the convolution kernels for the filter gate and the gating gate, respectively, $*$ denotes the dilated convolution operation, $\odot$ is element-wise multiplication, and $\sigma$ is the sigmoid function. The gating mechanism allows the network to **adaptively control the flow of information**—the sigmoid gate determines which time-step information should be retained or forgotten. This is crucial for distinguishing wake words from similar non-target speech.

#### 2.3 Residual Connections

Each gated activation layer is equipped with residual connections:

$$\text{output} = \text{Conv}_{1\times1}(z) + x$$

Residual connections allow gradients to propagate directly through deep networks, solving the **vanishing gradient problem** in stacked deep dilated convolutions, making it possible to train a 24-layer network.

#### 2.4 Skip Connections

The outputs of all layers are summed after passing through 1x1 convolutions and aggregated into the final output, ensuring that information from different receptive field scales is preserved.

### 3. Custom "End-of-Keyword" Labeling Scheme

This is another key innovation in this paper. Traditional frame-level labeling schemes label every frame within a keyword as positive, whereas the **"end-of-keyword" labeling scheme** in this paper only marks the target frame as positive within a specific time window ($\delta = 160$ms) after the keyword ends.

- **Advantages**: The loss function backpropagates only from these specific attention frames, meaning the model only needs to learn to detect the **end position** of the keyword, rather than precisely matching every frame of the entire keyword. This simplifies the learning task and improves detection robustness.
- **Training Labels**: Each audio segment is assigned a binary target label (positive/negative), and during training, the cross-entropy loss is calculated only on frames within the $\delta$ window after the keyword ends.

### 4. Architectural Comparison with Traditional Methods

| Feature | LSTM RNN | Standard CNN | This WaveNet-style |
|------|----------|---------|-------------------|
| Temporal Context | Recursive state, theoretically infinite | Limited by kernel size | Exponential growth via dilated convolutions |
| State Management | Requires periodic resetting | Stateless | Stateless |
| Parameter Efficiency | Medium | Low | High (due to dilation strategy) |
| Parallelization | Limited (sequential dependency) | Fully parallel | Fully parallel |

## Main Contributions

1. **First Application of WaveNet-style Architecture to KWS**: Introduces the combination of dilated convolutions, gated activations, and residual connections to the field of keyword detection, achieving efficient stateless long-range temporal modeling in streaming scenarios, eliminating the state management issues of RNNs.

2. **Custom "End-of-Keyword" Labeling Scheme**: Proposes backpropagating loss only from a specific time window after the keyword ends, simplifying the detection task—the model only needs to detect the end position of the keyword rather than precise frame alignment for the entire keyword.

3. **Publication of the "Hey Snips" Public Dataset**: Publicly released the "Hey Snips" wake-word corpus containing **over 2,200 different speakers**, establishing an open benchmark for wake-word detection. The training set consists of approximately 11K positive samples and 86.5K negative samples, with a test set serving as an independent evaluation set. At the time, this was one of the few publicly available wake-word detection datasets.

4. **Significant Performance Improvement**: Compared to the LSTM RNN baseline trained with max-pooling loss, significantly reduced the false rejection rate on the "Hey Snips" wake-word detection task, proving the superiority of the dilated convolution + gating architecture.

5. **Published at ICASSP 2019**, representing a significant contribution by Snips to end-to-end KWS.

## Experimental Results

### Dataset: Hey Snips
- **Training Set**: Approximately 11,000 positive samples (containing wake words) and 86,500 negative samples (without wake words)
- **Test Set (Development Set)**: Contains clean wake-word samples and noisy samples
- **Number of Speakers**: Over 2,200 different speakers

### Main Performance Comparison

| Model | Clean Environment FRR | Noisy Environment FRR |
|------|------------|------------|
| WaveNet (This Paper) | **0.12%** | **1.60%** |
| LSTM Baseline | 2.09% | 11.21% |

Under the same false alarm rate (FAH = 0) conditions, the WaveNet model had a false rejection rate of only 0.12% in the clean environment, while the LSTM baseline was 2.09%. The gap was even more significant in noisy environments: 1.60% for WaveNet versus 11.21% for LSTM.

### Ablation Studies

| Model Variant | Clean FRR | Noisy FRR |
|----------|---------|---------|
| Full Model (with gating) | 0.12% | 1.60% |
| Without Gating (tanh activation only) | 0.15% | 4.17% |

The contribution of the gating mechanism is particularly significant in noisy environments—removing the gating caused the FRR in noisy environments to rise sharply from 1.60% to 4.17%, an increase of approximately 2.57 percentage points. This proves that the gating mechanism is crucial for suppressing irrelevant signals in noisy environments.

### Training Details
- **Loss Function**: Binary cross-entropy, calculated only within the $\delta=160$ms window after the keyword ends
- **Optimizer**: Adam
- **Positive Sample Weight**: Higher weights were assigned to positive samples during training to handle class imbalance

## Limitations and Future Work

### Technical Limitations
- **Intermediate Activation Storage Overhead**: Compared to RNNs, dilated convolutions require storing intermediate layer activations during inference (especially since skip connections need to save outputs from all layers), which increases memory usage during inference. On extremely constrained embedded devices, this may become a deployment bottleneck.
- **Dilation Strategy Requires Tuning**: The optimal dilation rate sequence and repetition pattern need to be adjusted according to the length of different keywords. Shorter keywords may not require a receptive field of 182 frames, and an excessively large receptive field may lead to unnecessary computational waste.
- **Causality Constraint**: To ensure real-time processing in streaming scenarios, dilated convolutions must use a causal (causal) mode, meaning they can only utilize information from the current frame and past frames, which may limit the model's detection accuracy.

### Experimental Design Limitations
- Evaluation focused solely on a single wake word ("Hey Snips"), without demonstrating generalization capabilities for multi-class KWS (such as the 35 classes in Google Speech Commands).
- Lack of direct comparison with other contemporary stateless methods (such as TCN, Quasi-RNN).
- Noise robustness evaluation only used limited noise types, without systematic evaluation under various real-world environmental noises.

### Future Directions
- Extend to scenarios involving simultaneous detection of multiple keywords, exploring the applicability of dilated convolution architectures in multi-task KWS.
- Research adaptive dilation rate mechanisms to dynamically adjust the receptive field size based on input content.
- Combine with attention mechanisms to further enhance the modeling of key time steps while maintaining the advantages of being stateless.
- Explore model compression techniques (such as knowledge distillation, quantization) to further reduce model size and adapt to lower-end hardware platforms.
- Leverage the public availability of the "Hey Snips" dataset to promote method comparisons within the community on standardized wake-word detection benchmarks.
