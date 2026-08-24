# End-to-End Streaming Keyword Spotting

- **Authors/Affiliations**: Raziel Alvarez, Hyun-Jin Park (Google)
- **Date**: December 2018 / February 2019 (ICASSP 2019)
- **Link**: https://arxiv.org/abs/1812.02802
- **Keywords**: Keyword Spotting, End-to-End, Streaming, Memorized Neural Network, Memory, DNN, Temporal Context

## Problem Statement

Keyword Spotting (KWS) is a core component of speech interaction systems, widely used in wake-word detection scenarios for smart assistants. Traditional KWS systems typically adopt a multi-stage pipeline architecture, consisting of independent feature extraction (e.g., MFCC or LFBE), acoustic modeling (e.g., HMM or DNN), and decoding/decision stages. This modular design presents several core pain points:

1. **High System Complexity**: Multiple independent modules require separate optimization, and the interface design between modules increases engineering complexity. Hyperparameters for feature extraction, acoustic modeling, and decoding stages need to be tuned collaboratively, making overall optimization difficult.
2. **Information Loss**: Traditional methods pass acoustic features (such as MFCC) as fixed intermediate representations to subsequent modules. This hardcoded feature representation cannot be optimized end-to-end for the downstream KWS task, potentially leading to suboptimal performance.
3. **Difficulty in Managing Temporal Context for Streaming Inference**: In streaming scenarios, audio signals are input frame by frame, and the system must make real-time detection decisions under low-latency conditions. Although Recurrent Neural Networks (RNN/LSTM) can model temporal dependencies, their internal state management (state reset, vanishing gradients, etc.) suffers from state saturation issues in continuous input streams, and it is difficult to precisely control inference latency.
4. **Deployment Constraints**: Practical KWS systems need to meet strict resource constraints—low latency (typically requiring <100ms), low memory footprint, and low computational cost—to satisfy the continuous operation requirements on edge devices (such as smart speakers and mobile phones).

Therefore, the core challenge is to design an end-to-end streaming KWS system that is entirely contained within a single DNN model, which must simultaneously satisfy: (a) efficient management of temporal context for streaming inference; (b) direct output of keyword detection scores in a single forward pass; and (c) maintaining a sufficiently small model size and computational cost to adapt to resource-constrained devices.

## Methodology

This paper proposes a fully end-to-end streaming KWS system, the core innovation of which is the **Memorized Neural Network** topology.

### 1. System Architecture Overview

The entire KWS system (excluding the frontend components used for feature generation) is contained within an end-to-end trained DNN model. The system consists of the following key components:
- **Frontend Feature Extraction**: Converts raw audio signals into time-frequency representations (such as Log Mel Filterbank Energies, LFBE). This component is processed separately outside of end-to-end training.
- **Memorized DNN Backbone**: Receives frontend features, processes them through multiple DNN layers, utilizes the memorized topology to manage temporal context, and finally directly outputs keyword detection scores.

### 2. Memorized Neural Network Topology

This is the most core technical innovation of this paper. Traditional RNN methods compress temporal context into a fixed-dimension hidden state vector, where information compression loss increases as the sequence length grows. In contrast, the memorized topology proposed in this paper adopts a different strategy:

- **Distributed Memory Storage**: Unlike RNNs that concentrate memory in a single hidden state, the Memorized DNN distributes previously activated memories across multiple depth layers of the network. Each layer maintains its own memory buffer, storing the activation values of that layer at previous time steps.
- **Memory Retrieval and Fusion**: The input at the current time step participates in computation together with historical activation values from the memory buffers of all layers. This design allows the network to utilize historical information at different time scales across different levels of abstraction—shallow layers may focus on short-term acoustic detail changes, while deeper layers capture semantic patterns over longer time spans.
- **Efficient Memory Management**: Memory buffers are implemented using fixed-length FIFO (First-In-First-Out) queues. For each new input frame, the activation value of the corresponding layer is stored in the queue, and the oldest activation value is discarded. This design ensures constant memory occupancy and deterministic inference latency.

### 3. End-to-End Training Strategy

- **Unified Loss Function**: The entire DNN (including the memorized components) is optimized end-to-end directly for the keyword detection objective function, eliminating the need for staged training of the acoustic model and decoder.
- **Direct Output of Detection Scores**: The output layer of the DNN directly generates the detection probability of keywords (rather than traditional frame-level acoustic features or phoneme posterior probabilities), simplifying the mapping from acoustic features to detection decisions.
- **Streaming-Compatible Training Paradigm**: The training process simulates streaming inference behavior, i.e., inputting frame by frame and utilizing historical information from memory buffers, ensuring consistency between training and inference.

### 4. Key Differences from Traditional Methods

| Aspect | Traditional RNN/LSTM Methods | Memorized DNN Method |
|------|-----------------|-------------|
| Temporal Context Storage | Centralized hidden state | Distributed multi-layer memory |
| Information Capacity | Limited by hidden state dimension | Limited by total capacity of memory buffers across layers |
| Inference Latency | Dependent on sequence processing order | Deterministic, determined by memory window size |
| State Management | Requires periodic resetting | FIFO queue manages automatically |

## Main Contributions

1. **Memorized Neural Network Topology**: Proposes for the first time an architectural design that distributes the memory of temporal context across the depth of the DNN, breaking the limitation of RNNs compressing memory into a single hidden state. This distributed memory mechanism enables the network to utilize historical information at different time scales across different levels of abstraction, making better use of parameters and related computational resources.

2. **Fully End-to-End Streaming KWS**: Achieves fully end-to-end training from acoustic features to detection decisions, eliminating the complexity of traditional multi-stage pipelines. The DNN directly generates keyword detection scores through end-to-end training, without requiring independent decoders or post-processing modules.

3. **Superior Performance-Efficiency Trade-off**: Significantly outperforms previous methods in terms of detection quality, model size, and computational cost. The memorized architecture has higher parameter efficiency—requiring fewer parameters and computational costs for the same performance, which is crucial for resource-constrained edge deployment.

4. **Streaming Deployment Friendly**: The FIFO memory management mechanism ensures constant memory occupancy and deterministic inference latency, making it highly suitable for continuous streaming listening scenarios in actual products.

5. **Published at ICASSP 2019**, representing an important exploration by Google in end-to-end KWS systems.

## Experimental Results

This paper was evaluated on Google's internal KWS benchmark, with main results including:

- **Detection Quality**: The Memorized DNN system significantly outperforms CNN and LSTM baseline models of similar scale in detection accuracy (measured by DET curves). Under the same False Alarms per Hour (FAH) conditions, it achieves a lower False Rejection Rate (FRR).
- **Model Size**: The memorized topology achieves a smaller model size than traditional methods for the same detection performance through more efficient parameter utilization.
- **Computational Efficiency**: The computational cost per frame of inference (measured in multiply-accumulate operations) is significantly lower than that of comparative methods, which is particularly important for battery-powered mobile devices.
- **Ablation Studies**: Verified the performance advantage of distributed memory (distributed across multiple layers) compared to centralized memory (in only one layer), proving the effectiveness of the multi-layer memory design.

## Limitations and Future Work

### Technical Limitations
- **Memory Buffer Management Overhead**: Although the FIFO queue design ensures constant memory occupancy, the total storage requirement of multi-layer memory buffers remains considerable, especially when long temporal contexts are needed. Maintaining independent memory queues for each layer increases the complexity of memory access.
- **Frontend Features Not End-to-End**: The feature extraction frontend (such as LFBE calculation) is not included in end-to-end training, meaning the feature representation remains hand-designed, which may limit further improvements in system performance.
- **Selection of Memory Window Size**: The length of the memory buffer is a hyperparameter that needs to be set manually. The optimal window size may vary for different keywords, and there is a lack of adaptive mechanisms.

### Experimental Design Limitations
- This paper was primarily evaluated on Google's internal datasets, lacking standardized comparisons on public benchmarks (such as the Google Speech Commands Dataset), which limits the reproducibility of results and direct comparison with community methods.
- The experiments focused mainly on single-keyword detection scenarios, and the capability for simultaneous multi-keyword detection was not fully explored.

### Future Directions
- Incorporate the feature extraction frontend into the end-to-end training framework to achieve a truly end-to-end system from raw audio to detection decisions.
- Explore adaptive memory window mechanisms to dynamically adjust the memory range based on input content.
- Extend to multi-keyword scenarios and study the applicability of the memorized architecture in multi-task KWS.
- Combine the memorized idea with other architectures (such as attention mechanisms, Transformers) to further enhance long-range temporal modeling capabilities.
- Study the compatibility of quantization-aware training with the memorized architecture to further compress the model for adaptation to lower-end hardware platforms.
