# Streaming Keyword Spotting on Mobile Devices

**Authors/Affiliations**: Raziel Alvarez, Rohit Prabhavalkar, Arun Narayanan, Michiel Bacchiani, Naveen Gaur, Ryichiro Higa, Yanzhang He, Michael Riley, Shankar Kumar, Ian McGraw (Google Research)

**Date**: May 2020 (arXiv:2005.06720)

**Link**: https://arxiv.org/abs/2005.06720

**Keywords**: Keyword Spotting, Streaming Inference, Mobile Deployment, End-to-End, CRNN

## Problem Statement

Deploying KWS systems on mobile devices requires meeting strict real-time constraints:
- **Streaming Processing**: Audio arrives continuously in frames (typically one frame every 10ms), and the system must process them instantaneously.
- **Low Latency**: The latency from the utterance of a keyword to the detection trigger should be minimized (typically requiring <500ms).
- **Small Memory Footprint**: The model must fit within the memory constraints of mobile devices.
- **Low Power Consumption**: The power consumption of continuous listening modes must not significantly impact battery life.

Non-streaming KWS models typically perform global operations over the entire audio window (e.g., bidirectional RNNs, future-context convolutions) and cannot be directly applied to streaming scenarios. Adapting end-to-end KWS models to streaming mode faces the following technical challenges:
- **Causality Constraints**: Only current and past audio frames can be used; future frames are inaccessible.
- **State Management**: Internal states must be maintained and updated between frames.
- **Accuracy Loss**: Unidirectional (causal) models typically exhibit lower accuracy than bidirectional models.

## Methodology

### Convolutional Recurrent Neural Network (CRNN) Architecture

Combines the feature extraction capabilities of CNNs with the sequence modeling capabilities of RNNs:

**Convolutional Layers**:
- Uses causal convolutions: only current and past inputs are used, avoiding future context.
- Convolutions expand unidirectionally along the time axis, ensuring compatibility with streaming processing.
- Multi-layer convolutions progressively extract high-level features while reducing temporal resolution.

**Recurrent Layers**:
- Uses unidirectional LSTMs (Long Short-Term Memory).
- LSTMs are naturally suited for streaming processing: they update hidden states frame by frame.
- Hidden states are passed between frames, capturing long-term temporal dependencies.
- Stacking multiple LSTM layers increases modeling capacity.

**Fully Connected Output Layer**:
- Maps LSTM outputs to keyword class probabilities.
- Uses Softmax to output probabilities for each keyword and the "non-keyword" class.

### Streaming Processing Mechanism

**Frame-Level Processing Flow**:
1. Upon receiving a new audio frame, compute its spectral features (e.g., 40-dimensional Log-Mel).
2. Feed the feature frame into the causal convolutional layer to update the convolutional state.
3. Feed the convolutional output into the LSTM layer to update the LSTM hidden state.
4. Pass the LSTM output through the fully connected layer to obtain the keyword probability for the current frame.
5. Smooth probabilities across consecutive frames and apply thresholding to make the final detection decision.

**State Management**:
- **Convolutional Layers**: Maintain a time buffer to store historical frames covered by the convolutional kernel.
- **LSTM Layers**: Maintain the hidden state and cell state.
- States are persisted across consecutive frames, eliminating the need for recalculation.

### Model Quantization

To run efficiently on mobile devices:
- **Post-Training Quantization**: Converts floating-point weights and activations to 8-bit integers.
- **Quantization-Aware Training**: Simulates quantization effects during training to minimize accuracy loss due to quantization.
- **Typical Results**: Model size reduced by approximately 75%, with negligible accuracy loss.

## Main Contributions

1. **Streaming End-to-End KWS System**: Proposes a complete streaming end-to-end KWS system suitable for mobile devices, featuring a full pipeline design from audio frame input to detection decision output. This summarizes Google's engineering experience in mobile KWS systems.
2. **Causal CRNN Architecture**: Designs a causal convolution + unidirectional LSTM architecture that maintains strong sequence modeling capabilities while satisfying streaming causality constraints.
3. **Practical Deployment Considerations**: Discusses key technical issues in practical deployment, such as quantization and state management, providing a comprehensive technical reference for industrial implementation.
4. **End-to-End Optimization**: Directly optimizes for the keyword detection objective, avoiding the loss-metric mismatch issues present in traditional DNN-HMM methods.

## Experimental Results

### Experimental Setup
- Large-scale speech dataset.
- Evaluation: Detection rate and false alarm rate of streaming KWS.
- Latency and power consumption measurements on mobile devices.

### Key Results
- **Detection Accuracy**: The streaming CRNN achieves competitive detection accuracy under low-latency constraints.
- **Latency**: End-to-end latency meets the real-time requirements of mobile devices.
- **Quantization Effects**: 8-bit quantization incurs almost no accuracy loss, reducing model size by approximately 75%.
- **Efficiency**: The CRNN architecture achieves a good balance between accuracy and computational cost.
- **Real-Time Processing**: Achieves real-time processing of continuous audio streams on mobile hardware.

### Comparison with Non-Streaming Methods
- Streaming models exhibit slightly lower accuracy than non-streaming models (due to the inability to use future context).
- However, latency is significantly reduced, meeting practical deployment requirements.
- With careful design, the accuracy gap between streaming and non-streaming models can be kept within a small range.

## Limitations and Future Work

### Methodological Limitations
- **Streaming Constraints Limit Model Capacity**: Causality constraints limit the model's expressive power (e.g., inability to use bidirectional RNNs).
- **Latency-Accuracy Trade-off**: Careful tuning is required between response latency and detection accuracy.
- **Platform Specificity**: Some design decisions are specific to Google's mobile deployment infrastructure.
- **Gap with Non-Streaming SOTA**: There is a certain accuracy gap compared to the latest non-streaming methods.

### Future Directions
- Research adaptive context windows to dynamically use limited future context under low-latency constraints.
- Explore streaming Transformer architectures to leverage the parallel computing advantages of attention mechanisms.
- Combine the advantages of streaming and non-streaming models (e.g., using a streaming model for fast initial screening, followed by a non-streaming model for precise confirmation).
- Research device personalization by fine-tuning KWS models on-device.
- Explore more aggressive model compression methods (e.g., structured pruning, knowledge distillation).
