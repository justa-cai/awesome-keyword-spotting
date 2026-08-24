# Extending SE Blocks with Temporal Feedback for End-to-End Raw Waveform Keyword Spotting

- **Authors/Affiliations**: Taejun Kim, Juhan Nam (KAIST, Korea Advanced Institute of Science and Technology)
- **Date**: November 2019 (arXiv)
- **Link**: https://arxiv.org/abs/1911.01803
- **Keywords**: Keyword Spotting, Squeeze-and-Excitation, Temporal Feedback, CRNN, Raw Waveform, End-to-End, Top-Down

## Problem Statement

Standard CNN-based keyword spotting models employ a **bottom-up** feature extraction flow, where information is passed progressively from lower layers to higher layers. This design has the following limitations:

1. **Lack of top-down information flow**: Lower-layer convolutions, when extracting local acoustic features, cannot utilize long-range temporal context information from higher network layers. For example, when higher layers have already identified partial keyword patterns, this high-level semantic information cannot be fed back to lower layers to guide more precise local feature extraction.
2. **Limitations of standard SE blocks**: Squeeze-and-Excitation (SE) blocks achieve channel-wise adaptive weighting through global average pooling, but they only utilize statistical information from the current frame, lacking awareness of temporal context.
3. **Trade-off between handcrafted features and end-to-end learning**: While methods based on MFCC/LFBE are computationally efficient, the feature extraction process cannot be optimized specifically for the KWS task. End-to-end raw waveform methods avoid information loss but face challenges related to high computational cost and difficulty in feature learning.

Therefore, the core challenge is to design an architecture that can integrate **temporal context feedback from the top layers** back into the lower-layer feature extraction, achieving a top-down information flow while maintaining end-to-end raw waveform processing capabilities.

## Methodology

This paper proposes the **Temporal Feedback SE Block**, which extends the standard SE block to introduce recurrent temporal information.

### 1. Review of Standard SE Block

The standard Squeeze-and-Excitation block consists of two operations:
- **Squeeze**: Compresses the spatial/temporal dimensions via global average pooling to obtain a channel-wise global descriptor.
- **Excitation**: Learns inter-channel dependencies through fully connected layers to generate channel weights.

$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(X)))$$
$$\tilde{X} = s \odot X$$

### 2. Temporal Feedback Extension

This paper introduces **recurrent temporal feedback** into the standard SE block:

#### 2.1 Recurrent Module

- A recurrent module (such as GRU or LSTM) is introduced during the excitation phase of the SE block.
- The recurrent module captures temporal dependencies from the **top-layer representations**.
- The high-level representation from the previous time step is injected as a feedback signal into the low-layer SE block of the current time step.

#### 2.2 Top-Down Information Flow

```
Top-layer features -> RNN temporal modeling -> Feedback signal -> Low-layer SE block -> Improved feature extraction
```

This top-down feedback mechanism enables:
- Lower-layer convolutions to utilize **temporal context from higher layers** when extracting local features.
- Keyword pattern information learned by higher layers to **guide** lower layers to focus on more relevant local features.
- The formation of a **bidirectional information flow**: bottom-up feature extraction + top-down context feedback.

### 3. End-to-End Raw Waveform Processing

The entire system operates end-to-end directly on raw waveforms:
- **Frontend**: A 1D convolutional layer extracts low-level acoustic features from the raw waveform.
- **Feature Learning**: Multiple convolutional layers equipped with temporal feedback SE blocks.
- **Backend**: Fully connected layers output keyword classes.
- **No handcrafted features**: The entire feature extraction process is learned automatically through end-to-end training.

### 4. CRNN Integrated Architecture

The overall architecture combines the advantages of CNN and RNN:
- **CNN part**: Convolutional layers extract local time-frequency patterns.
- **RNN part**: Recurrent modules provide temporal feedback within the SE blocks.
- **SE blocks**: Channel attention mechanisms enhance feature representation.

## Main Contributions

1. **Temporal Feedback SE Block**: Proposes for the first time the integration of a temporal feedback mechanism into SE blocks, enabling them to utilize temporal context information from top layers in addition to channel attention capabilities. This design elegantly achieves top-down information flow within the framework of SE blocks.

2. **Combination of Top-Down and Bottom-Up**: Demonstrates that combining top-down temporal feedback with bottom-up feature extraction improves the quality of lower-layer feature representations. After high-level semantic information is fed back to lower layers, the lower-layer features become more discriminative.

3. **End-to-End Raw Waveform KWS**: Implements a complete system for end-to-end training directly from raw waveform audio, without any handcrafted feature extraction steps.

4. **Effective Integration of CNN+RNN**: Integrates the CRNN architecture within the framework of SE blocks—CNN handles local feature extraction, while RNN provides temporal feedback within the SE blocks—allowing both mechanisms to work synergistically.

## Experimental Results

- On the **Google Speech Commands dataset**, the proposed model achieves higher keyword spotting accuracy compared to baseline CNN and standard SE models.
- The improvement of the temporal feedback SE block over the standard SE block demonstrates the effectiveness of temporal context feedback in enhancing lower-layer features.
- End-to-end raw waveform processing avoids the information bottleneck associated with handcrafted features such as MFCC.

## Limitations and Future Work

### Technical Limitations
- **Increased computational requirements**: Compared to MFCC-based methods, end-to-end raw waveform processing increases computational demands—the sampling rate of raw waveforms is much higher than the feature frame rate of MFCCs, requiring the first convolutional layer to process longer sequences.
- **Latency from recurrent feedback**: The temporal dependencies of the recurrent module increase inference latency—processing the current frame requires waiting for feedback signals from previous frames. This may pose problems for strict real-time applications, such as wake-word detection.
- **Noise robustness**: There is limited analysis of the model's robustness to noise and varying acoustic conditions. The performance of raw waveform end-to-end methods in noisy environments requires more systematic evaluation.

### Future Directions
- Explore the use of causal convolutions (e.g., dilated convolutions) instead of RNNs to provide temporal feedback, avoiding latency issues caused by recurrent dependencies.
- Investigate multi-scale temporal feedback—SE blocks at different layers receive feedback information from different temporal scales.
- Systematically evaluate the robustness benefits of temporal feedback SE blocks under noisy, reverberant, and far-field conditions.
- Explore combining the temporal feedback mechanism with other attention mechanisms, such as spatial attention or self-attention.
- Investigate the generalizability of temporal feedback SE blocks to other audio classification tasks, such as environmental sound classification and music tagging.
