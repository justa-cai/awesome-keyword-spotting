# End-to-end Models with Auditory Attention in Multi-channel Keyword Spotting

- **Authors/Affiliations**: Haitong Zhang, Junbo Zhang, Yujun Wang (Xiaomi AI Lab)
- **Date**: November 2018 (arXiv:1811.00350)
- **Link**: https://arxiv.org/abs/1811.00350
- **Keywords**: multi-channel keyword spotting, auditory attention, transfer learning, spectral mapping, end-to-end, Xiaomi

## Problem Statement

Smart speakers and IoT devices are typically equipped with multiple microphones to enable far-field voice interaction. A multi-channel keyword spotting system needs to effectively fuse information from multiple microphone channels to improve detection robustness in far-field noisy environments.

**Limitations of Traditional Approaches**

Traditional multi-channel speech processing adopts a cascaded architecture: signal processing techniques are first applied for channel fusion and noise suppression (e.g., beamforming and acoustic echo cancellation, AEC), and the processed single-channel signal is then fed into the keyword spotting model. This cascaded architecture suffers from fundamental flaws:

- The objective functions of the signal preprocessing modules (beamforming, AEC), such as SNR maximization and echo cancellation, are misaligned with the final keyword spotting objective (minimizing false alarms and missed detections), resulting in suboptimal overall performance
- Beamforming requires estimating the sound source location (DOA), which is inaccurate in multi-speaker, high-noise scenarios
- Each module in the cascaded system is optimized independently and cannot be jointly tuned in an end-to-end manner
- The extra reference signal channel (e.g., the AEC echo reference) increases system complexity

**Key Challenges This Paper Aims to Solve**

- How to replace the traditional fixed signal processing pipeline with a learnable attention mechanism
- How to maintain reliable detection performance under extreme noise conditions (SNR as low as -20dB)
- How to effectively fuse multi-channel information within an end-to-end framework

## Methodology

### Overall Architecture Design

The paper proposes an end-to-end multi-channel keyword spotting framework based on auditory attention, which consists of three core modules:

**Module 1: Multi-channel Auditory Attention**

This is the core innovation that replaces traditional beamforming. For a device equipped with 6 microphones:
- The acoustic features (40-dimensional log-mel + PCEN) of the 6 channels are fed in separately
- The attention mechanism learns a dynamic weight for each channel: $\alpha_i = \text{softmax}(f(h_i))$, where $h_i$ is the feature representation of the $i$-th channel
- A weighted sum produces the fused single-channel feature: $h_{fused} = \sum_{i=1}^{6} \alpha_i \cdot h_i$
- The attention weights change dynamically over time, adaptively selecting the most reliable channel according to the current acoustic environment

**Module 2: GRU Encoder and Sequence-to-Sequence Training**

- A GRU (Gated Recurrent Unit) is used as the encoder to encode the fused feature sequence into a high-level semantic representation
- The output layer predicts the keyword presence probability frame by frame
- Training uses a frame-level binary cross-entropy loss (with labels generated from TDNN-LSTM alignments)

**Module 3: Posterior Smoothing Decoding**

- A moving average is applied to the keyword probabilities of consecutive frames
- A threshold is used to decide whether to trigger keyword detection

### Enhancement Techniques

**1. Spectral Mapping Multi-task Learning**

In addition to the primary keyword spotting task, an auxiliary task is added: predicting the spectrum of clean speech. The loss function of multi-task learning is:

$$L = \alpha \cdot L_{KWS} + \beta \cdot L_{spectral}$$

where $L_{KWS}$ is the keyword spotting loss and $L_{spectral}$ is the spectral mapping loss (the L1 or L2 distance between the predicted spectrum and the clean spectrum). Spectral mapping acts as a denoising autoencoder, helping the attention mechanism learn noise-invariant feature representations.

**2. Transfer Learning**

The model is first pre-trained on clean speech data and then fine-tuned on noisy data. This strategy addresses the scarcity of noisy data, while also letting the model first acquire basic acoustic pattern recognition capability.

**3. Multi-target Spectral Mapping**

On top of transfer learning, the fine-tuning stage simultaneously uses multiple spectral mapping targets (e.g., spectrograms at different time resolutions), further improving noise robustness.

### Technical Differences and Comparison

| Aspect | Traditional Methods | This Method |
|------|---------|--------|
| Channel fusion | Fixed beamforming | Learnable attention |
| Echo cancellation | AEC signal processing | Learned implicitly |
| Optimization | Independent per-module optimization | End-to-end joint optimization |
| Reference signal | Requires an AEC reference channel | Not required (6 channels only) |
| Noise adaptation | Relies on signal processing | Data-driven transfer learning |

## Main Contributions

1. **Attention replacing beamforming**: For the first time in multi-channel keyword spotting, the traditional fixed beamforming and AEC preprocessing are replaced with a learnable attention mechanism. The attention weights can adapt dynamically to the current acoustic environment, achieving absolute improvements of 4% and 7% in non-echo and echo scenarios, respectively.

2. **Breakthrough under extreme noise conditions**: Transfer learning + multi-target spectral mapping (Tran Multi Map) achieves a 30% absolute performance improvement under extreme noise conditions with an SNR of about -20dB, proving that data-driven noise adaptation strategies are far superior to traditional signal processing methods.

3. **Fewer input requirements**: The attention model uses only 6 microphone channels, whereas the traditional baseline uses 7 channels (with an additional AEC reference signal required), which simplifies hardware design.

4. **Advantages of end-to-end optimization**: All parameters (attention weights, encoder, classifier) are jointly optimized under a unified objective function, avoiding the suboptimality problem of cascaded systems.

## Experimental Results

### Datasets and Evaluation Setup

- Device: Xiaomi AI speaker (6-microphone array)
- Target keyword: the Xiaomi wake word
- Test conditions: clean, echo (device playing audio), noise (SNR from -20dB to 0dB)
- Baseline: traditional signal processing (beamforming + AEC) + single-channel KWS model
- Evaluation metric: FRR @ 0.5 FA/hour

### Core Results

**Non-echo and Echo Scenarios**

- The attention model outperforms the baseline by 4% (absolute) in non-echo scenarios and by 7% in echo scenarios
- The attention mechanism alone (without additional enhancement techniques) already significantly outperforms the signal processing baseline

**Noise Scenarios (SNR of about -20dB)**

- Attention mechanism alone: 40-60% improvement over the baseline (relative)
- Tran Multi Map (transfer + multi-target mapping): 30% improvement on hard noise data, 10% on easy noise data
- Transfer learning plays a key role in noise adaptation

**Conditional Dependence of Multi-task Learning**

- When training and testing conditions match, spectral mapping multi-task learning provides slight improvements
- When the conditions mismatch (trained on clean data, tested on noisy data), transfer learning is necessary
- Transfer learning + single-target mapping fails to outperform the attention-only method under noise conditions, indicating that excessive auxiliary tasks may interfere with the primary task

### Attention Weight Visualization Analysis

- The attention weights can adjust dynamically according to the direction of the noise source, lowering the weights of the channels severely affected by noise
- In echo scenarios, the attention tends to select the microphone channel closest to the user

## Limitations and Future Work

### Technical Limitations of the Method

- **Sensitivity of multi-task learning weights**: The weighting hyperparameters (alpha, beta, theta, delta) in the loss function need to be carefully tuned; inappropriate weights may cause the auxiliary task to dominate training or the auxiliary signal to be ignored
- **Conditional dependence of transfer learning**: When the training and testing noise conditions mismatch, the combination of transfer learning and spectral mapping yields unstable results, and more robust transfer strategies require further study
- **Expressiveness of the attention mechanism**: Soft attention is essentially a weighted average of channel features and cannot achieve the signal cancellation effect in beamforming (e.g., eliminating interference sources through phase inversion)

### Shortcomings of the Experimental Design

- No comparison with deep-learning-based beamforming methods (e.g., neural network adaptive beamforming)
- The interpretability analysis of the attention weights is not sufficiently in-depth
- The effect of different microphone counts (2, 4, and 8) on the attention mechanism was not evaluated

### Future Improvement Directions

- Explore more sophisticated channel fusion mechanisms (e.g., cross-attention-based channel interaction)
- Introduce phase information to assist attention learning, compensating for the limitation of magnitude-only attention
- Study self-supervised noise adaptation methods to reduce the dependence on predefined noise types
- Extend to distributed microphone array scenarios

### Implications for the KWS Field

- End-to-end learning can replace multiple hand-designed modules in the traditional signal processing pipeline
- Data-driven noise adaptation (transfer learning) is far superior to fixed algorithms under extreme conditions
- Multi-task learning is an effective means of improving noise robustness, but the relationships among the tasks must be carefully balanced
- Xiaomi's production experience shows that the attention mechanism has practical deployment value in multi-channel KWS
