# Sequence-to-Sequence Models for Small-Footprint Keyword Spotting

- **Authors/Affiliations**: Haitong Zhang, Junbo Zhang, Yujun Wang (Xiaomi AI Lab)
- **Date**: November 2018 (arXiv:1811.00348)
- **Link**: https://arxiv.org/abs/1811.00348
- **Keywords**: sequence-to-sequence, keyword spotting, GRU, LSTM, PCEN, frame-level alignment, Xiaomi

## Problem Statement

Although attention-based end-to-end keyword spotting models have made significant progress in accuracy, several critical issues have surfaced in actual deployment:

**Training-decoding mismatch**
Attention models typically adopt a "sequence-to-one" training scheme: a fixed-length audio window (e.g., 189 frames) is mapped to a single keyword/non-keyword label. In actual streaming decoding, however, the system must run a sliding window over a continuous audio stream, and there is no clear theoretical guidance for choosing the window size and stride. This architectural inconsistency between training and decoding leads to performance degradation.

**Limitations of the sliding window**
Attention-based methods require the sliding window size to be set in advance: a window that is too large increases latency, while one that is too small may truncate the keyword. For keywords of different lengths, the optimal window size also differs, which complicates system tuning.

**Lack of frame-level labels**
Purely end-to-end methods lack frame-level supervision signals, so the attention mechanism has to learn "where to attend" on its own. This increases training difficulty and may cause the attention to focus on suboptimal positions.

The core problem this paper aims to solve is: how to design a framework in which training and decoding are consistent, eliminating the sliding-window dependency, while keeping a small model size and low computational complexity.

## Methodology

### Overall Architecture Design

The paper proposes a sequence-to-sequence (Seq2Seq) keyword spotting framework whose core idea is: map each audio frame independently to a keyword/non-keyword frame-level label, so that training and decoding use a completely consistent frame-level inference pipeline.

**Stage 1: Frame-level label generation**
This is the key premise of the entire method. The paper leverages a TDNN-LSTM acoustic model pre-trained on large-scale data (about 3000 hours) to generate frame-level alignment labels. The specific pipeline:
1. The TDNN-LSTM model outputs the posterior probability of each phone at each frame
2. Forced alignment determines the precise time boundaries of the keyword's phone sequence in the audio
3. Frame-level labels are generated according to the time boundaries:
   - Frames containing the complete keyword are labeled 1 (keyword frames)
   - Frames not containing the keyword are labeled 0 (non-keyword frames)
   - Frames with ambiguous keyword boundaries are labeled -1 (ignored frames) and excluded from the loss computation during training

**Stage 2: Seq2Seq model training**
- **Encoder**: RNN layers (LSTM or GRU), taking PCEN features as input
- **Output layer**: a fully-connected layer + sigmoid activation, outputting the keyword presence probability frame by frame
- **Loss function**: binary cross-entropy loss, masked on frames labeled -1

**Stage 3: Streaming decoding**
- The model outputs probabilities on a per-frame basis, natively supporting streaming processing
- Posterior smoothing: a moving average is taken over the probabilities of n consecutive frames to smooth the detection results and reduce false alarms caused by frame-level noise

### Core Technical Details

**PCEN features**
Per-Channel Energy Normalization replaces conventional log-mel features, providing adaptive spectral normalization:
$$\text{PCEN}(t, f) = \left(\frac{x(t, f)}{(\epsilon + S(t, f))^{\alpha}}\right)^{\delta} + \beta$$
The learnable parameters of PCEN allow the features to adapt to different recording conditions and noise environments.

**Model configuration exploration**
The paper systematically explores a variety of RNN configurations:
- **Depth vs width**: multi-layer narrow networks (e.g., 3 layers of 64 units) are compared with single-layer wide networks (e.g., 1 layer of 128 units)
- **LSTM vs GRU**: LSTM has a more complex gating mechanism (input gate, forget gate, output gate), while GRU is simpler (update gate, reset gate)

### Comparison with Attention-Based Methods
| Property | Attention model | Seq2Seq model |
|------|----------|------------|
| Training scheme | Sequence-to-one | Sequence-to-sequence |
| Decoding scheme | Sliding window | Frame-by-frame inference |
| Training-decoding consistency | Inconsistent | Fully consistent |
| Latency | Dependent on window size | Fixed (single frame) |
| Requires frame-level labels | No | Yes |

## Experimental Results

### Dataset
- Target keyword: "Xiao Ai Tongxue" (the wake word of Xiaomi's smart assistant)
- Training data: Xiaomi's internal large-scale Mandarin speech dataset
- Source of frame-level labels: a TDNN-LSTM acoustic model trained on about 3000 hours of data

### Core Performance Comparison

| Model | FRR (%) @ 0.1 FA/hr | Parameters |
|------|---------------------|--------|
| Seq2Seq GRU (1 layer, 128 units) | 3.05 | 73.3K |
| Seq2Seq LSTM (1 layer, 128 units) | 6.08 | 86.8K |
| Baseline GRU (attention) | 4.47 | 77.5K |
| Baseline LSTM (attention) | 11.86 | 103K |

### Key Findings
- **GRU outperforms LSTM**: under the Seq2Seq framework, GRU outperforms LSTM in all configurations, suggesting that for a relatively simple temporal modeling task such as keyword spotting, GRU's simpler gating mechanism is already sufficient
- **Width advantage**: the single-layer 128-unit network performs best across all configurations, indicating that keyword spotting needs sufficient feature representation capacity rather than deep abstraction
- **Posterior smoothing**: the moving average of frame-level probabilities effectively reduces false alarms, and the smoothing window size has little impact on performance

## Main Contributions

1. **Eliminating the training-decoding mismatch**: by training with frame-level labels and decoding with frame-by-frame inference, training and decoding become fully consistent, removing both the performance loss and the tuning burden caused by the sliding window in attention models.

2. **Significant accuracy improvement**: the Seq2Seq GRU model achieves an FRR of 3.05% at 0.1 FA/hour, a relative improvement of about 20% over the baseline attention GRU model (4.47% FRR), with only about 73K parameters.

3. **Width beats depth in architecture**: it is found that for keyword spotting, a wider single-layer network (128 units) significantly outperforms deeper narrow networks (multiple layers of 64 units), providing clear guidance for the architecture design of KWS models.

4. **Effectiveness of posterior smoothing**: a simple moving average of frame-level probabilities substantially improves the robustness of streaming decoding, a zero-cost inference optimization.

## Limitations and Future Work

### Technical Limitations of the Method
- **Dependence on frame-level labels**: generating frame-level labels requires training a separate TDNN-LSTM acoustic model on large-scale data (about 3000 hours), which increases the complexity and cost of building the system. For scenarios without a large-scale pre-trained model, the method is not directly applicable.
- **Label noise**: the quality of frame-level labels depends on the alignment accuracy of the TDNN-LSTM; alignment errors introduce label noise, especially near keyword boundaries. The handling of ambiguous frames (labeled -1) mitigates this problem but does not fully resolve it.
- **Single-keyword testing**: validation was performed only on one Mandarin keyword, "Xiao Ai Tongxue", without exploring multi-keyword or cross-lingual scenarios.

### Shortcomings in Experimental Design
- No comparison with more baseline methods (e.g., Deep KWS, CNN-based methods)
- No analysis of performance differences across speaker groups
- No evaluation of robustness in real noisy environments

### Future Directions
- Explore self-supervised or weakly supervised frame-level label generation methods to reduce the dependence on large-scale pre-trained models
- Extend the Seq2Seq framework to multi-keyword spotting scenarios
- Use a CTC loss instead of binary cross-entropy, which may better handle the ambiguity of frame-level labels
- Explore end-to-end frame-level label learning, unifying alignment generation and keyword spotting within the same training framework

### Implications for the KWS Field
- Training-decoding consistency is a key principle in designing end-to-end KWS systems, and frame-level methods naturally satisfy it
- The advantage of GRU on keyword spotting tasks shows that not all temporal modeling tasks need LSTM's complex gating
- The streaming-friendliness of frame-level methods gives them a natural advantage in industrial deployment
- Xiaomi's production experience demonstrates that the frame-level Seq2Seq approach is a strong alternative to attention models
