# Multi-Task Network for Noise-Robust Keyword Spotting and Speaker Verification using CTC-based Soft VAD and Global Query Attention

**Authors/Affiliations**: Seungbin Kim, Minchan Kim, Jihwan Kim, Sunmukjoung Cho, Changhan Cho, Ji-Hoon Kim (KAIST, Korea Advanced Institute of Science and Technology)

**Date**: May 2020 (arXiv:2005.03867)

**Link**: https://arxiv.org/abs/2005.03867

**Keywords**: Keyword Spotting, Speaker Verification, Multi-Task Learning, VAD, Attention Mechanism

## Problem Statement

Keyword Spotting (KWS) and Speaker Verification (SV) are typically two independently operating modules in practical speech interaction systems:
- **KWS Module**: Detects whether a specific keyword has been spoken
- **SV Module**: Verifies whether the speaker is an authorized user

Independent operation presents the following issues:
1. **Computational Redundancy**: Both modules perform front-end feature extraction and acoustic encoding separately, resulting in significant redundant computation.
2. **Insufficient Feature Utilization**: KWS and SV share underlying acoustic features (such as spectral envelopes and fundamental frequency), but independent training fails to leverage this complementarity.
3. **Insufficient Noise Robustness**: Independent systems handle noise interference separately, lacking joint noise reduction capabilities.

In noisy environments, the performance of both tasks degrades severely. If the two tasks are modeled jointly, sharing feature extraction and noise processing modules, it not only improves efficiency but may also enhance the performance of each individual task through the regularization effect of multi-task learning.

## Methodology

### Multi-Task Network Architecture

The design consists of a Shared Encoder + Two Task-Specific Heads:

**Shared Encoder**:
- Multi-layer CNN + RNN structure
- Extracts high-level acoustic representations from audio spectral features
- This representation serves both downstream tasks: KWS and SV

### CTC-based Soft VAD

One of the key innovations of this paper—CTC-based Soft Voice Activity Detection:

**Design Motivation**: In noisy environments, distinguishing between speech and non-speech segments is crucial for both tasks. Traditional hard-decision VAD (binary decision) loses boundary information.

**Technical Implementation**:
- A CTC (Connectionist Temporal Classification) branch is added on top of the shared encoder
- The CTC branch outputs frame-level speech/non-speech probabilities (Soft VAD probabilities)
- Soft VAD probabilities are used as attention weights to weight the shared features:
  - Speech frames receive high weights (preserving useful information)
  - Non-speech frames receive low weights (suppressing noise)
- The entire Soft VAD module is end-to-end trainable

**Mathematical Representation**:
- CTC Output: $p_{vad}(t) = \text{softmax}(\text{CTC\_branch}(h(t)))$, taking the probability of the speech class
- Weighted Feature: $h'(t) = p_{vad}(t) * h(t)$

### Global Query Attention

Another key innovation—Global Query Attention Mechanism:

**Design Motivation**: The SV task requires extracting speaker identity features from the entire utterance, while the KWS task focuses on the acoustic content of the keyword. A mechanism is needed to guide KWS information into the SV task.

**Technical Implementation**:
- The output of the KWS branch (keyword embedding) is used as the Query
- The features from the shared encoder are used as the Key and Value
- Through a Cross-Attention mechanism, KWS information guides the SV branch to focus on speaker features related to the keyword
- For example: Knowing that "Hey Siri" was spoken allows the SV branch to focus more on the speaker features in that specific context

### Training Objective

Multi-task Joint Loss:
- $L_{total} = L_{KWS} + \lambda * L_{SV} + \mu * L_{VAD}$
- $L_{KWS}$: Keyword classification cross-entropy loss
- $L_{SV}$: Speaker verification loss (e.g., AM-Softmax, ArcFace)
- $L_{VAD}$: CTC Voice Activity Detection loss
- $\lambda$ and $\mu$ are balancing coefficients

## Main Contributions

1. **Unified KWS-SV Multi-Task Framework**: Proposes a multi-task network that simultaneously performs keyword spotting and speaker verification; the shared encoder reduces computational overhead during deployment.

2. **CTC-based Soft VAD**: Innovatively uses a CTC branch to implement soft voice activity detection as a feature weighting mechanism to enhance noise robustness. Soft VAD avoids information loss from hard decisions and can be optimized end-to-end.

3. **Global Query Attention**: Proposes using keyword information as a query to guide speaker feature extraction, achieving knowledge transfer from KWS to SV and enhancing the synergy between the two tasks.

4. **Noise Robustness**: Through thesynergistic effect of Soft VAD and the shared encoder, the system demonstrates stronger robustness under noisy conditions.

## Experimental Results

### Experimental Setup
- **KWS Task**: Keyword classification
- **SV Task**: Speaker Verification (EER, Equal Error Rate)
- **Noise Conditions**: Environmental noise at different SNR levels

### Main Results
- **Multi-Task > Single-Task**: Multi-task training outperforms corresponding single-task baselines on both KWS and SV tasks.
- **Soft VAD Effect**: CTC-based Soft VAD significantly improves the performance of both tasks under noisy conditions.
- **Global Query Attention Effect**: Keyword-guided attention mechanism improves the Equal Error Rate of the SV task.
- **Noise Robustness**: The advantage of the multi-task system is more pronounced under high noise conditions (SNR < 0dB).
- **Efficiency Improvement**: The shared encoder reduces total computation compared to two independent systems.

### Ablation Studies
- **Removing Soft VAD**: Significant performance drop for both tasks under noise.
- **Removing Global Query Attention**: SV performance drops, while KWS is less affected.
- **Shared Encoder vs. Independent Encoder**: The shared encoder outperforms the independent encoder on both tasks.

## Limitations and Future Work

### Method Limitations
- **Model Complexity**: The architecture of the multi-task network is more complex than single-task systems, increasing training and tuning difficulties.
- **Data Requirements**: Requires paired training data containing both KWS labels and speaker labels.
- **Task Trade-offs**: There may be conflicts in optimization directions between the KWS and SV tasks, requiring careful adjustment of loss weights.
- **Noise Evaluation Scope**: Insufficient evaluation across more diverse noise types and conditions.
- **Real-time Performance**: Inference latency for multi-task processing may be higher than that of a standalone KWS system.

### Future Directions
- Research dynamic task routing mechanisms to adaptively select tasks for execution based on input features.
- Explore joint optimization of more tasks (e.g., adding language identification, emotion recognition).
- Research lightweight multi-task network designs suitable for embedded deployment.
- Extend to streaming processing scenarios to support real-time multi-task inference.
- Combine with adversarial training to further enhance noise robustness.
