# Domain Aware Training for Far-field Small-footprint Keyword Spotting

**Authors/Affiliations**: Ming Li, Yan Jia (Duke Kunshan University)

**Date**: August 2020 (Interspeech 2020)

**Link**: https://arxiv.org/abs/2005.09653

**Keywords**: Keyword Spotting, Domain Adaptation, Far-field, Data Augmentation, CORAL Loss

## Problem Statement

Far-field Keyword Spotting (KWS) faces severe acoustic mismatch issues in practical applications:
- **Training Data**: Typically consists of near-field recordings with high Signal-to-Noise Ratio (SNR) and clear direct acoustic paths.
- **Deployment Environment**: Far-field conditions involving the following distortions:
  - **Reverberation**: Sound waves reflect off walls, floors, etc., and superimpose, blurring the temporal structure of speech.
  - **Distance Attenuation**: Sound intensity decays with the square of the distance, reducing the SNR.
  - **Environmental Noise**: Steady-state noise from air conditioners, fans, etc., as well as non-stationary interfering noise.
  - **Aliasing Effects**: Reverberation tails overlap with subsequent speech, causing self-masking and mutual masking.

This near-field to far-field acoustic mismatch causes a significant degradation in KWS model performance when deployed in far-field conditions. Small-footprint models, due to their limited parameter capacity, are more sensitive to domain shift, leading to particularly severe performance drops under far-field conditions.

## Methodology

### Far-field Data Simulation

Near-field data is converted into simulated far-field data using signal processing methods:
- **Room Impulse Response (RIR) Convolution**: Near-field speech is convolved with simulated RIRs to introduce reverberation effects.
- RIR simulation parameters include: room size, wall reflection coefficients, source-to-microphone distance, source direction, etc.
- The Image Method is used to generate RIRs.
- Different room configurations and speaker distances are simulated by adjusting parameters.

### Multi-Condition Training

Training data is divided into three parts:
1. **Original Near-field Data**: Maintains the diversity of the training set.
2. **Simulated Far-field Data**: Generated via RIR convolution and noise superposition.
3. **Mixed Data**: Combinations of near-field and far-field data in varying proportions.

### CORAL Loss (Correlation Alignment)

Core domain adaptation technique:
- CORAL (CORrelation ALignment) loss aligns the second-order statistics (covariance matrices) of features between the source domain (near-field) and the target domain (far-field).
- Mathematical definition: $L_{CORAL} = ||C_s - C_t||_F^2 / (4d^2)$, where $C_s$ and $C_t$ are the covariance matrices of the source and target domain features, respectively, and $d$ is the feature dimension.
- Intuition: If features from two domains have the same covariance structure, a classifier trained on one domain can be better transferred to the other.
- CORAL does not require one-to-one correspondence between source and target domain samples, making it suitable for unsupervised domain adaptation.

### Training Strategy

Total loss function: $L_{total} = L_{CE} + \lambda * L_{CORAL}$
- $L_{CE}$: Standard cross-entropy classification loss.
- $L_{CORAL}$: Domain alignment loss.
- $\lambda$: Balancing coefficient controlling the strength of domain adaptation.

## Main Contributions

1. **Far-field KWS Domain Adaptation Framework**: Proposes a complete domain-aware training framework that systematically addresses the near-field to far-field acoustic mismatch. This framework combines dual domain adaptation at the data level (simulation) and feature level (CORAL).

2. **First Application of CORAL Loss in KWS**: Introduces the CORAL domain adaptation loss into KWS training, demonstrating the effectiveness of aligning feature second-order statistics for far-field adaptation.

3. **Multi-Condition Training Strategy**: Improves model robustness to different acoustic conditions by training with a mixture of near-field and simulated far-field data.

4. **Near-field to Far-field Gap Analysis**: Provides a systematic analysis of performance degradation from near-field to far-field, establishing a baseline for understanding the problem in subsequent research.

## Experimental Results

### Experimental Setup
- **Near-field Training Data**: Standard KWS training set.
- **Far-field Evaluation**: Simulated far-field data (varying distances, reverberation, noise) and real far-field data.
- **Baseline**: Standard KWS model trained on near-field data.
- **Evaluation Metrics**: Classification accuracy, Detection Error Rate (DER).

### Main Results
- **Domain-aware Training Significantly Improves Far-field Performance**: Accuracy under far-field conditions is substantially increased.
- **Incremental Effect of CORAL Loss**: On top of multi-condition training, the CORAL loss provides additional performance gains.
- **Effectiveness of Multi-Condition Training**: Training with mixed near/far-field data performs better under far-field conditions than training with pure near-field data.
- **Distance Robustness**: Improvements are observed across different speaker distances (1m, 3m, 5m).
- **Noise Robustness**: The domain-adapted model exhibits more gradual performance degradation under different SNR conditions.

### Ablation Studies
- RIR simulation augmentation alone yields significant effects.
- CORAL loss further improves performance on top of simulation augmentation.
- The degree of improvement varies across different keywords.

## Limitations and Future Work

### Method Limitations
- **Limitations of Simulated RIRs**: Simulated Room Impulse Responses may not fully capture the complex acoustic characteristics of real rooms (e.g., diffusion, non-Gaussian reverberation).
- **CORAL Assumptions**: CORAL assumes that source and target domain features have similar covariance structures, which may not hold under extreme domain shifts.
- **Target Domain Data Requirements**: CORAL alignment requires representative data from the target domain (even if unlabeled).
- **Limited Room Diversity**: The number of evaluated room configurations is limited.

### Future Directions
- Use deep learning-driven room acoustic simulation (e.g., Neural Acoustic Simulation).
- Explore adversarial domain adaptation methods (e.g., DANN), which may be more powerful than CORAL.
- Investigate online domain adaptation, allowing the model to continuously adapt to new environments after deployment.
- Combine with front-end processing such as beamforming to further improve far-field performance.
- Extend to multi-microphone scenarios, leveraging spatial information to assist far-field detection.
