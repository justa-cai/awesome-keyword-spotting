# ConvMixer: Feature Interactive Convolution with Curriculum Learning for Small Footprint and Noisy Far-Field Keyword Spotting

- **Authors/Affiliations**: Dianwen Ng, Yunqi Chen, Biao Tian, Qiang Fu (Alibaba Group, Beijing); Eng Siong Chng (Nanyang Technological University, Singapore)
- **Date**: January 2022
- **Link**: https://arxiv.org/abs/2201.05863
- **Keywords**: keyword spotting, small footprint, noisy far-field, ConvMixer, curriculum learning, MLP mixer, depthwise separable convolution

## Problem Statement

### Problem Background and Domain Pain Points
Keyword recognition faces two simultaneous and severe challenges in real-world smart home, in-vehicle, and industrial IoT scenarios: robustness to far-field noise and extremely tight resource constraints. Far-field environments introduce a compound effect of multiple signal degradation factors: room reverberation (RIR, causing time-frequency blur due to multipath propagation, with typical T60 reverberation times of 0.3-0.8 seconds) "smears" the transient features of speech; environmental noise (appliances, traffic, wind, other speakers) can completely drown out keyword signals under low signal-to-noise ratio (SNR) conditions (as low as -10dB); and energy attenuation due to propagation distance reduces signal amplitude by 10-20dB. Meanwhile, KWS systems are typically deployed on cost-sensitive embedded devices (such as smart light bulbs, thermostats), where processor clock speeds may be below 100MHz, SRAM is less than 256KB, and Flash memory is less than 1MB. This requires model parameters to be in the range of 100K and computational cost to be in the tens of MMACs.

There is a fundamental tension between these two goals (noise robustness and minimal model size): noise robustness typically requires models to have a large receptive field (to capture global spectral patterns to distinguish signal from noise) and rich parameter capacity (to learn complex noise-speech separation boundaries), whereas minimal models have limited receptive fields and capacity.

### Specific Deficiencies of Existing Methods
- **Roots of Noise Vulnerability in Small CNN Models**: Traditional small CNN models (such as DS-CNN, TC-ResNet, BC-ResNet) reduce computational cost through depthwise separable convolutions and residual connections. However, their core limitation lies in their local receptive fields. A standard 3x3 convolution covers only 3 time frames x 3 frequency channels, and stacking multiple layers results in a limited effective receptive field. Under low SNR conditions, local features are severely polluted by noise, and the model needs to extract "global consistency" from a larger time-frequency context to determine the presence of a keyword—this exceeds the capability of local convolutions. Experimental data shows that DS-CNN-S achieves an accuracy of approximately 94.4% in clean conditions, but this may plummet to below 60% at -10dB SNR.
- **High Computational Overhead of Transformer Models**: Models based on self-attention mechanisms (such as KWT-1: 607K parameters, KWT-2: approx. 1.1M parameters, AST-Tiny: approx. 5M parameters) model long-range dependencies through global attention, performing excellently in noise robustness. However, their core issue is the $O(N^2 \cdot d)$ computational complexity of self-attention ($N$ is sequence length, $d$ is feature dimension). For typical KWS inputs (40 time frames, 64-dimensional features), the computational cost of a single self-attention layer can reach several million MACs. After stacking 12 layers, the total computational cost reaches hundreds of MMACs, which is more than 10 times that of small CNNs.
- **Limitations of Fixed Noise Augmentation Training Strategies**: Most KWS models use fixed noise augmentation strategies (such as random masking in SpecAugment, or randomly superimposing noise at fixed SNR) for training. This "one-off" augmentation strategy faces a dilemma: if augmentation is too weak (e.g., training only at 20dB SNR), the model lacks generalization ability under harsh conditions; if augmentation is too strong (e.g., training directly at -10dB SNR), the model struggles to converge in the early training stages because extreme noise samples are akin to pure noise for under-trained models, failing to provide effective learning signals.

### Key Challenges Addressed by This Paper
Design a micro-model with only about 100K parameters that replaces "expensive self-attention" with "efficient global feature interaction," enabling it to compete with Transformer models 5-50 times larger in parameter count under far-field noise conditions (SNR as low as -10dB), while maintaining extremely low computational overhead (approx. 22M MACs). Simultaneously, systematically build noise robustness through a progressive curriculum learning strategy, allowing the model to learn from "easy to hard" in a systematic order.

## Methodology

### Overall Architecture Design and Design Motivation
The design philosophy of the ConvMixer architecture is "replacing expensive attention with efficient MLP mixing," inspired by the MLP-Mixer in the visual domain (Tolstikhin et al., 2021). MLP-Mixer proved in image classification that "global feature interaction can be achieved using only MLPs, without attention or convolution." This paper adapts this idea to the time-frequency spectrograms of KWS. The core adaptation is: spectrograms have a clear two-dimensional structure (time axis and frequency axis), and the physical meanings of the two axes differ (the time axis encodes dynamic changes in speech, while the frequency axis encodes the spectral envelope). Therefore, mixing strategies must be designed separately for the two axes.

The overall architecture consists of three parts:
1. **Pre-conv block**: Uses 1D depthwise separable convolution to perform initial feature extraction on the input spectrogram along the time dimension, mapping the original 40-dimensional Mel-filterbank features to a high-dimensional feature space (128 dimensions). This step captures local time-dynamic patterns (such as consonant-to-vowel transitions).
2. **ConvMixer block (repeated N=4 times)**: The core innovative component. Each block contains three sub-operations: 2D depthwise separable convolution in the frequency direction, 1D depthwise separable convolution in the time direction, and a dual-dimension MLP mixing layer.
3. **Post-conv block**: Integrates the output of the ConvMixer blocks into final classification logits through average pooling and a fully connected layer.

Total parameters: 119K (approx. 0.12M), Computational cost: 22.2M MACs.

### Mathematical Principles of Core Algorithms: Dual-Dimension Mixing Layer

The mixing layer in the ConvMixer block is the key innovation of this architecture, containing two complementary MLP operations:

**Frequency-Channel Mixing MLP**:
Let the feature map output by the ConvMixer block be $X \in \mathbb{R}^{B \times T \times F \times C}$ ($B$ is batch size, $T$ is number of time frames, $F$ is number of frequency channels, $C$ is number of feature channels). The frequency mixing operation is:

$$X_{freq\_mix} = X + \text{MLP}_F(\text{LayerNorm}(X))$$

where $\text{MLP}_F$ performs a global linear transformation along the frequency dimension $F$: $X$ is flattened along the frequency dimension to $(B, T, C \times F)$, passed through a fully connected layer $\mathbb{R}^{C \cdot F} \to \mathbb{R}^{C \cdot F}$ (followed by GELU activation and LayerNorm), and then reshaped back to the original dimensions.

The physical significance of this design: The energy distribution patterns of different keywords on the spectrogram are discriminative (e.g., vowels concentrate in the formant region of low frequencies 200-800Hz, the fricative /s/ is distributed in the high-frequency region of 3000-7000Hz, and plosives /p/, /t/ manifest as broadband short-time energy bursts). Global frequency mixing enables the model to capture these cross-frequency channel energy distribution patterns, rather than just the adjacent frequency differences of local convolutions.

**Time-Channel Mixing MLP**:
Similarly, time mixing performs a global linear transformation along the time dimension $T$:

$$X_{time\_mix} = X + \text{MLP}_T(\text{LayerNorm}(X))$$

The physical significance of time mixing: The temporal structure of keywords (e.g., the phoneme sequence /s/->/t/->/a/->/p/ in "stop") determines their discriminability. Global time mixing allows the model to focus on key frames within the entire time window (e.g., the steady-state part of vowels, the transition part of consonants) without the $O(T^2)$ computational cost of self-attention.

**Computational Complexity Analysis**:
- Self-attention (KWT-1): $O(T^2 \cdot d)$. For $T=40, d=64$, the computational cost is approx. $102K$ per layer.
- MLP mixing (ConvMixer): $O(T \cdot d)$ or $O(F \cdot d)$. For $T=40, F=40, d=128$, the computational cost is approx. $5K$ per layer.
- The computational cost of MLP mixing is only about 1/20 of that of self-attention, yet it provides global time/frequency feature interaction capabilities.

### Multi-Condition Training Strategy Based on Curriculum

Five-stage progressive training process:

**Stage 1 (Basic Training)**: Trains using only clean audio. Establishes basic acoustic feature representations—the model learns phoneme-level discriminative features. Lasts approx. 30 epochs.

**Stage 2 (Light Noise Adaptation)**: Clean + 0dB noise augmentation. Introduces light noise, allowing the model to learn initial noise robustness—learning to ignore noise below speech energy. Lasts approx. 20 epochs.

**Stage 3 (Moderate Noise Adaptation)**: Clean + 0dB + -5dB. Increases moderate noise difficulty, enabling the model to learn to recognize keywords even when noise masks partial spectral information. Lasts approx. 15 epochs.

**Stage 4 (Extreme Noise Adaptation)**: Clean + 0dB + -5dB + -10dB. Introduces extreme noise conditions, allowing the model to learn to extract residual discriminative information from signals almost completely drowned by noise. Lasts approx. 10 epochs.

**Stage 5 (Far-Field Comprehensive Adaptation)**: Adds far-field Room Impulse Response (RIR) augmentation on top of the above—simulating multipath propagation effects through convolution. The model learns to cope with time-frequency blur caused by reverberation.

**Stage Transition Criteria**: The duration of each stage is determined by an adaptive criterion. The paper defines a composite score: $\text{Score} = \text{Normalized\_Accuracy} - \text{Normalized\_Loss}$. When accuracy improvement slows down (the model has learned features of the current difficulty) while loss continues to decrease (the model begins to overfit), the Score peaks and starts to decline, automatically triggering the next stage of training. This mechanism avoids training collapse caused by introducing difficult samples too early, and also avoids staying in a stage too long, leading to overfitting.

### Technical Differences from Existing Methods
- Compared to MLP-Mixer (Visual): This paper decomposes channel mixing into two independent operations: frequency mixing and time mixing, corresponding to the two physical dimensions of the spectrogram. The patch mixing and channel mixing of visual MLP-Mixer do not distinguish spatial dimensions.
- Compared to KWT (Keyword Transformer): KWT uses self-attention to achieve global interaction, while ConvMixer uses MLP mixing. The former has a computational cost of $O(T^2)$, while the latter is $O(T)$. When parameter counts are similar, ConvMixer has significantly higher computational efficiency.
- Compared to SpecAugment Augmentation Training: SpecAugment randomly masks parts of the spectrogram, which is a "passive" data augmentation technique. The "progressive augmentation" strategy of curriculum learning is an "active" training strategy—it controls the difficulty gradient of training samples, making the model's learning process more consistent with human cognitive learning patterns.

## Main Contributions

1. **ConvMixer Architecture for Micro KWS**: For the first time, the MLP-Mixer idea is introduced into the KWS domain, and a dual-dimension (time + frequency) mixing layer is designed specifically for the two-dimensional structure of time-frequency spectrograms. A model with only 119K parameters achieves 98.2% accuracy in clean conditions, comparable to or better than large models with 5-6 times the parameters (e.g., KWT-1 with 607K parameters achieves 97.7%). This proves that "global feature interaction" can be achieved not only through attention but also through more efficient MLP mixing.

2. **Efficient Global Feature Interaction Mechanism**: By replacing self-attention with MLP mixing layers, global feature interaction across channels is achieved with $O(n)$ linear complexity. In spectral analysis, this means the model can establish feature correlations across the entire frequency range or entire time span, rather than just the limited receptive field of local convolutions. This design provides a new answer to "how small models can gain a global perspective under limited capacity."

3. **Five-Stage Progressive Curriculum Learning**: A complete noise robustness training workflow is designed, transitioning gradually from clean to extreme noise, accompanied by an adaptive stage transition criterion. This strategy improves the accuracy of the micro-model at -10dB SNR by approximately 5.4 percentage points (from 66.50% to 71.88%). The improvement is particularly significant under low SNR conditions—because the "learning space" under difficult conditions is larger. Curriculum learning is orthogonal to the ConvMixer architecture and can be independently applied to other KWS models.

4. **Extreme Pursuit of Parameter Efficiency**: While maintaining competitiveness, the model has 5-50 times fewer parameters than Transformer-based KWS models, validating the assertion that "architecture design is more important than model scale." This provides new design ideas for resource-constrained KWS deployment.

## Experimental Results

### Datasets Used and Their Scale
- **Google Speech Commands V2-12 (GSC-12)**: 12 command word classes, approx. 105,000 1-second speech samples. The training set has approx. 84,000 samples, the validation set has 10,000 samples, and the test set has 11,000 samples. Version 2 increases the diversity of recorders and background noise compared to Version 1.
- **Noise-Augmented Test Set**: Generated using the MUSAN noise library (containing approx. 900 types of environmental noise, music clips, and interfering speech) and simulated Room Impulse Response (RIR) to create test data at different SNR levels (20dB, 0dB, -5dB, -10dB) and far-field conditions. Approx. 11,000 augmented test samples are generated for each SNR level.

### Definition and Rationale for Evaluation Metrics
- **Top-1 Accuracy (%)**: Standard classification accuracy, reported separately for clean and different SNR conditions. The reason for evaluating separately at different SNRs is that a single average accuracy may mask the model's deficiencies under extreme conditions.
- **Parameters (K)**: Measures the model's storage requirements. For embedded deployment, the parameter count directly determines Flash storage usage.
- **MACs (M)**: Measures the model's computational complexity. It directly affects inference latency and power consumption.

### Detailed Comparison with Baseline and SOTA Methods

**Performance Comparison in Clean Conditions**:
| Model | Parameters (K) | MACs (M) | Accuracy (%) |
|:---|:---:|:---:|:---:|
| ConvMixer (This Paper) | 119 | 22.2 | **98.2** |
| KWT-1 | 607 | ~50 | 97.7 |
| KWT-2 | ~1100 | ~100 | 98.0 |
| MHAtt-RNN | 784 | 347 | 97.3 |
| DS-CNN-S | 24 | 5.4 | 94.4 |
| BC-ResNet-1 | 30 | 5.2 | 95.4 |

ConvMixer achieves the highest accuracy (98.2%) with one of the smallest parameter counts (119K), surpassing KWT-1 which has 5 times more parameters and MHAtt-RNN which has 6.5 times more parameters.

**Curriculum Learning Effect under Noise Conditions (ConvMixer+ vs. Baseline ConvMixer, i.e., without curriculum learning)**:
| SNR | ConvMixer+ (%) | ConvMixer (%) | Improvement (%) |
|:---:|:---:|:---:|:---:|
| 20dB | 90.83 | 87.85 | +2.98 |
| 0dB | 83.04 | 78.10 | +4.94 |
| -5dB | 78.39 | 72.78 | +5.61 |
| -10dB | 71.88 | 66.50 | +5.38 |

Key Finding: The improvement from curriculum learning increases as SNR decreases (from +2.98% at 20dB to +5.38% at -10dB). This validates that curriculum learning has greater value under difficult conditions—because progressive training allows the model to better utilize learning signals from extreme noise samples.

### Findings from Ablation Studies

**Necessity of Dual-Dimension Mixing**:
- Full ConvMixer -> 71.88% at -10dB SNR
- Remove frequency dimension mixing -> approx. 68.9% at -10dB SNR (drop of approx. 3.0%). Frequency mixing is crucial for preserving the discriminability of spectral envelopes in noise.
- Remove time dimension mixing -> approx. 69.4% at -10dB SNR (drop of approx. 2.5%). Time mixing is crucial for capturing the temporal structure of keywords.
- Remove both -> approx. 66.5% at -10dB SNR (degenerates to a pure convolution model), indicating that mixing in both dimensions provides complementary information gains.

**Contributions of Each Stage of Curriculum Learning**:
- Stages 1-4 (Noise Augmentation) contribute approx. 70% of the total robustness improvement.
- Stage 5 (RIR Augmentation) contributes approx. 30%, indicating that reverberation is a degradation factor independent of noise and requires specialized training.

**MLP Mixing vs. Self-Attention (Under Same Parameter Budget)**:
- The accuracy of the MLP mixing layer is approx. 1.5% higher than that of the self-attention layer (under a 119K parameter budget), because MLP mixing layers have higher parameter utilization efficiency—the $Q, K, V$ projection matrices of self-attention consume a large number of parameters, whereas MLP mixing layers use all parameters for global feature transformation.

## Limitations and Future Work

### Technical Limitations of the Method
- **Absolute Performance Bottleneck at Extremely Low SNR**: Despite the significant improvement brought by curriculum learning, the absolute accuracy of 71.88% at -10dB SNR is still far below actual application requirements (usually requiring >90%). There is still a gap of approx. 10% compared to large Transformer models (e.g., AST-Tiny can reach 80%+ at -10dB). The linear modeling capability of the MLP mixing layer is still limited under extreme noise—linear transformations cannot "selectively focus" on inputs like attention mechanisms (i.e., dynamically adjusting weights based on input content).
- **Limitations of Long-Range Dependency Modeling in Mixing Layers**: MLP mixing layers are essentially global linear transformations of feature dimensions (fully connected layers). Although they have a larger receptive field than local convolutions, they lack the ability of self-attention mechanisms to "model interaction relationships between input elements." In keyword recognition requiring precise time alignment (e.g., distinguishing "yes" and "yet", which differ only in the phonemes of the last 50ms), this limitation may lead to insufficient fine-grained discrimination capability.
- **Dependence on Fixed-Length Inputs**: MLP mixing layers require fixed input lengths (because the weight dimensions of fully connected layers are fixed), making them less adaptable to variable-length keywords.

### Deficiencies in Experimental Design
- **Single Dataset**: All experiments are evaluated only on Google Speech Commands. The limitations of this dataset include: (1) Speech samples are short (approx. 1 second), so its applicability to multi-word wake words (such as "Hey Siri", "Xiao Ai Tong Xue") has not been verified; (2) It is mainly recorded in a North American English environment, so its generalization to tonal languages (such as the different meanings of the Chinese four-tone "ma") and morphologically rich languages (such as Turkish affix changes) is unknown.
- **Time Cost of Curriculum Training**: The total training time for the five-stage training is approx. 3-5 times that of standard training (each stage requires independent training to convergence). Although there is no additional overhead during inference, the increased training cost may become a bottleneck in fast-iterating product development.
- **Unreported Measured Data on Power Consumption and Latency**: Only MACs are reported; inference latency and power consumption were not measured on actual embedded hardware.

### Possible Directions for Future Improvement
- **Exploration of Hybrid Architectures**: Introduce lightweight attention modules (such as linear attention, local attention, or random attention) on the basis of ConvMixer to enhance dynamic feature selection capabilities without significantly increasing the parameter count.
- **Adaptive Curriculum Learning**: Use reinforcement learning or meta-learning to automatically optimize the stage division and transition strategies of curriculum learning, replacing the manually designed Score criterion. Automatic trigger mechanisms based on model training curves (statistical characteristics of learning rate, gradients) can also be explored.
- **Cross-Lingual and Multimodal Expansion**: Apply the ConvMixer + Curriculum Learning paradigm to multilingual KWS (such as Chinese command word recognition) and audio-visual multimodal KWS (combining lip movement information to improve far-field robustness).
- **Inspiration for the KWS Field**: The success of ConvMixer indicates that dual-dimension (time + frequency) modeling specifically for speech time-frequency spectrograms is more effective than generic 2D convolutions. This design principle—structuring architecture based on the physical structure of the data modality—is worth promoting in other speech processing tasks (such as speech separation, speech enhancement). The orthogonality of curriculum learning and architecture design indicates that the two can be independently optimized and their effects stacked.
