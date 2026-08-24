# Implicit Acoustic Echo Cancellation for Keyword Spotting and Device-Directed Speech Detection

- **Authors/Affiliations**: Universita Politecnica delle Marche; Amazon Alexa Team
- **Date**: 2021.11 (arXiv), 2023 (IEEE SLT)
- **Link**: https://arxiv.org/abs/2111.10639
- **Keywords**: Implicit Acoustic Echo Cancellation, Keyword Spotting, Device-Directed Speech Detection, Far-field Speech, Reference Signal, Neural Network AEC

## Problem Statement

Smart speakers and voice assistant devices often play audio simultaneously during operation (e.g., music, podcasts, TTS synthesized speech). This sound emitted from the device's own speakers is captured by the microphone, forming acoustic echo. Echo causes severe interference to Keyword Spotting (KWS) systems, potentially leading to:

1. **Increased False Alarms**: When TTS-synthesized speech happens to contain phoneme sequences similar to the wake-word, the KWS system may be falsely triggered.
2. **Decreased Recall**: When the echo signal overlaps with genuine user speech, the KWS system may fail to correctly recognize the user's wake-word.
3. **Failure in Device-Directed Detection**: The system may fail to distinguish between speech directed at the device (device-directed) and content played by the device itself.

Traditional solutions involve cascading an explicit Acoustic Echo Cancellation (AEC) module at the front end of the KWS system, using the speaker reference signal for adaptive filtering. However, this approach has significant drawbacks:
- The AEC module increases system complexity and tuning difficulty.
- The AEC module itself may introduce distortion, affecting downstream KWS performance.
- The AEC and KWS modules are optimized independently, preventing joint optimization.

The core problem this paper addresses is: Can the KWS model implicitly learn echo cancellation capabilities during training, thereby eliminating the dependency on an independent AEC module and achieving better end-to-end performance?

## Methodology

### Overall Architecture Design
The core idea of the Implicit AEC (iAEC) method is to feed both the microphone signal and the speaker reference signal as inputs to the KWS model, allowing the model to automatically learn how to utilize the reference signal to cancel echo interference during end-to-end training.

### Dual-Branch Input Architecture
The model adopts a Dual-Branch input processing architecture:
- **Main Branch**: Processes the signal captured by the microphone (containing user speech + echo + environmental noise).
- **Reference Branch**: Processes the reference signal from the speaker (i.e., the audio played by the device).

The two branches extract features through independent encoders and then fuse them at an intermediate layer:

1. **Feature Extraction Stage**:
   - Microphone signal -> Main Encoder -> Microphone Features $F_{mic}$
   - Reference signal -> Reference Encoder -> Reference Features $F_{ref}$

2. **Feature Fusion Stage**:
   - $F_{mic}$ and $F_{ref}$ are fused via attention mechanisms or concatenation followed by convolution.
   - The fused features contain information about "what was heard" (microphone signal) and "what the device is playing" (reference signal).

3. **Task Prediction Stage**:
   - Fused features -> Classification Head -> Keyword Spotting / Device-Directed Speech Detection results

### Reference Signal Processing
The reference signal is the digital audio about to be played by the speaker (obtained before digital-to-analog conversion), and is therefore clean and noise-free. By learning the mapping relationship between the reference signal features and the echo components in the microphone signal, the model achieves "echo subtraction" in the feature space.

### Training Strategy
- **Synthetic Training Data**: Training samples containing echo are created by mixing clean speech (user speech) with device-played audio under real or simulated Room Impulse Responses (RIR).
- **Multi-Task Training**: Simultaneously training for two tasks: KWS (whether a keyword is present) and Device-Directed Detection (whether the speech is directed at the device).
- **Loss Function**: Weighted sum of multi-task losses: $L = \alpha * L_{KWS} + \beta * L_{DDD}$

### Inference Flow
During inference, the model simultaneously receives the microphone audio stream and the speaker reference signal stream, outputting KWS and device-directed detection results end-to-end, without requiring any independent AEC preprocessing steps.

## Main Contributions

1. **Proof of Feasibility of Implicit AEC**: This is the first demonstration that a KWS model can implicitly learn echo cancellation capabilities through end-to-end training without an explicit AEC preprocessing module. This finding challenges the traditional design philosophy that "AEC must be implemented independently before KWS."

2. **Reduced System Complexity**: By eliminating the independent AEC module and its tuning requirements, the overall architecture of voice assistant systems is simplified. This is particularly important for commercial products requiring rapid iteration.

3. **Simultaneous Improvement in KWS and Device-Directed Detection under Echo Conditions**: Implicit AEC not only outperforms the "Explicit AEC + KWS" pipeline in terms of KWS accuracy but also performs excellently in the device-directed speech detection task.

4. **Provision of a Unified End-to-End Framework**: Echo processing, keyword spotting, and device-directed detection are unified within a single model, providing a new design paradigm for multi-task speech front-end processing.

## Experimental Results

### Dataset
- Uses a synthetic dataset containing echo scenarios simulated with real Room Impulse Responses (RIR).
- Evaluation Scenarios: User speech only, Device playback only, User speech + Device playback (overlapping), Quiet/Noisy environments.

### KWS Performance
- The Implicit AEC method significantly improves KWS accuracy in scenarios with echo (compared to the baseline without any AEC).
- Compared to the traditional "Explicit AEC + KWS" pipeline, iAEC achieves comparable or better accuracy.
- The advantages of iAEC are more pronounced in difficult scenarios where echo and user speech overlap.

### Device-Directed Speech Detection Performance
- Device-directed detection also benefits from implicit echo processing.
- When the device is playing content, traditional methods are prone to misjudging the playback as user commands; iAEC effectively mitigates this issue.

### Ablation Studies
- **Dual-Branch vs. Single-Branch**: Performance drops significantly after removing the reference signal branch, proving that the reference signal is a key input for implicit AEC.
- **Fusion Strategy**: Attention-based fusion outperforms simple feature concatenation.
- **Training Data Diversity**: Training with diverse echo conditions is crucial for generalization capability.

## Limitations and Future Work

### Technical Limitations
- **Dependency on Reference Signal**: This method requires the device to provide a reference signal of the speaker playback. The method may be limited in scenarios where the reference signal is unavailable or has significant latency.
- **Training Data Bias**: Model performance depends on the diversity and realism of echo conditions in the training data. If the echo characteristics of the actual deployment environment differ significantly from the training data, performance may degrade.
- **Unexplored Multi-Device Scenarios**: When multiple devices in the environment play audio simultaneously, the correspondence between the reference signal and the echo becomes more complex.

### Insufficiencies in Experimental Design
- Insufficient detail in the comparison of computational costs with traditional AEC pipelines.
- Lack of A/B test results in real user environments.
- Performance retention after quantization deployment was not explored.

### Future Improvement Directions
- Explore achieving implicit AEC without reference signals (utilizing only microphone signals).
- Combine self-supervised learning to further improve robustness under echo conditions.
- Extend iAEC to multi-channel microphone array processing.
- **Implications for the KWS Field**: End-to-end learning can replace independent modules in traditional signal processing pipelines; implicit learning often outperforms explicit cascading. This concept can be generalized to more front-end processing tasks such as speech enhancement and noise reduction.
