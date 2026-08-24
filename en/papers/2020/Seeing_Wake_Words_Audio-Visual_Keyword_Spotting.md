# Seeing Wake Words: Audio-Visual Keyword Spotting

**Authors/Affiliations**: Ruizhi Li, Shuang Yang, Dong Yu (University of Oxford)

**Date**: September 2020 (arXiv:2009.01225)

**Link**: https://arxiv.org/abs/2009.01225

**Keywords**: Audio-Visual Keyword Spotting, Multimodal Learning, Lip Reading, Wake Word Detection, Fusion Strategies

## Problem Statement

Traditional audio-based Keyword Spotting (KWS) systems suffer from severe performance degradation in noisy environments. As the Signal-to-Noise Ratio (SNR) decreases, keyword information in the audio signal is masked by noise, leading to a sharp decline in detection accuracy. This is a core pain point in practical applications, as KWS systems are often deployed in noisy environments such as when televisions are playing, in traffic noise, or during multi-person conversations.

Humans can utilize visual information (the speaker's lip movements) to assist speech understanding in noisy environments, a phenomenon known as the "cocktail party effect." Visual information is not interfered with by acoustic noise and is highly correlated with speech content. Introducing visual information into KWS systems can:
- Provide complementary information when the audio signal is degraded
- Improve the system's robustness under extreme noise conditions
- Enable audio-visual joint detection, thereby reducing the false alarm rate

## Methodology

### Dual-Stream Architecture Design

This paper proposes an Audio-Visual Dual-Stream KWS architecture:

**Audio Stream**:
- Input: Spectral features of the audio signal (Mel-filterbank features)
- Uses CNN to extract audio features
- In quiet environments, the audio stream alone can provide sufficient detection information

**Visual Stream**:
- Input: Sequence of video frames of the speaker's lip region
- Uses 3D CNN or 2D CNN+RNN to extract lip movement features
- Lip movements provide visual cues of the speech production process

### Feature-Level Fusion Strategy

- Audio and visual features are fused at the feature level (early fusion)
- The fused joint features contain information from both audio and visual modalities
- The fused features are fed into a shared classifier for keyword detection

### Multimodal Inference

The system supports three inference modes:
1. **Audio-only mode**: When visual information is unavailable
2. **Visual-only mode**: When the audio signal is extremely poor or unavailable
3. **Audio-Visual joint mode**: Utilizes complementary information from both modalities to achieve optimal performance

This flexibility allows the system to adapt to different deployment scenarios.

## Main Contributions

1. **Audio-Visual Fusion KWS Framework**: Proposes a complete audio-visual keyword detection framework, demonstrating the value of visual information in the KWS task. This is one of the early explorations introducing multimodal learning to KWS.

2. **Significant Improvement in Noise Robustness**: In noisy environments, the performance of the audio-visual joint system far exceeds that of the pure audio system, proving the natural immunity of visual information to noise.

3. **Flexible Multimodal Inference**: The system can automatically switch inference modes based on the availability of input data, offering good adaptability in practical deployments.

4. **Verification of Cross-Modal Complementarity**: Experimental results clearly demonstrate the complementarity between audio and visual modalities—audio is stronger in quiet environments, while visual information provides critical supplementation in noisy environments.

## Experimental Results

### Dataset
Audio-visual KWS benchmark dataset (containing synchronized audio and lip video)

### Key Results
- **Quiet Environment**: The performance of the audio-visual joint system is comparable to that of the pure audio system, with limited additional gain from the visual modality.
- **Noisy Environment (SNR < 0dB)**: The audio-visual joint system significantly outperforms the pure audio system, with the performance gap widening as noise increases.
- **Extreme Noise (SNR < -5dB)**: The pure audio system almost fails, while the audio-visual joint system maintains acceptable performance.
- **Visual-Only Mode**: Performance is lower than the audio mode but far superior to the pure audio mode in noisy environments.

### Fusion Analysis
- Feature-level fusion outperforms decision-level fusion (voting after separate classification)
- The visual stream primarily provides information on visually salient phonemes such as consonants
- Vowel information comes mainly from the audio stream

## Limitations and Future Work

### Methodological Limitations
- **Camera Requirement**: Requires continuous video input, which limits deployment scenarios (smart speakers typically do not have cameras)
- **Increased Computational Overhead**: The visual stream adds significant computational load, which may not be suitable for resource-constrained devices
- **Dependence on Video Quality**: Performance depends on the accuracy of lip detection and video quality (lighting, angle, occlusion)
- **Privacy Concerns**: Continuous video recording raises serious privacy concerns
- **Synchronization Requirements**: Audio and video streams require precise time synchronization

### Future Directions
- Research lightweight visual feature extraction methods to reduce the computational overhead of the visual stream
- Explore privacy-preserving visual KWS solutions (e.g., extracting only lip contours rather than the full face)
- Research dynamic audio-visual fusion strategies that adaptively adjust the weight of visual information based on audio quality
- Extend the framework to multi-speaker scenarios (lip tracking + speaker separation)
- Research cross-domain adaptation to enable the model to adapt to different cameras and lighting conditions
