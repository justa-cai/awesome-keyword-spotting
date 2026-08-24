# A Depthwise Separable Convolutional Neural Network for Keyword Spotting on an Embedded System

**Authors/Affiliations**: Peter Molgaard Sorensen, Bastian Epp, Tobias May (Technical University of Denmark)

**Date**: October 2020 (EURASIP Journal on Audio, Speech, and Music Processing)

**Link**: https://doi.org/10.1186/s13636-020-00176-2

**Keywords**: Keyword Spotting, Speech Recognition, Embedded Software, Deep Learning, Depthwise Separable Convolutional Neural Network, Quantization

## Problem Statement

Deploying Keyword Spotting (KWS) systems onto low-power embedded microprocessors faces severe resource constraint challenges. Embedded microprocessors (such as the ARM Cortex-M series) typically possess:
- **Extremely limited memory**: SRAM is usually in the range of hundreds of KB.
- **Limited computational power**: Clock frequencies are typically in the 100-200 MHz range.
- **No dedicated AI acceleration hardware**: All computations must rely on the CPU.

Standard CNN architectures have too many parameters and excessive computational costs to run in real-time on these platforms. Although Depthwise Separable Convolution (DSConv) has been proven to effectively reduce the computational complexity of CNNs, the impact of quantization on model accuracy when deploying DSConv-based KWS systems to embedded platforms has not been systematically studied.

## Methodology

### Network Architecture Design

A 10-word KWS system was designed based on the Depthwise Separable CNN (DS-CNN) architecture:
- **Depthwise Convolution**: Performs spatial convolution independently for each input channel, significantly reducing computational load.
- **Pointwise Convolution**: Uses 1x1 convolutions to perform linear combinations along the channel dimension, achieving cross-channel information fusion.
- Compared to standard convolution, the computational complexity of DSConv is reduced to approximately $1/N + 1/K^2$ (where $N$ is the number of output channels and $K$ is the kernel size).

### Hyperparameter Grid Search

To find the optimal balance between accuracy and complexity:
- A systematic grid search was conducted on key network hyperparameters.
- Search dimensions included: number of convolutional filters, number of network layers, and kernel sizes.
- Objective: Achieve maximum complexity reduction with minimal accuracy loss.
- It was found that network complexity could be significantly reduced with almost no impact on classification accuracy.

### Quantization Strategy

This paper conducts an in-depth study of quantization, proposing two fixed-point quantization schemes:

**Mixed Fixed-Point Quantization**:
- Different layers of the network use different bit-widths.
- Layers sensitive to quantization use higher bit-widths (e.g., 16-bit).
- Layers insensitive to quantization use lower bit-widths (e.g., 8-bit).
- Achieves an optimal trade-off between accuracy and memory footprint.

**Dynamic Fixed-Point Quantization**:
- The decimal point position is determined independently for each layer.
- Quantization parameters are dynamically adjusted based on the numerical range of weights and activations in each layer.
- Avoids overflow or accuracy loss that may result from fixed quantization parameters.

### Real-World Noise Data Augmentation

Unlike traditional methods using artificial noise (such as Gaussian white noise), this paper uses real-world environmental noise for data augmentation:
- Collected various types of indoor noise (air conditioning, fans, keyboard typing, speech, etc.).
- Mixed noise under various Signal-to-Noise Ratio (SNR) conditions: covering a range from -5 dB to +20 dB.
- Real-world noise better reflects the noise characteristics encountered in actual deployment than artificial noise.

### Embedded Deployment

Target Platform: ARM Cortex-M4 microprocessor
- Implemented real-time KWS processing for continuous audio streams.
- Both audio front-end processing (MFCC feature extraction) and neural network inference are performed on the MCU.
- Optimized memory usage: Model parameters are stored in Flash, while intermediate activation values during inference are stored in SRAM.

## Main Contributions

1. **Systematic study of DS-CNN complexity-accuracy trade-offs**: For the first time, systematically investigated the complexity scaling characteristics of DS-CNNs on KWS tasks, finding that the network can maintain high accuracy while significantly reducing complexity.

2. **Comprehensive evaluation of quantization on DS-CNN KWS**: Conducted a thorough study of mixed fixed-point and dynamic fixed-point quantization schemes, finding that 8-bit fixed-point quantization significantly reduces memory and computational requirements with minimal accuracy loss. This conclusion has important guiding significance for the practical deployment of embedded KWS systems.

3. **Real-world noise augmentation strategy**: Used diverse real-world indoor noise for data augmentation, improving the model's generalization ability to unseen acoustic conditions.

4. **Complete validation on embedded platforms**: Successfully implemented real-time KWS for continuous audio streams on the ARM Cortex-M4, verifying the practical feasibility of the entire technical solution.

5. **Evaluation under matched and unmatched noise**: Evaluated under both matched background noise (noise types seen during training) and unmatched background noise (noise types not seen during training), comprehensively measuring the system's robustness.

## Experimental Results

### Accuracy and Complexity Trade-off
- The original DS-CNN model achieved approximately 94% accuracy on the Google Speech Commands dataset.
- Through hyperparameter grid search, model complexity could be reduced by several times, with only a slight decrease in accuracy.
- This proves that KWS models contain a large amount of redundant parameters, and pruning them has a minimal impact on accuracy.

### Quantization Effects
- **8-bit quantization**: Almost no accuracy loss; model size reduced by approximately 75%.
- **4-bit quantization**: Accuracy began to show significant degradation.
- **Mixed fixed-point quantization** outperformed uniform quantization schemes: achieving higher accuracy at the same average bit-width.

### Noise Robustness
- Models augmented with real-world noise performed better under unseen noise conditions compared to those augmented with artificial noise.
- Training across multiple SNR conditions improved the model's robustness under different noise levels.

### Embedded Deployment
- The complete KWS system achieved real-time processing on the ARM Cortex-M4.
- Memory usage satisfied the MCU's SRAM and Flash constraints.
- Inference latency met real-time requirements (processing speed was much faster than the audio acquisition speed).

## Limitations and Future Work

### Methodological Limitations
- **Vocabulary Limitation**: Evaluated only on the 10-word Google Speech Commands dataset; scalability to larger vocabularies was not verified.
- **Single-Channel Audio**: Supports only single-channel audio input, without utilizing multi-channel information for spatial filtering.
- **Platform Generality**: The fixed-point quantization scheme is primarily targeted at the ARM Cortex-M4; its applicability to other embedded platforms (such as RISC-V, DSP) requires further verification.
- **Limited Noise Types**: Real-world noise evaluation was limited to specific indoor noise types.

### Future Directions
- Extend to larger vocabularies and custom keyword scenarios.
- Investigate binarization or lower bit-width (e.g., 4-bit, 2-bit) quantization methods.
- Integrate multi-channel front-end processing (beamforming) to improve far-field performance.
- Explore Neural Architecture Search (NAS) to automatically discover optimal DS-CNN architectures suitable for specific hardware.
- Investigate online adaptation methods to allow deployed models to adapt to new noise environments.
