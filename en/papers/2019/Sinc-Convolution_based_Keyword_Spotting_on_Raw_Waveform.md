# Sinc-Convolution based Keyword Spotting on Raw Waveform

- **Authors/Affiliations**: Simon Mittermaier, Ludwig Kürzinger, Bernd Waschneck, Gerhard Rigoll (Technical University of Munich)
- **Date**: November 2019 (ICASSP 2020)
- **Link**: https://arxiv.org/abs/1911.02086
- **Keywords**: Keyword Spotting, Sinc Convolution, Raw Waveform, Depthwise Separable Convolution, Parameterized Filters, Learnable Filterbanks

## Problem Statement

The performance of traditional Keyword Spotting (KWS) systems relies heavily on the quality of input features. Standard feature extraction methods face the following issues:

1. **Sub-optimality of hand-crafted features**: Features such as MFCCs and LFBEs are general acoustic features designed based on human auditory models, not specifically optimized for the KWS task. The parameters of these fixed features (e.g., frequency band division of the mel filterbank, number of DCT coefficients) may not be optimal for keyword discrimination.
2. **Information loss**: Compression and transformation during feature extraction (e.g., log compression, DCT transformation) may lose information useful for KWS. For instance, the cepstral representation of MFCCs may lose certain phase and fine spectral structure information.
3. **Challenges in processing raw waveforms**: Learning features directly from raw audio waveforms avoids information loss, but when standard convolution layers are applied to raw waveforms with high sampling rates (typically 16kHz), the first convolutional layer requires a large number of parameters to cover a sufficient frequency range, leading to low parameter efficiency.

Therefore, the core challenge is to design a parameter-efficient method that learns **filterbanks optimized for the KWS task** directly from raw audio waveforms, achieving high accuracy while maintaining an extremely small model size.

## Methodology

This paper proposes using **Sinc Convolution (SincNet)** as the first layer of the KWS neural network, operating directly on raw waveforms to achieve a parameter-efficient learnable filterbank.

### 1. Principle of Sinc Convolution

The core idea of Sinc convolution is to parameterize the filters of the first layer as **parameterized bandpass filters**, rather than learning convolution kernels of arbitrary shapes:

- **Ideal Bandpass Filter**: In the frequency domain, an ideal bandpass filter is a rectangular function, which can be represented as the difference between two sinc functions:

$$g(f_1, f_2)[n] = 2f_2 \text{sinc}(2\pi f_2 n) - 2f_1 \text{sinc}(2\pi f_1 n)$$

where $f_1$ and $f_2$ are the low and high cutoff frequencies of the bandpass filter, respectively.

- **Learnable Parameters**: Each sinc filter has only **two learnable parameters** ($f_1$ and $f_2$), rather than $K$ independent weights in standard convolution (where $K$ is the kernel length). This significantly reduces the number of parameters.

- **Physical Meaning**: The learned parameters directly correspond to the **cutoff frequencies** of the bandpass filters, possessing clear physical meaning—the model learns which frequency bands are most important for KWS.

### 2. Network Architecture

#### 2.1 Sinc Convolution First Layer

- Input: Raw audio waveform (16kHz sampling rate)
- 64 learnable sinc bandpass filters
- Each filter has only 2 parameters (low and high cutoff frequencies), totaling 128 parameters
- Compared to the first layer of a standard convolution (64 x 251 = 16,064 parameters), the number of parameters is reduced by more than 99%

#### 2.2 Depthwise Separable Convolution Subsequent Layers

After the sinc convolution, **Depthwise Separable Convolution** is used to further maintain parameter efficiency:
- **Depthwise Conv**: Each input channel is convolved independently to capture local patterns within channels.
- **Pointwise Conv**: 1x1 convolution is used to fuse information between channels.
- Compared to standard convolution, depthwise separable convolution significantly reduces the number of parameters and computational cost.

#### 2.3 Overall Architecture

```
Raw Waveform -> Sinc Convolution (Bandpass Filterbank) -> Depthwise Separable Conv xN -> Fully Connected -> Output
```

### 3. Interpretability

An important advantage of Sinc convolution is **interpretability**:
- The learned cutoff frequency pairs $(f_1, f_2)$ can be directly visualized, showing the **frequency response of the learned bandpass filters**.
- By analyzing which frequency bands are activated, one can understand which frequency ranges the model considers most important for KWS.
- This provides valuable insights into the decision-making process of KWS models.

## Main Contributions

1. **Introduction of Sinc Convolution to KWS**: For the first time, the parameterized filter idea of SincNet is applied to keyword spotting as a learnable alternative to traditional fixed filterbanks (such as MFCCs). The model automatically learns optimal frequency band divisions rather than relying on hand-crafted mel scales.

2. **Extreme Parameter Efficiency**: Achieved **96.4%** accuracy on Google Speech Commands with only **62K parameters**. This parameter count is far smaller than most contemporary KWS models, demonstrating the significant advantage of sinc convolution in parameter efficiency.

3. **End-to-End Learning from Raw Waveforms**: Enables end-to-end training from raw audio waveforms to keyword classification, eliminating the need for hand-crafted feature extraction steps and removing the information bottleneck of feature engineering.

4. **Interpretable Filter Learning**: The learned sinc filters can be interpreted as bandpass filters, and their cutoff frequencies reveal the frequency band distribution the model focuses on, providing an intuitive understanding for the design and optimization of KWS systems.

5. **Compact Combination of Sinc + Depthwise Separable Convolution**: Combines two parameter-efficient techniques (sinc convolution + depthwise separable convolution) to achieve an extremely compact KWS model.

## Experimental Results

### Google Speech Commands Dataset

| Metric | Value |
|------|---|
| Accuracy | **96.4%** |
| Total Parameters | **62K** |

### Key Results
- The 62K parameter model achieved 96.4% accuracy on Google Speech Commands, competing with baseline methods using MFCC features and larger models.
- The number of parameters in the first layer of Sinc convolution is only about 1% of that in the first layer of standard convolution.
- The learned filters exhibit a non-linear frequency distribution similar to the mel scale, but optimized and adjusted for the KWS task.

### Interpretability Analysis
- The learned bandpass filters cover the main frequency range of speech.
- Some filters are concentrated in the low-frequency region (corresponding to the fundamental frequency and harmonics of vowels), while others cover the high-frequency region (corresponding to frication noise of consonants).
- The distribution of filters provides direct evidence of the importance of frequency bands in the KWS task.

## Limitations and Future Work

### Technical Limitations
- **Initialization Sensitivity**: Performance may be sensitive to the initialization of sinc filter cutoff frequencies—improper initial values may cause training to fall into local optima, affecting the final filter configuration.
- **Noise and Far-field Conditions**: Evaluation under noisy or far-field conditions is limited. Noise in raw waveforms enters the sinc convolution layer directly, lacking noise reduction steps present in feature extraction like MFCCs, which may lead to poor performance under extreme noise conditions.
- **Qualitative Limitation of Interpretability**: The interpretability of learned filters is primarily qualitative (visual analysis) rather than strictly quantitative—there is a lack of quantitative metrics to measure filter quality.

### Future Directions
- Research multi-scale sinc convolution, using filters with different bandwidths in the same layer to capture frequency patterns at different granularities.
- Explore the combination of sinc convolution with attention mechanisms to adaptively weight the contributions of different frequency bands.
- Systematically evaluate the robustness of sinc convolution-based KWS under noisy and far-field conditions, and compare it with MFCC-based methods.
- Investigate initialization strategies for sinc convolution, such as using the Mel scale or Bark scale as initial cutoff frequencies.
- Extend sinc convolution to multi-channel inputs (e.g., microphone arrays) to leverage spatial information to enhance KWS performance.
