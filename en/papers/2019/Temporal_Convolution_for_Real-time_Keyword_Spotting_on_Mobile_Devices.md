# Temporal Convolution for Real-time Keyword Spotting on Mobile Devices

- **Authors/Affiliations**: Seungwoo Choi, Seokjun Seo, Beomjun Shin, Hyeongmin Byun, Martin Kersner, Beomsu Kim, Dongyoung Kim, Sungjoo Ha
- **Date**: April 2019 (Interspeech 2019)
- **Link**: https://arxiv.org/abs/1904.03814
- **Keywords**: Keyword Spotting, Temporal Convolution, Mobile Devices, ResNet, Low Latency, Real-time, 1D Convolution

## Problem Statement

Deploying real-time KWS systems on mobile devices requires a delicate balance between **accuracy and inference latency**. Traditional KWS methods based on 2D convolution face the following key issues:

1. **Computational overhead of 2D convolution**: Standard 2D CNNs convolve simultaneously across both the time and frequency dimensions of the spectrogram. While this approach can capture joint time-frequency patterns, it requires deep architectures and a large number of operations to fully capture low-frequency and high-frequency features. The computational cost of 2D convolution is typically $O(C_{out} \times C_{in} \times K_t \times K_f \times T \times F)$, where $K_t$ and $K_f$ are the kernel sizes in the time and frequency dimensions, respectively.
2. **Lack of practical latency evaluation**: Although the trade-off between high accuracy and low latency is critical for real-time applications, most KWS studies only report model parameters and floating-point operations (FLOPs), rather than actual inference latency on real mobile hardware. FLOPs are not always linearly correlated with actual latency—factors such as memory access patterns, cache utilization, and hardware-specific optimizations significantly impact actual performance.
3. **Necessity of deep architectures**: 2D CNNs require sufficiently deep networks to capture time-frequency patterns at different scales, but deeper networks imply higher latency.

Therefore, the core challenge is: to design a KWS architecture that enables real-time inference on mobile devices while maintaining or surpassing the accuracy of 2D CNNs, and providing quantitative latency validation on actual hardware.

## Methodology

This paper proposes the use of **1D Temporal Convolution** combined with a compact ResNet architecture for real-time KWS on mobile devices.

### 1. 1D Temporal Convolution vs. 2D Convolution

The core design shift:
- **2D Convolution**: Convolves simultaneously across the time ($t$) and frequency ($f$) dimensions, with a kernel size of $K_t \times K_f$.
- **1D Temporal Convolution**: Convolves only along the time dimension, with a kernel size of $K_t$. The frequency dimension is treated as the input channel dimension.

Key advantages of 1D Temporal Convolution:
- **Fewer parameters and computations**: Each convolution kernel requires only $K_t$ parameters (instead of $K_t \times K_f$).
- **Shallower architecture**: The receptive field of 1D convolution grows faster along the time axis, allowing it to cover sufficient temporal context without requiring a very deep network.
- **More efficient hardware utilization**: 1D convolution is more maturely optimized on mobile CPUs/GPUs, with more regular memory access patterns.

### 2. Compact ResNet Architecture

Adopts a compact network design with residual connections:
- Multiple residual blocks are stacked, each containing 1D Convolution + BatchNorm + ReLU.
- Residual connections allow gradients to propagate directly, supporting effective training.
- The overall architecture is shallower and narrower compared to contemporary 2D CNNs.

### 3. Practical Hardware Evaluation

A distinctive feature of this paper is the detailed latency measurement performed on a **Google Pixel 1** smartphone:
- Not simulated or estimated, but end-to-end inference time on real mobile hardware.
- Measurements include the complete pipeline latency: preprocessing, inference, and postprocessing.
- Provides a Pareto frontier analysis of accuracy versus latency for different model configurations.

### 4. Feature Processing

- Input features: Acoustic features extracted from audio (e.g., MFCCs or spectrograms).
- The frequency dimension of the spectrogram is treated as the channel dimension.
- 1D convolution slides along the time axis to capture temporal patterns.

## Main Contributions

1. **Over 385x speedup on mobile devices**: Achieved **over 385x inference speedup** compared to state-of-the-art 2D CNN models on the Google Pixel 1. This remarkable speedup ratio demonstrates the significant efficiency advantages of 1D temporal convolution on mobile devices.

2. **1D Convolution as an efficient alternative**: Proposes temporal (1D) convolution as an efficient alternative to 2D convolution in KWS, demonstrating that temporal modeling along the time axis may be more efficient than modeling joint time-frequency patterns in the KWS task.

3. **Rare practical latency analysis**: Provides a quantitative latency analysis on actual mobile devices, which is rare in the field. While most KWS studies only report FLOPs or parameter counts, this paper’s end-to-end measurements on real hardware offer valuable practical references for the community.

4. **Accuracy improvement without compromise**: While significantly increasing inference speed, the model surpassed the accuracy of state-of-the-art models on the Google Speech Command dataset. This proves that 1D convolution is not only faster but potentially more effective for the KWS task.

5. **Open-source implementation**: Released a complete implementation including end-to-end training and mobile evaluation pipelines, promoting research on KWS in practical deployment scenarios.

6. **Published at Interspeech 2019**, serving as an important benchmark work for efficient mobile KWS.

## Experimental Results

### Performance and Efficiency Comparison

| Metric | 2D CNN Baseline | 1D Temporal Convolution (This Paper) |
|------|-----------|-------------------|
| Google Speech Commands Accuracy | Baseline | **Surpasses Baseline** |
| Google Pixel 1 Inference Latency | Baseline | **385x+ Speedup** |
| Model Parameters | Larger | Significantly Reduced |

### Key Results
- The 1D Temporal Convolution model achieved real-time inference on the Google Pixel 1 (latency well below the 100ms threshold).
- Accuracy did not decrease; instead, it surpassed baseline models that used deeper 2D CNNs.
- Detailed latency analysis revealed the non-linear relationship between FLOPs and actual inference time.

## Limitations and Future Work

### Technical Limitations
- **Limitations in frequency-domain modeling**: 1D Temporal Convolution treats the frequency dimension as a channel dimension, processing it independently along the time axis. This may be less effective than 2D convolution at capturing **local patterns within the frequency domain** (such as harmonic structures or formant patterns), as 1D convolution cannot directly learn spatial relationships between adjacent frequency bands.
- **Hardware platform specificity**: The evaluation focused on the Google Pixel 1; performance may vary across different mobile platforms (e.g., different Android devices, iOS devices). Differences in CPU/GPU architecture, cache size, and memory bandwidth across chips will affect actual inference speed.
- **Need for architecture tuning**: The choice of the compact ResNet architecture (number of layers, number of channels, kernel size) requires careful tuning to achieve the optimal accuracy-speed trade-off on specific hardware.

### Future Directions
- Evaluate 1D Convolution KWS on more diverse mobile hardware platforms (including different Android/iOS devices, DSPs, NPUs).
- Explore hybrid architectures combining 1D and 2D convolutions—using 2D convolution in key layers to capture frequency-domain patterns, while using 1D convolution in other layers to maintain efficiency.
- Investigate automated architecture search methods to find the optimal 1D convolution configuration given specific hardware constraints.
- Combine 1D Temporal Convolution with attention mechanisms to enhance modeling of critical time steps while maintaining efficiency.
- Explore the efficiency advantages of 1D convolution in other speech tasks (such as Voice Activity Detection, emotion recognition).
