# End-to-end Keyword Spotting using Xception-1d

- **Authors/Affiliations**: Ivan Valles-Perez, Juan Gomez-Sanchis, Marcelino Martinez-Sober, Joan Vila-Frances, Antonio J. Serrano-Lopez, Emilio Soria-Olivas - Intelligent Data Analysis Laboratory (IDAL), University of Valencia
- **Date**: 2021.10
- **Conference**: ESANN 2021 (European Symposium on Artificial Neural Networks)
- **Link**: https://arxiv.org/abs/2110.07498
- **Code**: https://github.com/ivallesp/Xception1d
- **Keywords**: Xception, Depthwise Separable Convolution, Keyword Spotting, End-to-end, Raw Audio, Instance Normalization, TinyML

## Problem Statement

Keyword Spotting (KWS) is the first step in enabling voice interaction for voice assistants (such as Google Assistant, Amazon Alexa, Apple Siri). It requires identifying a limited vocabulary of speech commands with low latency and high accuracy on resource-constrained edge devices. The core challenge currently facing the KWS field is: how to improve recognition accuracy under a limited vocabulary size while ensuring lightweight computation, thereby surpassing human annotation levels.

Existing KWS methods have several limitations: (1) Bidirectional attention models based on RNNs achieve high accuracy but have high computational overhead, making them unsuitable for resource-constrained devices; (2) Gated Convolution LSTM structures increase model complexity; (3) Standard CNN methods have advantages in computational efficiency but their accuracy on complex tasks needs improvement; (4) Transfer learning methods require additional pre-training data and lack flexibility.

The core research question of this paper is: **Can the Xception architecture (depthwise separable convolution), which has achieved outstanding results in the field of computer vision, be effectively adapted to one-dimensional audio processing to achieve end-to-end keyword detection directly from raw audio waveforms, and surpass existing baseline methods and even human annotation levels across multiple task difficulties?**

## Methodology

### Overall Architecture Design

The paper proposes **Xception-1d**, a CNN architecture that adapts the original Xception architecture from two-dimensional image processing to one-dimensional time-series audio processing. Xception-1d takes raw audio waveforms as input directly, without the need for manual extraction of acoustic features such as MFCCs, achieving true end-to-end keyword detection. The architecture consists of three core modules:

### Architecture Details

**Input Features**: A 1-second audio segment, raw waveform sampled at 16 kHz (16,000 samples), requiring no pre-computed features.

**Entry Module (Entry Flow)**: Contains 5 Xception-1d blocks, responsible for compressing the raw audio waveform into a compact intermediate representation. The number of channels increases stepwise: 32 -> 64 -> 128 -> 256 -> 728, while the time dimension decreases stepwise from 16,000 to 256. The convolution strides for each block are 4, 2, 2, 2, 2, respectively, and the kernel sizes decrease stepwise from 9 to 3.

**Middle Module (Middle Flow)**: Contains 8 Xception-1d blocks. Each block uses 3 depthwise separable convolution layers. The number of channels remains 728, and the time dimension remains unchanged at 256 (stride of 1). This module is responsible for learning high-level abstract representations.

**Classification Module (Classification Flow)**: Contains 3 depthwise separable convolution layers with channel numbers of 1024, 1536, and 2048, respectively. After flattening and Dropout (p=0.75), a fully connected layer outputs the classification results.

### Core Technical Improvements

**Depthwise Separable Convolution**: The core computational unit of Xception-1d. Compared to standard convolution, depthwise separable convolution decouples spatial convolution and channel mixing. The computational cost is only 1/(N*S) of that of standard convolution, where N is the number of output channels and S is the depthwise convolution kernel size. This decomposition significantly reduces computational costs while maintaining expressive power.

**Instance Normalization**: Replaces standard Batch Normalization, performing normalization independently for each sample. This choice stems from thespecificity of audio signals—the amplitude and energy distribution vary significantly across different recordings. Instance normalization adapts better to these variations.

**Regularization Strategy**:
- Dropout: A dropout rate of 75% is applied after the last convolutional layer to address potential overfitting caused by the approximately 65,000 parameters in the fully connected layer.
- Weight Decay: Global L2 regularization is applied with a coefficient lambda = 10^{-3}.

### Data Augmentation Methods

The paper proposes an efficient audio data augmentation pipeline that expands the training data volume by a factor of 5. Five augmentation techniques are applied simultaneously with random parameters and intensities to each audio clip:
1. **Resampling**: Stretching/compressing audio length, which also affects pitch.
2. **Saturation**: Applying random amplitude amplification.
3. **Time Shifting**: Translating the audio segment along the time axis.
4. **White Noise Addition**: Adding Gaussian noise with random amplitude.
5. **Pitch Shifting**: Applying pitch distortion transformations.

All augmentations are applied only to the training data, and the transformed data is appended to the original dataset.

### Training Configuration

- Optimizer: Adam
- Initial learning rate: 10^{-4}, halved if validation set performance does not improve for 4 epochs
- Batch size: 32
- Training epochs: 50 epochs, selecting weights corresponding to the best performance on the development set
- Evaluation method: 5 models with different random initializations are trained for each task, and the mean and standard deviation are reported

### Dataset and Task Definition

The **Google TensorFlow Speech Commands** dataset is used, containing 1-second speech command recordings from thousands of non-professional speakers (16 kHz, 16-bit WAV format).

- **V1 Version**: 64,721 audio clips, 30 words
- **V2 Version**: 105,829 audio clips, 35 words (adding visual, follow, learn, forward, backward)

Four evaluation tasks:
1. **35-words-recognition**: Classify all 35 words (most complex task)
2. **20-commands-recognition**: 20 subset words + "unknown" class
3. **10-commands-recognition**: 10 subset words + "unknown" class
4. **left-right-recognition**: Distinguish only between "left" and "right" + "unknown" class

## Main Contributions

1. **Proposed Xception-1d Architecture**: For the first time, the one-dimensional adaptation of the Xception architecture is applied to end-to-end keyword detection. Through the combined design of depthwise separable convolution and instance normalization, accuracy surpassing human annotation levels is achieved without manual features. This contribution demonstrates that successful architectural designs from the field of computer vision can be effectively transferred to speech processing.

2. **Efficient Audio Data Augmentation Methodology**: A combined augmentation strategy containing five transformations is designed to expand the training data volume by a factor of 5. This strategy applies multiple transformations simultaneously with randomized parameter intensities, generating highly variable augmented samples that effectively improve model generalization.

3. **Human Performance Quantitative Baseline**: By having 4 human annotators label approximately 1,000 commands, a human-level baseline is established for the first time on the Google Speech Commands dataset. A Student's t-test (alpha=0.05) is used to verify the statistical significance of the difference between model and human performance.

4. **Open Source and Reproducibility**: The complete code repository (https://github.com/ivallesp/Xception1d) is made public, facilitating benchmark comparisons and method reproduction in subsequent research.

## Experimental Results

### V1 Dataset Results

| Method | 35-words | 20-commands | 10-commands | left-right |
|------|----------|-------------|-------------|------------|
| Andrade et al. [3] | 94.30 | 94.10 | 95.60 | **99.20** |
| McMahan & Rao [7] | 84.35 | 85.52 | - | 95.32 |
| Warden [9] | - | - | 85.40 | - |
| **Xception-1d** | **95.85 +/- 0.12** | **95.89 +/- 0.06** | **97.15 +/- 0.03** | 98.96 +/- 0.09 |
| Human | 94.15 +/- 1.03 | 94.56 +/- 0.98 | 97.22 +/- 0.85 | 99.54 +/- 0.16 |

- Xception-1d **statistically significantly surpasses human performance** on the 35-words and 20-commands tasks (p < 0.05).
- Accuracy exceeds 95% across all four tasks.

### V2 Dataset Results

| Method | 35-words | 20-commands | 10-commands | left-right |
|------|----------|-------------|-------------|------------|
| Andrade et al. [3] | 93.90 | 94.50 | 96.90 | **99.40** |
| Zhang et al. [6] | - | - | 95.40 | - |
| Warden [9] | - | - | 88.20 | - |
| **Xception-1d** | **95.85 +/- 0.16** | **95.96 +/- 0.16** | **97.54 +/- 0.08** | 99.25 +/- 0.07 |
| Human | 94.15 +/- 1.03 | 94.56 +/- 0.98 | 97.22 +/- 0.85 | 99.54 +/- 0.16 |

- On the 35-words task of the V2 version, it reaches **95.85%**, statistically significantly surpassing humans (94.15%, p = 0.015).
- On the 20-commands task, it reaches **95.96%**, also statistically significantly surpassing humans (94.56%, p = 0.027).
- On the 10-commands task, it reaches **97.54%**, with no significant difference from humans (97.22%).
- On the left-right task, it is slightly lower than Andrade et al.'s 99.40% (difference < 0.5%).

### Per-Class Analysis

In the per-class precision and recall analysis of the 35-words task:
- The precision and recall of most categories are between 90% and 100%.
- Confusable word pairs include "three" and "tree", "follow" and "four", "bed" and "bird", etc.
- An interesting finding: In the 35-words multi-class classification task, the per-class performance of "left" and "right" is actually higher than in the binary left-right task. This suggests that auxiliary tasks (larger vocabulary) may help improve feature extraction capabilities for the main task.

## Limitations and Future Work

### Technical Limitations

1. **Lack of Streaming Inference**: Xception-1d takes a whole 1-second audio segment as input and does not consider the streaming/real-time inference requirements in actual deployment. In voice assistants, audio arrives in continuous streams, requiring the model to support incremental processing.
2. **Limited Vocabulary**: Validated only on the 35 words of Google Speech Commands; the scalability to larger vocabularies (e.g., hundreds of keywords) has not been verified.
3. **Fixed Input Length**: The model assumes the input is a fixed 1-second audio segment. Shorter or longer speech commands require additional alignment or truncation processing.
4. **Cost of End-to-End**: Processing raw waveforms (16,000 points) directly instead of compressed acoustic features simplifies the pipeline but increases computational load, especially in the higher layers of the entry flow.

### Experimental Design Shortcomings

1. **Incomplete Baseline Comparison**: Lacks direct comparison with other contemporary end-to-end methods (e.g., SincNet, fine-tuned wav2vec 2.0).
2. **Edge Device Deployment Not Verified**: Although claimed to be suitable for TinyML scenarios, latency, memory usage, and energy consumption were not measured on real edge devices.
3. **Paper Length Constraints**: ESANN conference papers are limited to 5 pages, restricting the depth of experimental analysis, such as the lack of ablation studies.
4. **Insufficient Trade-off Analysis of Model Size vs. Accuracy**: The impact of different model configurations (e.g., reducing the number of middle flow blocks) on accuracy and inference speed was not explored.

### Future Improvement Directions

1. Explore the adaptation of Xception-1d in streaming inference scenarios, such as using causal convolutions and cumulative inference mechanisms.
2. Validate the scalability of the architecture on larger-scale speech command datasets.
3. Combine model compression techniques (e.g., quantization, pruning, knowledge distillation) to enable deployment on edge devices.
4. Study the combination of depthwise separable convolutions and attention mechanisms to improve the modeling of long-range dependencies.
5. Explore multi-task learning, using auxiliary tasks (such as speech activity detection, speaker identification) to improve main task performance.

### Implications for the KWS Field

Xception-1d demonstrates that successful architectural designs in computer vision can be directly applied to audio processing through reasonable dimensional adaptation. Its end-to-end design philosophy (processing raw waveforms directly) simplifies the system pipeline and avoids information loss that may be introduced by manual feature design. The efficiency of depthwise separable convolutions in the audio domain provides a new architectural choice for deploying more powerful KWS models on resource-constrained devices. Furthermore, the finding that auxiliary tasks help improve main task performance provides inspiration for multi-task learning research in the KWS field.
