# Behavior of Keyword Spotting Networks Under Noisy Conditions

- **Authors/Affiliations**: Vishal Passricha, Jan-Tobias Sohns, Fabien Jolivaldt, Claudio Greco, Michael Gref, Arjun Sharma - Indian Institute of Technology; University of Tübingen; RWTH Aachen University
- **Date**: 2021.09
- **Link**: https://arxiv.org/abs/2109.07930
- **Keywords**: Noise Robustness, Keyword Spotting, Data Augmentation, Model Analysis, SNR, Environmental Noise, Multi-Condition Training

## Problem Statement

Keyword Spotting (KWS) systems must face various noisy environments during practical deployment: television and air conditioning sounds in home environments, traffic noise outdoors, and human chatter in offices. Understanding the behavioral patterns of KWS networks under different noise types and intensities is crucial for designing robust KWS systems.

Although there is extensive research on noise-robust KWS, most works focus on proposing new methods, lacking a systematic analysis of the noise behavior of existing KWS architectures. Specifically, the following issues have not been sufficiently studied:

1. **Differences in the impact of different noise types on KWS**: Do different types of noise, such as human chatter, traffic noise, and music noise, cause equal degrees of performance degradation in KWS?
2. **Relationship between architecture selection and noise robustness**: Are there systematic differences in noise robustness between CNN and RNN architectures?
3. **Impact of training strategies**: To what extent can multi-condition training and noise data augmentation mitigate noise interference?
4. **Internal behavior of models under noisy conditions**: How do the internal representations and decision boundaries of KWS networks change when receiving noisy inputs?

This paper provides a systematic experimental analysis of these questions.

## Methodology

### Experimental Design Framework
The paper adopts a systematic experimental analysis approach, with controlled variables including:
- **Noise Type**: Different categories of environmental noise
- **Signal-to-Noise Ratio (SNR) Levels**: Different noise intensities from clean to -5dB
- **Model Architecture**: CNN series and RNN series
- **Training Strategy**: Clean data training vs. Multi-condition training

### KWS Architectures Tested
The paper evaluates various mainstream KWS architectures:
- **CNN Series**: Including standard CNNs, Depthwise Separable CNNs (DS-CNNs), and variants with different depths and widths
- **RNN Series**: Including recurrent network variants such as LSTM and GRU
- **Hybrid Architectures**: CNN+RNN combined models

### Noise Condition Settings
- **Noise Types**:
  - Environmental noise (wind, rain, air conditioning)
  - Traffic noise (cars, trains, airplanes)
  - Music noise (different types of background music)
  - Human chatter (babble noise, multi-speaker background noise)
  - White and pink noise (synthetic noise)
- **SNR Levels**: 20dB (slight noise), 10dB (moderate noise), 0dB (strong noise), -5dB (very strong noise)
- **Noise Injection Method**: Noise is superimposed onto clean test speech at the specified SNR during testing

### Comparison of Training Strategies
- **Clean Training**: Trained only on original clean data
- **Multi-Condition Training**: Various noises are superimposed on the training data, allowing the model to be exposed to noisy samples during training
- **Data Augmentation Training**: Using frequency-domain augmentation strategies such as SpecAugment

### Analysis Methods
- **Accuracy vs. SNR Curves**: Plotting curves of accuracy changes with SNR under different architectures and noise types
- **Confusion Matrix Analysis**: Analyzing classification error patterns under noisy conditions
- **t-SNE Visualization**: Visualizing the distribution changes of intermediate layer features of the model under noisy conditions
- **Decision Boundary Analysis**: Analyzing how noise affects the model's classification decision boundaries

## Main Contributions

1. **Provided a comprehensive systematic analysis of the noise robustness of KWS networks**: This work is one of the most comprehensive analyses of KWS model noise behavior as of 2021, covering multiple architectures, multiple noise types, and multiple SNR levels.
2. **Identified the most challenging noise types**: It was found that babble noise is the most destructive noise type for KWS systems because its spectral characteristics highly overlap with target speech, making it difficult for the model to separate.
3. **Revealed the relationship between architecture selection and noise robustness**: CNN models exhibited better robustness than RNN models under most noise conditions, possibly because CNN's local feature extraction is more stable in noise.
4. **Quantified the benefits of multi-condition training**: Multi-condition training provided consistent performance improvements across almost all noise types and SNR levels, validating its effectiveness as a standard KWS training strategy.

## Experimental Results

### Datasets
- **Google Speech Commands (GSC) v2**: 12-class and 35-class tasks
- **Noise Data**: From standard noise libraries such as DEMAND and MUSAN

### Impact of Different Noise Types
- **Babble Noise** is the most challenging: Even at 10dB SNR, the accuracy of all architectures decreased significantly (15-30%)
- **Music Noise**: Moderate impact, with accuracy decreasing by approximately 5-15%
- **Environmental Noise** (wind, rain): Relatively small impact, with a decrease of approximately 3-10%
- **White Noise**: Surprisingly, the impact of white noise was less than that of babble noise because its spectral characteristics overlap less with speech

### Performance Changes at Different SNR Levels
- **20dB SNR**: Most architectures showed slight performance degradation (1-3%), close to clean conditions
- **10dB SNR**: CNN architectures decreased by approximately 5-10%, RNN architectures decreased by approximately 10-15%
- **0dB SNR**: All architectures suffered severe performance degradation; CNN decreased by approximately 15-25%, RNN by approximately 20-35%
- **-5dB SNR**: The accuracy of almost all architectures dropped to unusable levels (close to random guessing)

### CNN vs RNN
- **CNN is more robust under noisy conditions**: Across all SNR levels, the performance degradation of the CNN series was smaller than that of the RNN series
- **Possible reasons**: The local feature extraction and translation invariance of CNN make it more stable in noisy environments; the sequence modeling of RNN may be more susceptible to the time-accumulation effect of noise interference

### Effect of Multi-Condition Training
- Multi-condition training provided significant performance recovery at SNR levels of 0dB and above (accuracy increased by 5-15%)
- At -5dB SNR, the improvement from multi-condition training was limited (still far below clean conditions)
- The effect of SpecAugment data augmentation was complementary to multi-condition training, with the best results achieved when both were combined

### t-SNE Analysis
- Noise caused significant changes in the feature distribution of different classes, making the boundaries between classes blurrier
- The change in feature distribution caused by babble noise was the most severe
- Models after multi-condition training maintained better class separation under noisy conditions

## Limitations and Future Work

### Limitations of the Analysis
- **Limited to the GSC dataset**: Google Speech Commands is an isolated word dataset, which differs from KWS scenarios in continuous audio streams
- **No new architecture proposed**: This is an analytical study and does not propose improved noise-robust KWS architectures
- **Limited noise types**: Although major noise types were covered, some noises common in practical scenarios (such as echo, reverberation, and nonlinear distortion) were not included
- **Advanced front-end processing not explored**: Such as speech enhancement front-ends, adaptive noise cancellation, etc.

### Guidance for Practice
- Noise robustness testing should focus on testing babble noise conditions
- Multi-condition training should be adopted as a standard practice for KWS model training
- CNN architectures should be prioritized in noise-sensitive applications
- Implications for the KWS field: Although systematic analysis does not propose new methods, it provides valuable design guidelines and experimental benchmarks for the community
