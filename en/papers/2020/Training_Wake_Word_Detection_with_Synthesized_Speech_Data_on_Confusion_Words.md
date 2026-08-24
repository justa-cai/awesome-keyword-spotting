# Training Wake Word Detection with Synthesized Speech Data on Confusion Words

**Authors/Affiliations**: Yan Jia, Zexin Cai, Murong Ma, Zeqing Zhao, Xuyang Wang, Junjie Wang, Ming Li (Duke Kunshan University, Duke University, AI Lab of Lenovo Research)

**Date**: November 2020 (arXiv:2011.01460)

**Link**: https://arxiv.org/abs/2011.01460

**Keywords**: Keyword Spotting, Multi-Speaker Text-to-Speech, Confusion Words, Wake Word Detection, Data Augmentation, CORAL Loss

## Problem Statement

In practical Keyword Spotting (KWS) applications, confusion words are one of the primary factors causing severe degradation in system performance. Confusion words refer to vocabulary that sounds similar to the target keyword, such as "nǐ hǎo" (hello) and "nǐ hào" (your number), or "xiǎo yì" (Xiao Yi, a voice assistant name) and "xiào yì" (smile) in Chinese. These words are highly similar to the target keyword in acoustic features but are not semantically keywords.

Existing KWS training methods typically treat all non-keywords as a single negative class, lacking targeted strategies for mining hard negative samples. This leads to false alarms when the model encounters non-keywords that are acoustically similar. Especially in complex acoustic environments, the false alarm rate caused by confusion words can reach up to 100%, severely impacting user experience.

## Methodology

### Overall Framework

This paper proposes two complementary data augmentation strategies combined with a domain adaptation loss function to improve the robustness of end-to-end KWS systems against confusion words:

### 1. Masked Audio Augmentation

Inspired by the masked training concept in face recognition:
- Random masking is applied to positive sample audio: 40%~60% of the audio signal is replaced with Gaussian white noise.
- The masked audio loses the complete acoustic information of the keyword but retains some spectral features.
- These processed samples are labeled as negative samples (non-keywords) to train the model to distinguish incomplete keywords from true keywords.
- The core intuition of this method is to force the model to not only rely on partial features in the audio but to understand the complete acoustic pattern of the keyword.

### 2. TTS Adversarial Augmentation

The core innovation of this paper:
- Adversarial samples of confusion words are generated using a multi-speaker Text-to-Speech (TTS) system.
- Specific implementation: A speaker-embedding-conditioned TTS system is used, employing speaker embedding vectors extracted from 10,000 different speakers to control the speaker characteristics of the synthesized speech.
- The text of the target confusion words is input into the TTS system to generate a large number of confusion word utterances from different speakers as negative samples.
- These TTS-generated samples are highly consistent with real confusion words in acoustic features, providing high-quality hard negatives.

### 3. CORAL Loss (Correlation Alignment Loss)

To reduce the distributional discrepancy between synthesized data (source domain) and real speech (target domain):
- The CORAL (CORrelation ALignment) loss function is used.
- CORAL loss achieves domain alignment by minimizing the difference in second-order statistics (covariance matrices) between the source and target domain features.
- Mathematical form: Minimizing the Frobenius norm distance between the covariance matrix of source domain features and the covariance matrix of target domain features.
- In TTS-enhanced training, the CORAL loss ensures that the features learned by the model on synthetic data are consistent with the feature distribution of real speech.

### Baseline Model Architecture

- CNN-based end-to-end KWS model.
- Input features: 80-dimensional Log-Mel Filterbank features.
- Multi-layer convolution + pooling structure, followed by a fully connected classification layer.

## Main Contributions

1. **First use of multi-speaker TTS for KWS confusion word adversarial training**: Utilizes the TTS system to generate large-scale, high-quality confusion word samples, solving the problem of insufficient hard negative samples in traditional methods. The TTS system can precisely control the content and speaker characteristics of the generated words, providing ideal adversarial training data.

2. **Masked Audio Augmentation Strategy**: Introduces the masked training concept from the face recognition field to KWS; it is simple, effective, and requires no additional resources.

3. **Integrated Application of CORAL Loss**: Introduces domain adaptation technology in the TTS-enhanced scenario, effectively bridging the domain gap between synthetic speech and real speech.

4. **Significant Practical Value**: The false alarm rate is reduced from 100% to 0.083%; this improvement is decisive for the usability of actual product systems.

## Experimental Results

### Experimental Setup
- Chinese wake word dataset, containing target keywords and confusion words.
- Baseline model: CNN end-to-end KWS model with 80-dimensional Log-Mel feature input.
- Evaluation metric: False Alarm Rate in the confusion word scenario.

### Key Results
- **Baseline Performance**: The model without augmentation had a false alarm rate of up to 100% in the confusion word scenario, meaning all confusion words were misclassified as keywords.
- **Masked Audio Augmentation**: Significantly reduced the false alarm rate, but was still not ideal when used alone.
- **TTS Augmentation**: Reduced the false alarm rate from 100% to a very low level (approximately 0.083%), with effects far surpassing masked audio augmentation.
- **TTS + CORAL**: The combination of CORAL loss and TTS augmentation further improved robustness, validating the effectiveness of the domain adaptation strategy.

### Analysis
- TTS augmentation is effective because the generated confusion word samples are acoustically highly realistic, forcing the model to learn more fine-grained discriminative features.
- Although simple, masked augmentation plays a complementary role in supplementing TTS augmentation.
- CORAL loss ensures that the model does not overfit to the specific distribution of synthetic speech.

## Limitations and Future Work

### Method Limitations
- **Language Dependency**: Validated only on Chinese wake words; the effect of TTS augmentation on other languages (especially low-resource languages) needs further study.
- **TTS System Dependency**: Requires a high-quality multi-speaker TTS system, which may not be available in certain languages and scenarios.
- **Definition of Confusion Words**: In practical applications, there may be undefined confusion word patterns; the generalization ability of the method to these unseen patterns is limited.
- **Architecture Limitations**: Validated only on CNN architectures; applicability to other KWS architectures (e.g., RNN, Transformer) has not been verified.

### Future Directions
- Explore automatic confusion word discovery mechanisms that do not require manually predefined confusion word lists.
- Research end-to-end adversarial training methods to automatically generate adversarial samples rather than relying on TTS.
- Extend this method to multi-lingual and cross-lingual scenarios.
- Combine with Active Learning strategies to continuously discover and address new confusion word patterns.
