# BBS-KWS: The Mandarin Keyword Spotting System Won the Video Keyword Wakeup Challenge

- **Authors/Affiliations**: Yuxuan Wang, Zhengdong Wang, Qiang Huo et al. - NetEase Yidun NISP Team
- **Date**: 2021.12
- **Link**: https://arxiv.org/abs/2112.01757
- **Keywords**: Chinese Keyword Spotting, Video Keyword Wakeup, BBS-KWS, CNN, Attention Mechanism, Transfer Learning, Data Augmentation, Challenge

## Problem Statement

With the rapid development of short-video and live-streaming platforms, keyword wakeup detection in video content has become an important technical application scenario. Video keyword wakeup requires accurately detecting preset specific keywords (such as brand names, sensitive words, prohibited terms, etc.) from continuous audio streams, while maintaining an extremely low false alarm rate. Unlike traditional smart speaker wakeup word detection, video scenarios face unique challenges:

1. **Extremely Complex Acoustic Environment**: Video content contains various interference factors such as background music, environmental noise, overlapping speakers, and sound effects. The diversity and intensity of these interferences far exceed those in home environments. Background music may contain segments with phonemes similar to the target keywords, and overlapping speakers can severely mask the speech of the target speaker.

2. **High Speaker Diversity**: Speakers of different genders, ages, accents, and dialect backgrounds use the same keywords, resulting in significant differences in acoustic characteristics. Speakers in video content may come from all over the country, with a wide variety of dialects and accents.

3. **High Cost of False Alarms**: In content moderation scenarios, false alarms lead to a large amount of manual review work (wasting human resources), while missed detections can cause compliance risks and legal consequences. Therefore, the system needs to maintain an extremely high recall rate while controlling the false alarm rate at a very low level.

4. **Specificity of the Chinese Language**: Chinese is a tonal language, where different tones of the same syllable represent different meanings (e.g., the four tones of "ma" correspond to "妈" (mother), "麻" (hemp), "马" (horse), and "骂" (scold)). The model needs to be highly sensitive to tonal variations. Additionally, the syllable structure of Chinese (initial consonant + final vowel + tone) is fundamentally different from languages like English, so directly applying English KWS experience may yield poor results.

5. **Real-time Requirements**: Video content moderation requires detection to be completed quickly after content publication, imposing strict requirements on system latency and throughput.

The BBS-KWS system introduced in this paper is designed for the Video Keyword Wakeup Challenge, with the goal of achieving the highest detection accuracy and recall rate on the Chinese dataset provided by the competition, while maintaining feasibility for practical deployment.

## Methodology

### Overall System Architecture
BBS-KWS adopts a multi-module integrated system architecture, with core components including: an acoustic feature extraction frontend, a CNN backbone network with attention mechanisms, a transfer learning module based on large-scale ASR models, a multi-scale feature fusion layer, and a two-stage detection pipeline (coarse screening + fine ranking). This multi-module integration design strategy is common in competition scenarios, maximizing system performance through the careful combination of various complementary technologies.

### Backbone CNN Architecture and Attention Mechanism
The system uses Depthwise Separable Convolution as the basic building block, combined with Squeeze-and-Excitation (SE) attention and temporal attention mechanisms:

- **Depthwise Separable Convolution**: Decomposes standard convolution into depthwise convolution (independent convolution for each input channel) and pointwise convolution (1x1 convolution to mix channel information), significantly reducing the number of parameters and computational load. While maintaining expressive power, it makes the model more suitable for efficient inference on the server side.

- **SE (Squeeze-and-Excitation) Channel Attention**: Learns channel-dimension attention weights through a path of global average pooling -> fully connected dimensionality reduction (compression ratio r=16) -> ReLU -> fully connected dimensionality expansion -> Sigmoid. The SE module can adaptively enhance discriminative feature channels (such as frequency channels related to target keyword phonemes) and suppress noise or irrelevant channels. In KWS, different keywords activate different frequency channel patterns, and SE attention dynamically adjusts channel weights based on the input.

- **Temporal Attention**: Applies attention weights in the time dimension, enabling the model to focus on key time segments of the keyword. For Chinese keywords, tonal information is mainly reflected in the time trajectory of the fundamental frequency (F0). Temporal attention helps the model focus on time periods containing key tonal changes. The implementation involves: aggregation along the channel dimension -> FC layer -> Softmax -> time-step weights.

### Transfer Learning Strategy
Transfer learning from large-scale Chinese Automatic Speech Recognition (ASR) models is a key technical decision in BBS-KWS and the main source of its performance advantage:

- **Pre-trained Model Selection**: Uses an ASR encoder (based on Conformer or Transformer architecture) trained on large-scale Chinese speech data (thousands of hours) as the initialization for the feature extractor. The ASR encoder has learned rich acoustic feature representations of Chinese, including phoneme-level acoustic patterns, tonal features, and speaker-independent speech representations.

- **Fine-tuning Strategy**: Employs a fine-tuning approach with hierarchical learning rates—the lower-level feature extraction layers use smaller learning rates (to preserve general acoustic knowledge learned by ASR), while higher-level task-specific layers use larger learning rates (to rapidly adapt to the KWS task). The progressive unfreezing strategy, which freezes the lower layers (1-4), fine-tunes the middle layers (5-8), and fine-tunes the top layers (9-12 + classification head), achieved the best results on the validation set.

- **Domain Adaptation**: Further fine-tunes on the target domain data provided by the challenge to reduce the distribution difference between training data and the actual deployment environment. Uses domain mixup strategies (mixing samples from source and target domains) to further improve domain adaptation effects.

### Data Augmentation Strategy
To address insufficient training data and acoustic environment diversity, BBS-KWS employs a rich and carefully tuned data augmentation pipeline:

- **Speed Perturbation**: Resamples training audio at speeds of 0.9x, 1.0x, and 1.1x to increase diversity in the time dimension. Speed changes not only alter duration but also affect tonal frequency, helping the model learn robustness to speech rate variations.

- **Volume Perturbation**: Randomly adjusts the audio volume level (+-6dB) to enhance robustness to volume changes. Speaker volume differences in video content can be significant.

- **Noise Injection**: Randomly selects background noise from a noise library (containing music noise, environmental noise, babble, etc.) and overlays it on the training audio, with SNR ranging from -5dB to 20dB. The diversity of noise types is particularly important for video scenarios.

- **Room Impulse Response (RIR) Simulation**: Simulates reverberation effects in different rooms (T60 from 0.1s to 0.8s) to help the model handle reverberation in different recording environments.

- **SpecAugment**: Applies time masking (up to 20 frames) and frequency masking (up to 8 frequency bins) on the spectrogram to further improve generalization.

- **Time Shift**: Randomly shifts the starting position of the audio within +-100ms to simulate variations in the position of keywords within audio segments.

### Multi-scale Feature Fusion
The model extracts multi-scale features from different layers of the CNN and fuses them:

- **Shallow Features (Layer 2-3)**: Capture basic time-frequency patterns, such as phoneme-level spectral features and short-term energy changes. These features are relatively robust to noise but contain less semantic information.
- **Middle Features (Layer 5-6)**: Capture medium-level acoustic patterns, such as syllable-level structures and tonal change trajectories.
- **Deep Features (Layer 8-10)**: Capture high-level semantic information, such as word-level and phrase-level representations. These features are semantically rich but more sensitive to noise.

The fusion method involves upsampling (to match spatial resolution) and concatenation to fuse feature maps of different resolutions into a unified representation, followed by dimensionality reduction via 1x1 convolution. Multi-scale fusion allows the model to simultaneously utilize fine-grained acoustic patterns (helpful for distinguishing acoustically similar keywords) and coarse-grained semantic information (helpful for overall keyword recognition).

### Two-Stage Detection Pipeline
To effectively control the false alarm rate while maintaining high recall, BBS-KWS adopts a two-stage detection approach:

- **Stage 1 (Coarse Screening)**: Uses a computationally lighter model to quickly scan the entire audio stream and filter out candidate segments that may contain keywords. The threshold in this stage is set towards high recall (better to have false positives than miss detections), allowing some false alarms to pass.
- **Stage 2 (Fine Verification)**: Uses a more precise (and larger) model to perform secondary verification on candidate segments, filtering out false alarms through stricter thresholds. The fine-ranking model typically uses more attention layers and larger feature dimensions.

The advantage of the two-stage pipeline is that the fast filtering in the first stage significantly reduces the amount of audio entering the second stage (usually only 5-10% of the total audio needs to be processed), thereby maintaining high overall computational efficiency.

## Main Contributions

1. **Built a keyword detection system that won the Video Keyword Wakeup Challenge**: Under the strict evaluation criteria of the competition, the BBS-KWS system ranked first with optimal comprehensive performance (balance of accuracy and recall). This result proves the effectiveness of multi-module integration and careful engineering optimization in competition scenarios.

2. **Proved the effectiveness of transfer learning from large-scale ASR pre-trained models to KWS tasks**: ASR pre-training provides high-quality acoustic feature initialization for KWS tasks. Even for models trained on different tasks, the learned acoustic feature representations exhibit high universality and transferability. This finding is significant for Chinese KWS tasks with limited data.

3. **Proposed a multi-scale feature fusion method to improve KWS accuracy**: By combining CNN features from different levels, the model can simultaneously utilize fine-grained acoustic patterns and coarse-grained semantic information, effectively solving the problem of insufficient expressive power of single-scale features.

4. **Designed a two-stage detection pipeline**: Effectively controls the false alarm rate while maintaining high recall, making it suitable for practical application scenarios such as content moderation that are sensitive to false alarms. This design paradigm offers reference value for the deployment of KWS in industrial scenarios.

5. **Summarized technical experience for Chinese KWS**: Systematically explored best practices for data augmentation, attention mechanisms, and transfer learning in Chinese KWS, providing valuable engineering experience for the Chinese KWS community.

## Experimental Results

### Challenge Performance
- **Ranked first in the Video Keyword Wakeup Challenge**, with a comprehensive score significantly higher than other participating systems.
- Significantly outperformed other participating systems on the official evaluation metrics of the competition (comprehensive score combining accuracy and recall, as well as recall at specific false alarm rates).

### Comparison with Baseline Systems
- Compared to the official baseline system (based on standard DS-CNN), detection accuracy was significantly improved (an absolute improvement of about 10-15%).
- While maintaining an extremely low false alarm rate, recall was significantly improved (an increase of about 8-12%).
- Compared to single-stage methods, the two-stage pipeline reduced the false alarm rate by an order of magnitude (from about 5% to about 0.5%), while recall decreased only by about 1-2%.

### Ablation Study Findings
- **Transfer learning contributed the most**: After removing ASR pre-training initialization, system performance dropped significantly by about 5-8% in absolute accuracy. This indicates that acoustic knowledge learned from large-scale ASR pre-training is crucial for KWS performance.
- **Necessity of data augmentation**: After removing data augmentation, model performance on the test set dropped by about 3-5%, with a severe decline in robustness to noisy audio and background music scenarios.
- **Gain from multi-scale fusion**: Performance showed an observable decline (about 1-3%) when using only single-scale features (especially only shallow features), with multi-scale fusion providing complementary information gains.
- **Effect of attention mechanisms**: SE attention brought a 1-2% accuracy improvement, and temporal attention provided an additional 0.5-1% improvement, with complementary effects between the two.
- **Two-stage vs. Single-stage**: The two-stage pipeline significantly outperformed the single-stage approach in terms of false alarm rate (reducing false alarms by about 80-90%), but increased inference time by about 50% (because the model needs to be run twice).

### Robustness Analysis
- The system maintained stable performance under various acoustic conditions (quiet, noisy, background music), with performance degradation in noisy scenarios controlled within 3%.
- Showed good generalization ability for keyword detection across different speakers (gender, age, accent), with performance differences between male and female speakers less than 1%.
- Maintained stable detection performance across different speech rates (0.8x to 1.2x of normal speed).

## Limitations and Future Work

### Technical Limitations
- **High System Complexity**: Multi-module integration (CNN backbone + attention + transfer learning + multi-scale fusion + two-stage detection) makes the system complexity far exceed that of simple KWS models. Developing and maintaining multiple components requires significant engineering effort. The two-stage pipeline also requires careful tuning of thresholds for both stages to balance recall and false alarm rates.
- **Focus on Chinese**: The system is optimized for Chinese acoustic characteristics (especially the tonal system) and has not been validated for generalization to other languages (especially non-tonal languages). The ASR model used for transfer learning is also specific to Chinese.
- **Challenge-Specific**: Some design decisions (such as specific combinations of data augmentation, threshold settings, model size) may be optimized for the competition dataset and may not apply to all KWS scenarios. The number and types of keywords in the competition dataset may not represent all practical application scenarios.
- **Dependency on ASR Pre-trained Models**: System performance heavily relies on the availability of high-quality Chinese ASR pre-trained models. For languages lacking large-scale ASR pre-trained models, the effectiveness of this method may be limited.

### Experimental Design Shortcomings
- Insufficient discussion on deployment feasibility on edge devices (computational requirements, latency, power consumption), as the paper focuses on server-side deployment scenarios.
- Lack of comparison with end-to-end KWS methods (such as RNN-T, CTC-based, attention-based end-to-end models), making it difficult to judge the pros and cons of multi-module integration compared to end-to-end methods.
- Lack of detailed analysis of the computational overhead of each system module, making it difficult to evaluate which components have the highest computational return on investment.
- Did not analyze the scalability of the system with respect to the number of target keywords.

### Future Improvement Directions
- Simplify the system into a more compact end-to-end architecture to reduce deployment and maintenance complexity. Explore whether single-stage models can directly achieve the performance of two-stage systems through more powerful backbone networks.
- Explore online learning and continuous learning mechanisms to support the dynamic addition of new keywords without retraining the entire system.
- Cross-lingual validation and adaptation to enable multi-language keyword detection capabilities. Specifically, explore multilingual pre-trained models (such as XLSR) as sources for transfer learning in Chinese KWS.
- Explore pre-training methods based on contrastive learning or self-supervised learning to reduce dependence on large-scale ASR labeled data.
- **Insights for the KWS Field**: Although competition systems achieve optimal performance, their core technologies (such as transfer learning, data augmentation, attention mechanisms, multi-scale fusion) can be distilled into general best practices for application in production systems. In particular, the paradigm of transferring learning from large-scale ASR models to KWS is worth promoting and applying in more languages and scenarios.
