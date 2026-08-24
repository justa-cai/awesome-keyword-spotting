# DONUT: CTC-based Query-by-Example Keyword Spotting

- **Authors/Affiliations**: Loren Lugosch, Samuel Myer, Vikrant Singh Tomar (Fluent.ai)
- **Date**: 2018.11 (NeurIPS 2018 Workshop, arXiv:1811.10736)
- **Link**: https://arxiv.org/abs/1811.10736
- **Keywords**: CTC, query-by-example, keyword spotting, custom wake word, embedded systems, zero-shot learning

## Problem Statement

Modern voice assistants (Siri, Alexa, Google Assistant) all use preset, fixed wake words (e.g., "Hey Siri", "Alexa", "OK Google"). However, users may want to set personalized wake words for different devices—for example, waking their phone with their own name, or waking their car with a specific phrase. This demand for "custom wake words" faces a fundamental technical challenge.

**Pain points in the field**
- Neural network-based keyword spotting methods typically require collecting large amounts of training data and retraining the model for every new keyword, which is unrealistic for ordinary users
- DTW-based template matching methods do not require retraining, but they are hard to interpret and debug (why was a particular sample matched or rejected?), and their performance is limited by hand-engineered features
- For embedded devices (smart watches, IoT sensors), computational resources and storage are extremely limited, making it impossible to run a full ASR system

**Key challenges this paper aims to solve**
- How to let users define a new wake word by providing only a few audio samples (rather than text transcriptions)
- How to adapt quickly to a new keyword without retraining the network
- How to keep the detection process interpretable (users can understand why the system made a particular decision)
- How to achieve all of the above goals on embedded devices

## Methodology

### Overall Architecture Design

DONUT (Dynamic Online UNsupervised Training) combines CTC-based keyword spotting with the user convenience of Query-by-Example (QbE). The core idea is: use a pretrained phoneme recognition model as a universal acoustic analyzer, map the user-provided audio samples to phoneme label sequences via the CTC mechanism, and then search for matching label sequences in new audio.

**Stage 1: Pretrain a phoneme recognition model**
- Train a phoneme-level sequence-to-sequence model on a large-scale speech dataset
- Input: acoustic features (e.g., MFCC or log-mel spectrograms)
- Output: phoneme posterior probabilities for each frame (trained with the CTC loss)
- The model learns a general-purpose phoneme recognition capability, not limited to any specific keyword
- This model serves as "train once, use forever" infrastructure

**Stage 2: The user enrolls a new keyword**
1. The user records a small number (e.g., 1-5) of training samples, speaking the wake word they wish to set
2. Each training sample is fed into the pretrained phoneme recognition model
3. Greedy decoding or beam search is used to obtain the phoneme label sequence for each sample
4. The phoneme label sequences from all samples are collected to form a candidate label set $H = \{h_1, h_2, ..., h_K\}$
5. Advantage of audio-form input: the samples provided by the user directly reflect their own accent, speaking rate, and pronunciation habits

**Stage 3: Online detection on new audio**
1. New audio frames are fed into the pretrained model in real time
2. For each hypothesized label sequence $h_k$ in the candidate label set, compute the CTC forward probability $P(h_k | x_{1:t})$
3. Aggregate the CTC scores of all hypotheses: $s(t) = \sum_{k=1}^{K} P(h_k | x_{1:t})$
4. When the aggregated score exceeds a threshold, a keyword detection is triggered

### Mathematical Principle of CTC Score Aggregation

When the CTC forward algorithm computes the probability of observing a given phoneme label sequence, it considers all possible alignments (through the insertion of blank labels). For keyword spotting, this means the system does not need to know the exact location of the keyword in the audio—CTC handles the alignment problem automatically.

Aggregating the scores of all candidate hypotheses provides robustness:
- A single user sample may produce imperfect phoneme labels
- The label hypotheses from multiple samples complement each other, covering the natural variation in the keyword's pronunciation
- If a hypothesis has a high CTC score on the audio, it indicates that the audio contains a phoneme sequence similar to the corresponding sample

### Interpretability Design
- The label-sequence output of CTC provides an explicit "intermediate representation"—users can see the phoneme sequence the system understood
- If a detection fails, the user can check whether the phoneme labels are correct and re-record better samples
- Compared with a "black-box" neural network, this offers a significant advantage for debugging

## Main Contributions

1. **Retraining-free custom wake words**: Once DONUT's pretrained model has been trained, any new keyword can be added through a simple audio enrollment procedure, without any additional network training or fine-tuning. This fundamentally changes the deployment model of keyword spotting systems.

2. **Interpretable detection process**: With CTC label sequences as an intermediate representation, users can understand and debug the system's behavior. When a detection fails, it can be traced back to phoneme recognition errors or enrollment sample quality issues, rather than facing an unexplainable black box.

3. **Audio-form user enrollment**: Users provide keyword examples as audio rather than text, which ensures the enrollment samples exactly match the user's own pronunciation habits and avoids errors that G2P (grapheme-to-phoneme) conversion could introduce.

4. **Privacy protection**: All processing (enrollment and detection) is done locally on the device, with no need to upload the user's speech data to the cloud. This is especially important for privacy-conscious smart home and wearable devices.

5. **Low computational requirements**: The detection process involves only forward inference with the pretrained model and CTC score aggregation, with no backpropagation or model updates, making it well suited for embedded systems.

## Experimental Results

### Datasets
- Google Speech Commands dataset (used to validate generality)
- Fluent.ai internal dataset (used to test embedded device scenarios)

### Key Performance Metrics
- Users can enroll a new keyword with only a few (1-5) audio samples
- Both learning and inference have low computational requirements, suitable for embedded devices
- CTC score aggregation provides more robust detection than any single hypothesis
- The effectiveness of the method was validated on multiple keywords

### Analysis of Method Characteristics
- **Enrollment speed**: Enrolling a new keyword takes only a few seconds (recording samples + phoneme decoding)
- **Inference efficiency**: Each frame requires only one forward inference pass + CTC score computation
- **Scalability**: Adding more keywords only increases the size of the candidate label set, not the model complexity
- **Robustness**: The hypotheses from multiple enrollment samples complement each other, covering pronunciation variation

## Limitations and Future Work

### Technical Limitations of the Method
- **Dependence on phoneme recognition quality**: System performance is limited by the quality of the pretrained phoneme recognition model. If the model has a high phoneme error rate on certain speakers or accents, the generated label hypotheses will contain errors, directly hurting detection performance.
- **CTC spike behavior**: CTC output exhibits a "spike" pattern—most frames output the blank label, and only a few frames output actual phonemes. This means that if some phonemes of the keyword are missed between spikes, it may lead to missed detections.
- **Sensitivity to sample quality**: Performance depends on the number and quality of the enrollment samples provided by the user. Samples recorded in noisy environments reduce detection accuracy.
- **Insufficient quantitative comparison**: The paper does not provide detailed quantitative ROC curve comparisons with other QbE methods (such as DTW-based methods).

### Shortcomings in Experimental Design
- As a NeurIPS Workshop paper, the experimental section is relatively brief
- Lacks systematic evaluation under difficult conditions (noise, far-field)
- Does not analyze the performance curve as a function of the number of enrollment samples

### Future Improvement Directions
- Replacing CTC with an attention mechanism might handle keyword localization better
- Introducing speaker adaptation mechanisms to improve robustness across different speakers
- Exploring multilingual phoneme sets to support cross-lingual custom wake words
- Combining endpoint detection to build a fully self-contained wake word system

### Insights for the KWS Field
- Custom wake words are an important direction for personalizing voice assistants, and DONUT provides a clean and elegant solution to this problem
- The paradigm of "pretrain a general-purpose model + lightweight enrollment" can be extended to other personalized speech tasks
- The idea of using CTC as an intermediate representation inspired subsequent CTC-based end-to-end KWS methods
- Interpretability has significant value in real-world deployment, helping with user trust and system debugging
- The privacy-preserving design philosophy is highly aligned with privacy-computing trends such as federated learning
