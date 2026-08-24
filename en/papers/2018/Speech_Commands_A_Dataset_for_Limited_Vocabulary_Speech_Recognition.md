# Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition

- **Authors/Affiliations**: Pete Warden (Google Brain)
- **Date**: 2018.04 (arXiv:1804.03209)
- **Link**: https://arxiv.org/abs/1804.03209
- **Keywords**: speech dataset, keyword spotting, limited vocabulary, Google Speech Commands, benchmark evaluation

## Problem Statement

Speech recognition research has long been dominated by a handful of large tech companies that possess vast amounts of labeled speech data. For academia and independent researchers, the lack of publicly available, well-structured speech datasets has been a major obstacle to conducting keyword spotting (KWS) research.

**Pain points in the field**
- Most high-quality speech datasets (e.g., Switchboard, Fisher) either require paid licenses, restrict commercial use, or are too small
- Existing speech datasets are typically designed for large-vocabulary continuous speech recognition (LVCSR) and contain long conversations, rather than the short command words needed for keyword spotting
- Different research teams evaluate on their own private datasets, making experimental results impossible to compare fairly
- Academic researchers need a standardized small-vocabulary speech recognition dataset to quickly iterate on and validate new methods

**Key challenges this paper aims to solve**
- Create a public dataset designed specifically for the keyword spotting task
- Establish a standardized evaluation methodology so that different methods can be compared fairly
- Release the dataset under an open license that supports both academic and commercial use
- Provide baseline model results to lower the entry barrier for new researchers

## Methodology

### Dataset Design Principles

**Purpose-built design for keyword spotting**
Compared with traditional ASR datasets (e.g., LibriSpeech, Switchboard), the Speech Commands dataset is designed specifically for keyword spotting scenarios:
- **Short audio clips**: each utterance is about one second long, corresponding to the pronunciation of a single word
- **Fixed vocabulary**: 30 target words covering common command and control words
- **Large number of speakers**: thousands of different speakers, providing rich speaker variability
- **Diverse recording conditions**: recorded with different devices (phones, laptops) in different environments

### Data Collection Process

**Crowdsourced collection**
- Volunteers were recruited through Google's crowdsourcing platform
- Volunteers recorded through a web browser, using the Web Audio API to capture audio
- Each volunteer recorded multiple utterances of different target words

**Target word selection**
The 30 target words fall into the following categories:
- **Command/control words**: "yes", "no", "up", "down", "left", "right", "forward", "backward", "go", "stop", "on", "off"
- **Digits**: "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
- **Auxiliary words**: "zero", "learn", "visual", "follow", "tree", "bed", "cat", "dog", "bird", "happy", "house", "Marvin", "Sheila", "wow"
- **Functional words**: "no", "yes", "on", "off"

The target words were selected with the following considerations:
- Common commands in real-world voice interaction
- Phonemic diversity (covering the major phonemes of English)
- Variation in word length (short words and long words)

### Data Validation and Quality Control

1. **Automatic validation**: check basic metrics such as audio length, energy level, and signal-to-noise ratio
2. **Manual review**: suspicious samples undergo human inspection
3. **Speaker deduplication**: ensure that no single speaker's utterances are over-represented
4. **Background noise**: natural noise from the real recording environments was retained (rather than recording clean speech in a laboratory)

### Dataset Versions
- **Version 1**: about 65,000 utterances, 30 target words, thousands of speakers
- Later versions expanded the amount of data and the vocabulary size

### Evaluation Methodology

The paper proposes a standardized evaluation method:
1. **Classification task**: given an audio clip, predict the target word it contains (12-class classification: 10 target words + unknown + silence)
2. **Standard split**: fixed training/validation/test splits
3. **Evaluation metric**: classification accuracy (Accuracy) as the primary metric
4. **Baseline model**: provides a simple CNN baseline implementation and results

### Baseline Model
- Architecture: a simple 4-layer CNN
- Input: 40-dimensional log-mel spectrogram (one second of audio)
- Training: standard cross-entropy loss
- Result: about 86% accuracy on the 12-class task

## Main Contributions

1. **The standard benchmark dataset for keyword spotting**: the Speech Commands dataset quickly became the standard benchmark for KWS research. Almost every KWS paper published after 2018 reports results on this dataset, enabling fair comparison across different methods.

2. **CC BY 4.0 open license**: released under the Creative Commons BY 4.0 license, allowing anyone (including commercial companies) to freely use, modify, and distribute it, with attribution as the only requirement. This dramatically lowered the barrier to entry for KWS research.

3. **Balance between scale and diversity**: 65,000 utterances, 30 target words, and thousands of speakers make it large enough to train meaningful models, yet small enough for fast experimentation on ordinary hardware. Any researcher can complete full training and evaluation on a single GPU.

4. **Standardized evaluation method**: proposes fixed training/test splits and standard evaluation metrics, eliminating the problem of "results cannot be compared because the datasets differ."

5. **Baseline model**: provides a simple CNN baseline implementation so that new researchers can get started quickly and understand the task.

## Experimental Results

### Dataset Statistics
- Total utterances: about 65,000
- Number of target words: 30
- Length per utterance: about 1 second
- Sampling rate: 16kHz
- Number of speakers: thousands (from different regions, genders, and age groups)
- Format: WAV (mono, 16-bit)

### Baseline Model Performance
- 12-class classification task: about 86% accuracy
- This baseline result provides a clear starting point for improvement in subsequent research

### Follow-up Impact
- Version 1 has been cited by hundreds of papers
- Catalyzed the proposal and comparison of a large number of new KWS architectures
- Became part of the official TensorFlow tutorials
- A later version (v2) expanded to more than 100,000 utterances and 35 target words

## Limitations and Future Work

### Limitations of the Dataset
- **English only**: all 30 target words are English words, limiting the dataset's applicability to cross-lingual KWS research
- **Mostly near-field recordings**: most utterances were recorded at close range via phone or laptop microphones, which differs considerably from the far-field scenario of smart speakers
- **Imbalanced class distribution**: some words have more utterances than others, which may bias models toward high-frequency words
- **Limited background noise**: although natural recording noise was retained, extreme noise conditions (streets, factories) are underrepresented
- **Limitation of one-second clips**: the fixed one-second length may truncate longer words or phrases, and cannot represent keyword spotting in continuous speech
- **Insufficient speaker diversity**: although there are thousands of speakers, the dataset may skew toward native English speakers and particular age groups

### Challenges for KWS Evaluation
- A single accuracy metric cannot reflect the FAR/FRR trade-off
- The 12-class classification task differs considerably from the actual wake word detection (binary classification) scenario
- No standardized FAR/FRR evaluation curves are provided

### Future Improvement Directions
- Extend to multilingual versions (e.g., Chinese, Spanish, or Hindi Speech Commands)
- Add far-field recordings and noise-augmented versions
- Add keyword annotations in continuous speech
- Provide more standardized evaluation metrics (e.g., DET curves, EER)
- Combine data augmentation strategies to generate more challenging test sets

### Profound Impact on the KWS Field
- The Speech Commands dataset's contribution to the KWS field is no smaller than ImageNet's contribution to computer vision—it unified a fragmented research area onto a common benchmark
- The dataset enables academic researchers (including students and professors without large-scale private data) to participate in cutting-edge KWS research
- Accelerated rapid iteration and objective comparison of KWS models
- The open-source spirit of Google Brain and Pete Warden set a benchmark for the entire AI community
- The existence of this dataset directly gave rise to a large number of innovative KWS architectures (e.g., various CNN variants, DS-CNN, attentive models, etc.)
