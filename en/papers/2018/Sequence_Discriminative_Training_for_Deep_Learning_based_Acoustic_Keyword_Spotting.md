# Sequence Discriminative Training for Deep Learning based Acoustic Keyword Spotting

- **Authors/Affiliations**: Zhehuai Chen, Yanmin Qian, Kai Yu (Department of Computer Science and Engineering, Shanghai Jiao Tong University; Qing Yuan Research Institute)
- **Date**: 2018.08 (arXiv:1808.00639)
- **Link**: https://arxiv.org/abs/1808.00639
- **Keywords**: sequence discriminative training, acoustic keyword spotting, sMBR, MMI, CTC, phone lattice

## Problem Statement

Sequence discriminative training (SDT) — including criteria such as sMBR (state-level Minimum Bayes Risk) and MMI (Maximum Mutual Information) — is an indispensable training technique in large vocabulary continuous speech recognition (LVCSR) systems. The transition from frame-level cross-entropy (CE) training to sequence-level discriminative training typically yields a 10-20% relative reduction in word error rate (WER). In the keyword spotting (KWS) domain, however, the vast majority of deep learning approaches still rely on frame-level cross-entropy training and have failed to fully exploit the potential of sequence-level optimization.

**Pain points in the field**
- Keyword spotting is inherently a sequence decision problem (deciding whether a segment of audio contains a specific phone sequence), yet frame-level training processes each frame independently, ignoring the temporal dependencies between frames and global sequence consistency
- Frame-level training can produce isolated mispredicted frames (e.g., only one or two frames in the middle of an audio clip being misclassified as keyword phones), whereas sequence-level training can suppress such implausible predictions
- Techniques such as sMBR/MMI, which are widely applied and highly effective in LVCSR, had not yet been systematically introduced into keyword spotting

**Shortcomings of existing methods**
- The fundamental difference between KWS and LVCSR is that the competing hypotheses in LVCSR are all possible word sequences, which can be efficiently represented by a word lattice; KWS, in contrast, only needs to distinguish between the two cases "contains the keyword" and "does not contain the keyword," so the way competing hypotheses are constructed is entirely different
- For CTC-based end-to-end KWS models, how to define "competing hypotheses" is even less obvious — the output space of CTC contains all possible phone/blank paths
- Prior attempts to apply SDT to KWS were either restricted to specific model architectures or failed to provide a unified framework

**Key challenges this paper aims to solve**
- How to uniformly construct competing sequence hypotheses for both the fixed-vocabulary and unlimited KWS paradigms
- How to implement effective sequence discriminative training in both generative models (HMM-DNN) and discriminative models (CTC)
- How to obtain consistent and significant performance gains without substantially increasing training complexity

## Methodology

### Overall Framework

The paper proposes a unified sequence discriminative training framework covering two acoustic KWS paradigms and two model types:

**Paradigm 1: Fixed-Vocabulary KWS**
Detecting a small predefined set of keywords (e.g., wake words), with the output being a keyword/non-keyword decision.

**Paradigm 2: Unlimited KWS**
Searching for arbitrary keywords in large audio archives (e.g., spoken document retrieval), with the output being the time locations of keyword occurrences.

### Sequence Discriminative Training for Generative Models (HMM-DNN)

**Construction of competing hypotheses: the word-independent phone lattice**

Conventional LVCSR uses word lattices to represent competing hypotheses, but the number of keywords in a KWS task is very small, and directly using word lattices would result in an overly sparse hypothesis space. The paper proposes using a "word-independent phone lattice":

1. Decode the training data once with the HMM-DNN model trained at the frame level with CE
2. During decoding, use a generic language model covering all phones (rather than a language model containing only the keywords)
3. The resulting phone lattice contains multiple possible phone sequence interpretations for each segment of audio
4. These "alternative phone sequences" constitute the competing hypotheses in sequence training

**The sMBR training criterion**
sMBR (state-level Minimum Bayes Risk) optimizes the expected error rate at the state level:
$$F_{sMBR} = \sum_{r} \sum_{s} \gamma_{ref}(s) \cdot \log P(s | x)$$
where $\gamma_{ref}(s)$ is the reference state occupancy (obtained from forced alignment), computed via the forward-backward algorithm on the phone lattice. sMBR directly optimizes state accuracy and is more stable than MMI.

### Sequence Discriminative Training for Discriminative Models (CTC)

**Competing hypotheses for CTC: the blank label**

The output space of a CTC model contains all phones plus one special blank label. For the KWS task:

1. The keyword phone sequence $h_{kw}$ is the "target hypothesis"
2. The CTC blank symbol $blank$ can be regarded as a proxy for the "non-keyword hypothesis" — if the model assigns most frames to blank, then the audio does not contain the keyword
3. Sequence discriminative training encourages the model to increase $P(h_{kw} | x)$ while decreasing $P(blank\_dominated | x)$

**Concrete implementation**
- Compute the CTC forward-backward probabilities for the training data
- Form the ratio of the numerator (keyword path probability) to the denominator (all path probabilities)
- Optimize this ratio with a criterion similar to sMBR

### Training Pipeline
1. Pre-train the baseline model with frame-level cross-entropy (CE)
2. Generate the competing hypotheses (phone lattices or CTC blank hypotheses)
3. Fine-tune the model with a sequence discriminative criterion (sMBR or CTC-based SDT)
4. Optional: multiple iterations of sequence training (regenerate hypotheses + retrain)

## Main Contributions

1. **Unified framework**: The first unified sequence discriminative training framework covering fixed-vocabulary and unlimited KWS as well as generative and discriminative models. Previous work either focused on only one paradigm or targeted only one model type.

2. **Word-independent phone lattice**: An innovative introduction of the word-independent phone lattice as the source of competing hypotheses for HMM-DNN KWS. This design both preserves rich competing information and avoids the degradation of word lattices in keyword-sparse scenarios.

3. **Sequence training for CTC models**: A sequence discriminative training method is proposed for CTC-based end-to-end KWS models, using the blank symbol as a proxy hypothesis for non-keywords. This fills the gap of missing sequence-level optimization in CTC KWS training.

4. **Consistent and significant improvements**: Consistent and significant performance gains were obtained across all experimental settings (fixed-vocabulary/unlimited, HMM-DNN/CTC, wake word/document retrieval), validating the generality of the framework.

## Experimental Results

### Evaluation Tasks

**Task 1: Spoken document retrieval (Unlimited KWS)**
- Search for specific keywords in a spoken document archive
- Evaluation metrics: Average Precision (AP), recall

**Task 2: Wake word detection (Fixed-Vocabulary KWS)**
- Detect a specific wake word
- Evaluation metric: FRR @ FAR operating points

### Core Results

**HMM-DNN models**
- sMBR training achieves consistent and significant improvements over the frame-level CE baseline
- Both fixed-vocabulary and unlimited KWS tasks benefit
- Sequence training improves the global consistency of frame-level predictions and reduces isolated mispredictions

**CTC models**
- Sequence training based on the blank hypothesis likewise brings significant improvements
- The value of sequence discriminative training is validated in the end-to-end KWS scenario
- The magnitude of improvement is comparable to that of sMBR training for HMM-DNN

**Cross-task consistency**
- Sequence training is effective for both short words (wake words) and long documents (spoken document retrieval)
- This demonstrates that sequence-level optimization is a general improvement lever for KWS systems rather than a trick for specific scenarios

## Limitations and Future Work

### Technical limitations of the method
- **Additional training complexity**: Sequence training requires first generating the competing hypotheses (phone lattices or CTC forward-backward) before parameter updates can be performed. This adds complexity and time cost to the training pipeline.
- **Dependence on phone lattice quality**: The effectiveness of sequence training for HMM-DNN depends on the quality of the phone lattice. If the CE baseline model is too weak, the generated phone lattice may contain a large number of erroneous hypotheses, which in turn hurts the effect of sequence training.
- **Limitations of CTC competing hypotheses**: The hypothesis that uses the blank symbol as a non-keyword proxy is relatively simple and may not provide competing information as rich as a phone lattice.

### Shortcomings of the experimental design
- No direct comparison with end-to-end attention-based KWS methods
- No in-depth analysis of the specific impact of sequence training on different error types (substitutions, deletions, insertions)
- Lacks robustness evaluations under different noise conditions

### Directions for future improvement
- Explore sequence discriminative training of Transformer-based sequence models for KWS
- Study online sequence training methods to reduce the dependence on pre-generated phone lattices
- Combine sequence training with end-to-end CTC/attention hybrid models
- Explore finer-grained competing hypothesis construction strategies (e.g., keyword confusion-set-oriented augmentation)

### Implications for the KWS field
- Sequence discriminative training is an underrated optimization dimension in KWS systems, and its effect can rival architectural innovations
- Transferring training techniques from LVCSR is an effective way to improve KWS performance
- This work fills the gap between frame-level training and end-to-end optimization, providing a systematic methodology for subsequent research
- The work from Shanghai Jiao Tong University demonstrates an important contribution of Chinese academia to research on KWS technology
