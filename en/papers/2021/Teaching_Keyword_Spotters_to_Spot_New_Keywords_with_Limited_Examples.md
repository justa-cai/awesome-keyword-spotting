# Teaching Keyword Spotters to Spot New Keywords with Limited Examples

- **Authors/Affiliations**: Abhijeet Awasthi, Kevin Kilgour, Hassan Rom - Google Research (Switzerland) / Indian Institute of Technology Bombay
- **Date**: 2021.06
- **Link**: https://arxiv.org/abs/2106.02443
- **Keywords**: Few-shot learning, Keyword Spotting, Speech Embeddings, Pre-trained Models, New Keyword Adaptation, Transfer Learning

## Problem Statement

Keyword Spotting (KWS) is a core component of smart voice assistants (such as Google Home, Apple HomePod, Amazon Echo), typically modeled as a task of classifying fixed-length speech segments into a predefined set of keywords. However, modern KWS models face a core pain point: **the rigid gap between training and deployment**. These models are usually trained on large-scale internal datasets comprising thousands of hours of data and are limited to a very small number of specific keywords (often no more than 20), making them inflexible for migration to user-defined new keywords.

In practical application scenarios, users expect voice assistants to recognize personalized wake words or command words, but collecting large amounts of training data for new keywords is both expensive and time-consuming. Existing solutions include: (1) initializing KWS models with ASR model weights, but this method relies on phoneme-grapheme dictionaries and has poor language scalability; (2) data augmentation (e.g., SpecAugment), which only alleviates the surface problem of data insufficiency; (3) few-shot learning methods such as Meta-Learning (MAML) and Prototypical Networks, which show limited performance in cross-lingual generalization; (4) the multi-task learning embedding model proposed by Lin et al., trained on 111K hours of YouTube speech, but each classification head learns to distinguish only 40 keywords, limiting its discriminative power.

The key challenge this paper aims to solve is: **How to design an efficient speech embedding representation that enables KWS models to accurately identify completely new keywords not seen during pre-training, while possessing cross-lingual generalization capabilities and supporting incremental learning of new keywords without forgetting existing ones, using only a very small number of training samples.**

## Methodology

### Overall Architecture Design

The paper proposes **KeySEM (Keyword Speech EMbedding)**, a speech embedding model specifically designed for keyword spotting tasks. The core idea of KeySEM is to learn highly discriminative speech representations by pre-training on a large-scale keyword classification task. The overall architecture consists of two stages:

1.  **Pre-training Stage**: KeySEM acts as a neural network $F: U \rightarrow \mathbb{R}^d$, mapping fixed-length speech segments to a $d$-dimensional real vector space. $F$ is pre-trained as part of a larger classification model $P(y|F(u))$, learning to classify speech segments into a large vocabulary containing 15,000 different keywords.
2.  **Downstream Adaptation Stage**: After pre-training is complete, the original linear classification layer is removed and replaced with a newly initialized random linear layer, which is trained on the target KWS dataset by minimizing the cross-entropy loss.

### Core Architecture Details

KeySEM adopts a CNN-based architecture (similar to Lin et al.), with specific designs as follows:

-   **Input Features**: 2-second long speech segments, converted into 40-dimensional log-mel features, using a 25ms window and 10ms stride, with a frequency range of 60Hz-7800Hz.
-   **Convolutional Modules**: Contains 6 convolutional blocks. The first 5 blocks each contain 5 layers: 4 layers of alternating 1x3 and 3x1 convolutions, followed by max-pooling layers. After the 5th block, the frequency dimension is reduced to 1. The 6th block contains two layers of 5x1 convolutions and average pooling.
-   **Channel Design**: Starts with 24 channels, adding 24 channels for each new block, up to a maximum of 96 channels.
-   **Output**: Finally maps the 2-second speech segment to a **96-dimensional** feature vector.

### Innovations in Pre-training Strategy

The core innovation of KeySEM lies in the design of the pre-training strategy:

**Loss Function**: Standard cross-entropy loss, traversing all keyword categories:
$$L = \sum_{y \in V} \sum_{u \in U_y} \log(P(y|F(u)))$$

**Pre-training Data Construction**: Utilizing the LibriSpeech corpus and its community forced alignment annotations, n-grams with length no more than 5 are extracted as keywords (requiring at least 10 characters in length and appearing at least 10 times), ultimately obtaining over **15,200 different keywords**, totaling **250 hours** of speech.

**Key Differences from Existing Methods**: Lin et al.'s method uses 125 classifiers, each distinguishing only 40 keywords, for multi-task learning, requiring 111K hours of data; whereas KeySEM uses a single classification task to distinguish 15K keywords simultaneously, requiring only 250 hours of data (three orders of magnitude less), yet producing more discriminative feature representations.

### Training Configuration

-   Optimizer: Adam, learning rate 5e-4
-   Batch size: 1024
-   Training steps: 1M steps
-   Downstream training supports two modes: freezing (fix) the embedding model and training only the linear layer, or fine-tuning (FT) the entire model

### Sequential Learning of New Keywords

KeySEM supports incremental learning: an independent sigmoid binary classifier is trained for each new keyword (distinguishing the new keyword from the negative class), using only 5 positive examples and 50 negative examples. At inference time, the keyword with the highest confidence among all classifiers is taken as the output. Freezing the embedding model avoids the problem of catastrophic forgetting.

## Main Contributions

1.  **Proposed KeySEM Speech Embedding Model**: By pre-training on a 15K keyword classification task, it learns highly discriminative speech representations. Compared to multi-task learning methods, a single large-scale classification task forces the model to learn richer discriminative features, allowing it to generalize better to unseen keywords. The technical significance of this contribution lies in proving that the vocabulary size of the pre-training task (rather than the amount of data) is the key factor for embedding quality.

2.  **Established the LibriKWS Benchmark Dataset**: Constructed an evaluation dataset containing 150 unseen keywords from LibriSpeech, which was the publicly available KWS dataset with the largest number of target categories at the time, providing a standard benchmark for evaluating the generalization ability of KWS models on diverse unseen keywords.

3.  **Discovery of Cross-Lingual Generalization Capability**: Although KeySEM was pre-trained only on English speech, it showed significant performance improvements on four non-English languages: Japanese, Esperanto, Polish, and Portuguese, proving that the learned representations are intrinsically related to the keyword spotting task rather than language-specific features.

4.  **Support for Sequential Learning of New Keywords**: Demonstrated that in device-side environments, by freezing the embedding model and training only a lightweight linear classifier, new keywords can be continuously learned without forgetting existing ones. The overall accuracy is only 0.4% lower than simultaneously training all categories, whereas the fine-tuning approach leads to catastrophic forgetting (accuracy drops to 24.3%).

## Experimental Results

### Datasets

-   **Google Speech Commands V2**: 35 words, 85.5K training, 10.1K development, 4.9K test samples
-   **LibriKWS**: 150 unseen keywords, 90 training samples/word, two test sets (test-clean and test-other) with zero speaker overlap
-   **Common Voice**: Word recognition in four languages (Japanese, Esperanto, Polish, Portuguese), 12-14 target keywords

### Full Training Results (Table 1)

| Method | SC | LK-c | LK-o | ja | eo | pl | pt |
|------|-----|-------|-------|------|------|------|------|
| Matchbox | 98.0 | 97.3 | 89.8 | 76.0 | 88.5 | 85.0 | 84.3 |
| MHAtt-RNN | 98.0 | 99.7 | 95.3 | 86.0 | 87.0 | 87.3 | 79.3 |
| MTLEmb (fix) | 96.6 | 95.1 | 87.2 | 86.7 | 87.4 | 89.5 | 82.6 |
| **KeySEM (fix)** | 93.9 | **99.8** | **97.8** | **92.3** | 91.2 | **90.3** | 82.4 |
| **KeySEM (FT)** | **98.2** | 97.8 | 93.2 | **92.9** | 89.1 | 90.3 | **84.7** |

KeySEM (fix) performs particularly well on LibriKWS (reaching 97.8% on test-other), indicating that the freezing approach is superior when data is limited.

### 5 Samples/Keyword Results (Table 2, Core Highlight)

| Method | SC | LK-c | LK-o | ja | eo | pl | pt |
|------|-----|-------|-------|------|------|------|------|
| Matchbox | 45.3 | 3.0 | 2.8 | 48.0 | 60.5 | 58.3 | 36.3 |
| MHAtt-RNN | 48.2 | 27.3 | 12.8 | 55.0 | 62.5 | 68.0 | 53.0 |
| MTLEmb (fix) | 79.1 | 43.9 | 33.1 | 76.5 | 74.5 | 76.3 | 71.3 |
| **KeySEM (fix)** | **86.5** | **98.5** | **94.5** | **86.2** | **87.4** | **86.3** | **80.8** |

-   Compared to MTLEmb, KeySEM provides an **absolute accuracy improvement of 7% to 61%** across all datasets.
-   Using only 5 samples/keyword, KeySEM (fix) reaches 94.5% on LibriKWS test-other, while MTLEmb achieves only 33.1%.
-   It is also effective on non-English languages, demonstrating cross-lingual generalization capability.

### Sequential Learning Experiments

-   After sequentially learning 10 categories of keywords, the overall accuracy is **86.1%**.
-   This is only **0.4%** lower than the result of simultaneously training all categories.
-   The fine-tuning approach leads to catastrophic forgetting, with a final accuracy of only **24.3%**.

### Analysis of Training Sample Quantity

-   When there are fewer than 10 samples/keyword, the frozen KeySEM approach is significantly superior to all other methods.
-   As training data increases, the differences between methods narrow.
-   With 1000 samples/keyword, fine-tuned KeySEM achieves the highest accuracy of 97.5%.

## Limitations and Future Work

### Technical Limitations

1.  **Model Size and Deployment Constraints**: KeySEM has approximately 410K parameters. Although smaller than some large KWS models, it is still too large for ultra-low-power microcontrollers (such as DSP chips). The paper does not discuss quantization or model compression strategies.
2.  **Fixed Speech Length**: The model assumes input is a fixed 2-second speech segment, requiring additional alignment processing for shorter or longer keywords.
3.  **Pre-training Language Bias**: Although cross-lingual generalization results are encouraging, KeySEM was pre-trained only on English. Its generalization capability to languages with significantly different acoustic characteristics (such as tonal language Chinese) has not been verified.
4.  **Upper Limit of Discriminative Power of Linear Classifiers**: While freezing the embedding model and training only the linear layer avoids overfitting, it also limits the upper bound of the model's performance under conditions of abundant data.

### Experimental Design Shortcomings

1.  **Limited Comparison Methods**: Lacks direct comparison with the latest meta-learning methods (e.g., improved MAML) and contrastive learning methods (e.g., COLA) available at the time.
2.  **Lack of Real-World Scenario Evaluation**: Experiments were conducted on relatively clean LibriSpeech data, without fully evaluating performance in real-world environments such as far-field, noise, and reverberation.
3.  **Sequential Learning Verified Only for 10 Categories**: As the number of categories continues to grow, the efficiency of combined inference using sigmoid classifiers may become a bottleneck.

### Future Improvement Directions

1.  Combine self-supervised learning objectives (e.g., contrastive loss from wav2vec 2.0) with KeySEM's supervised pre-training to further improve representation quality.
2.  Explore the integration of incremental learning algorithms (e.g., iCaRL, EWC) with KeySEM embeddings to more elegantly handle continuous learning scenarios.
3.  Extend KeySEM to multi-language pre-training data to enhance support for low-resource languages.
4.  Study model quantization (INT8/binary) and knowledge distillation to adapt to edge devices.

### Insights for the KWS Field

KeySEM proves that the "vocabulary size of the pre-training task" is more important than the "amount of pre-training data," an insight with profound implications for the design of pre-training strategies in the KWS field. It also provides a feasible technical path for device-side personalized KWS: pre-training a powerful embedding model in the cloud, and training only a lightweight linear classifier on the device using a small number of samples.
