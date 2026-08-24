# Multi-Task Learning for Speaker Verification and Voice Trigger Detection

**Authors/Affiliations**: Siddharth Sigtia, Erik Marchi, Sachin Kajarekar, Devang Naik, John Bridle (Apple Inc.)

**Date**: January 2020 (arXiv:2001.10816)

**Link**: https://arxiv.org/abs/2001.10816

**Keywords**: Multi-Task Learning, Speaker Verification, Voice Trigger Detection, Keyword Spotting

## Problem Statement

In smart voice assistants, two critical security features typically operate independently:
1. **Voice Trigger Detection (KWS)**: Detects whether the user has spoken a wake-word (e.g., "Hey Siri")
2. **Speaker Verification (SV)**: Verifies whether the speaker is an authorized user, preventing unauthorized users from triggering the device

Problems with independent operation:
- **Computational Redundancy**: Both systems have complete front-end processing and feature extraction pipelines, resulting in significant redundant computation in continuous listening mode.
- **Feature Non-Sharing**: Both KWS and SV require extracting acoustic features (spectral features, acoustic embeddings, etc.) from raw audio, but they do so independently.
- **Suboptimal Performance**: Both tasks share underlying acoustic representations (speech spectral characteristics encode both "what is said" and "who is speaking"), and independent training fails to leverage this complementarity.
- **Deployment Complexity**: Having two independent models increases the complexity of deployment and maintenance.

Multi-Task Learning (MTL) can address these issues simultaneously by sharing a feature encoder and jointly optimizing both tasks.

## Methodology

### Shared Encoder Architecture

**Design Principle**: Share low-level feature extraction while retaining task-specific output heads.

**Shared Network (Shared Encoder)**:
- Multi-layer CNN or CRNN structure.
- Extracts high-level acoustic representations from audio spectral features.
- This representation is designed to encode both speech content information (for KWS) and speaker identity information (for SV) simultaneously.
- Trained jointly on large amounts of training data to obtain richer feature representations.

**KWS Task Head**:
- Receives the output from the shared encoder.
- Uses additional network layers for keyword classification.
- Output: Probability distribution over keyword classes.

**SV Task Head**:
- Receives the output from the shared encoder.
- Uses additional network layers to extract speaker embeddings.
- Output: Speaker embedding vector, used for comparison with registered speaker templates.

### Joint Training Strategy

**Loss Function Combination**:
- $L_{total} = \lambda_{kws} * L_{kws} + \lambda_{sv} * L_{sv}$
- $L_{kws}$: Cross-entropy loss for keyword classification.
- $L_{sv}$: Speaker verification loss (e.g., contrastive loss, AM-Softmax, etc.).
- $\lambda_{kws}$ and $\lambda_{sv}$ control the optimization balance between the two tasks.

**Training Techniques**:
- Gradient balancing: Ensures the magnitude of gradients from both tasks is comparable, preventing one task from dominating the training.
- Learning rate scheduling: Uses potentially different learning rates for the shared part and the task-specific parts.
- Alternating training: Alternates the optimization of the two tasks in certain training steps.

### Feature Sharing Analysis

The feature representations learned by the shared encoder were investigated:
- Shallow features: Serve both tasks (general acoustic features).
- Deep features: May diverge into task-specific representations.
- Does the SV task benefit from the speech content awareness of KWS (knowing what is said helps extract who is speaking)?
- Does the KWS task benefit from the speaker awareness of SV (knowing who is speaking helps determine what is said)?

## Main Contributions

1. **KWS-SV Multi-Task Learning Framework**: Proposes a multi-task framework for jointly training voice trigger detection and speaker verification. The shared encoder reduces computational and memory overhead during deployment, while multi-task regularization may improve the performance of each individual task.

2. **Shared Feature Encoder**: Designs a shared feature encoder that serves both KWS and SV tasks, validating the complementarity of the two tasks at the feature level.

3. **Multi-Task Interaction Analysis**: Systematically analyzes the impact of multi-task training on the dynamics of both KWS and SV, providing an empirical basis for understanding the interaction between the two tasks.

4. **Large-Scale Production Validation**: Validates the effectiveness of the method on Apple's large-scale voice trigger data, with results holding high value for industrial practice.

## Experimental Results

### Experimental Setup
- Large-scale production voice trigger data (containing speaker annotations and keyword annotations).
- Baseline: Independently trained KWS and SV models vs. jointly trained multi-task models.
- Evaluation: KWS detection accuracy, SV Equal Error Rate (EER).

### Key Results
- **KWS Improvement**: Multi-task training improved the accuracy of voice trigger detection. The speaker awareness provided by the SV task helped KWS better distinguish similar phoneme sequences.
- **SV Improvement**: Speaker verification also benefited from multi-task training. The content awareness provided by the KWS task helped SV focus on speaker features within the keyword context.
- **Model Efficiency**: The shared encoder resulted in a total model size smaller than the sum of two independent models.
- **Regularization Effect**: Multi-task training acted as implicit regularization, reducing overfitting.

### Task Interaction Analysis
- The two tasks share general acoustic features at shallow layers (spectral shape, energy contour, etc.).
- The help KWS provides to SV is mainly reflected in: knowing which keyword was spoken allows for better modeling of the speaker's voice characteristics in that context.
- The help SV provides to KWS is mainly reflected in: speaker information helps distinguish pronunciation differences between different speakers.

## Limitations and Future Work

### Method Limitations
- **Loss Weight Tuning**: The weight balance between the losses of the two tasks requires careful tuning; different weights may lead to different performance trade-offs.
- **Task Conflict**: In some cases, the two tasks may have conflicting optimization directions (KWS wants to ignore speaker differences, while SV wants to leverage speaker differences).
- **Data Requirements**: Requires paired training data containing both KWS labels and speaker labels.
- **Deployment Assumptions**: Assumes that both tasks are always required to be executed simultaneously; if only one is needed, the computation for the shared encoder part may be wasteful.

### Future Directions
- Research dynamic task routing mechanisms to selectively activate task-specific heads as needed.
- Explore multi-task architecture search to automatically discover optimal feature sharing strategies.
- Research online multi-task learning, enabling the model to continuously adapt to new speakers and new keywords.
- Combine with contrastive learning to further enhance the discriminability of the shared embedding space.
- Explore multi-task frameworks incorporating more related tasks (e.g., language identification, emotion recognition).
