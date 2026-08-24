# Multi-task Voice Activated Framework using Self-supervised Learning

- **Authors/Affiliations**: Shuo Liu, Aditya Gourav, Sreyas Srinivasa, K.T. Kim - University of California; Qualcomm
- **Date**: 2021.10
- **Link**: https://arxiv.org/abs/2110.01077
- **Keywords**: Self-supervised Learning, Multi-task Learning, Keyword Spotting, Voice Activity Detection, Speaker Verification, Contrastive Predictive Coding, Parameter Sharing

## Problem Statement

Modern voice-activated systems (such as smart assistants) need to perform multiple speech tasks simultaneously: Keyword Spotting (KWS) to identify wake-up words, Voice Activity Detection (VAD) to determine if speech is present, and Speaker Verification (SV) to confirm the speaker's identity. The current mainstream approach involves training and deploying independent models for each task, which brings several key issues:

1. **High memory overhead**: Multiple models need to be stored simultaneously on edge devices, occupying precious storage and memory resources.
2. **Computational redundancy**: Models for different tasks may redundantly compute similar low-level features.
3. **Complex development and maintenance**: Multiple models require independent optimization and updates.
4. **Insufficient data utilization**: Each task's model can only utilize the labeled data for that specific task, unable to leverage supervisory signals from other tasks.

The core problem this paper aims to solve is: Can a unified model be constructed to handle KWS, VAD, and SV tasks simultaneously through self-supervised pre-training and a multi-task learning framework, thereby reducing the total number of parameters while maintaining or even improving the performance of each task?

## Methodology

### Overall Framework Design
The framework proposed in the paper consists of two stages:
1. **Self-supervised pre-training stage**: Learning general speech representations on large-scale unlabeled speech data.
2. **Multi-task fine-tuning stage**: Jointly fine-tuning the shared encoder and task-specific heads on labeled data.

### Self-supervised Pre-training
Pre-training is conducted using Contrastive Predictive Coding (CPC) or similar self-supervised objectives:
- **Input**: Large amounts of unlabeled speech data.
- **Encoder**: A multi-layer CNN or Transformer encoder that encodes raw audio or spectrograms into high-level representations.
- **Self-supervised Objective**:
  - Given the encoded representation $z_t$ at the current time step, predict the representations $z_{t+k}$ for the next $k$ time steps.
  - Use InfoNCE loss: Maximize the contrast between positive samples (true future representations) and negative samples (randomly sampled representations).
- **Learned Representations**: Through this process, the encoder learns to capture the basic structure of speech (phonemes, prosody, speaker characteristics, etc.) without any labeled information.

### Shared Encoder + Task-Specific Heads
The architectural design for the multi-task fine-tuning stage:
- **Shared Encoder**: All tasks share the same acoustic encoder (i.e., the encoder from self-supervised pre-training).
- **Task-Specific Heads**:
  - **KWS Head**: A fully connected classifier outputting keyword classes.
  - **VAD Head**: A binary classifier outputting speech/non-speech decisions.
  - **SV Head**: An embedding extractor + similarity calculation module.

### Multi-task Training Strategy
- **Joint Loss**: $L = \alpha * L_{KWS} + \beta * L_{VAD} + \gamma * L_{SV}$
- **Alternating Training**: In each training step, one or more tasks are randomly selected to compute the loss and update parameters.
- **Gradient Balancing**: Techniques such as GradNorm or similar methods are used to balance the gradient magnitudes of different tasks, preventing any single task from dominating the training process.

### Parameter Sharing Analysis
The paper analyzes different levels of parameter sharing strategies:
- **Hard Sharing**: All tasks completely share the encoder, with only the task heads differing.
- **Soft Sharing**: Tasks share the lower layers of the encoder, while higher layers use task-specific branches.
- **No Sharing**: Each task has a completely independent model (baseline comparison).

## Main Contributions

1. **Introduction of self-supervised pre-training for a voice-activated multi-task framework**: Demonstrates that self-supervised pre-training on unlabeled data benefits all downstream speech tasks (KWS, VAD, SV), providing an effective method for extracting general speech knowledge from unlabeled data.

2. **Demonstration of cross-task parameter sharing reducing total model size**: By sharing the encoder, the total parameter count of the multi-task framework is significantly less than the sum of three independent models, making it feasible to run multiple speech tasks simultaneously on edge devices.

3. **Provision of a practical deployment solution for multi-task speech systems**: The framework offers an end-to-end solution for deploying multi-functional speech systems on resource-constrained devices, from pre-training to multi-task fine-tuning and inference.

4. **Validation of the complementary effects of multi-task learning**: Supervisory signals from different tasks can complement each other, and the performance of each task after joint training is no worse than (and sometimes better than) that of models trained individually.

## Experimental Results

### Datasets
- **Pre-training Data**: Large-scale unlabeled speech corpora (e.g., LibriSpeech, VoxCeleb, etc.)
- **KWS Data**: Google Speech Commands
- **VAD Data**: Standard VAD datasets
- **SV Data**: Speaker verification datasets such as VoxCeleb

### Performance on Each Task
- **KWS**: The KWS accuracy of the multi-task framework is comparable to or slightly higher than the single-task baseline.
- **VAD**: VAD performance benefits from multi-task joint training, with a slight improvement in accuracy.
- **SV**: The Equal Error Rate (EER) for speaker verification remains within an acceptable range.

### Parameter Efficiency
- **Total Parameters**: The total parameter count of the shared encoder framework is approximately 40-50% of the sum of three independent models.
- **Inference Efficiency**: A single forward pass outputs predictions for all three tasks simultaneously, improving inference efficiency by 2-3 times.

### Effect of Self-supervised Pre-training
- Compared to random initialization, self-supervised pre-training provides consistent performance improvements across all tasks.
- The benefits of pre-training are particularly significant in scenarios with limited labeled data.
- Using more unlabeled data for pre-training can further enhance performance.

### Ablation Studies
- **Hard Sharing vs. Soft Sharing**: Hard sharing is superior in terms of parameter efficiency, but soft sharing performs better on certain tasks.
- **Pre-training vs. No Pre-training**: Pre-training improves KWS by 1-3% and VAD by 2-5%.
- **Task Combinations**: The combination of KWS+VAD yields the largest gain; adding SV maintains the performance of KWS and VAD.

## Limitations and Future Work

### Technical Limitations
- **Task Interference**: When tasks differ significantly (e.g., KWS requires temporal details, while SV requires speaker features), the shared encoder may not perfectly accommodate all tasks, leading to performance degradation in some tasks.
- **Training Complexity**: Multi-task training requires careful balancing of loss weights and learning rates for different tasks, making hyperparameter tuning more complex than in single-task scenarios.
- **Pre-training Computational Overhead**: The self-supervised pre-training stage requires substantial computational resources and time.

### Insufficiencies in Experimental Design
- Insufficient testing in noisy and far-field scenarios.
- Limited analysis of the contributions of different layers of the shared encoder.
- Incremental learning was not explored—specifically, how to avoid forgetting previously learned tasks when new tasks are added.

### Directions for Future Improvement
- Explore task-conditioned dynamic routing mechanisms to selectively activate different parts of the encoder based on the current task.
- Combine federated learning to achieve privacy-preserving multi-task pre-training.
- Investigate model distillation techniques to transfer multi-task knowledge to more compact student models.
- **Insights for the KWS field**: A unified multi-task speech frontend is the future trend for edge deployment, and self-supervised pre-training provides a powerful foundation for this.
