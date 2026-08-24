# Few-Shot Keyword Spotting With Prototypical Networks

**Authors/Affiliations**: Benjamin S. Acero, Zhiyao Duan (The University of North Carolina at Charlotte)

**Date**: July 2020 (arXiv:2007.14463)

**Link**: https://arxiv.org/abs/2007.14463

**Keywords**: Few-Shot Learning, Keyword Spotting, Prototypical Networks, Meta-Learning, Rapid Adaptation

## Problem Statement

Traditional KWS systems require collecting large amounts of training data for each keyword (typically thousands to tens of thousands of samples). This is impractical in the following scenarios:
- **Rapid Customization**: Users wish to instantly add new personalized command words.
- **Low-Resource Languages**: Certain languages lack large-scale speech data.
- **Niche Keywords**: Domain-specific jargon or brand names.

Few-Shot Learning enables models to learn to recognize new keyword categories from very few samples (1-5). This is of significant importance for achieving flexible, user-customizable KWS systems.

## Methodology

### Prototypical Networks

This paper adapts Prototypical Networks—a meta-learning method based on Metric Learning—to the KWS task:

**Core Idea**:
- Learn an embedding function (Embedding Function) to map speech signals to an embedding space.
- In the embedding space, samples of the same keyword cluster around their class prototype.
- The class prototype is the mean vector of the embeddings of all support set samples in that class.

**Mathematical Formulation**:
- Embedding function: $f_\phi: X \rightarrow \mathbb{R}^d$ (maps input audio to a $d$-dimensional embedding space)
- Class prototype: $c_k = \frac{1}{|S_k|} \sum f_\phi(x_i)$ for $x_i \in S_k$
- Classification probability: $p(y=k|x) = \text{softmax}(-d(f_\phi(x), c_k))$, where $d$ is the Euclidean distance.

### Episodic Training

Key training strategy for meta-learning:

**Training Set Construction**:
- Sample N-way K-shot classification tasks from the Google Speech Commands dataset.
- Each task (episode):
  - Randomly select N keyword classes.
  - Sample K samples per class as the Support Set.
  - Sample several Query Set samples for loss calculation.

**Training Process**:
1. Calculate the prototype vector for each class in the support set.
2. Calculate the distance from query samples to each prototype.
3. Optimize the embedding function parameters using cross-entropy loss.
4. Train on a large number of randomly sampled episodes, enabling the model to learn "how to learn."

### Embedding Network Architecture

- Input: Spectral features of audio (Mel filter banks).
- Use CNN as the backbone of the embedding network.
- Output: Fixed-dimensional embedding vector.

## Main Contributions

1. **First Application of Prototypical Networks to KWS**: Introduces the Prototypical Networks method from metric learning to the field of keyword spotting, validating the feasibility of few-shot learning in speech tasks.

2. **Flexible Keyword Addition**: Users only need to provide a small number (1-5) of new keyword samples, allowing the system to rapidly expand to new categories without retraining the entire model.

3. **Episodic Training Adaptation**: Designs an episodic training strategy suitable for audio KWS tasks, enabling meta-learning models to effectively generalize to unseen keywords.

4. **Practical Potential**: Demonstrates that KWS systems can operate under extremely low data conditions, opening up possibilities for personalized voice interaction.

## Experimental Results

### Experimental Setup
- Google Speech Commands dataset.
- Evaluation settings: 5-way 1-shot, 5-way 5-shot, etc.
- Embedding dimension: Configurable.
- Baselines: Simple distance classifier, Nearest Neighbor classifier.

### Main Results
- **1-shot**: Even with only 1 sample, the Prototypical Network achieves meaningful classification accuracy.
- **5-shot**: Performance improves significantly with 5 samples, proving that more support samples aid in more accurate class prototype estimation.
- **vs. Baselines**: The Prototypical Network significantly outperforms simple distance-matching baseline methods.
- **Generalization Capability**: Exhibits some generalization ability to keyword categories not seen during training.

### Key Findings
- The quality of the embedding space is crucial for few-shot performance.
- The diversity and representativeness of support samples directly affect the quality of prototype estimation.
- Larger embedding dimensions are not necessarily better in few-shot settings (curse of dimensionality).

## Limitations and Future Work

### Methodological Limitations
- **Accuracy Ceiling**: Compared to fully supervised methods with ample training data, the accuracy of few-shot methods still shows a significant gap.
- **Sensitivity to Support Samples**: Performance is highly dependent on the quality and representativeness of support samples; noisy or anomalous support samples can severely impact prototype estimation.
- **Limited Exploration of Embedding Architectures**: Only simple CNN embedding networks were used; more complex architectures (e.g., attention mechanisms, Transformers) were not explored.
- **Real-World Conditions**: Few-shot performance under complex conditions such as real-world noise, far-field, and multi-speaker scenarios was not fully validated.
- **Class Number Limitations**: As the number of classes increases, class discrimination in the embedding space becomes more difficult.

### Future Directions
- Combine data augmentation and TTS synthesis to improve support set quality.
- Investigate the application of more advanced meta-learning methods (e.g., MAML, Relation Networks) in KWS.
- Explore cross-lingual few-shot KWS, leveraging knowledge from high-resource languages to assist low-resource languages.
- Combine with active learning to intelligently select the most informative support samples.
- Research incremental few-shot learning to support the continuous addition of new keywords without forgetting old ones.
