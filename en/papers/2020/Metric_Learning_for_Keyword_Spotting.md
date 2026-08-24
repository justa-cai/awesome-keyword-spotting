# Metric Learning for Keyword Spotting

**Authors/Affiliations**: Jaesung Huh, Minjae Lee, Heesoo Heo, Seongkyu Mun, Joon Son Chung (Naver Corporation)

**Date**: May 2020 (arXiv:2005.08776)

**Link**: https://arxiv.org/abs/2005.08776

**Keywords**: Metric Learning, Keyword Spotting, Embedding Learning, Open-set Recognition, Flexible Scalability

## Problem Statement

Most Keyword Spotting (KWS) methods treat keyword detection as a closed-set classification problem: all target keywords and non-keywords are fixed and defined during training, and the model outputs a probability distribution over fixed classes. This paradigm has fundamental limitations:
- **Inability to flexibly add new words**: Adding new keywords requires retraining or fine-tuning the entire model.
- **Fixed vocabulary**: The vocabulary cannot be dynamically adjusted after deployment.
- **Insufficient handling of unknown words**: Non-keywords unseen during training may be incorrectly classified as known keywords.
- **Difficulty in personalization**: Different users may require different sets of keywords.

Metric Learning offers an alternative paradigm:
- Learn a general-purpose embedding function that maps audio signals to a discriminative embedding space.
- In the embedding space, embeddings of the same keyword are close to each other, while embeddings of different keywords are far apart.
- New keywords only require computing their embedding representations, without retraining the model.
- Supports Open-set Recognition.

## Methodology

### Embedding Network

- **Input**: Audio spectral features (Mel filterbanks)
- **Backbone**: CNN architecture (e.g., ResNet variants)
- **Output**: Fixed-dimensional embedding vector (e.g., 128 or 256 dimensions)
- The embedding vector is L2-normalized, mapping it to the unit hypersphere.

### Metric Learning Loss Functions

Beyond traditional Contrastive Loss and Triplet Loss, more advanced loss functions are employed:

**Proxy-based Loss**:
- Maintains a learnable proxy vector for each class.
- The loss is calculated based on the distance between sample embeddings and class proxies.
- Avoids the hard example mining problem for sample pairs/triplets found in traditional metric learning.

**Angular Margin Loss**:
- Adds angular margin constraints on the unit hypersphere.
- Forces embeddings of the same class to cluster more tightly and embeddings of different classes to separate more distinctly.
- Examples include ArcFace, CosFace, and other similar loss functions.

### Keyword Detection Pipeline

**Training Phase**:
- Train the embedding network using a large amount of keyword data.
- Optimize the loss function to enhance the discriminativity of the embedding space.

**Enrollment Phase**:
- The user provides a small number of samples for new keywords.
- Compute and store the embedding prototypes for the new keywords.

**Inference Phase**:
- Compute the embedding vector for the input audio.
- Determine whether it is a known keyword via nearest neighbor search or distance thresholding.
- Supports simple classification methods such as K-NN and cosine similarity.

### Open-set Detection

- Set a distance threshold: If the distance between the input embedding and all known keyword prototypes exceeds the threshold, it is classified as "unknown."
- No explicit training of an "unknown" class is required.
- Can handle non-keywords that were not seen during training.

## Main Contributions

1. **Introduction of Metric Learning Paradigm to KWS**: Introduces modern metric learning techniques to keyword detection, providing a new approach as an alternative to traditional classification paradigms. The embedding space approach enables KWS systems to possess open-set recognition capabilities and flexible scalability.

2. **Open-set KWS Capability**: The model can detect keyword types unseen during training and effectively reject unknown non-keywords. This is significant for practical deployment, where it handles the infinite variety of non-keyword inputs.

3. **Keyword Addition Without Retraining**: New keywords only require registering prototypes in the embedding space, without retraining or modifying the model. This greatly enhances the flexibility and customizability of KWS systems.

4. **Overcoming Traditional Metric Learning Difficulties**: By using proxy-based and angular margin losses, it overcomes the hard example mining challenges inherent in traditional contrastive and triplet losses.

## Experimental Results

### Experimental Setup
- Dataset: Google Speech Commands
- Evaluation: Closed-set classification accuracy and open-set detection capability
- Comparison: Standard classification models vs. Metric learning models

### Key Results
- **Closed-set Accuracy**: The metric learning method achieves accuracy comparable to traditional classification methods on Google Speech Commands.
- **Open-set Detection**: Significantly outperforms classification methods in detecting unknown keywords.
- **New Keyword Adaptation**: Effective detection is achieved for new keywords with only 1-5 registration samples.
- **Embedding Quality**: t-SNE visualizations show that different keywords form clear clusters in the embedding space.

### Comparison with Classification Methods
- **Closed-set Scenario**: The metric learning method has slightly lower accuracy than dedicated classification models (a gap of approximately 0.5-1%).
- **Open-set Scenario**: The metric learning method significantly outperforms classification methods.
- **Flexibility**: The cost of expanding the keyword vocabulary for the metric learning method is nearly zero.

## Limitations and Future Work

### Method Limitations
- **Sensitivity to Loss Functions**: Metric learning is sensitive to the choice of loss function and hyperparameters (e.g., margin values, learning rate).
- **Dependency of Embedding Quality on Training Diversity**: The quality of the embedding space depends on the diversity of keywords seen during training.
- **Inference Computational Overhead**: Nearest neighbor search requires calculating the distance between the input and all prototypes, which may affect efficiency when the number of keywords is large.
- **Accuracy Gap**: For closed-set scenarios with a fixed keyword set, dedicated classification models may provide higher accuracy.

### Future Directions
- Research more efficient embedding search methods (e.g., Approximate Nearest Neighbor, Hashing).
- Explore dynamic embedding space updates to support continuous learning of new keywords.
- Combine metric learning and classification methods to balance closed-set accuracy and open-set flexibility.
- Research cross-lingual metric learning to ensure consistency of the embedding space across different languages.
- Combine self-supervised pre-training to improve embedding quality.
