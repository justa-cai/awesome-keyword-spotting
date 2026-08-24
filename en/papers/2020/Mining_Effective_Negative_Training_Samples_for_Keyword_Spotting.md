# Mining Effective Negative Training Samples for Keyword Spotting

**Authors/Affiliations**: Jianyu Huang, Jinyu Li, Yujun Wang (Northwestern Polytechnical University, Mobvoi Inc.)

**Date**: May 2020 (ICASSP 2020)

**Link**: https://arxiv.org/abs/2005.02713

**Keywords**: Keyword Spotting, Negative Sample Mining, Data Augmentation, Training Strategy, Hard Negative Samples

## Problem Statement

The performance of Keyword Spotting (KWS) systems largely depends on the quality and diversity of negative (non-keyword) training data. In practical applications, false alarms are primarily caused by the following two types of negative samples:

1.  **Acoustically Similar Negative Samples**: Non-keywords that are highly similar to the target keyword in acoustic features (e.g., "Okay Google" vs. "Okay goggles").
2.  **Partial Match Negative Samples**: Long audio segments containing fragments of the keyword (e.g., the phoneme sequence similar to the wake word in "Hey, how about Siri?").

In traditional KWS training, negative samples are typically randomly sampled from general speech corpora, leading to the following issues:
-   **Lack of Targeting**: Random negative samples often differ significantly from keywords in acoustic features, failing to provide effective discriminative training signals.
-   **Insufficient Hard Negatives**: True hard negative samples, which are most likely to cause false alarms, constitute an extremely small proportion in the training set.
-   **Incomplete Coverage**: It is impossible to systematically cover all possible confusion patterns.

## Methodology

### ASR Confusion Analysis Mining

**Confusion Matrix Analysis Based on ASR**:
-   Use an Automatic Speech Recognition (ASR) system to recognize all training data.
-   Construct an ASR confusion matrix: statistically identify which non-keywords are misrecognized as the target keyword by the ASR.
-   Extract high-frequency confusion pairs from the confusion matrix.
-   The original audio corresponding to these high-frequency confusion pairs constitutes the most valuable hard negative samples.

**Technical Details**:
-   Utilize large-scale ASR models (e.g., Transformer-based or RNN-T models).
-   The confusion matrix is computed at the phoneme level, providing finer-grained confusion information.
-   Confusion patterns at both the single-phoneme and phoneme-sequence levels can be discovered.

### Embedding Similarity Mining

**Similarity Search Based on Acoustic Embeddings**:
-   Train an acoustic embedding model (e.g., x-vector, d-vector).
-   Map all negative and positive samples into the embedding space.
-   Find negative samples closest to the target keyword embedding via cosine similarity or Euclidean distance.
-   These nearest-neighbor negative samples are the hard negatives in the embedding space.

**Technical Details**:
-   The embedding model is trained using Contrastive Learning or Triplet Loss.
-   The discriminative nature of the embedding space makes similarity search more precise.
-   Approximate Nearest Neighbor (ANN) search can be used to accelerate mining on large-scale datasets.

### Combined Mining Strategy

Fuse the results of ASR confusion analysis and embedding similarity search:
-   The ASR method excels at discovering semantic-level confusions (phoneme sequence similarity).
-   The embedding method excels at discovering acoustic-level confusions (spectral feature similarity).
-   The hard negatives discovered by the two methods are complementary.
-   The union combination provides the most comprehensive coverage of hard negative samples.

## Main Contributions

1.  **Systematic Negative Sample Mining Framework**: Proposes a complete KWS negative sample mining framework, incorporating two complementary methods: ASR confusion analysis and embedding similarity. This framework systematically discovers and leverages hard negative samples, rather than relying on random sampling or manual selection.
2.  **ASR Confusion Analysis**: Innovatively utilizes the recognition error patterns of ASR systems to guide negative sample selection. The confusion patterns of ASR directly reflect acoustic-level similarity in speech, which is highly consistent with the confusion patterns of KWS systems.
3.  **Significant Reduction in False Alarms**: By enhancing training with mined hard negative samples, the false alarm rate of the KWS system is significantly reduced.
4.  **Architecture Agnosticism**: This method is a pure data-level improvement that requires no modification to the KWS model architecture and can be used with any KWS model.

## Experimental Results

### Experimental Setup
-   Large-scale KWS dataset, containing target keywords and a large number of negative samples.
-   Baseline: KWS model trained with randomly sampled negative samples.
-   Evaluation Metrics: False Alarm Rate and Detection Rate.

### Main Results
-   **Reduced False Alarm Rate**: The false alarm rate is significantly reduced after using mined hard negative samples.
-   **Effectiveness of ASR Confusion Mining**: Negative samples discovered by ASR confusion analysis are the most effective in reducing the false alarm rate.
-   **Complementarity of Embedding Method**: Embedding similarity search discovers hard negative samples missed by the ASR method.
-   **Optimal Combined Strategy**: The combination of both methods achieves the lowest false alarm rate.
-   **Maintained Detection Rate**: While reducing false alarms, the detection rate remains almost unaffected.

### Ablation Studies
-   **Proportion of Hard Negatives**: An appropriate proportion (e.g., 30-50%) of hard negative samples provides the best performance.
-   **Comparison of Mining Methods**: The ASR method is stronger in discovering phoneme-level confusions, while the embedding method is stronger in acoustic feature-level confusions.
-   **Data Scale**: More mined negative samples continuously improve performance, but with diminishing marginal returns.

## Limitations and Future Work

### Method Limitations
-   **Infrastructure Requirements**: ASR confusion analysis requires a high-precision ASR system, and the embedding method requires training an embedding model.
-   **Dependence on Mining Quality**: The mining effectiveness depends on the quality of the ASR model or embedding model used.
-   **Computational Cost**: The mining process on large-scale datasets (ASR decoding, embedding computation, similarity search) requires significant computational resources.
-   **Incomplete Coverage**: It may not capture all types of confusing negative samples (e.g., false alarms caused by non-speech noise).

### Future Directions
-   Research online hard negative sample mining to dynamically discover and utilize hard negative samples during the training process.
-   Combine generative models (e.g., GANs, TTS) to automatically synthesize hard negative samples.
-   Explore active learning strategies to annotate mined hard negative samples and add them to the training set.
-   Extend to negative sample mining in multilingual and cross-lingual scenarios.
-   Research adaptive mining strategies to dynamically adjust mining targets based on the current weaknesses of the model.
