# Sub-band CNN for Small-Footprint Keyword Spotting

- **Authors/Affiliations**: Chieh-Chi Kao, Ming Sun, Yixin Gao, Shiv Vitaladevuni, Chao Wang (Amazon)
- **Date**: July 2019 (Interspeech 2019)
- **Link**: https://arxiv.org/abs/1907.01448
- **Keywords**: Keyword Spotting, Sub-band CNN, Frequency Decomposition, Reduced Computational Cost, Small Footprint, Frequency Adaptation

## Problem Statement

Standard CNN-based keyword spotting models apply **identical** convolutional kernels across all frequency bands—i.e., using a uniform convolutional operation to process the entire frequency range. This "one-size-fits-all" approach suffers from efficiency issues:

1. **Uneven Distribution of Information**: The information content of speech signals is not uniformly distributed in the frequency domain. Low-frequency regions (such as the fundamental frequency and harmonics of vowels) typically carry the most discriminative information, while high-frequency regions contain relatively less information. Standard CNNs allocate the same computational resources to all frequency bands, leading to wasted computation.
2. **Bottleneck in Computational Efficiency**: On resource-constrained edge devices, every bit of computational saving is important. The computation performed by standard CNNs on frequency bands with low information content is redundant.
3. **Lack of Frequency Adaptation**: The convolutional kernels of standard CNNs use identical parameters at all frequency positions, preventing specialized modeling for the characteristics of different frequency regions.

Therefore, the core idea is to design a **frequency-adaptive** CNN architecture that allocates different amounts of computation to different frequency sub-bands based on their information content, thereby significantly reducing computational overhead while maintaining detection accuracy.

## Methodology

This paper proposes the **Sub-band CNN** architecture, which decomposes the input spectrogram into multiple sub-bands along the frequency dimension and applies independent convolutional kernels to each sub-band.

### 1. Frequency Sub-band Decomposition

The input spectrogram is divided into multiple **sub-bands** along the frequency dimension:
- Each sub-band covers a specific frequency interval of the original spectrogram.
- The division of sub-bands can be uniform (equal width) or non-uniform (e.g., based on the mel scale).
- Each decomposed sub-band becomes an independent input channel or a separately processed feature map.

### 2. Sub-band Specific Convolution

**Different convolutional kernels** are applied to each frequency sub-band:
- Each sub-band has its own independent convolutional parameters.
- Sub-bands with high information content (e.g., low-frequency regions) can use larger/more convolutional kernels.
- Sub-bands with low information content (e.g., high-frequency regions) can use smaller/fewer convolutional kernels.
- This design allows the model to allocate computation based on the information content of each frequency region.

### 3. Sub-band Feature Fusion

The independent convolutional outputs from each sub-band need to be fused:
- **Concatenation**: Concatenating the outputs of each sub-band along the channel dimension.
- **Aggregation**: Fusing cross-sub-band information through subsequent convolutional or fully connected layers.
- The fused features are used for final keyword classification.

### 4. Computational Optimization

The core mechanism by which Sub-band CNN reduces computational cost:
- **Reducing Kernel Size**: Using smaller convolutional kernels in sub-bands with low information content.
- **Reducing Channel Count**: Using fewer output channels in sub-bands with low information content.
- **Skipping Low-Information Sub-bands**: Using minimal operations or skipping sub-bands with extremely low information content.

### 5. Comparison with Standard CNN

| Aspect | Standard CNN | Sub-band CNN |
|------|---------|---------|
| Convolutional Kernel | Shared across all frequencies | Independent per sub-band |
| Computational Allocation | Uniform | Allocated based on information content |
| Frequency Modeling | Uniform | Frequency-adaptive |
| Computational Cost | Baseline | Significantly reduced |

## Main Contributions

1. **Frequency-Adaptive Sub-band Processing**: Introduced, for the first time in the KWS domain, an adaptive convolutional architecture based on frequency sub-band decomposition. Different frequency bands use convolutional kernels of varying complexity, achieving the goal of allocating computation based on information content.

2. **Significant Reduction in Computational Cost**: Achieved **39.7%** and **49.3%** reductions in computational cost compared to the baseline CNN architecture (under different configurations), demonstrating the great potential of frequency-adaptive strategies in computational efficiency.

3. **Verification of Unequal Band Contribution**: Ablation studies proved that the contribution of different frequency bands to keyword spotting is indeed unequal—low-frequency sub-bands are typically more important, while high-frequency sub-bands can be processed with less computation.

4. **Maintained Competitive Accuracy**: While significantly reducing computational requirements, the model maintained competitive keyword spotting accuracy. This proves that the reduction in computational cost primarily comes from optimizing redundant frequency regions rather than losing discriminative information.

5. **Published at Interspeech 2019**, representing an important contribution by Amazon to efficient KWS architectures.

## Experimental Results

### Google Speech Commands Dataset

| Configuration | Computational Reduction | Accuracy Impact |
|------|-----------|-----------|
| Sub-band CNN Config A | **39.7%** | Maintained competitiveness |
| Sub-band CNN Config B | **49.3%** | Maintained competitiveness |

### Key Findings
- The reduction in computational cost is concentrated in high-frequency sub-bands—these sub-bands use fewer/smaller convolutional kernels.
- Low-frequency sub-bands retain more complete convolutional operations to ensure that critical discriminative information is not lost.
- The optimal sub-band decomposition may vary for different keywords, but a unified decomposition strategy is effective across all keywords.

## Limitations and Future Work

### Technical Limitations
- **Generality of Sub-band Decomposition**: The optimal sub-band decomposition (number of sub-bands, frequency range division) may vary depending on different keywords or acoustic environments. The current fixed decomposition strategy may not be optimal for all scenarios.
- **Increased Architectural Complexity**: Compared to standard CNNs, Sub-band CNN increases additional architectural complexity—managing independent convolutions and feature fusion for multiple sub-bands increases the difficulty of implementation and deployment.
- **Interpretability of Frequency Features**: There is limited analysis of the learned frequency-specific features and their physical significance. Which frequency bands are most important for which keywords, and the relationship between this importance and phonetic knowledge, has not been explored in depth.

### Future Directions
- Research adaptive sub-band decomposition—dynamically adjusting the division of sub-bands based on input content or target keywords.
- Explore the connection between learned sub-band weights and phonetic knowledge (e.g., frequency characteristics of phonemes).
- Combine Sub-band CNN with other compression techniques (such as quantization and pruning) to further reduce computational cost.
- Evaluate the performance of Sub-band CNN under noisy and far-field conditions, studying the differential impact of noise on different sub-bands.
- Explore dynamic frequency band selection based on attention mechanisms to further optimize frequency-adaptive strategies.
