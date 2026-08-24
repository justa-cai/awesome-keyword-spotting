# Exploring Filterbank Learning for Keyword Spotting

**Authors/Affiliations**: Ivan Lopez-Espejo, Zheng-Hua Tan, Jesper Jensen (Aalborg University, Oticon A/S)

**Date**: June 2020 (arXiv:2006.00217)

**Link**: https://arxiv.org/abs/2006.00217

**Keywords**: Filterbank Learning, Keyword Spotting, Feature Extraction, End-to-End Learning

## Problem Statement

Traditional Keyword Spotting (KWS) systems use hand-designed Mel filterbanks or MFCC features for front-end feature extraction. These features are based on human auditory perception models (the Mel scale) and are not optimized for any specific speech application. Although Mel filterbanks perform well on general speech tasks, whether they constitute the optimal feature representation for the KWS task is a question worth investigating.

The KWS task has specific characteristics:
- **Limited number of keywords**: Only a small number of keyword classes need to be distinguished.
- **Short keyword duration**: Typically 1-2 seconds, with spectral feature patterns differing from continuous speech.
- **Discriminative frequency bands**: Differences between keywords may exist only in specific frequency bands.

Therefore, learning a filterbank optimized for the KWS task may provide a more discriminative feature representation than a generic Mel filterbank.

## Methodology

### Method 1: Constrained Filterbank Learning

Learning filter parameters under structural constraints of the Mel filterbank:
- **Learnable parameters**: Center Frequency, Bandwidth, and Gain of each filter.
- **Filter shape**: Gaussian-shaped filters, maintaining the smoothness and interpretability of the filters.
- **Operating domain**: Operates on the Power Spectrum domain.
- **Constraints**: Filters must satisfy basic constraints such as non-negativity and normalization.
- **Advantages**: The learned filters have clear physical meanings, allowing for analysis of differences from Mel filterbanks.

### Method 2: Unconstrained Filterbank Learning

Completely removing shape constraints on filters:
- Each filter is a freely learned weight vector.
- Can learn frequency responses of arbitrary shapes.
- **Advantages**: Maximum flexibility, capable of learning non-standard filtering patterns that Mel filterbanks cannot represent.
- **Risks**: Potential overfitting to training data, with uncertain generalization ability.

### End-to-End Training Pipeline

The entire KWS system is treated as an end-to-end trainable pipeline:
1. **Short-Time Fourier Transform (STFT)**: Computes the power spectrum of the audio (fixed operation, non-trainable).
2. **Learnable Filterbank Layer**: Projects the power spectrum into filterbank outputs (trainable).
3. **Log Compression**: Takes the logarithm of the filterbank outputs (fixed operation).
4. **Classifier Network**: A CNN classifier that performs keyword classification using the learned features (trainable).

Filterbank parameters and classifier parameters are optimized through joint training.

## Main Contributions

1. **First exploration of filterbank learning in KWS**: Systematically explores the application of filterbank learning in KWS tasks for the first time, filling a research gap in this direction.

2. **Comparison of constrained and unconstrained methods**: Compares constrained (Gaussian-shaped) and unconstrained filterbank learning methods, revealing the trade-off between flexibility and stability:
   - **Constrained method**: Good stability, strong interpretability, and robust performance improvements.
   - **Unconstrained method**: High flexibility, potentially learning superior filtering patterns, but with a higher risk of overfitting.

3. **End-to-end feature learning**: Incorporates front-end feature extraction into end-to-end training, eliminating the sub-optimality that may arise from hand-crafted feature design.

4. **Analysis of learned filters**: Visualizes and analyzes the shapes of learned filters, finding significant differences from Mel filterbanks in specific frequency bands. These differences are related to the discriminative requirements of the KWS task.

## Experimental Results

### Experimental Setup
- Google Speech Commands dataset.
- Baseline: Standard Mel filterbank (40 dimensions).
- Learned filterbank: Same dimensionality (40 dimensions).
- Evaluation of performance under different noise conditions.

### Main Results
- **Accuracy improvement**: Learned filterbanks outperform fixed Mel filterbanks in both clean and noisy conditions.
- **Constrained method**: Provides a stable accuracy improvement of approximately 0.5-1%, with improvements across all conditions.
- **Unconstrained method**: Achieves larger improvements under certain conditions, but the gains are less stable.
- **Noise robustness**: The advantages of learned filters are more pronounced under low SNR conditions.
- **Analysis**: Learned filters tend to allocate narrower bandwidths to frequency bands with high information content for keywords.

### Filter Visualization
- The shapes of learned filters show systematic differences from Mel filters.
- Some filters offer finer frequency resolution in discriminative frequency bands of keywords (e.g., the F1 and F2 formant regions of vowels).
- Some filters have narrower bandwidths than Mel filters, providing higher frequency selectivity.

## Limitations and Future Work

### Method Limitations
- **Increased training complexity**: Filterbank learning increases the complexity of the training process and the need for hyperparameter tuning.
- **Overfitting risk**: The unconstrained method may overfit to specific noise conditions in the training data.
- **Limited comparison with other front-end methods**: No comparison was made with other adaptive front-end methods (e.g., learnable Gammatone filters).
- **Computational overhead**: Additional computational overhead during the training phase.

### Future Directions
- Research multi-task filterbank learning (simultaneously optimizing multiple KWS-related tasks).
- Explore joint time-frequency learning, learning not only filter shapes but also time window parameters.
- Combine attention mechanisms to achieve dynamic filter selection.
- Validate the generalization ability of the method on larger and more diverse datasets.
- Investigate the transfer value of filterbank learning to other speech tasks (speech recognition, speaker recognition).
