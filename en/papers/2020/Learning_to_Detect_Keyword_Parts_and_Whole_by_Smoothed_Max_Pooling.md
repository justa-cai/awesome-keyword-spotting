# Learning to Detect Keyword Parts and Whole by Smoothed Max Pooling

**Authors/Affiliations**: Mohammadreza Tajik, Armand Navabi, Shankar Kumar, Michiel Bacchiani, Raziel Alvarez, Yanzhang He, Rohit Prabhavalkar (Google Inc.)

**Date**: January 2020 (arXiv:2001.09246)

**Link**: https://arxiv.org/abs/2001.09246

**Keywords**: Keyword Spotting, Smoothed Max Pooling, Part Detection, Temporal Modeling, Streaming Inference

## Problem Statement

In end-to-end streaming Keyword Spotting (KWS) models, audio is input as a sequence of frames, and the model produces predictions frame by frame. The final detection decision requires aggregating frame-level predictions into segment-level keyword detection scores. Existing aggregation methods suffer from the following issues:

**Max Pooling**:
- Takes the maximum value of frame-level predictions within a sliding window as the detection score.
- Advantages: Captures the most significant keyword evidence.
- Disadvantages: Extremely sensitive to outliers; noise predictions in a single frame can lead to false triggers.

**Average Pooling**:
- Takes the average of frame-level predictions within a sliding window.
- Advantages: Robust to outliers (smoothing effect).
- Disadvantages: May dilute the strong signal of keyword frames, leading to reduced sensitivity for detecting complete keywords.

**Fundamental Problem**: Keywords are temporal structures composed of multiple parts (e.g., "Hey Siri" consists of two parts: /HH EY/ and /S IY R IY/). An ideal aggregation method should:
1. Detect individual components of the keyword (Part-level Detection).
2. Aggregate part-level detections into a whole-keyword detection (Whole-level Detection).
3. Be robust to outliers while maintaining sensitivity to keyword signals.

## Methodology

### Part Detectors

**Design Motivation**: Keywords consist of multiple distinguishable phoneme segments. Detecting each part separately is more robust than directly detecting the entire keyword.

**Implementation**:
- Define $K$ sub-segments (Parts) for the keyword.
- Each sub-segment corresponds to a specific temporal segment of the keyword (e.g., the first phoneme, middle phonemes, last phoneme).
- Train frame-level classifiers, where each classifier is responsible for detecting the occurrence of one sub-segment.
- Output of part detectors: The probability of each part's occurrence at each time frame.

**Training Label Generation**:
- Use Forced Alignment to determine the temporal boundaries of each phoneme in the keyword.
- Divide the keyword into several sub-segments based on phoneme boundaries.
- Label each frame as belonging to a specific sub-segment or not belonging to any sub-segment.

### Smoothed Max Pooling

The core innovation of this paper:

**Mathematical Definition**:
- Standard Max function: $\max(x_1, x_2, ..., x_n)$ is non-differentiable and sensitive to outliers.
- Smoothed approximation: $\text{SmoothedMax}(x_1, ..., x_n; \alpha) = \frac{\sum(x_i \cdot \exp(\alpha \cdot x_i))}{\sum(\exp(\alpha \cdot x_i))}$
- $\alpha$ is an adjustable smoothing parameter:
  - $\alpha \to \infty$: Degenerates into standard Max Pooling.
  - $\alpha \to 0$: Degenerates into Average Pooling.
  - $0 < \alpha < \infty$: Provides a smooth transition between Max and Average.

**Properties**:
- Differentiable: Can be trained end-to-end.
- Robust to outliers: Does not ignore information from other frames due to a single high-value frame.
- Adjustable: Flexibly controls the balance between Max and Average behaviors via the $\alpha$ parameter.

### Part-Whole Detection Framework

**Aggregation Process**:
1. The frame-level network generates probabilities for keyword parts for each frame.
2. For each part, apply Smoothed Max Pooling over a temporal window to aggregate frame-level predictions.
3. Obtain the detection score for each part.
4. Combine the detection scores of all parts to obtain the overall keyword detection score:
   - Method: Geometric mean or minimum of all part scores (requiring all parts to be detected).
   - Overall Score = $\min(\text{part\_1\_score}, \text{part\_2\_score}, ..., \text{part\_K\_score})$

### Streaming Processing Adaptation

- The sliding window moves with the audio stream.
- Apply the above part-whole detection process at each window position.
- Output detection scores in real-time, supporting low-latency trigger decisions.

## Main Contributions

1. **Smoothed Max Pooling Operation**: Proposes a differentiable temporal aggregation operation that provides an adjustable smooth transition between the selectivity of Max Pooling and the robustness of Average Pooling. This operation is not only applicable to KWS but also serves as a reference for other tasks requiring temporal aggregation.

2. **Part-Whole Detection Framework**: Innovatively decomposes keyword detection into part-level detection and whole-level aggregation. This hierarchical approach leverages the internal temporal structure of keywords, making it more robust than end-to-end whole-keyword detection.

3. **Differentiable Aggregation Parameter**: The smoothing parameter $\alpha$ can be learned during training, allowing the model to automatically find the optimal balance between Max and Average.

4. **Streaming Compatibility**: The entire framework is designed to be compatible with streaming processing, meeting the low-latency requirements for practical KWS deployment.

## Experimental Results

### Experimental Setup
- Google Speech Commands dataset.
- Streaming KWS evaluation (detection on continuous audio streams).
- Baselines: Standard Max Pooling, Average Pooling, Smoothed Max Pooling.
- Evaluation Metrics: Detection accuracy, False Alarm Rate.

### Key Results
- **Smoothed Max > Max > Average**: Smoothed Max Pooling outperforms both standard Max and Average Pooling in terms of detection accuracy and false alarm rate.
- **Part Detection Gain**: The part detection framework is more robust than direct whole-keyword detection, especially when there are significant variations in keyword temporal alignment.
- **Smoothing Parameter Learning**: The $\alpha$ value learned automatically during training allows the model to adaptively adjust aggregation behavior across different scenarios.
- **Noise Robustness**: Smoothed Max Pooling is more robust to frame-level noisy predictions, reducing false triggers caused by outlier high-value frames.

### Ablation Studies
- Part-Whole vs. Whole Detection: The part detection framework significantly outperforms direct whole-keyword detection.
- Learning $\alpha$ vs. Fixed $\alpha$: Learning $\alpha$ provides better adaptive capabilities.
- Number of Parts: Dividing keywords into 3-5 parts achieves the best balance between performance and complexity.

## Limitations and Future Work

### Method Limitations
- **Additional Hyperparameters**: The $\alpha$ parameter and the part segmentation strategy require tuning.
- **Dependency on Part Definition**: Part detectors require phoneme-level temporal boundaries for the keyword (used to generate training labels).
- **Computational Overhead**: The computation of Smoothed Max Pooling is slightly higher than simple Max or Average pooling.
- **Part Segmentation Strategy**: The optimal method for dividing keyword sub-segments requires further research.

### Future Directions
- Research automated keyword part discovery methods that do not rely on phoneme-level annotations.
- Explore adaptive part segmentation, dynamically adjusting based on the acoustic characteristics of the input audio.
- Extend Smoothed Max Pooling to other sequence prediction tasks (e.g., Voice Activity Detection).
- Combine attention mechanisms to replace fixed part segmentation, enabling end-to-end part-whole learning.
- Research multi-granularity part-whole detection (phoneme-level, syllable-level, word-level).
