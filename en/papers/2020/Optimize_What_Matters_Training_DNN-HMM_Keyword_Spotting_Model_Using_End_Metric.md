# Optimize what matters: Training DNN-HMM Keyword Spotting Model Using End Metric

**Authors/Affiliations**: Ashish Shrivastava, Arnav Kundu, Chandra Dhir, Devang Naik, Oncel Tuzel (Apple Inc.)

**Date**: November 2020 (arXiv:2011.01151)

**Link**: https://arxiv.org/abs/2011.01151

**Keywords**: DNN-HMM, Keyword Spotting, Wake Word Detection, End-to-End Training, Differentiable HMM, Loss Function Optimization

## Problem Statement

In traditional DNN-HMM Keyword Spotting (KWS) systems, there is a fundamental training-deployment mismatch: the DNN acoustic model is trained independently at the phoneme state level using Cross-Entropy Loss, while the evaluation metric during actual deployment is based on the keyword detection score from the Viterbi decoder. This inconsistency between the loss function and the final evaluation metric is the primary reason for suboptimal performance in KWS systems, particularly in small models with limited capacity.

Specifically, the cross-entropy loss requires the DNN to accurately predict the posterior probabilities of all phoneme states at every frame, whereas the KWS system only cares about the final keyword detection score. The DNN allocates equal optimization effort to all phoneme states (including those irrelevant to the detection decision), diluting the modeling capability for critical discriminative states. This problem is especially severe in low-capacity models, as the limited model parameters cannot simultaneously model all phoneme states precisely.

## Methodology

### Core Technical Idea

This paper proposes an innovative end-to-end training strategy: transforming the HMM decoder (dynamic programming algorithm) into a differentiable module, allowing gradients to backpropagate through the decoder to directly optimize the final keyword detection score.

### Differentiable HMM Decoder

Traditional Viterbi decoding uses the `argmax` operation, which is non-differentiable. The key innovation in this paper includes:
- Replacing the discrete selection in the Viterbi path with a softmax probability-weighted sum, making the decoding process differentiable.
- Calculating the keyword detection score using the Forward-Backward Algorithm, where the score is the probability-weighted sum of all possible state paths.
- The detection score is defined as the log-likelihood ratio of the keyword HMM model likelihood to the filler model likelihood, given the observation sequence.

### IOU-Based Training Window Sampling

During training, the generation of positive and negative sample windows is based on the Intersection-over-Union (IOU) with the true keyword boundaries:
- **Positive Samples**: Windows with IOU above a threshold; the training objective is to maximize the detection score.
- **Negative Samples**: Windows with IOU below a threshold; the training objective is to minimize the detection score.
- This sampling strategy simulates the sliding window detection scenario in actual streaming inference.

### Training Objective Function

The overall loss function combines positive and negative samples:
- For positive samples (high IOU): maximize the detection score.
- For negative samples (low IOU): minimize the detection score.
- Margin Loss is used to ensure sufficient separation between positive and negative samples.

### Key Technical Advantages

1. **Zero Inference Overhead**: This method only changes the training algorithm; the model architecture and inference flow remain completely unchanged.
2. **Reduced Annotation Requirements**: Only start and end time annotations for keywords are needed, without requiring frame-level phoneme labels.
3. **Automatic Focus on Critical States**: End-to-end training enables the model to automatically learn to focus on the phoneme states most important for detection decisions.

## Main Contributions

1. **First End-to-End Training Strategy for DNN-HMM KWS Systems**: Solves the long-standing loss-metric mismatch problem by incorporating the HMM decoder into an end-to-end differentiable training framework, injecting the advantages of end-to-end optimization into traditional DNN-HMM architectures.

2. **Differentiable HMM Decoder Design**: Achieves gradient backpropagation through probabilistic Viterbi decoding, allowing the detection score to be used directly as the optimization target. This technical insight is inspiring for the entire speech recognition and keyword detection field.

3. **IOU Window Sampling Strategy**: Proposes a natural and effective method for constructing training data, making the training process closer to actual streaming detection scenarios and reducing the distribution gap between training and deployment.

4. **Significant Practical Value**: Achieves substantial performance improvements without adding any computational or memory overhead during inference. This is particularly important for deployment on resource-constrained edge devices.

## Experimental Results

### Experimental Setup
- Evaluated on Apple's internal large-scale real-world wake word dataset.
- Used the DNN-HMM KWS system from the actual production environment as the baseline.
- Evaluation metrics include the trade-off curve between False Rejection Rate (FRR) and False Trigger Experience.

### Key Results
- **FRR Reduced by Over 70%**: At the same false trigger level, the False Rejection Rate of the end-to-end trained model is reduced by over 70% compared to the traditionally independently trained DNN-HMM. This improvement is significant for actual product systems.
- **Smaller Models Benefit More**: Experiments show that the relative improvement brought by end-to-end training is more significant for models with smaller capacity. This is because smaller models are more susceptible to loss-metric mismatch.
- **Confusion Matrix Analysis**: The end-to-end trained model predicts critical discriminative states more accurately, while its performance on irrelevant states may be worse than that of the cross-entropy trained model, verifying the method's ability to automatically focus on critical states.

### Ablation Analysis
- End-to-end training enables the DNN to learn to allocate modeling capabilities unequally: critical states for detection decisions receive more modeling resources.
- This adaptive learning of state importance is unachievable with traditional cross-entropy training.

## Limitations and Future Work

### Method Limitations
- **Annotation Requirements**: Although frame-level phoneme labels are not required, precise start and end time annotations for keywords are still needed, which may be difficult to obtain in some scenarios.
- **Architecture Dependency**: The method is specifically designed for the DNN-HMM architecture; its applicability to pure end-to-end models (such as CTC, RNN-T, Attention-based) has not been verified.
- **Dataset Limitations**: Experiments were only validated on Apple's internal dataset, lacking comparisons on public benchmark datasets.

### Future Directions
- Extend the idea of differentiable decoding to other sequence decision tasks (such as speech recognition, speech activity detection).
- Explore weakly supervised training methods that do not require keyword boundary annotations.
- Combine this method with model compression techniques (quantization, pruning) to further optimize edge deployment performance.
- Investigate the applicability of this method in multi-keyword detection and open-vocabulary scenarios.
