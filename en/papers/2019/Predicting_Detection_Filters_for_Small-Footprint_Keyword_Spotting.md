# Predicting Detection Filters for Small-Footprint Keyword Spotting

- **Authors/Affiliations**: Theodore Bluche, Thibault Gisselbrecht (Snips)
- **Date**: December 2019 (arXiv)
- **Link**: https://arxiv.org/abs/1912.07575
- **Keywords**: Keyword Spotting, Open Vocabulary, Detection Filters, Small Footprint, Neural Networks, Keyword Registration, Streaming Detection

## Problem Statement

Traditional keyword spotting systems face a fundamental design dilemma:

1. **Limitations of Closed-Vocabulary Systems**: Most KWS systems are designed for a fixed set of keywords (e.g., "OK Google", "Alexa"). Adding new keywords requires retraining or at least re-tuning the model. This is inflexible in scenarios where users wish to customize wake words.
2. **High Resource Requirements of Open-Vocabulary Systems**: KWS systems based on large-vocabulary ASR can detect arbitrary keywords, but their model size and computational load far exceed the capacity of resource-constrained devices. Typical end-to-end ASR models may require hundreds of MB of storage space and significant computational resources.
3. **Rigidity of Fixed Filters**: Traditional keyword-specific filters (such as template matching, keyword HMMs) require manual design or separate optimization for each keyword, lacking flexibility and incurring high maintenance costs.
4. **Deployment Constraints**: The memory budget for actual edge devices is extremely limited—KWS systems are typically restricted to a total memory footprint of **<250KB**, which excludes most open-vocabulary methods.

Therefore, the core challenge is to design a **fully neural open-vocabulary KWS system** that runs with a minimal memory footprint while supporting flexible registration of arbitrary keywords without retraining the main model.

## Methodology

This paper proposes an innovative fully neural open-vocabulary KWS method, with the core idea being **predicting detection filters via a neural network**.

### 1. System Architecture

The system consists of two neural networks working in synergy:

#### 1.1 Main KWS Network (Fixed)

The main KWS network is responsible for detecting the presence of keywords in streaming audio:
- Receives a continuous stream of acoustic features as input
- Performs matching/correlation operations on the input using detection filters
- Outputs a confidence score for the presence of a keyword
- **The main network parameters are completely fixed after deployment** and do not change with the keyword.

#### 1.2 Auxiliary Filter Prediction Network

The auxiliary network **predicts** detection filters from registration samples of a keyword:
- Input: One or more registration audio samples of the keyword
- Output: A set of detection filter parameters, serving as matching templates for the main KWS network
- Training method: Jointly trained with the main KWS network to learn the mapping from audio samples to effective filters

### 2. Keyword Registration Process

The process for registering a new keyword is extremely concise:
1. The user provides audio samples of the keyword (one or more)
2. The auxiliary network processes the samples to predict and generate the corresponding detection filters
3. The detection filters are loaded into the main KWS network
4. **No retraining or modification of the main model is required**

This design decouples the "knowledge" of the keyword from the model parameters into the predicted filters, achieving the flexibility of open vocabulary.

### 3. Streaming Detection Mechanism

During the inference phase:
- The main KWS network continuously processes the input audio stream
- It calculates the match degree between the audio features and the keyword template using the current detection filters
- A keyword detection event is triggered when the match degree exceeds a threshold
- Supports real-time streaming processing with low latency characteristics

### 4. Implementation of Minimal Memory Footprint

The total model size of the entire system (main network + filters) is controlled within **<250KB**:
- Compact network architecture design
- The detection filters themselves occupy very little storage space
- Satisfies the memory constraints of MCU-level devices

## Main Contributions

1. **Fully Neural Open-Vocabulary KWS**: Proposes for the first time a fully neural network-based open-vocabulary KWS system. By predicting detection filters via an auxiliary network, it eliminates the need for manually designed keyword-specific components.

2. **Minimal Memory Footprint (<250KB)**: While supporting the flexibility of open vocabulary, the entire system is controlled within 250KB, which is very compact for similar systems. This demonstrates that open-vocabulary KWS does not have to come at the cost of large model size.

3. **Flexible Keyword Registration Mechanism**: Users only need to provide audio samples of the keyword, and the system automatically generates detection filters without retraining or modifying the main model. The registration process is fast and user-friendly.

4. **Decoupled Design of Fixed Main Model + Dynamic Filters**: Separates the general acoustic processing capability of the model (fixed main network) from keyword-specific knowledge (dynamic filters), achieving a "train once, deploy for any keyword" deployment mode.

## Experimental Results

- The proposed system achieves effective keyword spotting performance with a total model size of less than 250KB.
- The predicted detection filters enable reliable keyword detection across different vocabulary entries.
- The registration process is fast—the conversion from audio samples to usable detection filters is almost instantaneous.
- Compared to closed-vocabulary systems, the performance loss brought by the flexibility of open vocabulary is within an acceptable range.

## Limitations and Future Work

### Technical Limitations
- **Sensitivity to the Number of Registration Samples**: When the number of registration samples is very small (e.g., only 1 sample), the predicted detection filters may not be robust enough, and detection performance may degrade.
- **Robustness to Acoustic Conditions**: Evaluation under noisy or far-field acoustic conditions is limited. Detection performance may degrade when the acoustic conditions of the registration samples do not match the actual usage conditions.
- **Model Size-Accuracy Trade-off**: The accuracy-model size trade-off for open-vocabulary systems has not been fully explored compared to closed-vocabulary systems optimized for specific keywords.

### Future Directions
- Research methods to generate more robust detection filters from a small number of registration samples, such as data augmentation and multi-sample ensembling.
- Explore adaptive filter update mechanisms—continuously improving filters based on the user's actual speech during use.
- Extend the system to scenarios involving simultaneous detection of multiple keywords, studying the parallel management and resource allocation of multiple detection filters.
- Integrate speaker verification functionality to enable open-vocabulary KWS systems to support both arbitrary keywords and specific speaker constraints.
- Verify the practical deployment feasibility of the system on lower-end hardware platforms (such as ARM Cortex-M series MCUs).
