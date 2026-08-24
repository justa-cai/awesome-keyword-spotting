# The 2020 Personalized Voice Trigger Challenge: Open Database, Evaluation Metrics and the Baseline Systems

- **Authors/Affiliations**: Xiongjun Zhang, Yulong Wan, Jing Xu, Wei Rao, Shidong Shang - Xiamen University (XMUSPEECH); Duke Kunshan University
- **Date**: 2021.01
- **Link**: https://arxiv.org/pdf/2101.01935
- **Keywords**: Personalized Voice Trigger, Challenge, Open Database, Evaluation Metrics, Baseline Systems, Speaker Verification, Keyword Spotting, PVTC2020

## Problem Statement

Personalized Voice Trigger Detection requires a system to respond only to the wake-word spoken by a specific registered user, while rejecting the same wake-word spoken by anyone else, as well as non-wake-word speech. This technology has significant application value on personal devices (smartphones, smart speakers)—the device responds only to the owner's wake-word.

However, the field faces several key issues:
1. **Lack of standardized benchmarks**: Different research groups use different private datasets and evaluation protocols, making it difficult to fairly compare research results.
2. **Inconsistent evaluation metrics**: Some use Equal Error Rate (EER), some use miss rate at a fixed false alarm rate, and others use Detection Cost Function (DCF).
3. **Non-public datasets**: Most studies use internal company datasets, making it difficult for the academic community to reproduce and advance the field.
4. **Insufficient exploration of fusion strategies for KWS and Speaker Verification**: There is still no consensus on how to optimally combine keyword detection scores and speaker verification scores.

The 2020 Personalized Voice Trigger Challenge (PVTC2020) aims to address these issues by providing an open database, standardized evaluation metrics, and reproducible baseline systems.

## Methodology

### Open Database Construction
The challenge provided an open dataset specifically designed for personalized voice trigger research:
- **Wake-word**: "Nihao Wenwen" (Chinese for "Hello Wenwen")
- **Number of speakers**: Recordings from multiple speakers
- **Recording conditions**: Includes different distances (near-field/far-field) and different environments (quiet/noisy)
- **Data types**:
  - Registration speech: Target wake-word registration recordings for each speaker
  - Positive test samples: Wake-words spoken by registered speakers
  - Negative test samples: Wake-words spoken by non-registered speakers, non-wake-word speech, and noise

### Standardized Evaluation Metrics
The challenge defined multi-dimensional evaluation metrics:
- **False Rejection Rate (FRR)**: The proportion of registered speakers' wake-word utterances that are rejected
- **False Acceptance Rate (FAR)**: The proportion of non-registered speakers' speech or non-wake-word utterances that are accepted
- **Detection Cost Function (DCF)**: A weighted cost combining FRR and FAR
- **Equal Error Rate (EER)**: The error rate when FRR equals FAR
- **DET Curve (Detection Error Tradeoff)**: The tradeoff curve between FRR and FAR at different thresholds

### Baseline System Design
The challenge provided two baseline systems for participants to reference:

**Baseline 1 - Single-stage Method**:
- Uses a single neural network to perform keyword detection and speaker identification simultaneously
- The output contains joint predictions for both keyword class and speaker identity
- Advantages: Simple inference, single forward pass
- Disadvantages: Losses from the two tasks may interfere with each other

**Baseline 2 - Two-stage Method (KWS + SV Cascade)**:
- Stage 1: A generic keyword detection model detects the wake-word (regardless of speaker)
- Stage 2: A speaker verification model confirms whether it is a registered speaker
- The two modules are trained independently and executed in cascade during inference
- Score fusion: Explores different fusion strategies (multiplicative fusion, weighted fusion, SVM fusion)

### Evaluation Protocol
- **Task 1**: Joint wake-word detection and speaker verification (requires both correct keyword and matching speaker)
- **Task 2**: Personalized detection under more complex acoustic conditions (including noise, far-field, etc.)
- Evaluation uses standard cross-validation protocols to ensure result reproducibility

## Main Contributions

1. **Release of an open database specifically for personalized voice trigger research**: This is one of the first publicly available dedicated datasets in this field, filling the gap in personalized KWS data for academia, enabling meaningful research and comparison.

2. **Establishment of standardized evaluation metrics and protocols**: Defined a comprehensive evaluation metric system (FRR, FAR, DCF, EER), providing a unified measure for the community.

3. **Provision of reproducible baseline systems**: Including single-stage and two-stage baseline implementations, providing a comparison benchmark for subsequent research. The baseline systems use public architectures and training recipes, allowing any researcher to reproduce them.

4. **Analysis of different score fusion strategies**: Systematically compared the performance differences of methods such as multiplicative fusion, additive fusion, and SVM fusion when combining KWS and SV scores.

## Experimental Results

### Baseline System Performance
- **The two-stage method significantly outperforms the single-stage method**: Across almost all evaluation metrics, the two-stage method (KWS+SV cascade) outperforms the single-stage joint prediction.
- **EER of the two-stage method**: Achieved reasonable Equal Error Rates on the challenge dataset (specific numbers vary by task configuration).
- **Limitations of the single-stage method**: The single-stage model has weaker speaker discrimination capability compared to dedicated speaker verification models.

### Comparison of Score Fusion Strategies
- **Multiplicative fusion** (Score_KWS * Score_SV) is simple and effective, performing well in most scenarios.
- **Weighted fusion** (alpha * Score_KWS + beta * Score_SV) requires tuning the alpha and beta parameters.
- **SVM fusion** slightly outperforms simple fusion methods when training data is sufficient.
- **Importance of calibration**: Calibrating KWS and SV scores (e.g., using Platt Scaling) before fusion can significantly improve performance.

### Challenge Results
- Multiple participating teams achieved significant improvements over the baseline systems.
- The winning systems' performance far exceeded the baselines, demonstrating the technical potential of the field.
- The challenge provided valuable benchmark data for the community.

## Limitations and Future Work

### Dataset Limitations
- **Dataset scale**: Compared to private datasets used in the industry, the PVTC2020 dataset scale is relatively limited.
- **Language constraints**: The dataset only contains Chinese wake-words, which may limit the cross-lingual generalization of methods.
- **Acoustic diversity**: Although it includes various recording conditions, it may not fully represent the acoustic diversity in real-world deployments.
- **Number of speakers**: The limited number of registered speakers may affect the statistical reliability of the speaker verification module.

### Limitations of Evaluation Metrics
- The weight parameter selection for DCF may not be suitable for all application scenarios.
- Deployment-related metrics such as computational efficiency and latency were not considered.

### Baseline System Limitations
- Simpler architectures were used; more advanced models (e.g., ECAPA-TDNN for SV, Conformer for KWS) might achieve better results.
- End-to-end personalized KWS methods (e.g., using speaker information as conditional input to the KWS model) were not explored.

### Future Improvement Directions
- Extension to multi-language, multi-wake-word personalized detection.
- Exploration of robustness evaluation under adversarial attacks (e.g., voice cloning, replay attacks).
- Integration of anti-spoofing modules to prevent synthetic speech attacks.
- Insights for the KWS field: Standardized challenges and datasets play a crucial role in promoting academic research; more similar open benchmarks should be established in the future.
