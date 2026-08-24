# An Integrated Framework for Two-pass Personalized Voice Trigger

- **Authors/Affiliations**: Xiongjun Zhang, Yulong Wan, Jing Xu, Wei Rao, Shidong Shang - Xiamen University
- **Date**: 2021.06
- **Link**: https://arxiv.org/abs/2106.15950
- **Keywords**: Personalized Voice Trigger, Two-stage Detection, Speaker Verification, Keyword Spotting, False Rejection, Score Fusion, Feature Sharing

## Problem Statement

Personalized Voice Trigger is a critical feature in modern voice assistant devices (such as smartphones, smart speakers, and smart earbuds). The system needs to respond only to wake-up words spoken by specific registered users, while rejecting all of the following scenarios:
1. Other users speaking the same wake-up word (preventing unauthorized activation of the device)
2. Registered users speaking non-wake-up words in daily conversation (preventing false triggers)
3. Noise and non-speech sounds in the environment (preventing noise-induced false triggers)

Single-stage systems typically use a unified model to perform keyword detection and speaker discrimination simultaneously. However, this approach faces a core contradiction: to achieve a low False Rejection Rate (FRR, i.e., not missing the registered user's wake-up word) for the registered user, the model's decision boundary needs to be sufficiently loose, but this simultaneously leads to an increase in the False Alarm Rate (FAR, i.e., accepting wake-up words spoken by non-registered speakers) for non-registered speakers. Conversely, the opposite is also true. These two objectives are difficult to optimize simultaneously within the decision space of a single model.

The core problem this paper aims to solve is: How to design an integrated two-stage framework where the first stage focuses on high-quality keyword detection (regardless of speaker identity), and the second stage focuses on precise speaker verification. By employing efficient inter-module feature sharing and flexible score fusion strategies, it achieves personalized voice triggering—maintaining an extremely low FRR for registered users while effectively rejecting all non-registered users and non-wake-up words.

## Methodology

### Overall Framework Design
The paper proposes an integrated two-stage (Two-pass) framework consisting of three core components: a shared feature extractor, a keyword detection module, and a speaker verification module. The key design principle is "stage-wise processing with shared low-level features."

### Shared Feature Extractor
The foundation of the two-stage framework is a shared feature extractor that serves both KWS and SV tasks:

- **Architecture**: A multi-layer CNN (such as DS-CNN or a ResNet variant) is used as the shared feature extractor to extract general acoustic features from MFCC or spectrogram inputs.
- **Sharing Hierarchy Design**:
  - **Hard Sharing**: KWS and SV completely share the same encoder, with only the final classification heads differing. This approach is the most parameter-efficient, but the two tasks may interfere with each other.
  - **Partial Sharing**: The lower layers (1-4) are shared, while the higher layers (5-8) branch into KWS-specific and SV-specific branches. This approach strikes a balance between parameter efficiency and task specificity.
  - **Paper's Choice**: A partial sharing strategy is adopted—the lower 4 layers are shared, and the higher layers are divided into a KWS branch and an SV branch. The lower layers share the extraction of general acoustic features (such as phoneme-level spectral patterns), while the higher branches focus on task-specific information (the KWS branch focuses on the temporal structure of keywords, and the SV branch focuses on the speaker's vocal tract features).

- **Computational Efficiency of Feature Sharing**: The shared feature extractor allows the two-stage inference to obtain intermediate features with only one forward pass, significantly reducing computational overhead. Compared to running two completely independent models, feature sharing can reduce the total computation by approximately 40-50%.

### Stage 1: Keyword Spotting (KWS)
The goal of the first stage is to detect whether the wake-up word is present, regardless of speaker identity:

- **KWS Module**: On the KWS branch of the shared features, a few task-specific convolutional/fully connected layers are used for keyword classification.
- **Output**: Keyword detection score $S_{KWS}$ (range [0,1], representing the confidence that the keyword exists).
- **Threshold Setting**: The first-stage threshold $\theta_{KWS}$ is set relatively loosely (biased towards high recall), ensuring that the registered user's wake-up word is almost never rejected by the first stage. A loose threshold means that some wake-up words from non-registered users will pass the first stage and enter the second stage.
- **Frame-level Detection**: The KWS module operates at the frame level, continuously listening to the audio stream. When the keyword score in a certain time window exceeds the threshold, it triggers the second stage.

### Stage 2: Speaker Verification (SV)
The goal of the second stage is to confirm whether the audio that triggered the first stage comes from a registered user:

- **SV Module**: On the SV branch of the shared features, a network structure dedicated to speaker verification (such as an x-vector style architecture based on statistics pooling + fully connected layers) is used to extract speaker embeddings (speaker embedding/d-vector).
- **Registration Process**: The registered user records the wake-up word a few times on the device. The SV module extracts the speaker embedding for each recording and calculates the average to serve as the registration template $e_{reg}$.
- **Verification Process**: For audio triggered by the first stage, the SV module extracts its speaker embedding $e_{test}$, and then calculates the cosine similarity with the registration template:
  $$S_{SV} = \text{cosine\_similarity}(e_{test}, e_{reg}) = \frac{e_{test} \cdot e_{reg}}{||e_{test}|| * ||e_{reg}||}$$
- **Threshold Setting**: The second-stage threshold $\theta_{SV}$ is set relatively strictly (biased towards low false alarm rate), ensuring that wake-up words from non-registered users are effectively rejected.

### Score Fusion Strategies
The final decision requires combining the scores from both the KWS and SV stages. The paper systematically explores various fusion strategies:

1. **Cascading Strategy**:
   - Decision Rule: if $S_{KWS} > \theta_{KWS}$ AND $S_{SV} > \theta_{SV}$ then ACCEPT
   - The two stages make independent decisions; samples that pass the first stage enter the second stage.
   - Advantages: Simple, inference-efficient (the first stage quickly filters out a large number of non-keywords).
   - Disadvantages: Independent tuning of the two thresholds is difficult.

2. **Multiplicative Fusion**:
   - $S_{final} = S_{KWS} * S_{SV}$
   - if $S_{final} > \theta$ then ACCEPT
   - Multiplicative fusion requires both scores to be high to pass, naturally implementing the logic of "satisfying both conditions simultaneously."
   - Advantages: Mathematically elegant, the scales of the two scores are automatically aligned.
   - Disadvantages: If either score is zero, the final score becomes zero, which may be too strict.

3. **Weighted Fusion**:
   - $S_{final} = \alpha * S_{KWS} + (1 - \alpha) * S_{SV}$
   - if $S_{final} > \theta$ then ACCEPT
   - $\alpha$ controls the relative weight of KWS and SV and needs to be tuned according to the application scenario.
   - Advantages: Flexible, can balance the importance of the two tasks.
   - Disadvantages: Requires tuning the $\alpha$ parameter.

4. **SVM/Logistic Regression Fusion (Learned Fusion)**:
   - Train an SVM or logistic regression classifier using labeled validation data, taking $(S_{KWS}, S_{SV})$ as input features to output the final accept/reject decision.
   - Advantages: Can learn non-linear decision boundaries.
   - Disadvantages: Requires additional training data, increasing system complexity.

5. **Calibrated Fusion**:
   - Perform score calibration (e.g., Platt Scaling, Isotonic Regression) on the KWS and SV scores separately, so that the calibrated scores have probabilistic meanings.
   - Then perform fusion on the calibrated scores.
   - Calibration can significantly improve fusion performance, especially when the score distributions of the two modules are inconsistent.

### Training Strategy
- **Multi-task Training**: The shared feature extractor is trained using a joint loss for KWS and SV:
  $$L = \alpha * L_{KWS} + \beta * L_{SV}$$
  KWS uses cross-entropy loss, and SV uses contrastive loss or AAM-Softmax loss.
- **Stage-wise Training**: First, train the shared feature extractor + KWS branch (on KWS data), then freeze the lower layers and train the SV branch (on speaker verification data).
- **Data Augmentation**: Use SpecAugment, noise injection, speed perturbation, etc., to enhance the generalization ability of both stages.

## Main Contributions

1. **Introduction of an integrated two-stage framework for personalized voice triggering**: Decomposes personalized voice triggering into two independent sub-tasks: keyword detection and speaker verification. Each sub-task can be optimized independently, avoiding the conflict between the two objectives in single-stage methods. This "divide and conquer" design philosophy has proven very effective in engineering practice.

2. **Efficient feature sharing between KWS and SV modules**: Through the partially shared feature extractor, the computational overhead of the two-stage framework increases by only about 50-60% compared to a single-stage KWS system (rather than 200% for two independent models), making it feasible to deploy a two-stage personalized system on edge devices.

3. **Systematic analysis of different score fusion strategies**: The paper systematically compares various strategies such as cascading, multiplicative, weighted, learned, and calibrated fusion, providing clear guidance for practitioners to choose the optimal fusion strategy.

4. **Achieving a better trade-off between the registered user's False Rejection Rate and the non-registered user's False Alarm Rate**: The two-stage framework allows for independent adjustment of the thresholds for the two stages, enabling the maintenance of an extremely low FRR for registered users (<1%) while keeping the FAR for non-registered users at an extremely low level (<0.5%).

## Experimental Results

### Dataset and Setup
- **Dataset**: PVTC2020 Challenge dataset (Chinese wake-up word "Ni Hao Wen Wen") and internal datasets.
- **Evaluation Metrics**: Equal Error Rate (EER), FRR at fixed FAR, Detection Cost Function (DCF).
- **Number of Speakers**: Dozens of registered and non-registered speakers.

### Two-stage vs. Single-stage
- **Equal Error Rate**: The EER of the two-stage framework is approximately 2-3%, significantly lower than that of single-stage methods (approximately 5-8%).
- **Registered User FRR**: Under the condition of fixed FAR=1%, the FRR of the two-stage framework is approximately 2-3%, about 50% lower than that of single-stage methods.
- **Non-registered User FAR**: The two-stage framework effectively rejects over 90% of non-registered user wake-up words.

### Comparison of Score Fusion Strategies
- **Multiplicative Fusion**: Simple and effective, performs well in most scenarios (EER approx. 2.5%), requiring no additional training.
- **Weighted Fusion**: Performs best when $\alpha=0.4$ (slightly lower weight for KWS), with an EER of approx. 2.3%.
- **SVM Fusion**: Slightly outperforms simple methods when training data is sufficient (EER approx. 2.0%), but requires additional training data.
- **Calibrated Fusion**: Performing Platt Scaling calibration on raw scores before fusion reduces the EER by about 0.3-0.5 percentage points. The effect of calibration is particularly significant when the score distributions of the two modules differ greatly.

### Effect of Feature Sharing
- **Computational Efficiency**: The total computation of partial sharing (lower layer sharing + higher layer branching) is approximately 55-60% of two independent models, reducing inference time by about 40%.
- **Performance Impact**: The impact of feature sharing on KWS performance is minimal (accuracy drop <0.5%), while the impact on SV performance is slightly larger (EER increase approx. 0.5-1%), but still within acceptable limits.

### Ablation Studies
- **Number of Shared Layers**: Sharing 4 layers (out of 8 total) achieves the best balance between computational efficiency and task performance. Sharing too few layers (2 layers) weakens the efficiency advantage of feature sharing, while sharing too many layers (more than 6) causes the performance of both tasks to decline (task interference).
- **First-stage Threshold**: The overall performance of the two-stage system is best when $\theta_{KWS}=0.3$ (loose threshold). Thresholds that are too low (<0.2) increase the computational burden on the second stage, while thresholds that are too high (>0.5) cause some registered user wake-up words to be rejected in the first stage.
- **SV Loss Function**: AAM-Softmax loss outperforms standard Softmax loss and contrastive loss, providing better speaker discrimination capability on the speaker verification branch.

## Limitations and Future Work

### Technical Limitations
- **Two-stage Inference Latency**: Although feature sharing reduces computation, two-stage inference (especially when the SV branch needs to be run after the first stage is triggered) still increases latency by approximately 50-100% compared to single-stage methods. In scenarios extremely sensitive to latency (such as real-time voice interaction), this additional latency may be unacceptable.
- **Registration Data Requirements**: The second-stage SV module requires the registered user to record the wake-up word a few times to serve as the registration template. The registration process is an extra step for the user, affecting user experience.
- **Framework Performance Depends on Sub-module Quality**: If the KWS module's performance degrades under noisy conditions, the performance of the entire system will also be affected. The performance bottlenecks of both modules directly impact the final results.
- **Speaker Variability**: When a registered user is sick (sore throat, nasal congestion) or experiences emotional changes, their acoustic features may deviate from the registration template, causing the SV module to reject legitimate users.

### Experimental Design Shortcomings
- Evaluation under far-field and noisy conditions is limited, which are the most common challenges in actual deployment.
- System security under adversarial attacks (such as speech synthesis and voice conversion attacks) was not evaluated.
- The number of speakers is small (dozens), which may not be sufficient to evaluate the system's performance in large-scale user scenarios.
- Online learning and adaptive mechanisms (such as gradually updating the registration template based on the user's daily usage) were not explored.

### Future Improvement Directions
- Explore more efficient end-to-end personalized KWS architectures, incorporating speaker information as conditional input to the KWS model to avoid independent two-stage inference.
- Combine anti-spoofing modules to prevent speech synthesis and replay attacks, enhancing system security.
- Research adaptive registration template updates—continuously optimizing the registration template using the user's successful wake-up records (confirmed by the user) to adapt to long-term changes in speaker features.
- Explore unsupervised speaker adaptation—automatically identifying the registered user's voice features and updating the model after deployment without requiring explicit re-registration by the user.
- **Inspiration for the KWS field**: The two-stage framework is a classic and effective solution for personalized KWS problems. Its "divide and conquer" philosophy can be generalized to more KWS scenarios that need to satisfy multiple constraints simultaneously (such as satisfying both keyword accuracy and acoustic environment adaptation).
