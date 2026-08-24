# Keyword Spotting for Hearing Assistive Devices Robust to External Speakers

- **Authors/Affiliations**: Ivan Lopez-Espejo, Zheng-Hua Tan, Jesper Jensen (Aalborg University, Oticon)
- **Date**: June 2019 (arXiv)
- **Link**: https://arxiv.org/abs/1906.09417
- **Keywords**: Keyword Spotting, Hearing Assistive Devices, Multi-Task Learning, Speaker Verification, Deep Residual Networks, Own-Speech, External Speakers

## Problem Statement

Standard Keyword Spotting (KWS) systems are designed to be **speaker-independent**, meaning that anyone uttering the keyword can trigger the system. While this design is reasonable for consumer devices such as smart speakers and mobile phones, it constitutes a significant functional flaw for **Hearing Assistive Devices (HADs)**:

1.  **False Triggering Issue**: In the context of hearing aids, only the user wearing the device should be able to control it via voice commands. If anyone nearby can trigger the device by saying the keyword, it leads to frequent false activations and degraded user experience—this is particularly prominent in social scenarios where family members, colleagues, or strangers of the device user might inadvertently utter the trigger word.
2.  **Security and Privacy Risks**: Malicious attackers could manipulate the settings of hearing aids (e.g., volume adjustment, mode switching) by uttering specific commands, potentially causing harm to hearing-impaired users.
3.  **Small Footprint Constraints**: Hearing aids have extremely limited computational resources and battery capacity. Any solution must operate within a very small model size and low power budget.
4.  **Specific Acoustic Conditions**: Hearing aids are typically used in noisy, real-world environments, requiring robust performance in both noise robustness and speaker discrimination simultaneously.

Therefore, the core challenge is to design a KWS system running on hearing aids that jointly performs keyword detection and speaker verification—ensuring that only the device user's voice can trigger commands, while maintaining a minimal model footprint.

## Methodology

This paper proposes a **Multi-Task Learning (MTL)** framework based on **Deep Residual Networks** to jointly perform keyword detection and own-voice/external speaker discrimination.

### 1. Multi-Task Learning Architecture

The core idea of the system is to learn two related tasks simultaneously through a shared backbone network:

-   **Task 1: Keyword Spotting (KWS)** – Identifying whether the input audio contains a predefined keyword.
-   **Task 2: Speaker Verification** – Determining whether the detected keyword was uttered by the device user (own-voice) or an external speaker.

#### 1.1 Shared Backbone Network

A **Deep Residual Network (ResNet)** is adopted as the shared backbone network:
-   **Input**: Acoustic features (e.g., MFCCs or spectrograms).
-   **Feature Extraction**: Multiple layers of residual blocks extract high-level acoustic representations.
-   **Representation Learning**: The backbone network learns general acoustic features useful for both tasks.

#### 1.2 Task-Specific Branches

Above the shared backbone network, two task-specific output branches diverge:
-   **KWS Branch**: Outputs the probability distribution of keyword classes (including a "no keyword" class).
-   **Speaker Verification Branch**: Outputs a binary classification result (own-voice vs. external speaker).

### 2. Multi-Task Loss Function

The total loss function is a weighted sum of the losses for the two tasks:

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{KWS} + \beta \cdot \mathcal{L}_{speaker}$$

Where $\mathcal{L}_{KWS}$ is the cross-entropy loss for keyword classification, $\mathcal{L}_{speaker}$ is the binary cross-entropy loss for speaker verification, and $\alpha$ and $\beta$ are hyperparameters balancing the weights of the two tasks.

### 3. Simulated Hearing Aid Corpus

Due to the lack of public hearing aid KWS datasets, this paper generates a dedicated speech corpus simulating hearing aid acquisition conditions from the Google Speech Commands dataset:
-   Simulates the acoustic effects of hearing aid microphone characteristics and wearing positions (in-the-ear/behind-the-ear).
-   Adds typical hearing aid environmental noises (indoor reverberation, background speech, etc.).
-   Generates "own-voice" and "external speaker" labels for each speaker.

### 4. Parameter Efficiency

A key advantage of the multi-task architecture is parameter efficiency:
-   Parameters of the shared backbone network are utilized by both tasks.
-   Only the speaker verification branch adds a small number of extra parameters.
-   The overall parameter increase is **negligible**, preserving the small-footprint characteristic.

## Main Contributions

1.  **First Identification and Resolution of Speaker-Sensitive KWS Needs for Hearing Aids**: Formalizes the special requirement of hearing aid KWS—that only the device user can trigger commands—into a multi-task learning problem combining KWS and speaker verification.
2.  **Efficient Multi-Task Learning Framework**: A multi-task architecture based on deep residual networks achieves synergistic optimization of both tasks through a shared backbone network, with negligible parameter increase. This demonstrates the practical value of multi-task learning on resource-constrained devices.
3.  **Significant Performance Improvement**: Compared to standard KWS systems that do not account for external speakers, the multi-task method achieves an approximate **32% relative accuracy improvement**, proving that joint learning of speaker information significantly enhances KWS performance.
4.  **Hearing Aid Simulated Corpus**: Generates a dedicated speech corpus simulating hearing aid acquisition conditions, providing a data foundation for this research direction.

## Experimental Results

### Performance Comparison

| Method | KWS Accuracy | Relative Improvement |
|------|-----------|---------|
| Standard KWS (ignoring external speakers) | Baseline | — |
| Multi-Task KWS + Speaker Verification | Significant Improvement | **~32%** |

-   The multi-task deep residual network achieves a relative KWS accuracy improvement of approximately 32% compared to the standard system.
-   The improvement is particularly significant in scenarios with external speaker interference.
-   The increase in model parameters is negligible, satisfying the small-footprint constraints of hearing aids.

### Key Findings
-   The speaker verification branch not only provides speaker verification functionality but also improves the performance of the KWS branch through the regularization effect of multi-task learning.
-   Shared representation learning enables the KWS task to acquire implicit awareness of speaker characteristics.

## Limitations and Future Work

### Technical Limitations
-   **Requirement for Own-Speech Registration Data**: The speaker verification branch requires voice samples from the device user for training/fine-tuning. New users need to provide a certain amount of own-speech data, which may pose a user experience barrier during actual deployment.
-   **Acoustic Similarity Interference**: When the acoustic characteristics (e.g., pitch, timbre) of external speakers are highly similar to those of the device user (e.g., family members), the performance of speaker verification may degrade.
-   **Gap Between Simulated and Real Data**: The evaluation uses simulated hearing aid data generated from Google Speech Commands, rather than real hearing aid recordings. The microphone characteristics, wearing position effects, and environmental noise of real hearing aids may be more complex.

### Future Directions
-   Explore few-shot/zero-shot speaker adaptation methods to reduce the registration data requirements for new users.
-   Utilize on-device continuous learning technologies to enable the system to continuously adapt to changes in the user's acoustic characteristics during use.
-   Evaluate on real hearing aid hardware to verify the performance gap between simulated and real data.
-   Integrate voice biometrics technology to further improve the accuracy and robustness of speaker verification.
-   Investigate joint KWS and speaker verification performance under extreme noise conditions.
