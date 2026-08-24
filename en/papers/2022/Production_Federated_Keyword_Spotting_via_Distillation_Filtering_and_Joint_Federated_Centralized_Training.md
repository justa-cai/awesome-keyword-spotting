# Production Federated Keyword Spotting via Distillation, Filtering, and Joint Federated-Centralized Training

- **Authors/Affiliations**: Andrew Hard, Kurt Partridge, Neng Chen, Sean Augenstein, Aishanee Shah, Hyun Jin Park, Alex Park, Sara Ng, Jessica Nguyen, Ignacio Lopez Moreno, Rajiv Mathews, Francoise Beaufays (Google LLC; Sara Ng is also affiliated with the University of Washington)
- **Date**: April 2022 (Interspeech 2022)
- **Link**: https://arxiv.org/abs/2204.06322
- **Keywords**: federated learning, keyword spotting, distillation, filtering, domain adaptation, multi-task learning, privacy-preserving training

## Problem Statement

### Problem Background and Domain Pain Points
Google’s "Hey Google" voice wake-up system operates on over a billion Android devices globally, including smartphones, smart speakers, smart displays, watches, and more. The system's performance is directly tied to user experience: false rejects mean users must repeat the wake-word (surveys indicate that user satisfaction drops by approximately 15% for each additional repetition), while false accepts cause the device to respond unexpectedly (e.g., waking up while music is playing), which can even trigger privacy concerns if the device records conversations when it was not intended to be awakened.

To continuously improve KWS performance, it is necessary to utilize real user interaction data for model training. The value of this data lies in its inclusion of the most diverse acoustic conditions found in the real world: hundreds of environmental noises, various accents and speaking styles, speakers of different ages (from children to the elderly), various device positions (in pockets, on tables, at a distance), and various interference conditions (TV sound, other speakers). This diversity is unmatched by any laboratory-recorded dataset.

However, voice data is highly sensitive—it may contain private conversations, medical information, financial information, and personally identifiable information. Under privacy regulations such as GDPR and CCPA, uploading raw user audio to servers for centralized training poses severe legal compliance risks. Even if anonymized (e.g., by removing device IDs), raw audio can still be linked to specific individuals via voiceprint recognition technologies.

### Specific Shortcomings of Existing Methods
- **Privacy Risks of Centralized Training**: Traditional methods upload user data to servers for centralized training. Even with differential privacy (DP) or data anonymization, raw audio may still leak identifiable information. At Google’s scale (billions of devices), the impact of any privacy incident would be catastrophic.
- **Label Absence in Standard Federated Learning**: Federated Learning (FL) allows models to be trained on-device, uploading only model gradients rather than raw data, thereby fundamentally protecting user privacy. However, standard FL assumes that device-side data has labels—in image classification, user behaviors (such as clicks or browsing) can serve as labels; but in the KWS scenario, audio snippets cached on the device lack human-annotated labels. The device only knows that it "detected a keyword" (which might be a false alarm) or that the "user said the wake-word but the device did not respond" (which might be a miss), making it impossible to determine the true class of the audio.
- **Domain Mismatch Issues**: The distribution of device-side cached data differs from that of centralized training data. Device-side caches contain a large amount of "system-accepted" audio (which may include non-keywords that were false alarms) but lack "system-rejected" audio (because rejected audio is discarded under privacy designs). This incomplete data distribution may lead to poor generalization of pure FL models in certain domains.
- **Limitations of On-Device Training Capabilities**: Mobile devices have limited computational and storage resources. On-device training can only use simple data augmentation (such as SpecAugment) and cannot perform complex augmentation operations (such as noise mixing or room impulse response simulation), limiting the diversity of training data. Furthermore, on-device training must be completed within minutes (as users may use the device at any time), imposing strict requirements on training speed.

### Key Challenges Addressed by This Paper
How to train high-quality KWS models using large-scale real user data (approximately 300K clients, millions of audio clips) while protecting user privacy (without uploading raw audio or directly exposing user data), simultaneously addressing three core issues: label absence (device-side data lacks human annotation), label noise (inaccurate predictions by the teacher model for edge cases), and domain mismatch (incomplete distribution of device-side cached data).

## Methodology

### Overall Architecture Design
This paper proposes a production-grade federated KWS system that integrates three key technological innovations: Federated Distillation, Confidence Filtering, and Joint Federated-Centralized Training. These three techniques address label absence, label noise, and domain mismatch, respectively, forming a complete end-to-end federated training solution.

### Model Architecture
- **Two-Stage Stacked SVDF (Singular Value Decomposition Filter)**: Approximately 320K parameters
- The SVDF layer decomposes the fully connected layer $W \in \mathbb{R}^{m \times n}$ into low-rank matrix multiplication $W \approx U \cdot V$, where $U \in \mathbb{R}^{m \times r}$ and $V \in \mathbb{R}^{r \times n}$, with $r \ll \min(m, n)$. This decomposition reduces the number of parameters from $mn$ to $r(m+n)$ and correspondingly reduces the computational load. SVDF is particularly suitable for training and inference on mobile devices, as the low-rank decomposition reduces memory footprint and computational cost.
- **Bottleneck (Encoder-Decoder) Structure**: Bottleneck layers are added between SVDF layers to compress the dimensionality of intermediate representations, further reducing the size of gradients uploaded from the device (enhancing privacy) and communication bandwidth requirements.
- **Input Features**: Log-mel filter bank energies (40 dimensions, 10ms frame shift, 25ms frame length, approximately 1 second of audio = 100 frames), using 40 frequency channels to cover the 0-8kHz speech frequency range.

### Core Algorithm 1: Federated Distillation

**Problem Definition**: Device-side cached audio lacks human-annotated labels $y$. The training objective of standard FL is to minimize $\sum_i L(y_i, \hat{y}_i)$, but $y_i$ is not accessible.

**Mathematical Description of the Solution**:
1. Train a "teacher model" $T_\theta$ (which can be a more powerful model architecture) on centralized labeled data $\mathcal{D}_{labeled}$ on the server.
2. The teacher model generates "soft labels" for the device-side cached audio $x_i$: $\tilde{y}_i = T_\theta(x_i) = \text{softmax}(\text{logits}/\tau)$, where $\tau$ is the temperature parameter.
3. The soft label $\tilde{y}_i \in \mathbb{R}^C$ ($C$ is the number of classes) is a probability distribution representing the teacher model's confidence in each class—for example, $\tilde{y}_i = [0.02, 0.85, 0.10, 0.03]$ indicates that the teacher believes there is an 80% probability that the class is 2.
4. Soft labels are distributed to corresponding devices via a Secure Aggregation protocol.
5. Devices use soft labels as supervision signals for local training: $L_i = D_{KL}(\tilde{y}_i \| \hat{y}_i)$ (KL divergence loss).
6. Devices upload local gradients $\nabla L_i$ to the server for federated averaging (FedAvg aggregation).

**Advantages of Soft Labels vs. Hard Labels**:
- Soft labels contain similarity relationships between classes (e.g., the soft labels for "yes" and "yeah" might be $[0.7, 0.3]$ and $[0.3, 0.7]$, respectively, reflecting their acoustic similarity), providing richer supervision signals for the student model.
- The temperature parameter $\tau$ in soft labels controls the "softness" of the labels—a higher temperature makes the distribution flatter (smaller differences between classes), facilitating knowledge transfer.
- The teacher model has access to centralized labeled data (thousands of hours of annotated speech), providing high-quality "pseudo-labels" for the unannotated device-side data.

### Core Algorithm 2: Confidence Filtering

**Problem Definition**: The soft labels from the teacher model are not perfect. When the domain of the cached audio falls outside the teacher model's training distribution (e.g., extreme noise, special accents, children's voices), the teacher may provide incorrect soft labels. Training the student model with incorrect soft labels introduces noise and may even lead to "worse performance with more training."

**Solution—Confidence Assessment Based on User Feedback Signals**:
This paper cleverly utilizes user feedback signals naturally generated in the KWS system to assess the reliability of teacher labels:

1. **Server Reject Signal**: When the on-device KWS detects a keyword, the audio is sent to the server for secondary verification (by a more powerful server-side model). If the server rejects it (i.e., the server determines it is not a keyword), it suggests that the on-device detection might be a false alarm. For these samples, the teacher's soft label may be inaccurate (because the teacher and the on-device model disagree on the "borderline" nature of the sample).

2. **Retry Signal**: If the user repeats the wake-word within a short period (e.g., within 2 seconds), it suggests that the previous attempt may not have been detected (a miss). The cached audio from the previous attempt might be a sample that was "accepted" by the system but was actually a false alarm.

3. **Speaker ID Assistance**: Utilize speaker recognition information on the device (e.g., "voiceprint features of users registered on this device") to assist in judgment—if the speaker of the cached audio does not match the registered user, the reliability of the sample is reduced.

**Specific Implementation of Filtering Strategy**:
- If any of the above signals indicate that the teacher label may be incorrect, the sample is marked as "low confidence" and excluded from on-device training.
- The filtering is "conservative"—it is better to discard some samples with correct labels (false negative filtering) than to ensure that the retained samples have high label quality (low false positive rate).

**Label Accuracy Validation Experiment**:
The paper conducted a dedicated label accuracy study to validate the effectiveness of the filtering strategy:
- Recruited 117 professional annotators.
- Manually annotated 11,908 samples (each sample was independently annotated by 3 annotators, with the majority vote taken as the result).
- Results: The label accuracy of retained samples after filtering was approximately 15-20 percentage points higher than that of unfiltered samples, validating the effectiveness of confidence filtering.

### Core Algorithm 3: Joint Federated-Centralized Training

**Problem Definition**: Device-side caches lack certain data domains (such as specific background noise environments, non-target speech, or silence segments) because only audio "accepted" by the system is cached. This leads to insufficient generalization of pure FL models in these domains—the model may have encountered these domains in real user environments but did not see them during training.

**Solution**:
Combine FL training with server-based centralized training, merging their gradient/parameter updates on the server:

1. **FL Rounds**: Aggregate gradient updates from thousands of devices using FedAvg.
   - Data: Real user audio cached on devices (privacy-preserving, in-domain data).
   - Advantages: Real user distribution, personalized noise conditions.

2. **Centralized Rounds**: Train using publicly available datasets and simulated data.
   - Data: Google Speech Commands + simulated noise/reverberation augmented data (does not involve user privacy data).
   - Advantages: Complete domain coverage (including domains missing from device-side caches).

3. **Alternating Training**: FL rounds and centralized rounds are performed alternately (e.g., every 5 FL rounds + 1 centralized round).
4. **Parameter Merging**: The server maintains a global model parameter $\theta$, and updates from both FL and centralized rounds act on $\theta$.

**Why Joint Training is Needed Rather Than Pure FL**:
The domain missing in device-side caches is specifically manifested in: (1) Device-side caches hardly contain "obvious non-keywords" (because non-keywords are rejected on the device), making it difficult for the model to learn the precise boundary of "what is a non-keyword"; (2) Device-side caches lack audio under extreme noise conditions (because in extreme noise, the system may frequently false alarm or miss, leading to very poor label quality in cached samples); (3) The cache distribution varies greatly across different devices (the cache content of high-end smartphones and low-end smart speakers is completely different). Centralized training compensates for these missing domains.

### Privacy Protection Mechanisms
- All user data remains on the device; only model gradients are uploaded (encrypted via a secure aggregation protocol, making it impossible for the server to reverse-engineer any individual device's gradient from the aggregated gradients).
- Teacher soft labels are distributed via encrypted channels, not exposing user audio content.
- Complies with Google's privacy principles and operates under a DP framework.

## Main Contributions

1. **First Large-Scale Production-Grade Federated KWS System**: This is the first work to successfully deploy federated learning for KWS training on real user devices. It involves nearly 300,000 independent clients, with approximately 6,000 FL training rounds per day. This scale far exceeds the typical scale of academic FL research (usually hundreds to thousands of clients, often in simulated environments), proving the feasibility and scalability of FL in industrial-grade KWS systems.

2. **Federated Distillation Solves Label Absence**: Innovatively uses soft labels from the teacher model as supervision signals for on-device training, bypassing the fundamental obstacle of lacking human-annotated labels for device-side data. Federated distillation is a key enabling technology for FL in the KWS scenario—without it, FL training cannot proceed (as loss cannot be calculated without labels).

3. **Confidence Filtering Ensures Training Quality**: The first to utilize on-device user feedback signals (server rejects, retries, speaker ID) to filter unreliable teacher labels. The ingenuity of this strategy lies in its use of "secondary signals" (user behavior feedback) naturally generated in the system, requiring no additional cost for human annotation or privacy risks.

4. **Joint Training Solves Domain Mismatch**: Combines FL with centralized training to creatively solve the problem of "incomplete domain coverage in device-side cached data." The joint training paradigm is not only applicable to KWS but can also be generalized to other tasks requiring FL (such as input method prediction and recommendation systems).

## Experimental Results

### Datasets Used and Their Scale
- **Federated Data**: Approximately 300,000 independent client devices, with a median of 175 samples cached per client (average duration 1.7 seconds), totaling approximately 52.5 million samples.
- **Centralized Data**: Google's internal annotated speech data (thousands of hours), plus simulated augmented data.
- **Model**: SVDF model with approximately 320K parameters.

### Label Accuracy Study
- 117 annotators annotated 11,908 samples.
- The label accuracy of retained samples after confidence filtering was significantly higher than the unfiltered baseline (an improvement of approximately 15-20 percentage points).
- Inter-annotator agreement (Cohen's Kappa) > 0.85.

### Model Performance Comparison

**Offline Evaluation**:
- Federated Distillation + Confidence Filtering + Joint Training > Pure Centralized Training > Pure FL (without distillation).
- On standard KWS test sets, the detection rate of the federated training model improved by approximately 5-10% (relative improvement) at the same false alarm rate.
- Specifically, under the condition of fixed FPR=0.1%, the TPR of the federated model was approximately 3-5% higher than that of the pure centralized model.

**Online A/B Testing**:
A/B experiments conducted on real user groups (lasting several weeks, involving millions of devices):
- Experimental Group (Federated Training Model) vs. Control Group (Previous Version of Centralized Training Model).
- User satisfaction metrics (based on whether users continued to use the voice assistant, false wake-up feedback rate, etc.) significantly improved.
- False wake-up rate decreased by approximately 10-15% (relative decrease), and miss rate decreased by approximately 5-8%.

### Ablation Experiments for Each Component

**Necessity of Federated Distillation**:
- Removing distillation (using hard labels instead of soft labels) -> Model performance on hard cases decreased by approximately 3-5%.
- The inter-class similarity information encoded in soft labels is crucial for the training of the student model.

**Effect of Confidence Filtering**:
- Filtering removed approximately 10-20% of low-confidence samples.
- After removing these samples, the model's performance on hard cases (borderline samples, high-noise samples) improved most significantly.
- Performance degradation without filtering was mainly reflected in an increase in the false positive rate—low-quality labels caused the model to learn incorrect decision boundaries.

**Contribution of Joint Training**:
- Removing centralized training rounds -> Model performance under atypical noise conditions decreased by approximately 2-3%.
- Centralized data provided "negative samples" (clear non-keywords) missing from device-side caches, helping the model establish a more precise "keyword vs. non-keyword" decision boundary.

## Limitations and Future Work

### Technical Limitations of the Method
- **User Scale Requirements**: FL requires a sufficiently large user base to ensure statistical convergence of FedAvg. The paper used approximately 300K clients, but for products with a smaller user base (such as emerging voice assistants), FL may not provide enough training data to guarantee convergence quality.
- **Limitations of On-Device Training Capabilities**: The computational power of mobile devices limits the complexity of on-device training. Currently, only simple spectral augmentation (such as SpecAugment) is supported, and complex noise mixing or RIR augmentation cannot be executed on the device. This limits the diversity of on-device training data.
- **Communication Overhead**: Each FL round requires uploading model gradients (320K parameters x 4 bytes = approximately 1.3MB), which may generate non-negligible traffic and energy consumption overhead on mobile networks (especially in 3G/4G environments).
- **Impact of Heterogeneous Data Distribution**: Data distributions vary greatly across different devices (e.g., smartphones in quiet offices vs. smart speakers in noisy kitchens). Convergence of FedAvg under Non-IID data distributions may be slow or converge to suboptimal solutions.

### Shortcomings in Experimental Design
- **Lack of Quantitative Analysis of Privacy-Utility Trade-off**: The paper does not provide specific values for the differential privacy (DP) budget $\epsilon$, nor does it analyze performance loss under different privacy levels. In industrial deployment, the strength of DP guarantees directly affects legal compliance.
- **Incomplete Baseline Comparisons**: There is a lack of fair comparison with pure centralized training on the same scale of data (using the same amount of centralized labeled data vs. FL data), making it difficult to quantify the gains brought by FL itself.
- **Total Cost of FL Training Not Reported**: Including server-side aggregation computational costs, on-device training energy consumption, and communication bandwidth costs.

### Possible Future Improvement Directions
- **Stronger Differential Privacy Guarantees**: Introduce formal user-level DP guarantees (such as $(\epsilon, \delta)$-DP with $\epsilon=10$) in FL training, while minimizing the impact of privacy budgets on model performance. This is an active research direction—recent work indicates that through better gradient clipping and noise calibration strategies, the performance loss of DP-FL can be controlled within 1-2%.
- **Personalized Federated Learning**: Learn personalized KWS models for different users or device types, rather than a single global model. For example, learn noise-robust models for in-car devices and high-precision models for desktop devices in quiet environments. This can be achieved through meta-learning or clustering-based FL.
- **Asynchronous Federated Training**: Support asynchronous on-device training and server aggregation (currently using synchronous FedAvg), reducing dependence on device online rates and response times. Asynchronous FL allows devices to submit gradient updates at any time, with the server aggregating them immediately.
- **Inspiration for the KWS Field**: This paper proves that "privacy protection and model performance are not contradictory"—by using federated distillation and joint training, KWS performance can be improved using real data without sacrificing user privacy. This paradigm provides a replicable blueprint for privacy-compliant training in speech AI and has reference value for privacy-compliant practices across the speech technology industry.
