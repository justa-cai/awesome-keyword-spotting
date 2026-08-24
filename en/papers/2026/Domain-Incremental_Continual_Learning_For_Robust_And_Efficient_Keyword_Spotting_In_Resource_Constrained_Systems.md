# Domain-Incremental Continual Learning for Robust and Efficient Keyword Spotting in Resource Constrained Systems

- **Authors/Affiliations**: Prakash Dhungana, Sayed Ahmad Salehi - University of Kentucky
- **Date**: 2026.01 (arXiv 2601.16158v1, submitted January 22, 2026)
- **Link**: https://arxiv.org/abs/2601.16158
- **Keywords**: keyword spotting, domain-incremental continual learning, catastrophic forgetting, rehearsal buffer, class prototype, on-device learning, quantization

## Problem Statement

### Problem Background and Domain Pain Points
Keyword Spotting (KWS) serves as a persistent entry point for voice interaction on edge devices such as smart speakers and wearables. Models must simultaneously satisfy low complexity, small memory footprint, and high energy efficiency to run in real-time on microcontroller unit (MCU)-level hardware. However, there is a systematic **domain shift** between the training datasets of such models and the real-world deployment environments: new speakers, new accents, new background noises, and new recording conditions all alter the input distribution. Empirical evidence cited in the paper indicates that when the training and deployment environments are inconsistent, KWS model accuracy can degrade by up to 27% [Citation 1]. For always-on wake-word systems, this is not a one-time issue—moving a device to a different room, changing streets, or having a washing machine turned on nearby changes the domain.

The traditional response is to transmit data back to the cloud for retraining and then push the updated model, but this is infeasible under three constraints: First, transmitting raw audio back poses privacy risks (the paper explicitly lists privacy-sensitive scenarios as target applications); Second, the computational and human costs of repeated full-scale retraining are high; Third, manual intervention cannot be relied upon during the deployment cycle of edge devices. Continual Learning (CL) thus emerges as a candidate path—allowing the model to autonomously absorb new domain data on the device while preserving old knowledge and preventing catastrophic forgetting. CL is categorized into three scenarios by scene: class-incremental (new classes), task-incremental (new tasks with the same number of classes), and **domain-incremental (same classification task, input distribution changes with the domain)**. This paper chooses domain-incremental learning: the classification head remains "Yes/No," but the noise environment varies.

However, bringing CL to an MCU is significantly more difficult than bringing it to a server: training itself requires intermediate activation memory for backpropagation, while MCUs have only hundreds of kB of SRAM; runtime data lacks manual annotation, and pseudo-label errors can poison retraining; models are typically INT8 quantized, requiring additional mechanisms for gradient updates on quantization parameters. These constraints collectively define the problem space of this paper.

### Specific Deficiencies of Existing Methods
The paper reviews gaps in existing solutions along two lines: "on-device deep learning" and "domain-adaptive CL."

- **Inference-side models are too heavy to support on-device training.** Architectures such as MobileNet, MCUNet, and DSCNN used in ODDA have parameter counts between 23.7k and 480k. While sufficient for inference, enabling learning on the device significantly increases processing latency per sample due to the additional memory and computation required for backpropagation. The larger the model, the more constrained the selection of "which layers can be updated" becomes.
- **ODDL[4]: Updates only the classifier layer.** It re-trains the classifier layer of DSCNN using pre-labeled samples mixed with noise at 0 dB SNR, evaluating "seen" and "unseen" noise during training separately. This leaves three issues: The feature extraction backbone is completely frozen, unable to adapt to statistical drift in features of new domains; validation was performed only at a single 0 dB SNR; the implementation relies on the GAP9 multi-core parallel processor, creating a high hardware barrier. A more critical assumption flaw is that it assumes some deployment noise was seen during training, whereas in real deployments, noise is entirely new.
- **Lin et al.[14] (MCUNetv3 series) restrict on-device training to specific layers/parameters**, using a proprietary combination of Quantization Aware Scaling (QAS), sparse updates, and Tiny Training Engine (TTE) to validate in a simulated embedded environment on a standard GPU—far from real MCU deployment.
- **TAP-SLDA[16] updates only Gaussian prototypes without moving the backbone**, improving accuracy by at most 11% compared to not updating Gaussian prototypes, but its representational capacity is locked within prototype statistics.
- **Vu et al.[15]’s binary network CL targets class-incremental learning**—accommodating new classes by expanding the classifier layer, not addressing domain shift.
- **Rehearsal-based CL (iCaRL, GEM, DER, etc.)** has proven effective in using rehearsal buffers to combat forgetting, but previously focused mainly on class-incremental and task-incremental learning, without combining "runtime pseudo-label supervision" for domain-incremental learning.

In summary, the common shortcomings of existing methods are: **limited update scope (only moving the classifier layer or sparse layer selection) prevents backbone features from following the new domain; validation noise conditions are singular (mostly 0 dB); and it is assumed that some deployment noise was already seen during the training phase.**

### Key Challenges This Paper Aims to Solve
The core question the paper seeks to answer is: Can a domain-incremental CL closed-loop be constructed within an MCU-level parameter budget (thousands of parameters) that does not rely on manual annotation, does not require parallel accelerators, and maintains approximately 94% accuracy across a wide SNR range of −10 dB to 10 dB? This breaks down into five sub-challenges: (1) The model must be small enough to allow **full-parameter updates** rather than just moving the classifier layer, otherwise domain adaptation capabilities are limited; (2) Unlabeled data at runtime requires a reliable pseudo-label filtering mechanism to prevent erroneous samples from polluting retraining; (3) The reliability of input features must be guaranteed first—if denoising is unreliable, both classification and pseudo-label filtering will collapse; (4) Training must close the loop on INT8 quantized models; (5) Re-training must combat forgetting, ensuring old knowledge from clean domains is not lost.

## Methodology

### Overall Architecture Design and Design Motivation
The overall framework operates in two stages: **runtime effective sample determination** (judging at inference time whether "this sample is worth storing as training material") and **fixed-interval CL re-training** (accumulating to a cycle point, using "rehearsal buffer + augmented rehearsal buffer + effective samples" to form a mini-batch to re-train the entire model, while simultaneously updating class prototypes and distance thresholds).

Supporting this closed-loop is a four-component KWS pipeline: **Wavelet Denoising (raw waveform domain) → Feature Extraction (CMSIS-DSP integer MFCC + Mel Spectrogram) → Spectral Denoising (feature domain) → Quantized CNN Classifier**. The placement of each component has a clear "why":

- **Wavelet denoising is placed at the very beginning** because processing non-stationary noise in the waveform domain preserves the integrity of both temporal and spectral structures (the paper states it "removes nonstationary noise while preserving the temporal and spectral integrity"); the output is directly quantized to [−128, 127], keeping the entire frontend in integer arithmetic.
- **Feature extraction uses integer MFCC from the CMSIS-DSP library** to avoid the latency and memory costs of floating-point operations—this is the real bottleneck on low-end M4 cores.
- **Mel spectrogram as a second input incurs zero additional cost**: LogMel is inherently an intermediate product of the MFCC calculation pipeline; feeding it into the classifier is equivalent to getting a second perspective on features for free.
- **Spectral denoising is placed in the feature domain**, responsible for cleaning up noise components residual from the wavelet stage and drifting in feature statistics due to environmental changes; normalizing feature maps to [0,1] before subsequent operations is done to achieve magnitude invariance—ensuring mask thresholds remain stable across different recording volumes.
- **The classifier is a three-layer 5×5 Conv2D + Flatten + FC**, all without zero padding. Removing padding is not just for convenience: the paper explicitly states that experiments verified zero padding has minimal impact on accuracy; removing it results in smaller feature maps in subsequent layers, reducing both memory and computation.
- **Latent representation is taken from the output of the third Conv2D layer**, a choice driven purely by resources: this latent layer has only 160 bytes, whereas the first and second layers have 960 and 192 bytes respectively; calculating prototype distance using MAE requires only 160 arithmetic operations (MSE requires 320). Prototype comparison occurs for every runtime sample, so every saving here is multiplied by the total traffic.

### Mathematical Principles of Core Algorithms

**Level 1: Wavelet Domain VisuShrink Denoising.** Audio is framed into non-overlapping 1024-sample frames, each undergoing single-level Haar wavelet decomposition, yielding approximation coefficients and detail coefficients that carry high-frequency information. Noise variance is estimated using the median absolute deviation (MAD) of the detail coefficients:

$$MAD = \frac{\text{median}(|d - \text{median}(d)|)}{0.6745}$$

where $d$ represents the detail coefficients of the Haar wavelet. The global shrinkage threshold is:

$$\tau = MAD \cdot \sqrt{2 \log N}$$

where $N$ is the window length. Coefficients below $\tau$ undergo soft thresholding (attenuation rather than hard zeroing to avoid ringing), followed by inverse Haar transform to reconstruct the denoised frame, which is then spliced with adjacent frames and finally quantized to [−128, 127]. VisuShrink is characterized by "a single global threshold + preference for smooth reconstruction," with extremely low implementation overhead.

**Level 2: Feature Domain Spectral Denoising.** Let $x_n$ be the feature map (MFCC or Mel spectrogram) normalized to [0,1]. Mean subtraction is performed along the time and frequency axes respectively:

$$x_t = x_n - \mu_t(x_n), \quad x_s = x_n - \mu_f(x_n)$$

Then, binary masks are constructed, retaining only components higher than their respective means (the paper's intuition: background noise tends to be uniformly distributed across both axes; after mean subtraction, only weak components below the mean remain; significant structures like speech harmonics and transients will be above the mean):

$$M_t = \mathbb{1}[x_t > \mu_t(x_t)], \quad M_s = \mathbb{1}[x_s > \mu_f(x_s)]$$

The final denoised feature is reconstructed from mask components and original features using an attenuation coefficient $\alpha$ (Equation 5, recorded verbatim from the paper):

$$x_d = (1-\alpha)\big(\alpha\, x_s M_s + (1-\alpha)\, x_t M_t\big) + \alpha\, x_n$$

$\alpha$ controls the trade-off between denoising aggressiveness and signal preservation: smaller $\alpha$ favors denoising, larger $\alpha$ preserves original data (degenerating to pass-through when $\alpha = 1$). The paper combines spectral subtraction, masking, and normalization into this lightweight formula, specifically targeting time-frequency representations. (Writing detail: The caption for Fig 6 cites "Equation 7," while the main text formulas are grouped and numbered only up to (5); if each line of Equations 3/4 were numbered individually, $x_d$ would be the 7th equation—indicating an inconsistency in the paper's own numbering.)

**Class Prototypes and Distance.** The prototype for each class is the mean vector of the latent representations of samples belonging to that class, with dimensions matching the latent layer (160 bytes for a single-input model, doubled to 320 bytes for a dual-input model). Similarity between a sample and the prototype is measured using MAE. The prototype acceptance threshold is not a fixed constant but is adaptive to the class distribution:

$$d_{th} = \mu + n\sigma$$

where $\mu$ and $\sigma$ are the mean and standard deviation of the distances from samples of the same class to the prototype within the mini-batch, and $n$ is an empirical coefficient. This design scales the definition of "outlier" with the compactness of the class distribution: if the class distribution is scattered, the acceptance window is wide; if the distribution is tight, the acceptance window automatically narrows.

**CL Update (Algorithm 2).** For quantized models, de-quantization is performed first ($F_\theta, F_h \leftarrow D_Q\{f_\theta, f_h\}$), cross-entropy loss is calculated on the mini-batch $l \leftarrow L_{CE}(F_h(F_\theta(X_i)))$, gradients are obtained via backpropagation, and weights are updated at the same learning rate of 0.001 as initial training (the paper's rationale: using the same learning rate controls the magnitude of updates and ensures retraining stability). The updated weights are then re-quantized to INT8, and finally, class prototypes $P_{class} \leftarrow \frac{1}{N}\sum_{i=1}^{N} f_\theta(X_i)$, distance means, and standard deviations are recalculated using the new model. The update step in the original text is written as $w_m \leftarrow |F|$ and $F' \leftarrow f - \delta_\theta \odot w_m$, meaning element-wise scaling of gradients by parameter magnitude—the design motivation for this step is not elaborated in the paper, representing a fuzzy point in reproducibility.

### Key Technical Innovation 1: Compact CNN with Dual Feature Inputs and Resource Argument for "Full Model Update"
The classifier has two versions: single-input and dual-input. Single-input version: Input is a 20×16 MFCC map (1 second of audio), passing through three layers of 5×5 no-padding Conv2D, with feature maps sequentially becoming 16×12 → 12×8 → 8×4 (each layer reduces both dimensions by 4, consistent with the geometry of no-padding convolution), followed by Flatten and an FC layer (described in the text as containing 160 parameters). Dual-input version: MFCC and LogMel flow in parallel, each passing through independent three-layer Conv2D (structure identical to the single-input version), flattened and concatenated into a single vector entering the Dense layer for classification; **the output of the concatenation layer is selected as the carrier for latent representation/class prototypes**, justified by the fact that it carries information from both cepstral and time-frequency domains, while adhering to the single-input version's memory argument for "taking the latent layer at the last convolutional position"—the dimension is double that of the single-input version (320 bytes).

The benefits of dual-input are backed by clean data: 97.45% on the GSCD v2 test set for single-input, 99.63% for dual-input, an improvement of 2.18 percentage points, with zero cost (LogMel is an intermediate product of the MFCC pipeline). The paper attributes the gain to the complementary spectral-temporal features provided by MFCC (cepstral domain) and Mel spectrogram (time-frequency domain).

The deeper role of this architecture is to **provide a feasibility argument for "full model update"**—this is the watershed between this paper and all previous works. The total model parameters are only 1,595 (Table IV records 1.64k), and the activation memory and update overhead for full-parameter backpropagation are bearable within the MCU budget. Previous works were forced to "only move the classifier layer" or "sparse layer selection" because the models were too large; this paper compresses the model to the thousands-of-parameters level, raising the ceiling of domain adaptation capability from "fine-tuning classification boundaries" to "the entire feature extractor following the new domain." (The paper does not explicitly give the output channels for each convolutional layer; inferring from the 8×4 spatial positions (32 total) and 160-dimensional latent layer, the third layer's output channels can be deduced as 5. This is a note inference, not a statement in the original text.)

### Key Technical Innovation 2: Two-Level Integerized Denoising in Wavelet and Feature Domains
The two-level denoising is not redundant design; each targets different layers of noise residuals. The first level uses Haar-VisuShrink in the waveform domain to clear non-stationary noise; the second level uses dual-axis mean subtraction + masking + $\alpha$ reconstruction in the feature domain to clean up noise components residual in feature statistics. The pre-normalization of the second level ([0,1] scaling) provides magnitude invariance, making mask thresholds stable across different recording gains—this is a necessary protection when deployment environment volume is uncontrollable.

There is also an easily overlooked indirect role of the denoising design: **it simultaneously serves effective sample determination**. Pseudo-label filtering relies on classification confidence and latent-prototype distance, both of which presuppose "reliable incoming features"; the paper explicitly lists "ensuring feature reliability for classification and effective sample determination" as one of the goals of dual-level denoising in its contribution statement. Ablation data also supports the division of labor between the two levels: without retraining, "spectral denoising only" at −10 dB (88.62%) is actually higher than "wavelet + spectral" (86.38%) (wavelet introduces distortion under heavy pollution); but with retraining enabled, "wavelet + spectral + retraining" reaches 92.94%, becoming the best—leading the paper to argue that those distortions mainly come from the noise source rather than the audio features themselves, and can be absorbed by the CL stage.

### Key Technical Innovation 3: Confidence × Prototype Distance Dual-Gating Effective Sample Filtering
Each runtime sample undergoes INT8 quantized inference, obtaining both the final prediction and latent representation, then follows Algorithm 1 for two-stage gating:

1. **Confidence Gate**: The maximum predicted probability must exceed $P_{th}$ (set at 85%, an empirical value) to proceed; otherwise, it is directly discarded. Checking confidence before calculating distance is an intentional computational saving—distance calculation is not free (160 MAE operations), and the vast majority of low-confidence samples are not worth calculating.
2. **Prototype Distance Gate**: The MAE between the latent representation and the prototype of the predicted class is calculated; only if the distance falls within the $|d_p - \mu|$ threshold is it marked as an **effective sample**, stored in memory for the next CL use, with the pseudo-label being the predicted class.

The logic for why both gates are indispensable: Using only confidence allows overconfident erroneous samples (not rare in noise domains) to enter the retraining set with wrong labels; using only prototype distance cannot handle ambiguous samples on the class boundary that are "close to both prototypes." The $\mu + n\sigma$ adaptive threshold restricts the acceptance window to a range that "provides new information without introducing ambiguous information to the class prototype"—the paper illustrates this area as a green band in its figures, with samples inside (yellow dots) selected and outside (red dots) rejected. The essence of this mechanism is transforming an unsupervised runtime stream into a **semi-supervised stream of a high-confidence subset**.

### Key Technical Innovation 4: Full-Parameter CL Closed-Loop for Quantized Models and Feature Domain Rehearsal Buffer
The complete closed-loop of the CL stage (Algorithm 2) is: de-quantization → cross-entropy loss → backpropagation → learning rate 0.001 update of **all** weights → re-quantization to INT8 → updating class prototypes and mean/standard deviation thresholds for prototype distances using the retrained model. The mini-batch for retraining is composed of three parts: rehearsal buffer (MFCC + LogMel features from a subset of GSCD training data), augmented rehearsal buffer (generated by mixing 1-second noise spectrogram features recorded in the deployment environment with buffer features), and effective samples accumulated at runtime.

Two resource-oriented designs are worth expanding on. **First, the rehearsal buffer stores features, not audio**: This directly saves raw waveform storage, and since augmented samples are already in the feature domain, mixing them requires only one pass through spectral denoising, skipping the wavelet denoising and feature extraction frontend stages—minimizing the computational cost of augmentation. **Second, retraining is triggered at fixed intervals**: The interval between two consecutive retrainings is 1024 samples; the mini-batch is composed of 8 seconds of random noise taken from 300 seconds of environmental noise mixed with 64 "Yes" and 64 "No" samples each; the paper reports the average accuracy after 25 retraining iterations. This periodic design simplifies "when to learn" from event-driven to count-driven, incurring zero additional judgment overhead in implementation.

### Technical Differences with Existing Methods
Comparing with the two closest lines, differences concentrate on four dimensions. **Update Scope**: ODDL[4] only re-trains the DSCNN classifier layer, Lin et al.[14] perform sparse updates on specific layers, while this paper updates the complete model plus prototypes—the premise is compressing the model to 1.64k parameters, which buys CL freedom through architectural choice. **Noise Assumption**: ODDL divides noise into "seen/unseen during training" categories and validates only at 0 dB; this paper assumes all noise appears only after deployment (all treated as unseen) and validates across five levels from −10 to 10 dB. **Hardware Dependency**: ODDL requires the GAP9 multi-core parallel processor, Lin et al. simulate embedded environments using GPUs, while this paper targets a single-core Cortex-M4 (TM4C123GXL), requiring no parallel capability. **Supervision Source**: Prototype methods like TAP-SLDA[16] only update prototype statistics, whereas this paper combines prototype updates with full-model supervised retraining (pseudo-label effective samples + labeled rehearsal buffer)—the paper describes this as bridging the gap between "prototype semi-supervised strategies and supervised CL."

## Experimental Results

### Datasets Used and Their Scales
- **Training/Clean Test**: Google Speech Commands Dataset v2 (GSCD v2), comprising over 105,000 audio clips of 1 second or shorter at 16 kHz, covering 35 keywords and 2,618 speakers. This paper selects the two words "Yes" and "No" for binary classification, evaluating clean performance using the official GSCD v2 test set.
- **Noise Sources**: DEMAND dataset, six environmental categories (Domestic, Nature, Office, Street, Public, Transportation), with 3 recording environments per category, each noise segment 300 seconds long, recorded at 48 kHz and resampled to 16 kHz. Four environments were actually selected: DWASHING (domestic dishwasher), NFIELD (natural field), OOFFICE (office), TCAR (transportation car). (The original text first says "six categories" then says "four out of five," showing inconsistent numerical expression; the actual usage is the above 4.)
- **Evaluation Set Construction**: GSCD v2 test samples are mixed with single-second noise from DEMAND. SNR ranges from −10 dB to 10 dB in steps of 5 dB, totaling five levels. 169,695 evaluation samples are generated for each noise environment.
- **CL Simulation Process**: Initial adaptation is performed by mixing spectral domain features using the rehearsal buffer plus random segments from 300 seconds of noise in that environment; then entering inference mode, where effective samples are determined and saved at runtime; retraining is triggered every 1024 samples (mini-batch composed of 8 seconds of random noise mixed with 64 Yes/No samples each); the reported metric is the average accuracy after 25 retraining iterations.

### Definition and Rationale for Evaluation Metrics
The primary metric is **classification accuracy**, chosen directly because it is the core measure for binary classification tasks and allows direct comparison with previous works like ODDL and MCUNet on the same dataset. For the embedded dimension, the TinyML quartet is used: **parameter count, FLOPS, latency per sample (ms), and energy consumption per sample (µJ)**—these four quantities jointly determine whether the solution fits within the MCU budget. It should be noted that FA/FRR (False Alarm Rate/False Rejection Rate), AUC, DET curves commonly used in the KWS domain, and forgetting metrics (regression of old domain accuracy after adaptation) commonly used in the CL domain, **are not reported in the paper**.

### Detailed Comparison with Baseline Methods and SOTA
**Clean Performance (Table I)**: Single-input (MFCC) 97.45%, Dual-input (MFCC + LogMel) 99.63%. All subsequent CL experiments use the dual-input model.

**CL Results for Four Noise Environments × Five SNR Levels (Table II, Dual-Input Model)**:

| Adaptation Environment | −10 dB | −5 dB | 0 dB | 5 dB | 10 dB |
|---|---|---|---|---|---|
| DWASHING | 93.84 | 94.88 | 95.62 | 95.22 | 95.20 |
| NFIELD | 91.44 | 92.56 | 94.91 | 96.22 | 95.09 |
| OOFFICE | 92.94 | 92.50 | 94.34 | 94.94 | 94.31 |
| TCAR | 94.56 | 95.25 | 95.28 | 94.78 | 95.22 |

The worst point is NFIELD at −10 dB (91.44%), and one of the best points is NFIELD at 5 dB (96.22%); all four environments maintain above 91% at −10 dB, and all exceed 94% starting from 0 dB. The paper's interpretation is: only 25 retraining iterations significantly improve noise robustness, and the slope of accuracy degradation with worsening SNR is flattened.

**System-Level Comparison with On-Device Learning Frameworks (Table IV)**:

| Solution | Task/Data | Hardware | Parameters | FLOPS | Latency (ms) | Energy (µJ) | Accuracy (%) |
|---|---|---|---|---|---|---|---|
| MCUNetv3 | VWW | STM32F746 (M7) | 480k | 46M | 546 | 6899* | 89.3 |
| ODDL DSCNN-S | GSCD | GAP9 | 23.7k | 2.95M | 6.74 | 384 | 90.68 |
| ODDL DSCNN-M | GSCD | GAP9 | 138.1k | 17.2M | 16.34 | 974 | 92.64 |
| ODDL DSCNN-L | GSCD | GAP9 | 416.7k | 51.1M | 32.95 | 2028 | 93.47 |
| This Paper | GSCD | TM4C123GXL (M4) | 1.64k | 0.89M | 95.11** | 200.87** | 94.31 |
| This Paper | GSCD | Optimized M4 | 1.64k | 0.89M | 95.11** | 64.45** | 94.31 |

(*, ** are estimated values based on ARM M7/M4 datasheets, not board-level measurements.) This paper surpasses all three levels of ODDL and MCUNetv3 in accuracy with 1.64k parameters and 0.89M FLOPS: it has 93% fewer parameters than the smallest ODDL DSCNN-S while being 3.6 percentage points higher in accuracy, and 99.6% fewer parameters than the largest DSCNN-L while being 0.8 percentage points higher. The paper claims 98.81% fewer parameters than "high-complexity networks"—arithmetically, 1.64k vs 138.1k (DSCNN-M) is exactly 98.81%, but the original text parenthetically notes DSCNN-S (23.7k, which should be 93.1%), suggesting a typo. In terms of cost dimensions, the 95.11 ms single-sample latency is significantly higher than the 6.74–32.95 ms on GAP9, but the hardware tiers are completely different (single-core entry-level M4 vs multi-core GAP9), and for a 1-second analysis window, it remains within real-time limits. The last row "Optimized M4" demonstrates the energy consumption benefits of hardware-software co-optimization: for the same accuracy and computation, energy consumption drops from 200.87 µJ to 64.45 µJ.

### Findings from Ablation Experiments
**Component Ablation (Table III, OOFFICE Environment, Dual-Input Model)**:

| Retraining | Wavelet Denoising | Spectral Denoising | −10 dB | −5 dB | 0 dB | 5 dB | 10 dB |
|---|---|---|---|---|---|---|---|
| No | No | Yes | 88.62 | 94.50 | 96.72 | 97.97 | 98.03 |
| No | Yes | Yes | 86.38 | 92.16 | 95.47 | 95.69 | 96.41 |
| Yes | No | Yes | 92.88 | 94.53 | 96.59 | 96.31 | 97.03 |
| Yes | Yes | Yes | 92.94 | 92.50 | 94.34 | 94.94 | 94.31 |

Three findings: (1) **The benefit of retraining concentrates at low SNR**: At −10 dB, retraining pulls accuracy from 88.62/86.38 to 92.88/92.94, an improvement of over 4 percentage points; whereas at 10 dB, retraining slightly decreases accuracy (98.03 → 94.31), indicating that when domain shift is small, CL provides no gain, and there is a slight mismatch between the mixed distribution of the rehearsal buffer and the clean distribution. (2) **The value of wavelet denoising is only realized in conjunction with retraining**: Without retraining, it actually loses 2.24 percentage points at −10 dB (88.62 → 86.38), indicating that wavelet denoising introduces distortion under heavy noise; with retraining, it surpasses the spectral-denoising-only configuration, leading the paper to argue that the distortion originates from the noise rather than the audio features and can be absorbed by CL. (3) Spectral denoising alone is quite capable—without retraining, spectral denoising only still achieves 88.62% at −10 dB, making it the most cost-effective level of denoising among the two levels.

**Sensitivity of Spectral Denoising Attenuation Coefficient $\alpha$ (Fig 6)**: Performance is stable across the entire $\alpha$ range of [0.4, 0.9], with accuracy in all environments from 0–10 dB remaining above 94–96%; only NFIELD and OOFFICE at −10 dB occasionally dip when $\alpha$ is too strong or too weak, with no sharp optimum—concluding that the denoising algorithm is robust to parameter selection, with no critical degradation points.

**Sensitivity of Confidence Threshold $P_{th}$ (Fig 7)**: Scanning the threshold from 0.70 to 0.85, accuracy changes across all environments and SNRs by less than 0.5%, with curves almost flat. This indicates that prediction confidence after INT8 quantization is well-calibrated, with reliable separation between correct and incorrect predictions, and the CL mechanism is insensitive to this hyperparameter.

**Sensitivity of Prototype Distance Threshold (Fig 8)**: Accuracy changes by less than 1–1.5% across the entire threshold range of [1.7, 2.4], with no peak optimum, and slightly larger variance for NFIELD at −10 dB. The conclusion is the same: distance gating does not rely on fine-tuning, and the decision boundaries in the latent space are naturally well-separated.

## Main Contributions
1. **New Domain-Incremental CL Framework**: Driven jointly by "effective samples" of runtime pseudo-labels and rehearsal buffers, achieving incremental updates of the complete quantized model (rather than just the classifier layer or sparse layers), with simultaneous updates of class prototypes—pushing the upper limit of on-device domain adaptation capability from classification boundaries to the entire feature extractor.
2. **Dual-Level Denoising Embedded in Feature Pipeline**: Two-level integerized denoising in the wavelet domain (Haar-VisuShrink) and feature domain (spectral subtraction + masking + normalization), simultaneously ensuring classification accuracy and feature reliability for effective sample determination.
3. **Dual-Feature Input Classifier**: Parallel streams of MFCC + LogMel, achieving 99.63% on clean data (2.18 percentage points higher than single-input), with the second feature stream obtained at zero additional computational cost as an intermediate product of the MFCC pipeline.
4. **Wide-Condition Validation and Extreme Resource Footprint**: Maintaining above 91% accuracy across five SNR levels from −10 to 10 dB and four noise environments; 1.64k parameters, 0.89M FLOPS, deployable on Cortex-M4 level TM4C123GXL, with 98.81% fewer parameters than ODDL's DSCNN-M and higher accuracy.

## Limitations and Future Work

### Technical Limitations of the Method
- **Binary Classification Boundary**: The entire framework (dual-class prototypes, FC output, $\mu+n\sigma$ threshold, balanced rehearsal of 64 samples each) is designed for Yes/No binary classification. In multi-class scenarios, the growth of prototype numbers, rehearsal ratios for class imbalance, and maintenance of per-class distance thresholds would need redesign—the paper acknowledges this as the current boundary in the conclusion.
- **High Latency and Estimated Values**: The 95.11 ms single-sample latency is real-time for a 1-second window but is on the high side for low-latency streaming wake-word scenarios; moreover, this number, like energy consumption, comes from ARM M4 datasheet estimates (marked with ** in Table IV), not board-level measurements, and the optimization method for the "Optimized M4" row (64.45 µJ) is not explained in the paper.
- **Dependency on Retraining Occurrence**: Ablation shows that wavelet denoising introduces distortion without retraining at −10 dB (88.62 → 86.38). This implies that the pipeline's robustness depends on "retraining occurring periodically"; if severe domain changes occur between two retrainings, there exists a performance dip window.
- **Hyperparameters are Empirically Set**: The confidence threshold of 85%, $\alpha$, and $n$ in $\mu+n\sigma$ are all empirical values. Sensitivity analysis proves flatness (which is itself a contribution), but no methodology for selection transferable to other datasets is provided.
- **Low Latent Dimension**: The 160/320 byte latent representation determines the upper limit of prototype expressiveness; when phoneme overlap between classes is high, the margin for distinction is questionable; the paper does not perform ablation on latent layer position/dimension.
- **Reproducibility Gaps in Algorithm Details**: The element-wise scaling step $w_m \leftarrow |F|$, $F' \leftarrow f - \delta_\theta \odot w_m$ in Algorithm 2 lacks explanation of design motivation; details on the optimizer, number of epochs, data augmentation specifics, and QAT configuration for initial training are not reported in the paper.

### Deficiencies in Experimental Design
- **94.31% in Table IV lacks specified context**: It is unclear which noise environment and SNR it corresponds to (the value matches OOFFICE@10 dB in Table II, suspected to be that configuration, but not explicitly stated in the paper), making the accuracy comparison with ODDL/MCUNet not perfectly aligned (the latter's context is also not unified to the same noise conditions).
- **Abstract is overly optimistic**: The abstract claims "exceeds 94% in all noise environments at −10 dB," but Table II's −10 dB column ranges from 91.44–94.56, with only TCAR exceeding 94%.
- **Numerical and Writing Flaws**: Contradiction between DEMAND's "six categories" and "four out of five"; Fig 6 caption cites "Equation 7" while the main text formula group numbering stops at (5); the parenthetical note for "98.81% fewer parameters" cites DSCNN-S, which arithmetically should correspond to DSCNN-M.
- **Missing Key Dimensions**: The specific capacity of the rehearsal buffer (number of samples or bytes) and total system RAM/Flash usage are not reported—these are precisely the core resource accounts for on-device CL; accuracy on the clean test set after CL (forgetting metric) is not reported; KWS business metrics like false alarm rate are not reported; random seeds and variance are not reported (making it impossible to judge if results are stably reproducible); the pseudo-label error rate of effective samples is not reported (no direct evidence of the filtering purity of the dual-gating mechanism); and there is no ablation of "rehearsal buffer only vs. rehearsal + effective samples" to isolate the independent contribution of the effective sample mechanism.
- **Narrow Definition of Domain**: Domain shift is simulated only using additive environmental noise, not covering reverberation, far-field, microphone channel distortion, speaker and accent changes—several of which are also listed as sources of domain shift in the paper's introduction.
- **Lack of Head-to-Head Comparison with CL Methods**: TAP-SLDA, iCaRL, etc., are only cited with numbers in the related work section, without direct comparison under the same framework.

### Possible Directions for Future Improvement
- **Multi-Class Domain-Incremental Extension**: The next step stated by the paper, targeting multi-class tasks like KWS-35 (the paper also lists visual datasets like MNIST, CIFAR-10, CIFAR-100, reflecting that the authors position this framework as a general domain-incremental solution rather than KWS-specific).
- **Board-Level Measurements and Real Deployment**: Replace datasheet estimates with measured energy consumption/latency and validate the behavior of fixed-interval retraining on real always-on devices.
- **Expansion of Domain Types**: Incorporate reverberation, far-field, channel distortion, and speaker adaptation into the definition and evaluation of domain-incremental learning.
- **Refinement of CL Mechanisms**: Rehearsal ratios under class imbalance, online detection of prototype drift, event-driven retraining triggers (replacing count-driven to compress the performance dip window after domain changes).
- **Integration with Federated Learning**: The paper's conclusion envisions the role of domain-incremental learning in generalizing across heterogeneous sensors in TinyML and federated learning—cross-device collaborative domain adaptation is a natural extension of this framework.
