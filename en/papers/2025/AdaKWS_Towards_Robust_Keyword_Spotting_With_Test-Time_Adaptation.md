# AdaKWS: Towards Robust Keyword Spotting with Test-Time Adaptation

- **Authors/Affiliations**: Yang Xiao (The University of Melbourne / Fortemedia Singapore); Tianyi Peng (Nanyang Technological University); Yanghao Zhou (National University of Singapore); Rohan Kumar Das (Fortemedia Singapore)
- **Date**: May 2025 (arXiv:2505.14600v1, submitted May 20, 2025)
- **Link**: https://arxiv.org/abs/2505.14600
- **Keywords**: test-time adaptation, keyword spotting, entropy minimization, pseudo-keyword consistency, domain shift, batch normalization, noise robustness

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) refers to the identification of specific wake words or command words from continuous audio streams. It is a core component of voice entry points such as Apple Siri and Google Home, and is almost always deployed on edge devices. Unlike large-scale continuous speech recognition, the product form of KWS dictates two unique engineering attributes: First, the vocabulary is small, and the model must be extremely lightweight because it must run continuously on microcontrollers or digital signal processors under a milliwatt-level power budget. Second, the cost of errors is asymmetric; missed wake-ups make users feel the device is malfunctioning, while false wake-ups directly infringe upon privacy boundaries. Therefore, any fluctuation in performance is immediately perceived by users. An essential characteristic of such systems is **always active**: the device listens around the clock, and it is impossible to stop and retrain every time the environment changes. Thus, it must carry a continuous inference load with a small-footprint model. Current deep learning KWS systems (such as BC-ResNet, DS-CNN, ED-sKWS, etc.) primarily trade clever structural designs for efficiency and use a limited set of keywords for training to compress computational and memory overhead. This route is quite mature on clean test sets, but it implicitly relies on a strong assumption: the test distribution is close to the training distribution.

The engineering pain point lies in the fact that the training data for these models is predominantly close-talking corpus, while real-world deployment faces low Signal-to-Noise Ratio (SNR), far-field pickup, room reverberation, and various background noises. The paper uses a set of motivation experiments (Figure 2) to demonstrate the severity of the problem: a BC-ResNet-3 model that performs well on the clean GSC test set drops to 41.34% accuracy after adding additive Gaussian noise with $\delta=0.03$ (Table 1, Unadapted row), and falls to just 26.44% under real Babble noise at −10 dB (Table 2). When the training and test distributions are similar, performance is good; when the distribution shifts, it collapses. This is the bankruptcy of the "training distribution assumption" in real-world deployment. More troublingly, the acoustic domain is not static: changing rooms, background noises, or placement positions all generate new domain shifts, while the model remains static once delivered. The paper addresses this fundamental contradiction: "continuously facing unknown domains after deployment without allowing for retraining."

### Specific Deficiencies of Existing Methods

The paper systematically reviews four existing routes and points out their infeasibility in deployment scenarios:

- **Multi-environment pooled data training** (Refs [10-15]): Mixing labeled speech from different environments for training is the most direct idea. It has two shortcomings: First, each deployment environment requires labeled data, but labeled domain data is scarce. Second, new environments that appear after deployment are not in the training pool, merely delaying the problem rather than solving it.
- **Post-deployment supervised fine-tuning**: The paper gives three reasons for its infeasibility: (1) The new environment will continue to drift, and fine-tuning can never catch up; (2) It requires additional labeled data, which is costly; (3) User data tends to stay on local devices due to privacy concerns, and device computing power is limited, making training-time fine-tuning impossible to run on the edge.
- **Unsupervised Domain Adaptation (UDA)**: Methods such as domain adversarial training, knowledge distillation, and feature alignment can indeed narrow the domain gap, but they require **simultaneous access to source data and a sufficient amount of target domain samples**. Privacy and storage constraints often make source data unavailable during the adaptation phase—this is the main shortcoming of UDA implementation.
- **Directly applying ASR Test-Time Adaptation (TTA)**: Methods like SUTA have achieved good results in ASR through entropy minimization, but the paper points out two specific obstacles for KWS. First, **small-footprint sensitivity**: KWS models are made extremely lightweight for edge efficiency, so even minor parameter changes can significantly affect performance, and high-entropy samples are prone to accumulating errors during adaptation. Moreover, existing ASR TTA methods (Refs [31-33]) are designed only for domain mismatch of a single type of noise. Second, **test speech is short and inconsistent**: KWS processes short speech, lacking the long-term information available in ASR, making prediction inherently more difficult. Therefore, it is necessary to actively select samples that are "less likely to be predicted incorrectly" to participate in adaptation; otherwise, errors will amplify relay-style between batches.

### Key Challenges to be Solved by This Paper

Under the strict constraints of "no source data, no test labels, and only the test stream itself is available," designing a test-time adaptation mechanism for small-footprint KWS models requires resolving three intertwined difficulties: (1) Small models are extremely sensitive to parameter perturbations; standard online entropy minimization has a trivial collapse solution where all samples are predicted as the same class, and estimating a single normalization statistic from a mini-batch mixed across multiple noise distributions will be distorted; (2) Discriminative information in short speech is limited; relying solely on entropy as a single criterion will miss samples contaminated by transient noise where the features themselves are unreliable; (3) Adaptation must be lightweight enough not to violate the efficiency prerequisites of edge deployment.

## Methodology

### Overall Architecture Design and Design Motivation

The overall process of AdaKWS (Figure 1) can be summarized as "one forward pass probing twice, two-level filtering, and then weighting":

1. The test mini-batch enters the model, and the first-level filtering is performed based on prediction entropy, retaining low-entropy reliable samples where $L_{ent} < \tau_{ent}$;
2. Perturbation copies $x'$ are obtained by applying time/frequency masking transformations to the surviving samples. Pseudo-keyword consistency $L_{pkc}$ is calculated, and the second-level filtering is performed based on $L_{pkc} > \tau_{pkc}$, yielding $x_{pkc}$ (Cluster 4 in Figure 1);
3. Dual-channel weights $\alpha(x)$ are calculated for the selected samples according to Equation (5);
4. The weighted entropy loss (Equation 6) is used to **update only the BN layer parameters**, completing one step of test-time adaptation.

The backbone network chosen is BC-ResNet-3 (a lightweight CNN proposed in Interspeech 2021 for edge KWS), with 40-dimensional MFCC features and a 160 ms frame shift as input. Why this choice: BC-ResNet-3 is a representative backbone for small-footprint KWS. It is precisely the model category that the paper claims is most vulnerable to domain shifts, making validation most persuasive and easiest to align with community results.

Why only update BN parameters: (a) Tent has proven that updating only BN affine parameters is sufficient to carry domain adaptation, with a very small number of tunable parameters, naturally fitting the premise that "adaptation must be lightweight"; (b) Small-footprint models are sensitive to parameter changes, and the risk of full-parameter updates is explicitly listed as a negative lesson by the paper; (c) KWS small models generally use BN rather than the LayerNorm commonly used in ASR (as pointed out in Section 2.2.1 of the paper), and the statistics of BN inherently encode information about the training distribution—when domain shift occurs, the mismatch in BN statistics is the root cause, so modifying them is targeted treatment.

### Mathematical Principles of Core Algorithms

**Problem Formulation (Section 2.1)**: A trained KWS model $M_\theta$ obtains parameters $\theta$ on the source domain $D_{train}=(x_i^{train}, y_i^{train})_{i=1}^{N_{train}}$; The goal of TTA is to adapt **only using** the input part of the target domain test data $D_{test}=(x_i^{test}, y_i^{test})_{i=1}^{N_{test}}$ during the adaptation phase (neither $D_{train}$ nor $y^{test}$ is available), completing adaptation without retraining the model. Compared to fine-tuning and UDA, this setting requires less data and computing power.

**Entropy Loss (Equation 1)**: There are no labels during testing, and the only computable confidence proxy is the entropy of the output distribution:

$$L_{ent}(x) = -P(x)\cdot\log P(x) = -\sum_{i=1}^{C} p(x)_i \log p(x)_i$$

where $P(x)$ is the model's output probability for $x$, and $C$ is the number of classes (35 on GSC). Minimizing entropy sharpens the class distribution and is a standard unsupervised objective for domain adaptation.

**Entropy Filtering (Equation 2)**: $x_{ent} = \{x \mid L_{ent} < \tau_{ent}\}$, with $\tau_{ent}=0.4$ in implementation.

**Pseudo-Keyword Consistency (Equations 3, 4)**:

$$L_{pkc}(x, x') = p(x)_c - p(x')_c, \qquad x_{pkc} = \{x \mid L_{pkc}(x, x') > \tau_{pkc}\}$$

where $c$ is the pseudo-label class, $p(x)_c$ and $p(x')_c$ are the pseudo-label confidences of the model for the original input and the masked transformed input, respectively, and $\tau_{pkc}=0.05$.

**Dual-Channel Weight (Equation 5)**:

$$\alpha(x) = \frac{1}{\exp\{(L_{ent}(x)-\sigma)\}} + \frac{1}{\exp\{-L_{pkc}(x, x')\}}$$

$\sigma$ is a normalization factor (taken as 0.5). Observing its monotonicity: The first term equals $e^{\sigma - L_{ent}}$, increasing as entropy decreases, rewarding samples where the "classifier is confident"; The second term equals $e^{L_{pkc}}$, increasing as pseudo-keyword consistency increases, rewarding samples where "prediction evidence is rooted in keyword features." Two orthogonal signals are fused into soft weights in an additive form.

**Total Loss (Equation 6)**: $L_{AdaKWS} = \alpha(x)\cdot L_{ent}(x_{pkc})$, meaning first hard filter by PKC, then soft weight by dual-channel, and finally sum the weighted entropy loss for selected samples, backpropagating only to BN parameters.

**Semantic Analysis of PKC (Key to Understanding This Paper)**: $L_{pkc}$ measures "how much the pseudo-label confidence drops when a portion of the time-frequency region is masked." If it drops significantly ($L_{pkc}$ is large), it indicates that the evidence the model relies on for this prediction is hidden within the masked region—meaning the prediction is truly "rooted" in the acoustic features of the keyword itself (phoneme consistency, spectral consistency); If the confidence remains almost unchanged after masking ($L_{pkc}$ is small), it indicates that the prediction basis lies in stationary components spread across the entire frame, which is often noise rather than keyword evidence. Based on this, the paper selects samples with $L_{pkc} > \tau_{pkc}$ into the update set, and the expression in the introduction is also "selecting samples that are more clearly rooted in keywords." It should be noted that Section 2.2.2 simultaneously contains the statement "the smaller the drop, the more stable, we focus on stable samples," which is opposite to the mathematical direction of Equations (4) and (5), constituting a contradiction in the paper's writing (see discussion in the Limitations section).

### Key Technical Innovation 1: Selective Entropy Minimization

Why use entropy: There are no labels during testing, and entropy is the only confidence proxy that can be optimized without supervised signals, and entropy minimization has ample precedent in domain adaptation (Tent pioneered using entropy loss for TTA and updating only BN parameters).

Why add a threshold $\tau_{ent}$ on top of entropy: The paper gives two explicit motivations. First, online entropy minimization has a **trivial collapse solution**—the model predicts all samples as the same class, and entropy still tends to a minimum, but adaptation fails completely; Second, BN statistics reflect a specific distribution. If a single set of statistics is estimated from a test mini-batch spanning multiple noise distributions, they will contaminate each other. Using $\tau_{ent}=0.4$ to retain only low-entropy samples is equivalent to "only letting samples the model is confident about participate in gradient voting," suppressing error accumulation from the source. This design directly responds to the challenge in the Problem Statement of "high-entropy sample error accumulation in small-footprint models." The paper's expression is: After this selective entropy minimization, the KWS model becomes more stable during adaptation.

### Key Technical Innovation 2: Pseudo-Keyword Consistency (PKC) Resampling

Design Motivation: The entropy criterion only looks at the sharpness of the output distribution and cannot answer "whether the features relied upon for this prediction are reliable itself"—samples contaminated by transient noise or with non-discriminative features may also output low entropy. The paper cites the view of Ref [34] ("entropy is insufficient to support test-time adaptation") as the basis for its argument, believing that a second dimension needs to be added from the perspective of feature stability: phoneme stability and spectral consistency, which are exactly the properties most relied upon by KWS under noise and unknown conditions.

Transformation Design (Why time/frequency masking): Time masking artificially creates temporal discontinuities, simulating background interruptions or speech truncation; Frequency masking changes the spectral representation, simulating changes in audio equipment or environmental acoustics such as changing microphones. The two perturbations strike the evidence chains in the time and frequency domains respectively, jointly probing the vulnerabilities of the prediction. In implementation, two time masks (maximum length 20 frames) and two frequency masks (maximum length 5 frequency bands) are applied to each sample.

Relationship with SpecAugment: Formally homologous (both are time-frequency masks), but with opposite purposes—SpecAugment does data augmentation in the **training phase** to improve generalization, while PKC acts as a probe in the **testing phase** to measure feature criticality. One prevents overfitting, the other selects reliable samples.

Effect Mechanism: By retaining only samples where "prediction changes significantly after masking" (i.e., evidence is concentrated in keyword regions) and giving them high weights, the model is constrained to adapt to reliable, noise-robust features that are truly critical for keyword discrimination, rather than overfitting to noise. The paper states that this allows AdaKWS to effectively identify and utilize reliable samples even in harsh acoustic environments.

### Key Technical Innovation 3: Sample Weighting with Dual-Criterion Fusion

Equations (5) and (6) use both entropy and PKC criteria at two levels: **selection** (deciding which samples enter the update set) and **weighting** (deciding the magnitude of each sample's impact on the gradient). Why weight in addition to filtering: Ablation experiments (Table 4) show that removing reweighting drops accuracy from 57.02% to 56.84%, indicating that soft weights have an independent contribution in balancing the impact of various features on the update magnitude—hard filtering only decides "who enters the field," while weights decide "who speaks loudly," and the two are not entirely equivalent. In the weight function, the entropy channel and PKC channel are added rather than multiplied, meaning a strong signal in one channel is sufficient to obtain a large weight, avoiding the issue of insufficient samples due to overly strict dual conditions.

### Technical Differences from Existing Methods

- **Vs. TBN**: TBN only re-estimates BN statistics using target data during the test forward pass (covariate shift adaptation), with no gradient updates or sample discrimination; AdaKWS overlays entropy gradient updates and dual-criterion selection weighting on top of statistic adjustment, averaging 1.53 percentage points higher in Table 1 (77.01 vs. 75.48).
- **Vs. Tent**: Tent performs entropy minimization updates on all samples within the batch to update BN affine parameters; for short speech small models, high-entropy noise samples will contaminate the gradient. AdaKWS is equivalent to Tent plus three corrections: entropy threshold, PKC, and weighting. It averages 0.77 percentage points higher in Table 1 (77.01 vs. 76.24), and the advantage widens as noise increases.
- **Vs. ETA**: ETA uses high-entropy sample filtering to avoid noise gradients and forgetting, but it is still a single entropy criterion; AdaKWS adds the dimension of feature reliability (PKC), averaging 1.67 percentage points higher in Table 1 (77.01 vs. 75.34).
- **Vs. SAR**: SAR uses sharpness-aware and reliable entropy minimization to handle practical conditions such as small batches and online label shift, focusing on the stability of the optimization process; AdaKWS focuses on the sample-level issue of "whether short speech features are rooted in keywords." The two are orthogonal, and AdaKWS averages 1.94 percentage points higher in Table 1 (77.01 vs. 75.07).
- **Vs. ASR TTA like SUTA**: SUTA is oriented towards large ASR models (mostly using LayerNorm) and long speech, single-sentence adaptation scenarios; AdaKWS is oriented towards small-footprint BN models and batched short audio, and explicitly handles the KWS-specific problem of "which samples are trustworthy."

## Experimental Results

### Datasets Used and Their Scales

- **Google Speech Commands (GSC) [35]**: 105,829 short audio clips of 1 second duration, covering 35 keywords, split into training/validation/test sets at 80%/10%/10%, with all audio sampled at 16 kHz. Used as source domain training data.
- **Additive Gaussian Noise**: Added to the GSC test set in tiers according to severity $\delta$ to construct the target domain; Table 1 reports three tiers of $\delta=0.01/0.02/0.03$ and the mean (Table note claims five severity levels were set, but only three tier values are visible in the obtainable table body).
- **MS-SNSD [36] Test Set**: 8 types of real-environment noise (including Air Conditioner, Babble, Vacuum Cleaner, etc.), mixed with original audio at 5 SNR tiers; Table 2 reports results for Typing, Copy Machine, Air Conditioner, and Babble at −10 dB.
- **ESC-50 [37]**: 2,000 5-second environmental recordings, 50 categories, belonging to five major classes: Animals/Natural/Urban/Human/Domestic; Table 3 reports results for each of the five classes and all five classes combined (All) at three noise levels of −10/0/10 dB, forming a more difficult multi-source noise target domain.

### Definition and Rationale for Evaluation Metrics

The metric is Accuracy ACC (%). Rationale: GSC is a closed-vocabulary classification task with 35 classes, and ACC is the standard metric for this benchmark, allowing direct alignment and comparison with existing literature for the four TTA baselines (TBN/Tent/ETA/SAR). A gap to note: The paper does not report false alarm rate/missed detection rate trade-offs (e.g., miss rate at fixed false alarm rate) or detection metrics like AUC, which are more critical for streaming KWS engineering. It also does not report the computational and latency overhead introduced by the adaptation process—these missing dimensions in metrics are discussed in the Limitations section.

### Detailed Comparison with Baseline Methods and SOTA

**Gaussian Noise (Table 1)**: Unadapted average 55.92%, TBN 75.48%, SAR 75.07%, ETA 75.34%, Tent 76.24%, AdaKWS 77.01%. TTA brings an average recovery of 21.09 percentage points overall (77.01 vs. 55.92, calculated from Table 1 values). The advantage of AdaKWS over the strongest baseline Tent widens monotonically as noise increases: at $\delta=0.01$, 84.66 vs. 84.44 (+0.22); at $\delta=0.02$, 76.48 vs. 75.70 (+0.78); at $\delta=0.03$, 69.89 vs. 68.59 (+1.30). This is consistent with design expectations: the stronger the noise and the harder the domain, the larger the blind spot of the entropy criterion, and the higher the value of reliable samples selected by PKC. The paper attributes this to PKC's identification of reliable features enhancing adaptation capability and reducing overfitting to noise.

**Real Single-Source Environmental Noise (Table 2, −10 dB)**: AdaKWS ranks first or ties for first on all four noise types. Babble: Unadapted 26.44% improves to 49.56% (+23.12, surpassing the strongest baseline ETA's 45.71 by +3.85); Air Conditioner: 60.98% improves to 70.44 (+1.40 surpassing Tent's 69.04); Copy Machine: 21.06% improves to 38.01 (+0.32 surpassing ETA's 37.69); Typing: 31.65% improves to 62.33 (basically tying with Tent's 62.30, +0.03). An interesting pattern: the four baselines differ by less than 2 percentage points on Typing, while AdaKWS pulls the largest gap exactly on Babble, a **speech-like noise**—babble noise is most likely to create "high-confidence pseudo-keyword features," which are samples where the entropy criterion is most likely to be misled, and exactly the target scenario for the PKC probe. The dismal performance of Unadapted on Copy Machine (21.06%) and Babble (26.44%) again emphasizes that adaptation is necessary for real deployment.

**Multi-Source Noise (Table 3, ESC-50)**: The most informative is the All column (five noise types mixed): AdaKWS achieves 54.96/70.11/81.07 at −10/0/10 dB, ranking first in all three tiers; Unadapted is 45.74/62.98/74.82. Notably, Tent only achieves 49.75% at All −10 dB, which is not only lower than TBN's 54.58%, but even lower than its own level under single noise of the five types (approx. 52~57)—the single entropy criterion puts unreliable samples into the gradient when multi-source noises are mixed, causing adaptation backlash; AdaKWS stays stable at 54.96% thanks to the dual-criterion design. Looking at classification categories, AdaKWS leads in all 15 cells across 5 classes × 3 tiers (e.g., Urban 0 dB 74.86 vs. Tent 73.92; Domestic 10 dB 80.45 vs. TBN 79.78; Animals −10 dB 52.88 vs. Tent 52.25). The paper specifically emphasizes that the 70.11 at All 0 dB reflects the adaptive mechanism's ability to handle diverse acoustic characteristics.

**Noise Sensitivity Motivation Experiment (Figure 2)**: BC-ResNet-3, which performs well on clean GSC (Source in the figure), sees a significant drop in ACC under three conditions: Gaussian noise, MS-SNSD's Babble/Typing, and ESC-50's Animals/Natural (specific curve values are not given in the main text), providing motivation evidence that "real deployment requires adaptation."

**Overall Interpretation of Horizontal Patterns**: Looking at the three tables together, three judgments beyond single-table conclusions can be made. First, the benefit of TTA mainly comes from the layer of "normalization statistic alignment"—the simplest TBN (only re-estimating BN statistics, no gradient updates) already pulls the average accuracy under Gaussian noise from 55.92% to 75.48% (Table 1), accounting for more than 80% of the total recovery magnitude, indicating that the primary lesion of domain shift is indeed in the normalization statistics; Gradient-style adaptation (the Tent family) squeezes out another approx. 1 percentage point on top of this, while AdaKWS's selection and weighting mechanisms squeeze out another approx. 0.8 percentage points on top of gradient-style adaptation. The benefits decrease layer by layer, but each layer is positive. Second, the real watershed between methods appears in difficult domains rather than simple domains: at $\delta=0.01$, the gap between methods is less than 1 percentage point (Table 1), but at Babble −10 dB, the gap between Tent and TBN is 4 percentage points, and between AdaKWS and Tent is nearly 4 percentage points (Table 2). Difficult domains are where sample discrimination mechanisms are tested. Third, multi-source mixed noise (Table 3 All column) is the only condition where single-entropy criterion methods experience "adaptation backlash," making the value of the dual-criterion design greatest here.

### Findings from Ablation Experiments

**Component Ablation (Table 4, ESC-50 Domestic −10 dB)**:

| Entropy Sampling | PKC Sampling | Reweighting | ACC (%) |
|---|---|---|---|
| ✓ | ✓ | ✓ | 57.02 |
| ✓ | ✓ | ✗ | 56.84 |
| ✓ | ✗ | ✓ | 55.94 |
| ✗ | ✓ | ✓ | 56.59 |

Three findings: (1) Removing reweighting drops accuracy by 0.18 percentage points, contributing the least but positively, indicating that soft weighting has independent value; (2) Removing PKC sampling drops accuracy by 1.08 percentage points, the largest contribution among the three components, proving the core argument that "feature reliability criteria are needed in addition to entropy"; (3) Removing entropy sampling drops accuracy by 0.43 percentage points. The paper believes that the continuous decline below its relative competitiveness (56.59%) reflects its role in sharpening the class distribution. The conclusion is that the three components are individually effective and complementary, with the full combination being optimal.

**Batch Size Study (Figure 3, Domestic −10 dB)**: As batch increases from 32 to 128, accuracy rises from 50.61% to 57.02%; continuing to increase to 256 and 512 causes a drop back to 56.76% and 56.26%. The paper's explanation: Larger batches improve the statistical stability of channel-wise normalization, but excessively large batches introduce more noise samples and weaken the method's focus on individual samples, leading to diminishing returns or even negative returns. Based on this, batch=128 is selected as the balance point between computational efficiency and performance.

## Main Contributions

1. **Opening the Problem Space**: To the authors' knowledge, this is the first work to introduce test-time adaptation to KWS, shifting the question of "how to face unknown acoustic domains after edge KWS deployment" from the traditional framework of data collection/fine-tuning to a source-free, label-free, lightweight TTA framework.
2. **Selective Entropy Minimization**: Using a predefined threshold $\tau_{ent}$ to filter low-entropy samples and updating only BN parameters, specifically addressing the sensitivity of small-footprint models to perturbations and the collapse risk of online entropy minimization.
3. **Pseudo-Keyword Consistency (PKC)**: Pioneering the use of test-time time/frequency mask probes to measure the degree of dependence of predictions on keyword features, filling the blind spot of "unreliable feature samples" that the entropy criterion cannot identify; Ablation proves this is the primary source of gain (Table 4, +1.08 percentage points).
4. **Dual-Criterion Unified Sample Weighting Loss** (Equations 5/6): Unifying selection (hard filter) and weighting (soft weights) in one mechanism, with the entropy channel and PKC channel fused additively.
5. **Systematic Noise Domain Validation**: Three target domains of synthetic Gaussian (Table 1) + real environmental noise MS-SNSD (Table 2) + multi-source ESC-50 (Table 3), achieving the highest or tied-best results under all conditions, and demonstrating stability advantages over single-entropy criterion methods (Tent backlash dropping to 49.75%) under multi-source mixed noise.

## Limitations and Future Work

### Technical Limitations of the Method

- **Inference Overhead Not Quantified**: PKC requires an additional mask forward pass for each candidate sample (computing $p(x')_c$), nearly doubling the forward computation volume; The paper never reports the latency, VRAM, and energy overhead introduced by adaptation, which is a key missing piece for edge KWS premised on "always active low power." The paper's self-statement that future work will study resource efficiency amounts to admitting this gap.
- **Tension in the Realism of Batch Size 128**: Streaming KWS runs frame-by-frame; it is not realistic to gather 128 test samples at the same moment; And Figure 3 shows that at batch=32, accuracy is only 50.61% (6.41 percentage points lower than the peak), meaning the method's benefits are highly dependent on batched testing settings, creating a misalignment with real deployment forms.
- **Dependence on BN Structure**: The method only updates BN parameters, implicitly requiring the backbone to contain BN layers; It is not directly applicable to edge skeletons without BN (pure convolution structures or models using LayerNorm/GroupNorm), and generalization is constrained by architecture.
- **Unknown Hyperparameter Sensitivity**: $\tau_{ent}=0.4$, $\tau_{pkc}=0.05$, $\sigma=0.5$, batch=128, and mask quantity/length (2 time masks max 20 frames, 2 frequency masks max 5 bands) are all manually tuned fixed values. The paper provides no sensitivity analysis, and it is impossible to judge whether these values remain optimal after changing datasets or noise types.
- **Stability of Continuous Adaptation Not Evaluated**: Threshold-based sample filtering can only mitigate, not eliminate, the risks of entropy minimization collapse and error accumulation; The paper does not conduct long-term online adaptation or catastrophic forgetting experiments (the forgetting problem concerned by ETA is not measured in this paper).
- **Contradiction in PKC Directional Expression**: Equations (4) and (5) mathematically reward large $L_{pkc}$ (if confidence drops significantly after masking, the sample is selected and weighted heavily), which is consistent with the semantics of the introduction "selecting samples that are more clearly rooted in keywords"; However, Section 2.2.2 also writes "the smaller the drop, the more stable, we focus on these stable samples," which is the opposite direction. Reproducers need to verify independently which orientation is the effective implementation, which is an issue of writing clarity rather than conceptual logic.

### Deficiencies in Experimental Design

- **Single Task Form**: All experiments are completed on the GSC 35-class classification protocol, without validation on real streaming detection protocols (missed wake/false alarm trade-off); The gap between classification ACC and KWS engineering metrics is not discussed.
- **Clean Domain Forgetting Not Tested**: Table 1 does not have a result column for clean conditions ($\delta=0$); The paper does not report whether adaptation damages original performance without noise—this is exactly one of the core concerns of the ETA paper.
- **Inconsistency Between Main Text and Table Numbers**: For example, the main text claims Tent is 84.16% on Urban 10 dB and ETA is 81.54% on Human 0 dB, while Table 3 shows 83.25% and 67.90% at the corresponding positions; Tables should be cited as the standard, which also reflects insufficient careful proofreading of the paper.
- **Incomplete Information in Table 1**: The table note claims "five severity levels," but the obtainable table body only lists three tiers of $\delta=0.01/0.02/0.03$ and the average, with the values for the other two tiers missing.
- **No Statistical Significance Reported**: There are no repeated experiments with multiple random seeds or variance reports. Some leading margins (e.g., Typing +0.03, Human −10 dB +0.09, see Tables 2 and 3 respectively) are difficult to distinguish in quality within measurement noise.
- **Narrow Comparison Scope**: No comparison with more ASR-side TTA (SGEM, continuous TTA, etc.) or KWS-side custom wake words, incremental learning methods; Cross-dataset generalization is listed by the authors themselves as future work.

### Possible Future Improvement Directions

- **Directions Stated by the Paper**: Cross-dataset generalization and resource efficiency. Quantification of adaptation overhead should become the first priority.
- **Single-Sample/Small Batch Online Adaptation**: Combining single-sample normalization adaptation or sharpness-aware optimization (SAR route) to loosen the batch=128 constraint to a streaming-feasible level of 1~8 samples. This is a necessary path towards real deployment.
- **Automatic Threshold Mechanism**: Let $\tau_{ent}$ and $\tau_{pkc}$ adaptively adjust according to the intra-batch entropy distribution and consistency distribution, replacing fixed hyperparameters to reduce cross-domain tuning costs.
- **Detection-Style Evaluation**: Re-validate the method under fixed false alarm constraints for miss rate, streaming protocols, and edge-side latency/energy metrics to supplement engineering persuasiveness.
- **Extension of PKC**: Introduce more perturbation families (speed perturbation, additive perturbation, reverberation probes) or even learnable mask strategies, turning the "feature reliability probe" into a more general testing-phase tool.
- **Combination with Incremental Learning**: The author team has layouts in the direction of KWS incremental learning at the same time (AnalyticKWS, Dark Experience, etc., cited in this paper). TTA handles domain shift, and incremental learning handles class expansion. The joint use of the two is a natural extension direction.
