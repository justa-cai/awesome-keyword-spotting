# Personalized Keyword Spotting for User-Defined Keywords Leveraging Text-Independent Speaker Verification

- **Authors/Affiliations**: Ming-Hsiang Hu, Kuan-Tang Huang, Chien-Chun Wang, Berlin Chen - Dept. of Computer Science and Information Engineering, National Taiwan Normal University (NTNU); Hung-Shin Lee - United Link Co., Ltd. (Taiwan)
- **Date**: 2026.06 (arXiv 2606.20106v1, submitted June 18, 2026)
- **Link**: https://arxiv.org/abs/2606.20106 (Code: https://github.com/Padawan101/ZP-KWS)
- **Keywords**: User-Defined Keyword Spotting (UD-KWS), Text-Independent Speaker Verification, Dual Zero-Shot, GE2E Loss, Multiplicative Late Fusion, Phoneme Supervision, Edge Deployment

## Problem Statement

### Problem Background and Domain Pain Points

Wake-up mechanisms for speech interaction interfaces require efficiency, security, and personalization. Traditional Keyword Spotting (KWS) has long focused on small models on the device side. User-Defined Keyword Spotting (UD-KWS) further supports zero-shot wake-word detection from arbitrary text inputs, eliminating the need for retraining when new keywords are introduced. Cross-modal alignment methods (such as CMCD, PhonMatchNet) achieve this by matching phoneme sequences derived from the text side with audio representations. Subsequent works have explored multimodal fusion and scaling with dual data sizes.

However, this technical route carries a structural cost: to generalize across different speakers, the representations learned by the model must be "speaker-invariant." Such systems are fundamentally incapable of rejecting "impostors who say the keyword correctly"—for example, bystanders or playback audio reciting the correct wake word. In always-on edge deployments, such false activations degrade user experience and waste power. The paper refers to this scenario as the "dual zero-shot" problem: both the keyword and the speaker are unseen, placing pressure on two generalization dimensions simultaneously.

The natural candidate for introducing personalization into UD-KWS is Speaker Verification (SV). However, existing methods like PK-MTL follow a Text-Dependent (TD-SV) route: the enrollment and test speech must share the same preset keyword, benefiting from phoneme alignment. Once the keyword changes, both the KWS classifier and the speaker enrollment must be updated—this precisely destroys the core zero-shot flexibility of UD-KWS. Switching to Text-Independent (TI-SV) resolves the constraint that "enrollment and query speech must have consistent phonetic content," but TI-SV faces two practical obstacles: first, TI-SV is already mature for long speech (e.g., x-vector, ECAPA-TDNN, CAM++, ERes2NetV2), and its discriminative power degrades significantly on sub-second short speech; second, typical TI-SV models have over 10M parameters, which is too heavy for edge devices.

### Specific Shortcomings of Existing Methods

- **Speaker Blindness of UD-KWS Backbone Methods**: Phoneme-guided zero-shot KWS methods like PhonMatchNet perform well in standard detection modes, but their representations naturally erase speaker cues. The most striking number in the paper's experiments is: when PhonMatchNet switches to "Target Speaker Only" (TO-KWS) mode, the FRR@1% FAR on LibriPhrase Easy reaches 97.00%, and 93.12% on Qualcomm—equivalent to being completely unable to distinguish "the target user saying the word correctly" from "an impostor saying the word correctly."
- **Text-Dependent Assumption of PK-MTL**: PK-MTL jointly trains KWS and TD-SV using a shared encoder, then merges the keyword score and speaker score using task-specific scoring functions. The premise of TD-SV is that enrollment and test speech share the same preset keyword. If the keyword changes, the classifier and speaker enrollment must be redone, losing the zero-shot property. Additionally, the multi-task design of the shared encoder carries the risk of task interference: gradients from the semantic task can distort the speaker representation space.
- **Direct Transfer of TI-SV is Infeasible**: Mature TI-SV models have >10M parameters, exceeding edge budgets; moreover, their statistical pooling degrades noticeably on sub-second queries with limited temporal context.
- **Compensation Vulnerability of Additive Score Fusion**: Weighted addition of keyword and speaker scores allows a high keyword score to "compensate" for a low speaker score, allowing impostors to still pass the threshold—this is semantically incorrect for security gating.

### Key Challenges Addressed by This Paper

The proposed ZP-KWS (Zero-shot Personalized KWS) aims to satisfy five mutually constraining objectives within a 1.55M parameter budget: (1) Keyword zero-shot—changing words does not require model changes or speaker re-registration (this dictates the need for TI-SV rather than TD-SV); (2) Speaker zero-shot—the enrolled speaker is unseen during training; (3) Speaker embeddings with sufficient discriminative power on sub-second short speech; (4) Strict AND gating semantics—keyword content and speaker identity must be verified independently; if either fails, the activation is rejected; (5) Standard keyword detection accuracy must not degrade due to the addition of the speaker branch, and mode switching (standard/biased/target-only) must not require retraining. The combined force of these five points points to a design of "functionally decoupled dual branches + inference-time fusion," rather than continuing multi-task learning on a shared encoder.

## Methodology

### Overall Architecture Design and Design Motivation

ZP-KWS consists of two functionally completely separate branches: the TI-SV branch is responsible for speaker identity, and the phoneme supervision branch is responsible for keyword content. The core design decision is to place personalization in **inference-time late fusion** rather than joint training. The motivations for this are threefold: first, to prevent semantic task gradients from polluting the speaker embedding space (the old problem of PK-MTL's shared encoder); second, to give each branch independent veto power, requiring activation to pass both content and identity verification; third, the three operating modes rely only on threshold switching, without requiring retraining.

**Speaker Branch**: Uses EfficientTDNN-Small (approx. 0.9M parameters) to extract 192-dimensional speaker embeddings. The key choice is to **freeze** this encoder during KWS training—preventing semantic task gradients from distorting the speaker embedding space. During evaluation, each target speaker uses only one enrollment speech, and the keyword of the enrollment speech can differ from the query speech. This is the significance of the text-independent setting in products: users do not need to recite the wake word to complete voiceprint registration. The enrollment embedding $e^{enroll}_{spk}$ and the query embedding $e^{input}_{spk}$ are mapped to probability $p_{spk}$ via cosine similarity + a calibrated linear layer.

**Keyword Branch**: The audio encoder follows a dual-stream approach. A frozen pre-trained embedding (Lin et al., ICASSP 2020) calculates 96-dimensional features every 80ms, which are upsampled to a 20ms frame rate via transposed convolution and linearly projected (following PhonMatchNet) to obtain $E_{pt}$; the trainable stream uses 40-dimensional log-Mel passing through two Conv1D layers (128, 256 channels, kernel size 5, each followed by BN+ReLU) and two BiGRU layers (128 hidden units per direction), then fully connected projection to 128 dimensions to obtain $E_{stft}$. The two streams are fused as $E_{audio} = E_{pt} + \mathrm{LayerNorm}(E_{stft})$—the role of LayerNorm here is to control the scale of the trainable stream so that it does not drown out the frozen stream $E_{pt}$ carrying linguistic priors during addition. On the text side, G2P converts keywords into phonemes, which are then projected into semantic embeddings $E_{text} \in \mathbb{R}^{T_t \times 128}$. The concatenated sequence $E_{concat}=[E_{audio}; E_{text}]$ enters the pattern extractor (a single-head scaled dot-product self-attention with causal masking) to obtain $E_{joint}$; the pattern discriminator decodes outputs at two granularities from $E_{joint}$: utterance-level matching probability $p_{utt} \in [0,1]$ (single-layer GRU, 128 hidden units + FC + sigmoid) and phoneme-level matching sequence $p_{phon} \in \mathbb{R}^{T_t \times 1}$ (element-wise FC + sigmoid on the text-aligned part of $E_{joint}$).

Total parameters are approx. 1.55M: KWS branch 0.65M + Speaker encoder 0.90M. The runtime overhead during deployment is deliberately kept low: the speaker encoder is frozen, and enrollment embeddings are calculated only once per user; each query adds only one cosine similarity calculation and one scalar multiplication.

### Mathematical Principles of Core Algorithms

**Multiplicative Late Fusion** (Eq. 1):

$$p_{final} = p_{utt} \cdot p_{spk}, \qquad p_{utt}, p_{spk} \in [0,1]$$

The product form ensures that $p_{final}$ approaches 1 if and only if both probabilities are close to 1; if either probability approaches 0, the product is pulled to 0, meaning each branch has veto power. The essential difference from additive fusion $\alpha p_{utt} + (1-\alpha) p_{spk}$ is that additive terms can compensate for each other (high scores mask low scores), whereas multiplicative terms cannot.

**Calibrated Mapping of Speaker Matching Probability** (Eq. 2):

$$p_{spk} = \sigma\left(w_{spk} \cdot \cos(e^{input}_{spk}, e^{enroll}_{spk}) + b_{spk}\right)$$

where $\sigma(\cdot)$ is sigmoid, $w_{spk}=10$, $b_{spk}=-5$ are **fixed** hyperparameters (not involved in training), corresponding to a decision boundary of cosine similarity 0.5, with values based on the score distribution of the training set. The necessity of this step comes from dimensional alignment: $p_{utt}$ is a probability output by sigmoid, while the original cosine similarity clusters around 0.5 with insufficient dynamic range. Direct multiplication would cause the fusion to lose discriminative power (ablation experiments confirm: removing the calibration layer worsens the TO-KWS EER on LibriPhrase Easy from 8.16% to 15.24%).

**Label Smoothing Cross-Entropy for Frame-Level Phoneme Supervision** (Eqs. 3, 4):

$$L_{align} = -\frac{1}{|V|}\sum_{t \in V}\sum_{d=1}^{D} \tilde{y}_{t,d} \log \hat{y}_{t,d}, \qquad \tilde{y}_{t,d} = (1-\varepsilon)\,y_{t,d} + \frac{\varepsilon}{D}$$

where $V$ is the set of valid frame indices within the batch, $\hat{y}_{t,d}$ is the softmax prediction, $y_{t,d} \in \{0,1\}$ is the one-hot ground truth, $D=42$, and the smoothing parameter $\varepsilon = 0.1$. The source of the labels is the 20ms frame-level targets obtained by running Montreal Forced Aligner (MFA) on the LibriPhrase training set. The smoothing term models the temporal ambiguity at forced alignment boundaries: the attribution of boundary frames is inherently ambiguous, and hard one-hot labels would force overconfident predictions. The uniform component $\varepsilon/D$ explicitly writes this uncertainty into the target distribution.

**GE2E Loss** (Following the standard form of Wan et al., ICASSP 2018, used by the paper for second-stage fine-tuning): For a batch of $N$ speakers, with $M$ utterances per speaker, the embedding $e_{ji}$ is compared with each speaker centroid $c_k$ ($c_k = \frac{1}{M}\sum_{m} e_{km}$, excluding self in the same-speaker case) via cosine similarity and scaled with bias to $S_{ji,k} = w \cdot \cos(e_{ji}, c_k) + b$. The loss is

$$L_{GE2E} = \sum_{j,i}\left(S_{ji,j} - \log\sum_{k}\exp S_{ji,k}\right)$$

Intuitively, this pulls each embedding toward its own speaker's centroid and pushes it away from other centroids in the batch. The paper uses $N=16$, $M=10$, fine-tuning for 30 epochs on the 460-hour LibriPhrase training set (AdamW, learning rate $10^{-4}$).

**Total Objective** (Eq. 5): $L_{total} = L_{utt} + L_{phon} + L_{align}$, summed with equal weights. $L_{utt}$ and $L_{phon}$ are BCE losses on the utterance-level $p_{utt}$ and phoneme-level $p_{phon}$, jointly optimizing keyword matching at coarse and fine granularities; meanwhile, a key truncation is applied to the gradients—gradients from $L_{align}$ only backpropagate to the trainable stream, not entering the frozen pre-trained stream, preventing phoneme supervision from perturbing the already solidified representations of $E_{pt}$.

### Key Technical Innovation 1: Sub-second TI-SV Speaker Encoder via GE2E Two-Stage Pre-training

**Problem Addressed**: Statistical pooling suffers from insufficient temporal context in sub-second queries, leading to large embedding variance and collapsed discriminative power. The paper does not switch to a larger SV backbone (which would explode the parameter budget), but instead revives the 0.9M EfficientTDNN-Small using "out-of-domain pre-training + in-domain GE2E fine-tuning": first pre-trained on VoxCeleb2 to acquire speaker discriminative power across acoustic conditions, then fine-tuned with GE2E loss on the 460-hour LibriPhrase training set, which is dominated by sub-second keyword segments. The value of GE2E for short speech lies in its use of "batch centroids" as anchors for metric learning, directly constraining intra-class aggregation and inter-class separation of high-variance short embeddings, rather than relying on the assumption that pooling statistics are stable only on long speech.

**Evidence Chain**: Isolated experiments (evaluating speaker verification separately) show that GE2E fine-tuning reduces the EER on LibriPhrase from 22.19% to 8.41% (a relative reduction of 62%), and on GSC from 23.07% to 12.28% (a relative reduction of 47%); on Qualcomm, it only slightly drops from 6.52% to 6.23%. The paper explains that this dataset has longer speech, where embeddings are already stable—this pattern of "benefit disappearing as speech lengthens" indirectly confirms that the role of GE2E is indeed on short speech. When placed in the end-to-end system, removing GE2E pre-training causes the largest TO-KWS degradation in the entire system: FRR@1% on LibriPhrase Easy rises from 29.47% to 73.42%, and on Qualcomm from 33.12% to 67.44%, almost reverting to PK-MTL levels.

### Key Technical Innovation 2: Frame-Level Phoneme Supervision ($L_{align}$) Acting Only on the Trainable Stream

**Problem Addressed**: Downstream objectives ($L_{utt}$, $L_{phon}$) only supervise the end-to-end result of "keyword matching," providing only indirect constraints on the frame-level phonetic structure inside the audio encoder. The paper adds an **auxiliary classification head that exists only during training** to $E_{stft}$, projecting the trainable stream to predict frame-level phoneme posteriors of dimension $D=42$. The entire head is removed during inference, incurring zero runtime overhead.

Two design details explain "why": first, supervision is added only to $E_{stft}$ and not to the fused $E_{audio}$, to force the trainable stream to learn phoneme discrimination itself, rather than lying on the linguistic priors already present in the upstream frozen stream $E_{pt}$; second, label smoothing ($\varepsilon=0.1$) is used when generating labels via MFA forced alignment because the attribution of alignment boundary frames is inherently ambiguous. The benefit is reflected in hard cases: adding $L_{align}$ reduces Qualcomm's C-KWS EER from 10.81% (when removed) to 6.88%. The cost is that it is not a universal gain—on tasks with low phoneme overlap, it causes over-regularization (see the counter-example in GSC in the ablation section).

### Key Technical Innovation 3: Multiplicative Late Fusion and Re-training-Free Three-Mode Switching

**Problem Addressed**: How to cover the entire spectrum of strictness from "standard detection" to "strict speaker gating" without retraining. The answer is to make it a pure inference-time threshold problem: $p_{final} = p_{utt} \cdot p_{spk}$ (Eq. 1), combined with the calibrated $p_{spk}$ mapping (Eq. 2). The three operating modes are defined as:

| Mode | ts-tk (Target Speaker/Target Keyword) | nts-tk (Non-Target Speaker/Target Keyword) | ts-ntk | nts-ntk |
|---|---|---|---|---|
| C-KWS (Standard) | Accept | Accept | Reject | Reject |
| TB-KWS (Target Biased) | Accept | Neutral (not counted as positive/negative) | Reject | Reject |
| TO-KWS (Target Only) | Accept | Reject | Reject | Reject |

C-KWS sets $p_{spk} \equiv 1$ (multiplicative identity, bypassing the speaker branch). TO-KWS is the strictest mode—only "target speaker says target keyword" activates; impostors saying the word correctly are also rejected. The multiplicative form has an implicit benefit here: switching modes requires changing no weights, only the threshold. The paper also conducted an alternative operator experiment: the result of element-wise minimum $p_{final} = \min(p_{utt}, p_{spk})$ was comparable to multiplication, indicating that the real benefit comes from the "AND gating" semantics itself, rather than the specific operator of multiplication—this is a clean validation of the design essence.

### Technical Differences with Existing Methods

- **Difference from PK-MTL (Most Core)**: PK-MTL is TD-SV; enrollment and test share a preset keyword. Changing words requires retraining the classifier and re-enrolling, breaking the zero-shot property. ZP-KWS is TI-SV; enrollment speech content is arbitrary, and changing words is zero-cost. PK-MTL uses a shared encoder for joint multi-task training, where semantic and speaker gradients pull against each other in the same encoder. ZP-KWS freezes the speaker encoder entirely, physically isolating the two tasks. PK-MTL's Score Combination Module (SCM) uses additive/linear fusion (coefficient $\alpha$ requires tuning), while ZP-KWS uses multiplication to enforce AND semantics with fixed, non-trainable fusion weights. One can say ZP-KWS replaces PK-MTL's "parameter sharing" with "structural decoupling."
- **Difference from PhonMatchNet**: PhonMatchNet is pure content matching, speaker-independent. ZP-KWS retains the PhonMatchNet-style phoneme-text matching backbone (all systems in the experiments share this backbone to ensure fairness) and parallels a speaker branch on top of it. The increment is the capability of "personalized gating," not rebuilding the KWS backbone.
- **Difference from Mainstream TI-SV Systems**: x-vector/ECAPA-TDNN/CAM++/ERes2NetV2 etc. are oriented towards long speech and have >10M parameters. ZP-KWS's speaker branch has only approx. 0.9M parameters, and is specifically fine-tuned with GE2E in the sub-second domain, making it a TI-SV "customized for wake-word duration distribution," rather than a direct transfer of a general SV model.
- **Difference from the Naive Combination of "Additive Score Fusion + Speaker Branch"**: Naive weighted summation has compensation vulnerabilities, and weights are hard to determine when score dimensions are inconsistent. ZP-KWS's calibrated sigmoid mapping + multiplication maps both branches to the [0,1] probability space before performing non-compensable combination.

## Experimental Results

### Datasets Used and Their Scales

- **LibriPhrase (Easy / Hard split)**: In-domain benchmark, training set 460 hours (used for GE2E fine-tuning and main training). The Hard split consists of minimal pairs, specifically examining fine-grained phoneme-level discrimination.
- **Google Speech Commands (GSC)** and **Qualcomm Keyword Speech**: Out-of-domain generalization tests, introducing unseen acoustic conditions and different vocabularies. The paper does not report the specific number of trials and duration statistics for these two evaluation sets.
- All evaluations enforce strict dual zero-shot: target keywords and target speakers do not appear in the training data. During evaluation pairing, a 1:1 balanced ratio of target/non-target speakers is maintained.

The baseline configuration is carefully designed for fairness: all systems (including the two baselines) share PhonMatchNet as the UD-KWS backbone and feature extractor. PK-MTL was originally designed for a fixed vocabulary; the paper replaces its classifier-style keyword backbone with PhonMatchNet to enable zero-shot keyword settings, with the speaker score generated by its SCM module and linear fusion coefficient $\alpha$ tuned on the validation set—the authors explicitly state this is a stronger baseline configuration than the original PK-MTL. Implementation details: 40-dim log-Mel (25ms window, 10ms hop), AdamW learning rate $10^{-4}$, batch size 2048, single NVIDIA RTX 5090, equal loss weights, $\varepsilon=0.1$.

### Definition and Rationale for Evaluation Metrics

Following PK-MTL's evaluation protocol, each trial is defined by two binary attributes: "keyword match" and "speaker match," resulting in four pairing types (ts-tk, nts-tk, ts-ntk, nts-ntk), which are aggregated according to the table above into three modes: C/TB/TO. Three metrics are used: **EER** measures overall discriminative power; **FRR@1% FAR** (False Rejection Rate at 1% False Acceptance Rate) corresponds to a strict security working point; **FRR@10% FAR** corresponds to a relaxed working point. The rationale is that edge always-on deployment cares most about the low FAR region (the cost of false activation in power and experience is high), and 1% FAR is the working point closest to reality; on the Hard split, the FRR@1% for all systems exceeds 80%, at which point FRR@10% becomes the meaningful observation window. Significance testing uses paired bootstrap (1000 resamples), with improvements significant at p < 0.001 across all four datasets.

### Detailed Comparison with Baseline and SOTA Methods

**TO-KWS (Core Battlefield)**: ZP-KWS reduces the FRR@1% on LibriPhrase Easy from 72.79% (PK-MTL) to 29.47% (a relative reduction of approx. 60%), and from 97.00% (PhonMatchNet) to 29.47% (a relative reduction of approx. 70%); on Qualcomm, it reduces from 55.71% to 33.12% (a relative reduction of approx. 41%), and from 93.12% (PhonMatchNet) to 33.12% (a relative reduction of approx. 64%). EER improves synchronously: Easy 8.16% (PK-MTL 17.74%, PMN 25.16%), Qualcomm 8.81% (PK-MTL 13.73%, PMN 16.06%), GSC 11.05% (PK-MTL 13.14%, PMN 12.27%).

**C-KWS (Confirming no "accuracy loss from adding security")**: ZP-KWS achieves the best C-KWS EER on three of the four datasets (Easy 2.38% vs PMN 3.34%, PK-MTL 3.46%; Qualcomm 6.88% vs 8.67%/8.88%; Hard 17.48% vs 20.48%/21.31%). The exception is GSC (10.74%, slightly worse than PMN's 10.07%). On FRR@1%, Qualcomm 23.19% is significantly better than both baselines (24.57%/39.18%).

**Hard Minimal Pairs**: The C-KWS FRR@1% for all systems is pushed above 80%, indicating the task has entered an ultra-hard zone; from the FRR@10% perspective, ZP-KWS still leads significantly (C-KWS 31.69%, TO-KWS 19.64%, vs PMN's TO-KWS 74.94%), and achieves the best TO-KWS EER of 13.87% (PMN 31.86%, PK-MTL 24.04%)—speaker verification still plays a complementary role to keyword matching under extreme phoneme overlap.

**DET Curve (Fig. 2, TO-KWS Mode)**: In the most critical low FAR region for operation (FAR ≤ 5%), the miss rate for both baselines stays above 50%—speaker-independent systems have no ability to reject impostors under tight thresholds. ZP-KWS's EER is 13.6%, compared to PK-MTL 28.9% and PhonMatchNet 32.5%, and the gap widens as the working point becomes tighter. The paper does not specify the exact data split corresponding to this curve.

**TB-KWS**: ZP-KWS shows a prominent advantage on Qualcomm (EER 5.67% vs PK-MTL 8.55%, PMN 9.04%), and 2.36% on Easy, almost identical to C-KWS, indicating that in target-biased mode, speaker information only adds gain without dragging down performance.

### Findings from Ablation Experiments

Ablations are conducted inline in Table 1, with three components each having a clear profile, and proven to have complementary contributions:

- **Removing GE2E Pre-training → Largest TO-KWS Collapse**. TO-KWS FRR@1% on Easy worsens from 29.47% to 73.42%, and on Qualcomm from 33.12% to 67.44%; corresponding isolated SV experiments show EER 22.19%→8.41% (LibriPhrase, 62% relative) and 23.07%→12.28% (GSC, 47% relative). It is the primary contributor to personalization capability, and the loss falls almost entirely on TO mode, with C mode remaining basically unchanged—precisely locating the component's role.
- **Removing Calibration Linear Layer → Fusion Misalignment**. TO-KWS EER worsens from 8.16% to 15.24% (Easy), 8.81%→15.00% (Qualcomm). The mechanistic explanation is that the original cosine similarity clusters around 0.5 with insufficient dynamic range; uncalibrated scores multiplied by $p_{utt}$ lack discriminative power. This shows that multiplicative fusion is not just "multiply them"; dimensional alignment is a prerequisite.
- **Removing Frame-Level Phoneme Supervision $L_{align}$ → Content Branch Weakens, with Counter-Examples**. C-KWS EER worsens on three of the four datasets, with the largest gap on Qualcomm (6.88%→10.81%), and TO-KWS is also heavily penalized (Qualcomm FRR@1% 33.12%→43.64%). The counter-example is GSC: removing $L_{align}$ actually improves C-KWS EER from 10.74% to 10.08%, and FRR@1% from 35.96% to 28.05%. The paper provides a consistent explanation: on tasks with low inter-class phoneme overlap, forcing fine-grained phonetic structure is over-regularization—coarse-grained cues are already sufficient, and $L_{align}$ instead consumes capacity. This is a rare disclosure of a "negative boundary condition" in the paper, which is highly valuable.
- **Min Fusion vs. Multiplicative Fusion**: The results of element-wise minimum are comparable to multiplication, proving that the benefit comes from the AND gating semantics rather than the operator choice.

## Main Contributions

1. **Problem Formalization**: Defines "UD-KWS + Personalization" as a dual zero-shot problem (unseen keyword × unseen speaker), and points out that the TD-SV route (PK-MTL) is structurally incompatible with zero-shot flexibility, necessitating TI-SV. This definition, along with the three-mode (C/TB/TO) evaluation protocol, provides a reusable experimental framework for subsequent work.
2. **Feasible Path for Sub-second TI-SV**: Proves that a 0.9M EfficientTDNN-Small, after "VoxCeleb2 pre-training + GE2E fine-tuning on LibriPhrase," can stably discriminate on sub-second speech (isolated EER relative reduction of 62%/47%), breaking two default assumptions: "TI-SV must have >10M parameters" and "must require long speech."
3. **Multiplicative Late Fusion**: Achieves independent veto for dual branches via $p_{final} = p_{utt} \cdot p_{spk}$, enabling re-training-free switching of three operating modes; TO-KWS FRR@1% reduces by up to 60% relative to the strongest baseline, while C-KWS does not degrade (best EER on three of four datasets).
4. **Edge Feasibility**: The entire system has 1.55M parameters (0.65M KWS + 0.90M SV), enrollment embeddings are calculated only once per user, and each query adds only one cosine calculation and one scalar multiplication. The engineering landing background is clear (funded by Realtek, projects 113KK01103, 114KK01005), and code is open-sourced.

## Limitations and Future Work

### Technical Limitations of the Method

- **Absolute Performance of TO Mode is Still Far from Practical**. FRR@1% on Easy is still 29.47%, and on Hard 78.20%; even with the relaxed FRR@10%, Easy still has 6.93%. While the relative improvement (60%) is striking, the absolute numbers mean the target user might fail one out of every three wake-ups— the usability cost of strict gating remains high. The paper does not discuss trade-off strategies between false rejection rate and user experience (e.g., degraded retry after TO mode failure).
- **No Anti-Spoofing Capability**. The paper uses "playback audio" as one of the motivations, but the TI-SV branch has no protection against playback or TTS synthesized speech, and there are no related evaluations in the experiments. A recorded playback that passes voiceprint verification can deceive both branches simultaneously, which is a substantial gap in the narrative of personalized wake-up as a "security mechanism."
- **Calibration Uses Fixed Hyperparameters, Not Learned Ones**. $w_{spk}=10$, $b_{spk}=-5$ are derived from the training set score distribution (corresponding to a cosine 0.5 boundary). It is questionable whether this linear mapping remains optimal after domain transfer (TO-KWS on GSC and Qualcomm is significantly worse than in-domain); the paper itself lists "confidence calibration under noise and mismatch conditions" as future work.
- **No A Priori Criteria for the Applicability Boundary of $L_{align}$**. The negative gain on GSC shows that phoneme supervision is not a harmless regularization, but the paper does not provide a criterion for "when to turn it on" (e.g., using phoneme overlap degree or vocabulary confusion degree as a switch).
- **Speaker Encoder is Frozen Throughout**. This secures the representation space, but at the cost of inability to adapt to the deployment domain. The setting of a single enrollment speech also means the impact of enrollment quality (duration, channel, noise) on $e^{enroll}_{spk}$ is not studied. Evaluations cover only English (LibriSpeech series, GSC, Qualcomm), and the multi-lingual transferability of G2P and the phoneme table (D=42) is not verified.

### Shortcomings in Experimental Design

- **Efficiency Only Reports Parameter Count**. Beyond 1.55M parameters, the paper does not report inference latency, RTF, FLOPS/MACs, memory usage, or any real edge hardware measurements—"runtime overhead remains low" is only a qualitative assertion. For a work主打 edge deployment, this is the dimension that needs to be supplemented most.
- **Paper Does Not Report**: Specific number of evaluation trials, duration distribution of enrollment speech, number of speakers, specific composition of target/non-target keywords on each dataset; the data split corresponding to the DET curve in Fig. 2 is also not specified.
- **Baselines are Modified**. PK-MTL's keyword backbone is replaced with PhonMatchNet, and $\alpha$ is tuned on the validation set—this makes the comparison fairer, but also means the reported PK-MTL numbers are not from the original implementation, and cannot be directly compared with the original results of PK-MTL in the literature.
- **1:1 Speaker Balance Ratio** is an artificially constructed prior. In real scenarios, the frequency structure of impostor appearance is different, and the FRR/FAR ranking under this ratio may not transfer.
- **Operator Substitution Experiment Only Tested min**, not covering variants like weighted geometric mean or noise-robust soft AND; there is also no sensitivity analysis for the choice of equal loss weights.

### Possible Directions for Future Improvement

- **Directions Self-Identified by the Paper**: Confidence calibration under noise and mismatch conditions, which responds to both the fixed hyperparameter of the calibration layer and the degradation out-of-domain.
- **Add Anti-Spoofing**: Incorporate playback detection/TTS detection as a third gating layer, or mix synthetic speech into the training of the speaker branch for adversarial augmentation, to support the complete narrative of "biometric security."
- **Multiple Enrollment Speeches and Adaptive Thresholds**: Using the mean or uncertainty estimation of embeddings from multiple enrollment speeches to dynamically adjust the $p_{spk}$ threshold could help reduce the high false rejection rate in TO mode—this is a direct extension beyond the paper's setting (single enrollment).
- **Edge Testing and Streaming**: Provide latency and power numbers on real MCU/DSP, and examine the streaming inference compatibility of the causal masked attention + BiGRU structure (the bidirectionality of BiGRU requires special handling in streaming scenarios, which the paper does not discuss).
- **Adaptive Switch for $L_{align}$**: Automatically determine the weight of auxiliary supervision based on dataset phoneme confusion, turning the negative gain on GSC into an avoidable risk.
- **Multi-lingual and Cross-Channel Extension**: Verify the performance of G2P, the D=42 phoneme table, and the frozen speaker encoder in languages such as Chinese, and under far-field/multi-microphone channels.

---

**One-Sentence Summary**: ZP-KWS solves the "dual zero-shot" problem via "structural decoupling"—a frozen 0.9M TI-SV branch handles identity, a 0.65M branch with phoneme supervision handles content, and multiplicative fusion at inference time handles AND gating; the contributions of the three components do not overlap in ablation studies. It proves that personalization need not sacrifice zero-shot keyword generalization, but the absolute false rejection rate of TO mode, the missing anti-spoofing evaluations, and the unmeasured edge latency determine that it is currently more like a correct problem definition plus a feasible technical route, rather than a product solution ready for direct deployment.
