# KFC-KWS: Keyframe Fusion with CTC for User-Defined Keyword Spotting

- **Authors/Affiliations**: Jin Li, Wenbin Jiang (Corresponding Author), Ji Hu - Hangzhou Dianzi University (School of Electronic Engineering / School of Communication Engineering)
- **Date**: 2026.06 (arXiv:2606.10365v1, submitted June 9, 2026, cs.SD, Interspeech template)
- **Link**: https://arxiv.org/abs/2606.10365
- **Keywords**: user-defined keyword spotting, CTC peaky posterior, keyframe selection, multi-modal fusion (phoneme-text-audio), cross-attention, modality dropout, phonetically confusable keywords

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) determines whether a specified keyword appears in a speech segment and serves as the entry component for voice interaction devices such as smart speakers, mobile phones, and wearable devices. Mainstream KWS methods are trained on large-scale pre-defined keyword samples, inherently supporting only fixed vocabularies. To pursue an open-vocabulary approach, systems typically switch to CTC or Transducer frameworks with beam search decoding; however, such pipelines have high training costs and are difficult to efficiently adapt to new keywords on-device. From a product perspective, this represents an escalating adaptation cost: each new wake word or localized entry implies a round of re-collection, re-training, and re-deployment. What users truly want is the ability to say a self-selected word, with the device recognizing it on the spot.

User-Defined Keyword Spotting (UD-KWS) emerges from this need: users register a new keyword via text or audio, and the model learns to detect it on the fly. Two registration routes have their respective trade-offs: text registration extracts phoneme features and matches them with audio embeddings, offering stable performance but degrading under heavy accents or strong noise; audio registration (query-by-example) better aligns with the user's own pronunciation, typically implemented via metric learning or meta-learning. Recent work further combines text and audio registration into multi-modal frameworks (CLAD, MM-KWS, PLCL), achieving near-perfect accuracy on easy keywords.

The true deployment pain point is not easy words, but false triggers caused by phonetically confusable words: target words and distractors often differ by only one or two phonemes (e.g., the pair *night* and *light*, which differ only in the initial phoneme, as cited by the author). Full-sentence matching methods treat all frames equally; since differences exist only at a few phoneme positions, uniform attention dilutes this critical discriminative signal across the entire sequence. The cost of errors is also asymmetric: a miss makes the device seem "deaf," which users can correct by repeating; a false trigger occurs when the user has no interaction intent, causing greater experience damage, and accumulates linearly with the device's always-on listening time. Therefore, the paper explicitly identifies "reducing false activations caused by phonetically confusable words" as the core challenge for practical deployment. The paper's core observation lies here—confusable word pairs usually differ only at a few phoneme positions, so the detection system needs to concentrate discriminative resources on these positions rather than applying uniform effort across the whole sentence.

### Specific Deficiencies of Existing Methods

According to the paper's review, four representative methods have respective gaps:

- **CLAD [23] / MM-KWS [24]**: Cross-modal full-sequence matching treats all frames equally regardless of speech salience. Subtle phoneme-level differences are diluted by the entire sequence, which is the source of false triggers for confusable words.
- **PLCL [25]**: Introduces an external phoneme memory bank to enhance phoneme-level representations. While the discriminative granularity is correct, it requires additional external components, with a parameter count of 40.0M (Table 1), contradicting the lightweight requirements for edge devices.
- **AdaKWS [27]**: Uses adaptive instance normalization for fusion. Without explicit phoneme-level alignment, it lacks structural support for discriminating hard examples.
- **CED [10] and other phoneme-text isomorphic embedding methods**: Rely on global embedding similarity, achieving only 89.20%~94.02% AUC on the LPH subset (Table 1), clearly struggling with confusable words.

The common gap is: no method utilizes the model's own alignment capability to automatically locate "the few phoneme positions where keywords truly differ." Either they apply uniform effort or introduce additional components to do so.

### Key Challenges Addressed by This Paper

How to automatically select frames carrying the strongest phoneme discriminative information (keyframes) without adding external modules or increasing inference pipeline burden, using them for phoneme-aligned cross-modal matching, while retaining full-sentence representations to cover global context; and ensuring the entire framework achieves this within a budget of approximately 2.0M trainable parameters, simultaneously improving AUC and EER on high-confusion subsets like LibriPhrase-Hard compared to contemporary methods.

## Methodology

### Overall Architecture Design and Design Motivation

KFC-KWS is a "registration-query" multi-modal verification framework (Fig. 1). On the registration side, the input is a pair (audio + text). The text passes through a G2P converter to obtain phoneme features and a text encoder to obtain semantic features. On the query side, the query audio shares the same frozen pre-trained audio encoder (XLS-R 0.3B) as the registration audio—the motivation for sharing the encoder is to ensure that query and registration audio features naturally fall into the same space, eliminating the alignment cost between different encoders. Four types of features (query audio $E_a^q$, registration audio $E_a^s$, registration phoneme $E_p^s$, registration text semantics $E_t^s$, all of dimension $T \times D$) pass through linear projections to map to a 128-dimensional shared space, followed by positional encoding (to preserve temporal structure) and modality encoding (to identify modality type). The motivation for unified representation format is direct: let heterogeneous features follow the same processing pipeline and fall into the same latent space, so that cross-modal matching has a common coordinate system.

On top of this, two query branches run in parallel (QbyOmni and QbyKeyframe). In Fig. 1, the three pre-trained encoders are all frozen (snowflake markers); the trainable parts are only the projection layers, the two query branches, and the CTC head—this constitutes the 2.0M trainable parameters, explaining why training can be completed in 50 epochs on a single 4080 Super. The two branches are:

**QbyOmni (Full-Feature Query Branch)**: Concatenates query audio features with each registration modality feature along the time dimension, passes them through self-attention to obtain enhanced representations, and then maps them to a fixed-dimension sequence $F_m^c$ via GRU and fully connected layers (Eq. 1, 2). This branch retains information from all frames, responsible for global context—corresponding to scenarios where "easy words are sufficient with overall similarity."

**QbyKeyframe (Keyframe Query Branch)**: First, a CTC selector selects phoneme-aligned keyframes from the query audio embeddings. It calculates cosine similarity matrices with each of the three registration modalities separately. Then, using the similarity matrix as Query and the full-sentence features output by QbyOmni as Key/Value, it performs cross-attention (Eq. 5, 6). The results of the three modalities $F_a^k, F_p^k, F_t^k$ are summed to output sentence-level confidence. This branch is responsible for scenarios where "hard words must dig into phoneme details."

Why the dual-branch approach: The paper's comparison and trade-off analysis (Section 4.1.2, Table 1/2) show that full-sequence methods can achieve EER below 1% on LPE (easy samples are acoustically separable, with sufficient redundant cues), while sparse keyframe selection naturally discards some redundant cues, resulting in an LPE EER of about 2%. Conversely, on LPH, the keyframe branch is significantly stronger. The two branches attack opposite ends, fusing at the end to win on balanced metrics.

### Mathematical Principles of Core Algorithms

**CTC Frame-Level Phoneme Posterior** (Eq. 3): A linear projection followed by softmax is applied on the audio embedding $E_a$:

$$P(l_t \mid E_a) = \mathrm{Softmax}(W_c E_a + b_c) \in \mathbb{R}^{T \times (|\mathcal{V}|+1)}$$

where $\mathcal{V}$ is the phoneme vocabulary, and the extra 1 dimension corresponds to the CTC blank. Note that this CTC head is merely a linear layer; it does not decode or output transcriptions. Its sole duty is to assign a "which phoneme does this frame belong to" confidence score to each frame.

**Keyframe Criterion**: For each frame, take $l_t^* = \arg\max_l P(l_t \mid E_a)$. Frame $t$ is selected as a keyframe if and only if: (1) $l_t^* \neq \mathrm{blank}$; (2) $l_t^*$ has not been selected by any earlier frame (distinct-token constraint, keeping only the first occurrence of each unique predicted phoneme). After deduplication, the keyframe sequence length $T_p$ equals the number of distinct predicted phonemes, naturally aligning with the sequence form of the registration phoneme embedding $E_p^s$.

**Context Window Pooling** (Eq. 4): For each keyframe position $t_k$, take a symmetric window of $2w+1$ (experimentally $w=2$, i.e., 5 frames) and average:

$$\hat{e}_{t_k} = \frac{1}{2w+1} \sum_{j=t_k-w}^{t_k+w} e_j$$

**Cosine Similarity Matrix** (Eq. 5): Calculate normalized inner products between the query keyframe sequence and each registration modality feature:

$$M_m^a = \frac{(\hat{E}_a^q)^\top E_m^s}{\|\hat{E}_a^q\|_F \cdot \|E_m^s\|_F}, \quad m \in \{a, p, t\}$$

**Where-What Cross-Attention** (Eq. 6):

$$F_m^k = \mathrm{CrossAttn}\big(Q = M_m^a,\ K = V = F_m^c\big)$$

**Two Steps of QbyOmni** (Eq. 1, 2):

$$E_m^s = \mathrm{SelfAttn}(E_a^q \oplus E_m^s), \quad m \in \{a, p, t\}; \qquad F_m^c = \mathrm{FC}(\mathrm{GRU}(E_m^s))$$

(Note: The left-side $E_m^s$ refers to the representation after self-attention enhancement, following the paper's notation.)

**Training Objectives** (Eq. 7-10): Let $\mathrm{BCE}(\ell, y) = -[y \log \sigma(\ell) + (1-y)\log(1-\sigma(\ell))]$,

- Sentence-level loss: $L_u = \mathrm{BCE}(\ell_u, y_u)$, where $\ell_u$ is obtained by linear projection after summing $F_a^k + F_p^k + F_t^k$, and $y_u$ is the sentence-level ground truth label;
- Sequence-level frame supervision (phoneme-level and word-level): $L_m^s = \frac{1}{N_m} \sum_t \mathrm{mask}_{m,t} \cdot \mathrm{BCE}(\ell_{m,t}, y_{m,t})$, $m \in \{p, t\}$. The frame-level logit $\ell_{m,t}$ is extracted from the enhanced representation $E_m^s$ via indexing and linear projection, and the mask ignores padding;
- CTC loss: $L_c = \mathrm{CTC}(z^q, p^q) + \mathrm{CTC}(z^s, p^s)$, supervising both query and registration audio simultaneously. $z^{q/s}$ are linear projections of audio features, and $p^{q/s}$ are the ground truth phoneme sequences obtained from G2P;
- Total loss: $L_{total} = L_u + L_p^s + L_t^s + \lambda L_c$, with experimental $\lambda = 0.2$.

A key point: During inference, only registration text and query audio are needed. The CTC module acts solely as a keyframe selector and does not require the query audio's transcription.

The hierarchical nature of this composite loss has clear responsibility division: the sentence-level loss $L_u$ directly supervises the final decision; the two sequence-level losses $L_m^s$ provide direct per-frame gradient signals to each modality branch at phoneme and word granularities—this echoes the goal of modality dropout. Since a modality might be masked entirely at any time, each branch must be independently trainable and usable. The CTC loss $L_c$ applies supervision to both query and registration audio, ensuring the posterior output by the selector is sufficiently credible on both audio streams (peaks align accurately with phonemes). It hangs at the end of the total loss with a small weight $\lambda=0.2$, positioning it as an auxiliary task rather than the dominant objective—if the weight were too large, the backbone features would bias towards phoneme recognition itself, harming the matching decision. The paper does not report a scan of $\lambda$ values; this weight trade-off currently has only single-point evidence.

**Implementation Details**: Audio encoding uses frozen XLS-R (0.3B); G2P outputs 64-dimensional phoneme embeddings; text uses multilingual DistilBERT; projection to 128-dimensional shared space; QbyOmni is a 2-layer Transformer encoder (feedforward dim 512); GRU hidden dim 64; trainable parameters approx. 2.0M (excluding frozen encoders); single NVIDIA 4080 Super, batch 512, Adam (lr 0.001) trained for 50 epochs.

### Key Technical Innovation 1: CTC-Guided Keyframe Selection (Zero-Cost Frame Importance Metric)

The deep logic of the design motivation is "turning waste into treasure": models trained with CTC exhibit famous peaky behavior [26]—posterior probability mass is highly concentrated on a few frames aligned to phonemes, while the rest are assigned to blank. In alignment or robustness tasks, this is a pathological behavior that needs correction. This paper reverses this, using it as a free frame importance scorer: those high-confidence peak frames are precisely the positions carrying the strongest discriminative speech information and are natural anchors for cross-modal alignment. Compared to PLCL maintaining an external phoneme memory bank or MM-KWS using uniform frame-level attention, the posterior here comes only from a linear layer (Eq. 3). The selector itself has zero additional module cost, which the paper calls a "zero-cost frame importance metric."

The two selection rules each have their "why":

- **Non-blank criterion**: Blank frames are explicitly marked by CTC as "no phoneme content" frames. Excluding them directly reuses the CTC-learned knowledge of silence segments and transitions, avoiding the need to relearn "which frames are important";
- **Distinct-token deduplication**: Peaky behavior produces multiple peaks for the same phoneme consecutively. Without deduplication, keyframes would over-represent repeated phonemes, wasting matching slots. After deduplication, the sequence length equals the number of distinct phonemes, naturally aligning with the registration phoneme sequence, and also compresses the matching sequence length (for 1-4 word phrases, the number of keyframes is far less than the total frame count).

Window averaging (Eq. 4) addresses the fragility of peaks: single-frame features are sensitive to pronunciation rate differences and slight misalignments. Averaging ±2 frames preserves the discriminative information at the phoneme center while bringing in local acoustic context, remaining much more compact than full-sentence representations.

### Key Technical Innovation 2: Where-What Fusion with Similarity Matrix as Query

KFC-KWS does not feed keyframe features directly into cross-attention. Instead, it first calculates the cosine similarity matrix $M_m^a$ between query keyframes and registration features, then uses $M_m^a$ as Query and the full-sentence features $F_m^c$ as Key and Value (Eq. 6). The paper explicitly states this is a deliberate design: the similarity matrix encodes "where keyframes match across modalities" (where), while the full-sentence vector encodes "what global patterns exist in the entire segment" (what). Cross-attention between these two complementary views yields representations that are both locally precise (from similarity matching) and globally contextualized (from full-sentence features). In plain terms: keyframes tell the model "look here," and full-sentence features tell the model "what is the context around here." The fusion ensures the model doesn't look at the wrong position nor focus only on the local part while forgetting the whole.

### Key Technical Innovation 3: Modality-Level Random Masking (Modality Dropout)

During training, for each sample, the registration-side audio, phoneme, and text modality embeddings are entirely zeroed out with probability $p=0.5$. The paper specifically distinguishes it from SpecAugment [29] and standard dropout [30]: SpecAugment masks time-frequency regions within a single modality, standard dropout randomly deactivates neurons at the granularity level, while modality dropout acts at the modality granularity—forcing the model to still make predictions when an entire modality stream is absent. This encourages each modality branch to develop independent, informative representations rather than resting on a dominant modality. The benefit is Bal. AUC +0.67%, Bal. EER −0.83% (Δ column in Table 2); compared to the enhancement gains of CED† +1.70% and MM-KWS† +1.10%, this indicates that while the keyframe architecture itself is relatively robust, regularization still yields considerable returns.

### Technical Differences from Existing Methods

- **Vs. CLAD / MM-KWS (Uniform frame full-sequence matching)**: KFC explicitly selects phoneme-salient frames using CTC posteriors, concentrating discriminative resources on difference positions.
- **Vs. PLCL (External phoneme memory bank, 40.0M parameters)**: KFC's phoneme alignment signal comes from the model's internal CTC posterior, with no external components. Trainable parameters are 2.0M (Table 1), approximately 1/20th of PLCL's.
- **Vs. AdaKWS (Adaptive fusion, no explicit phoneme alignment)**: KFC's alignment has structural priors from CTC peaks, not relying purely on data-learned fusion weights.
- **Vs. Pure full-sequence methods**: KFC retains the QbyOmni branch to cover global context, winning on balanced metrics rather than attacking hard examples alone.

## Experimental Results

### Datasets Used and Their Scale

Experiments are conducted on the LibriPhrase benchmark, derived from LibriSpeech, with samples being short phrases of 1-4 words. The training set is taken from the train-clean-100 and train-clean-360 subsets, and the evaluation set from the train-other-500 subset; based on the phoneme similarity of positive-negative sample pairs, it is divided into LibriPhrase-Easy (LPE) and LibriPhrase-Hard (LPH). LPH contains many phonetically confusable word pairs and is the most discriminative subset. The paper does not report specific sample counts or duration distributions for the training and evaluation sets.

### Definition and Rationale for Evaluation Metrics

The paper reports AUC (Area Under the ROC Curve, measuring threshold-independent ranking quality) and EER (Equal Error Rate, the error rate when false rejection rate equals false acceptance rate, reflecting error at the operating point). The core design focuses on "Bal." (Balanced) metrics—taking the arithmetic mean of LPH and LPE results. The rationale is that in practical deployment, easy and hard words coexist; looking at a single subset would mislead model selection. This design also frames the evaluation philosophy of this method: not pursuing perfect scores on one side, but pursuing balanced leadership on both sides—a system that wins only on hard examples but lags significantly on easy ones (or vice versa) cannot be considered a winner in deployment.

### Detailed Comparison with Baseline and SOTA Methods

**Without Enhancement (Table 1)**: KFC-KWS achieves the highest Bal. AUC of 98.06% with 2.0M trainable parameters, surpassing the second-place HyperSpotter-c(4)'s 97.98%, while having only 1/2.75th the parameters (2.0M vs 5.5M). On LPH, it achieves the best AUC of 96.54% and best EER of 9.13%: leading the strong multi-modal baseline PLCL (40.0M) by +0.98% AUC and −0.83% EER, and leading HyperSpotter-c by +0.47% AUC—proving that CTC-guided keyframe selection effectively locates the differing phoneme positions of confusable words. Bal. EER of 5.68% is competitively close to DS-KWS-M1 (5.27%) and PLCL (5.59%). The paper acknowledges the gap mainly comes from LPE (2.22% vs DS-KWS-M1's 0.52%): full-sequence models dominate on easy samples, while this method prioritizes discriminative power for hard samples. For reference, early methods show significant gaps: EMKWS has an LPH AUC of only 73.58% and Bal. EER as high as 20.66%; CED is 89.20% / 9.60%.

**Horizontal Positioning of Parameter Efficiency (Table 1)**: In the lightweight group, SLiCK (0.6M) and iPhonMatchNet (0.7M) have the fewest parameters, but their LPH AUCs are only 94.90% and 88.23%, respectively, suffering significantly on hard examples. In the performance group, HyperSpotter-c(4) (5.5M), DS-KWS-M1 (4.1M), MM-KWS (3.9M), and CED (3.8M) have parameters more than double that of KFC, while PLCL (40.0M) is an order of magnitude larger. KFC-KWS stands at a favorable position between the two groups with 2.0M parameters: methods with fewer parameters perform worse on hard examples, and methods with hard-example performance close to KFC have at least double the parameters. "Replacing external phoneme memory banks with CTC peaks" is the source of this parameter-performance balance point.

**With Enhancement (Table 2)**: KFC-KWS with modality dropout achieves the best Bal. AUC of 98.73% across all scenarios. On LPH, it achieves 97.65% AUC / 7.75% EER, leading the strongest enhanced baseline PLCL† by +1.06% / −0.72%. Bal. EER of 4.85% is close to PLCL†'s 4.52%. Comparing enhancement gains horizontally (Bal. AUC increase): CED† +1.70%, MM-KWS† +1.10%, KFC-KWS† +0.67%, PLCL† +0.52%—methods relying on global context like CED and MM-KWS benefit more from data diversity, while KFC-KWS gains +0.67% with just a lightweight regularization.

**Trade-off Analysis (Section 4.1.2)**: The paper explicitly positions this method as sacrificing a small amount of easy sample performance (LPE EER approx. 2%) to gain significant benefits on hard samples (LPH EER approx. 7-9%). Mechanism explanation: LPE samples are acoustically separable; full-sequence methods can achieve EER below 1% based on redundant cues, while sparse keyframe selection naturally discards some redundant cues beneficial to easy samples. The lead in balanced metrics proves that hard-example gains far outweigh easy-example costs.

### Findings from Ablation Experiments

Table 3 performs sequential removal of the three registration encoders (other settings same as enhanced KFC-KWS†):

- **Removing the Phoneme Encoder causes the greatest harm**: LPH AUC drops from 97.65% to 91.90% (−5.75 percentage points), LPH AUC also drops from 99.81% to 97.52%, LPE EER worsens from 1.94% to 8.88%, and Bal. EER worsens from 4.85% to 13.60%. Conclusion: Phoneme-level information is a rigid requirement for the CTC keyframe strategy—without registration phoneme features, phoneme-aligned matching loses its alignment target, and the keyframe branch becomes ineffective.
- **Removing the Text Encoder shows an anomalous pattern**: LPH AUC degrades mildly (97.33%), but LPE AUC actually rises to 99.95%, and LPE EER drops to 0.77%. However, LPH EER surges from 7.75% to 18.90%. The paper's explanation: Text semantic features mainly provide supplementary word-level clues for hard example discrimination; removing them leaves easy examples with fewer interference sources. Notably, AUC remains almost unchanged while EER surges, indicating that after removing semantic anchoring, the tail of the score distribution deteriorates, making the operating point sensitive—the paper does not analyze this deeply.
- **Removing the Audio Encoder shows balanced degradation**: LPH 96.78%, LPE 99.07%, both sides dropping by less than 1 percentage point. LPH EER 9.18%, LPE EER 4.86% worsen synchronously, indicating that audio features provide consistent foundational support for both matching scenarios.

The three ablations together outline a clear picture of modality division of labor: the phoneme modality is the alignment anchor for the keyframe strategy; without it, hard-example discrimination collapses directly. The text modality provides word-level semantic clues, mainly suppressing false triggers on hard examples and stabilizing the tail of the decision score distribution. The audio modality is the foundational feature consumed by both branches; without it, both easy and hard sides drop slightly. This division of labor also explains the architecture design—why modality dropout randomly removes entire modalities during training: each of the three modalities has irreplaceable responsibilities, and any one might be absent in actual registration scenarios (user provides only text, or only audio). The model must remain usable under such incomplete inputs.

Ablation dimensions not reported by the paper: the keyframe selection itself (comparing random frame selection or uniform sampling), the role of the distinct-token deduplication constraint, the context window $w$, the loss weight $\lambda$, and the modality dropout probability $p$—direct comparative evidence for core mechanisms is absent in these dimensions.

## Main Contributions

1. **CTC-Guided Keyframe Selection Strategy**: Directly uses CTC's peaky posterior as a zero-cost frame importance metric to extract phoneme-aligned keyframes, achieving fine-grained matching across audio, phoneme, and text modalities, without any additional modules or external resources—this is a creative reuse of the "pathological characteristic" of CTC peaky behavior.
2. **Dual-Branch Fusion of Keyframe-Level and Sentence-Level Representations**: Where-what cross-attention using the similarity matrix as Query and full-sentence features as Key/Value combines local discriminative precision with global context, with the two branches covering hard and easy examples respectively.
3. **Empirical Results**: Best balanced performance on LibriPhrase (Bal. AUC 98.73% after enhancement, Table 2), significant lead on the hard subset LPH (97.65% AUC / 7.75% EER), requiring only 2.0M trainable parameters (approx. 1/20 of PLCL, 1/2.75 of HyperSpotter-c), without complex data augmentation or external resources.

## Limitations and Future Work

### Technical Limitations of the Method

1. **Dependence on CTC Peaky Behavior is a Double-Edged Sword**: The reliability of keyframe selection is entirely built on the premise that the posterior distribution is sufficiently "peaky." Under far-field, additive noise, or heavy accent conditions, CTC posteriors often flatten globally or shift peaks. In such cases, the selector may pick wrong frames or fail to pick any, causing the entire QbyKeyframe branch to fail. The paper itself lists noise robustness benchmark evaluation as future work, implying this premise has not yet been tested.
2. **Cost on Easy Examples is Structural**: In the non-enhanced setting, LPE EER is 2.22% vs DS-KWS-M1's 0.52% (Table 1). In actual product scenarios dominated by easy words, this cost may not be worth it. The paper defends with balanced metrics, but balance is an arithmetic mean; weights should depend on the prior difficulty of the target scenario.
3. **Distinct-Token Deduplication Constraint Does Not Discuss Repeated Phonemes**: When keywords contain repeated phonemes (the same phoneme appears twice), only the peak of the first occurrence is retained, and the discriminative information carried by the second occurrence (different surrounding phoneme context) is discarded. How the "natural alignment" between the keyframe sequence and the registration phoneme sequence is maintained under repeated phonemes is not explained by the paper.
4. **Hyperparameters Lack Sensitivity Analysis**: Window $w=2$, CTC loss weight $\lambda=0.2$, and modality dropout probability $p=0.5$ are all single-point values, without ablation or scan support. It is unknown whether a sweet spot exists, or if the range is too narrow or too wide.
5. **No Direct Evidence of Selector's Own Reliability**: The CTC head is a linear layer on frozen XLS-R features. The paper does not report the frame-level phoneme discrimination accuracy (keyframe selection rate) of this module—the upper bound of the entire method depends on this unquantified module.
6. **Parameter Count Underestimates Real Footprint**: The 2.0M trainable parameters do not include the frozen XLS-R (0.3B). During inference, every query audio still passes through the 0.3B parameter encoder. The paper does not report actual edge-side latency, throughput, or memory usage. The "lightweight" narrative only holds under the trainable parameter metric.

### Deficiencies in Experimental Design

1. **Single Benchmark, Single Language**: Only LibriPhrase (English, read style, derived from LibriSpeech). The architecture selected multi-modal components (XLS-R, multilingual DistilBERT), but no multi-lingual or Chinese evaluation was performed. Component selection does not match the evaluation scope.
2. **No Robustness Dimensions**: No experiments on noise, reverberation, far-field, accents, or registration-query speaker mismatch. The recognized weakness of the text registration route is precisely accents and noise (as pointed out by the paper itself in the introduction), yet the experiments do not cover this.
3. **Core Mechanism Lacks Direct Comparison**: Ablation (Table 3) only covers the three encoders. The core claim that "CTC frame selection is superior to uniform/random frames" lacks comparative experimental support; readers can only infer indirectly from end-to-end results.
4. **No Statistical Significance Reported**: Results are from a single run, with no variance across multiple random seeds. The lead of Bal. AUC over the second-place HyperSpotter-c is only 0.08 percentage points (98.06% vs 97.98%, Table 1), which is within the potential noise magnitude.
5. **Task Form and Deployment Narrative Have Distance**: LibriPhrase is phrase-level binary classification verification (AUC/EER), not deployment metrics for streaming KWS (false alarms per hour, streaming latency, wake-up response time). The argument that "actual deployment contains both easy and hard words" remains at the data distribution level rather than the system level.
6. **Anomalous Results Not Deeply Investigated**: The w/o text row shows LPH EER 18.90% coexisting with AUC 97.33% (Table 3). The paper brushes this off as "mild degradation" without analyzing the mechanism of EER tail out-of-control, nor discussing its impact on threshold deployment.
7. **Ambiguity in Inference Input Scope**: Section 2.3 states that inference only requires registration text and query audio (registration audio is not needed), but it is not explicitly stated whether the main results in Table 1/2 include registration audio input. If the main results indeed use only text registration, the actual inference configuration corresponds to the w/o audio row in Table 3 (LPH AUC 96.78%, Bal. EER 7.02%), creating a scope gap with the main table numbers (97.65% / 4.85%) that needs clarification. This ambiguity directly affects the fairness of comparison with other baselines.

### Possible Directions for Future Improvement

- **Paper's Self-Reported Directions**: Adaptive keyframe selection strategy (replacing fixed "non-blank + deduplication" rules, dynamically deciding which frames to keep based on confidence or context); evaluating real deployment performance on noise robustness benchmarks.
- **Author's Extensions**: Change hard selection to posterior confidence weighting to mitigate fragility under peak drift; design position-aware selection constraints for repeated phonemes; complete ablation and CTC head phoneme accuracy reports for $w$, $\lambda$, and frame selection strategies; distill the frozen XLS-R into a smaller encoder to convert parameter advantages into real edge-side footprints; perform score normalization or threshold calibration to solve the instability of EER tails after removing semantic anchoring; verify in multi-lingual and Chinese custom wake-word scenarios (XLS-R's multi-lingual prior is a ready-made foundation); extend evaluation from phrase verification to streaming detection (false alarms/hour plus response latency) to align with real product metrics.
