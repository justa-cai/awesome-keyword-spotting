# ProKWS: Personalized Keyword Spotting via Collaborative Learning of Phonemes and Prosody

- **Authors/Affiliations**: Jianan Pan, Yuanming Zhang, Kejie Huang (College of Information Science and Electronic Engineering, Zhejiang University)
- **Date**: March 2026 (arXiv:2603.18024v1 [eess.AS], submitted March 5, 2026)
- **Link**: https://arxiv.org/abs/2603.18024
- **Keywords**: user-defined keyword spotting (UDKWS), prosody modeling, prosodic signature, contrastive learning, FiLM modulation, personalized KWS, multi-modal enrollment

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) serves as the human-machine interface for speech interaction devices. Traditional KWS relies on a set of predefined wake-up words (e.g., "OK Google"), which lacks flexibility and personalization. This limitation has driven the development of User-Defined Keyword Spotting (UDKWS), allowing users to customize trigger words without re-collecting large amounts of data or retraining the model. Mainstream research in UDKWS focuses on **resolving speech ambiguity**, i.e., distinguishing the target keyword from phonetically similar distractors. This has led to two major paradigms: Query-by-Text (QbyT, which uses fine-grained alignment between text and acoustic representations for matching) and Query-by-Example (QbyE, which uses registered audio templates for matching).

However, the paper points out a blind spot ignored by the entire field: **existing UDKWS systems are both speaker-agnostic and intent-agnostic**—they only model "what is said" (what is said), completely ignoring "how it is spoken" (how it is spoken). Prosody (including rhythm, stress, and intonation) carries critical information: it can distinguish between imperative and interrogative sentences (e.g., "Turn on light." and "Turn on light?" have identical phoneme sequences but different intonation contours) and reflects the speaker's vocal style and emotional state. The consequence of not modeling prosody is that the system is prone to misjudging intent (triggering on questions as commands), is not robust to natural variations such as accents and emotional timbre, and cannot achieve "personalization." Although prosodic cues have been widely studied in speaker verification and emotion recognition, their application in personalized, intent-aware KWS remains largely blank. This is the meaning of "Personalized" in the title: allowing the wake-up behavior to reflect both **phonetic content accuracy** and **vocal style consistency**.

### Specific Shortcomings of Existing Methods

- **Only phoneme-level matching, discarding suprasegmental information**: State-of-the-art methods like PLCL (Phoneme-Level Contrastive Learning) demonstrate that enforcing feature separation at the phoneme granularity achieves strong robustness against confusable phrases; MM-KWS further constructs more reliable keyword representations using multi-modal registration of text and speech templates. However, these methods share the same implicit assumption—that the identity of a keyword is determined entirely by its phoneme sequence. For hard negatives of the same sentence but different intonation (imperative vs. interrogative), phoneme-level matching is theoretically incapable of distinction.
- **Registered audio is only used to align content, not to characterize style**: Although QbyE and multi-modal methods collect user registration audio, they only extract "which word is being said" from it. The user's unique intonation and rhythm patterns are treated as speaker variation (nuisance) to be erased rather than as a signal.
- **Lack of evaluation benchmarks for the prosodic dimension**: Even if methods attempt to model prosody, existing public benchmarks (such as LibriPhrase) lack annotations for accents and intent directions, making it impossible to quantify the property of "prosodic sensitivity."
- **High cost of the general large model route**: As shown in Table 2, Whisper-Large (1550M parameters) still has an EER of 19.57% on LibriPhrase-hard. It is neither accurate enough nor small enough. Edge-side KWS requires a dedicated architecture with a small parameter count that balances content and style.

### Key Challenges Addressed by This Paper

1. **Decoupling and Complementarity of Dual Information Sources**: Phoneme content (dominated by spectral envelope) and prosodic style (dominated by suprasegmental contours such as F0, energy, and phonation quality) are entangled in the same frame sequence at the signal level. How to extract "speaker-independent phoneme representations" and "speaker-dependent prosodic signatures" via two separate streams without crossing boundaries.
2. **Prosody Learning under Minimal Supervision**: Prosody lacks natural labels (unlike phonemes, which can use text + forced alignment for weak supervision). How to learn discriminative prosodic representations relying solely on the similarity constraints of "registration-query positive pairs" (referred to as "minimal supervision" in the paper).
3. **How Prosody Information Truly Participates in Decision-Making Rather Than Decorative Concatenation**: Simply concatenating prosody vectors to features often leads to them being ignored during training. It is necessary to let prosody deeply intervene in the formation of phoneme representations.
4. **Absence of Evaluation**: How to construct a controllable test set that separates the two prosodic variables of "accent" and "intent" while strictly controlling phoneme content consistency.
5. **Parameter Budget**: Reducing parameters to 2.9M while surpassing the performance of PLCL/MM-KWS (3.9M), proving that adding a prosody stream does not come at the cost of stacking parameters.

## Methodology

### Overall Architecture Design and Design Motivation

ProKWS is a framework consisting of a dual-stream encoder and a collaborative fusion module, processing multi-modal enrollment (enrollment text + enrollment audio) and query audio:

- **Phoneme Stream**: 80-dimensional FBank features → shared audio encoder composed of convolutional downsampling layers + stacked Conformer blocks → frame-level acoustic representations → Phoneme Pooler aggregates frame-level features into phoneme-segment-level features based on the start and end timestamps of each phoneme segment provided by MFA forced alignment. The enrollment text is processed by a pre-trained G2P text encoder to obtain text embeddings of the same dimension.
- **Prosody Stream**: 3-dimensional frame-level prosodic features (F0, Aperiodicity AP, RMS energy) → two-layer bidirectional GRU → frame-level prosodic representations; the registration side aggregates these into a single 64-dimensional context vector via an attention-based Prosody Pooler, which serves as the "prosodic signature."
- **Collaborative Fusion Module**: The FiLM layer uses the prosodic signature to perform affine modulation on phoneme features → the modulated phoneme features undergo cross-attention with text embeddings → the query-side prosody vector calculates cosine similarity with the registration prosodic signature → the three pieces of evidence (multi-modal fusion output, prosodic signature, prosodic similarity) are concatenated and passed through a GRU + FC layer to output a sentence-level score $s \in [0,1]$.

The core of the design motivation is "division of labor + intervention": the phoneme stream is responsible for discriminability (inheriting PLCL's phoneme-level contrastive learning route to ensure robustness against confusable words), while the prosody stream is responsible for personalization (extracting a compact style vector from a small number of registration samples). The fusion is not a simple concatenation of features but allows prosody to exert influence **before** the phoneme representation is formed via FiLM (as stated in the paper: "To incorporate a user's speaking style into phoneme extraction"), with explicit similarity evidence serving as a fallback at the decision end. The registration and query audio share the same audio encoder to ensure a shared feature space—this is a prerequisite for contrastive learning to take effect.

### Mathematical Principles of Core Algorithms

Let the batch size be $B$. The phoneme stream inputs registration text $T_e$ and registration/query audio $(X_e, X_q)$, extracting FBank features $F \in \mathbb{R}^{B \times T \times D_{fbank}}$, where $D_{fbank} = 80$; the audio encoder outputs $Z_p \in \mathbb{R}^{B \times T \times D_p}$ (the paper does not report the specific value of $D_p$). The G2P text encoder outputs $E'_t \in \mathbb{R}^{B \times T' \times D_p}$, and the Phoneme Pooler obtains phoneme-segment features $\hat{Z}_{pq} \in \mathbb{R}^{B \times T' \times D_p}$ using MFA timestamps. The prosody stream inputs $P \in \mathbb{R}^{B \times T \times 3}$, and the two-layer BiGRU outputs $Z_{pro} \in \mathbb{R}^{B \times T \times 64}$; attention pooling on the registration side yields the prosodic signature $v_{pro} \in \mathbb{R}^{B \times 64}$.

**FiLM Prosody Modulation** (Equations 1-2): The prosodic signature is projected into per-channel scaling and translation vectors,

$$\gamma = \text{Linear}_\gamma(v_{pro}), \quad \beta = \text{Linear}_\beta(v_{pro})$$

$$\hat{Z}_{pro}^q = \gamma \odot \hat{Z}_{pq}^q + \beta$$

where $\odot$ denotes element-wise multiplication. The meaning of this affine modulation is that the user's prosodic signature determines the "gain + bias" for each channel of the phoneme features. The same phoneme is mapped to different positions in the feature space under different users' styles—therefore, queries with the same content but different styles can be distinguished.

**Cross-Attention Multi-Modal Fusion** (Equation 3): The modulated phoneme features serve as the query, and the text embeddings serve as both key and value:

$$\hat{Z}_{at}^q = \text{Cross-Attention}(\hat{Z}_{pro}^q, Z_t, Z_t)$$

That is, the "phoneme representation with style" actively retrieves the corresponding phonemes of the registration text, and the attention weights themselves achieve soft alignment.

**Prosody Matcher**: The query prosody features $Z_{pro}^q$ are globally average-pooled to obtain $v_{pro}^q$, which calculates cosine similarity $s_{pro}$ with the registration prosodic signature. Finally, $[\hat{v}_{at}, v_{pro}, s_{pro}]$ are concatenated and passed through a GRU + FC to obtain the decision score $s \in [0,1]$.

**Training Objective** (Equation 4) is a composite loss of four terms:

$$L_{total} = L_{utt} + L_{phon}^{at} + L_{phon}^{aa} + \lambda L_{pro}$$

Sentence-level BCE loss (Equation 5):

$$L_{utt} = -\frac{1}{B} \sum_{i=1}^{B} \left[ y_i \log s_i + (1 - y_i) \log(1 - s_i) \right]$$

Phoneme-Text Phoneme InfoNCE loss (Equation 6): Aligned audio phoneme segments $z_j$ and corresponding text embeddings $e_j$ are positive samples, compared against negative samples $e_k$:

$$L_{phon}^{at} = -\sum_j \log \frac{\exp(\text{sim}(z_j, e_j)/\tau)}{\sum_k \exp(\text{sim}(z_j, e_k)/\tau)}$$

where sim is cosine similarity, and $\tau$ is a temperature hyperparameter (value not reported in the paper). The Phoneme-Audio Phoneme InfoNCE loss $L_{phon}^{aa}$ is similar in form, comparing paired audio phoneme segments of registration and query. The prosodic similarity loss (Equation 7) applies only to positive pairs:

$$L_{pro} = \frac{1}{|B_{pos}|} \sum_{i \in B_{pos}} \left(1 - \text{sim}(v_{pro}, v_{pro}^{q(i)})\right)$$

Design logic: $L_{phon}^{at}$ uses text as the anchor (text does not depend on audio quality and provides stable phoneme supervision), $L_{phon}^{aa}$ ensures consistency of phoneme representations on both registration and query sides, $L_{pro}$ constrains the prosodic signature space in the weakest form (only positive pairs, no negative pairs, no labels), and $\lambda$ balances the weight of the prosody term (value not reported in the paper).

### Key Technical Innovation 1: Dual-Stream Encoder and MFA Forced-Aligned Phoneme Segment Pooling

The phoneme stream follows the conclusion of PLCL—that feature separation at the phoneme granularity is necessary to combat confusable words—but engineers it using a Phoneme Pooler: MFA provides the start and end timestamps for each phoneme segment, and frame-level Conformer outputs are aggregated into phoneme-segment features based on these timestamps. This ensures that InfoNCE contrast occurs between phoneme segments rather than between frames. Why segment-level is necessary: frame-level contrast would treat alignment errors and co-articulation variations as discriminative signals, whereas phoneme segments are phonetically stable units; moreover, only segment-level features can be one-to-one mapped to G2P text embeddings for cross-modal phoneme-text contrast. The prosody stream forms an intentional **division of information** with it: it uses only 3 suprasegmental features (F0, AP, RMS energy), does not touch the spectral envelope, and thus prevents the prosody stream from learning phoneme content from the input stage, avoiding functional overlap with the phoneme stream (the motivation for this division is implicitly expressed through architectural choices in the paper, without explicit word-for-word argumentation). Registration and query share the encoder, which is standard practice for contrastive learning—the features of the two branches must be in the same space for similarity to be meaningful.

### Key Technical Innovation 2: Prosodic Signature Mechanism

The prosodic signature is the core concept of the paper (Point 2 of the contribution statement): compressing the user's vocal style into a fixed 64-dimensional vector. The three design points each have their "why":

1. **3-Dimensional Input** (F0, AP, RMS energy): The F0 contour carries intonation (rising intonation for questions, falling for commands), the RMS energy contour carries stress and rhythm, and AP reflects phonation quality (breathy/voiced degree). These three dimensions constitute the classic minimal complete set of prosody in phonetics. Computationally, they are far lighter than spectral inputs, and they naturally ensure that the prosody stream contains no phoneme information.
2. **Two-Layer BiGRU Encoding of Contours**: The discriminative information of prosody lies in the dynamic shape of the contour (timing of pitch drops, spacing of stresses). The bidirectional GRU captures forward and backward dependencies, making it more suitable for whole-sentence intonation patterns than purely forward networks. The output dimension is compressed to 64, far smaller than $D_p$—the signature must be a "compact summary" rather than frame-by-frame details, otherwise cross-sentence matching would be impossible.
3. **Asymmetric Design: Attention Pooling on Registration Side vs. Global Average Pooling on Query Side**: Registration audio serves as a reference template, and attention pooling allows the model to select frames with the most significant style to aggregate into an anchor; the query side uses parameter-free average pooling to keep it simple. The paper does not explicitly argue for the motivation of this asymmetry; it is a design choice inferred from results.

Minimal supervision is the selling point of the signature mechanism: no "style labels" are needed, only the positive sample similarity loss in Equation 7—the registration audio and positive queries from the same user are pulled closer in the signature space. The discriminative structure for intent/accents emerges spontaneously during training (verified by t-SNE in Section 4.2).

### Key Technical Innovation 3: Collaborative Fusion Module (FiLM Modulation + Cross-Attention + Three-Evidence Decision)

The fusion module solves the problem of "how prosody truly intervenes in decision-making," progressing in three layers:

1. **Prosody Adaptation (FiLM)**—Prosody acts as conditional modulation for phoneme representations (Equations 1-2). Compared to concatenating $v_{pro}$ into features and letting the network learn how to use it, FiLM injects style into each channel via an explicit affine form. It has a short gradient path and saves parameters (two linear layers). Moreover, modulation occurs **before** cross-attention, meaning that the stylized phoneme representation will alter the attention alignment itself.
2. **Multi-Modal Cross-Attention** (Equation 3)—Inherits the text + audio multi-modal registration idea of MM-KWS, but the query side carries prosody modulation; text embeddings serve as key/value to ensure that keyword content supervision remains anchored to the registration text.
3. **Three-Evidence Decision Head**—$[\hat{v}_{at}$ (content + style fusion evidence) $, v_{pro}$ (registration style passed as-is) $, s_{pro}$ (explicit prosodic similarity scalar) $]$ are concatenated and passed through GRU+FC. Sending $s_{pro}$ separately into the decision head is a key detail: even if the contribution of FiLM is ablated, this scalar still provides a prosody shortcut independent of representation learning. Ablation experiments (Table 4) show that removing only the FiLM layer reduces EER to 12.29%, indicating that the three pathways indeed complement each other.

### Key Technical Innovation 4: Accent-KWS and Intent-KWS Prosody Evaluation Benchmarks

Existing benchmarks lack prosodic dimension annotations. The paper constructs two controllable test sets using CosyVoice2 TTS (Point 3 of the contribution statement): **Accent-KWS** takes 100 hard keywords from LibriPhrase, synthesized with 4 accents (American, British, Indian, Australian), totaling 2400 samples (3 speakers per accent, 2 sentences each); **Intent-KWS** transforms the same 100 keywords into 3 intents (command, question, neutral), totaling 900 samples (3 intents × 3 sentences each). The construction logic is "control variables": phoneme content is strictly consistent, allowing only the accent or intent variable to change, thereby isolating the model's sensitivity to prosody from its content discriminability for measurement.

### Technical Differences from Existing Methods

- **vs PLCL**: PLCL demonstrates the power of phoneme-level contrastive learning but has no concept of personalization; ProKWS retains its phoneme-granularity contrast ($L_{phon}^{at}$ is isomorphic to it) and adds a prosody stream, reducing EER on LPH from 8.47% to 7.52%, while reducing parameters from 3.9M to 2.9M.
- **vs MM-KWS**: Both use text + audio multi-modal registration. MM-KWS uses audio templates to build "more reliable keyword representations" for multi-language open-vocabulary generalization; ProKWS redefines registration audio as a "style fingerprint" for personalization and intent awareness. Table 3 directly uses MM-KWS as a baseline (BL), showing a huge gap in the prosodic dimension (Intent-KWS AUC 61.35% vs 86.42%).
- **vs General Modulation Methods (AdaKWS series)**: From the citations, AdaKWS uses adaptive instance normalization for open-vocabulary KWS, with modulation statistics derived from the query itself; ProKWS's FiLM modulation source is the **registration-side prosodic signature**—the modulation direction is reversed from "normalizing away individual differences" to "injecting individual differences." This is the fundamental divide between the generalization route and the personalization route.
- **vs QbyE series**: QbyE uses only audio registration, where content and style are mixed in the template; ProKWS separates text (content anchor) and prosodic signature (style anchor) for registration, avoiding the collapse of content discriminability when QbyE registration audio quality is poor.
- **vs Whisper-like General Models**: Does not rely on large-scale weak supervision; with 2.9M parameters, it achieves an EER of 7.52% on LPH, far better than Whisper-Large (1550M) at 19.57%.

## Experimental Results

### Datasets Used and Their Scales

- **LibriPhrase**: Constructed from LibriSpeech train-others-500 according to references [2][4], divided into easy (LPE) and hard (LPH) subsets. The construction of easy and hard negatives is shown in Table 1: when the English anchor is "friend," easy negatives are phonetically unrelated words like guard, comfort, superior, while hard negatives are rhyming near-homophones like frind, rend, trend; for the Chinese WenetPhrase anchor "ning2yuan4," hard negatives are ting2yuan4, xing2yuan4, qing2yuan4 (sharing the "ing2yuan4" prosodic segment), and easy negatives are sha1mo4, de2zhi1, gong1wu4. The specific sample scales of LibriPhrase/WenetPhrase are not reported in the paper (following the construction protocols of the cited references).
- **WenetPhrase**: Constructed according to [12] for the WPE/WPH Chinese subsets (specific scales not reported in the paper).
- **Accent-KWS**: 100 LibriPhrase hard keywords × 4 accents × 3 speakers × 2 sentences = 2400 samples, synthesized by CosyVoice2.
- **Intent-KWS**: Same 100 keywords × 3 intents × 3 sentences = 900 samples, synthesized by CosyVoice2.
- Training data and train/test splits are not reported in the paper; training hyperparameters only include: AdamW optimizer, weight decay 1e-3, initial learning rate 3e-4, linear warmup to 3e-4 for the first 5 epochs followed by cosine annealing; batch size, total epochs, $\lambda$, $\tau$, $D_p$, and number of Conformer blocks are not reported.

### Definition and Rationale for Evaluation Metrics

**EER** (Equal Error Rate, the error rate when False Alarm Rate FAR equals False Rejection Rate FRR) and **AUC** (Area Under the ROC Curve) are used. Rationale: The registration-query matching in UDKWS is essentially a binary verification/retrieval task with no fixed keyword prior. EER eliminates dependence on a specific operating threshold, while AUC summarizes ranking quality across all thresholds. Both are standard metrics in this series of works (MM-KWS, PLCL, etc.), ensuring horizontal comparability. The cost is that these metrics do not directly reflect the experience at a fixed false alarm rate (e.g., 0.1 FA/h) during deployment—the paper does not report any metrics for fixed-threshold operating points.

### Detailed Comparison with Baseline and SOTA Methods

**LibriPhrase (Table 2)**:

| Method | Params | AUC LPH | AUC LPE | EER LPH | EER LPE |
|---|---|---|---|---|---|
| Whisper-Tiny | 39M | 73.37 | 89.19 | 33.04 | 17.31 |
| Whisper-Small | 224M | 82.90 | 95.92 | 21.45 | 8.14 |
| Whisper-Large | 1550M | 85.80 | 97.54 | 19.57 | 5.33 |
| CMCD | 0.7M | 73.58 | 96.70 | 32.90 | 8.42 |
| CLAD | 2.2M | 76.15 | 97.03 | 30.30 | 8.65 |
| EMKWS | 3.7M | 84.21 | 97.83 | 23.36 | 7.36 |
| PhonMatchNet | 0.7M | 88.52 | 99.29 | 18.82 | 2.80 |
| CED | 3.6M | 92.70 | 99.84 | 14.40 | 1.70 |
| AdaKWS-Tiny | 15M | 93.75 | 99.80 | 13.47 | 1.61 |
| MM-KWS | 3.9M | 96.25 | 99.95 | 9.30 | 0.68 |
| PLCL | 3.9M | 96.59 | 99.97 | 8.47 | 0.57 |
| **ProKWS** | **2.9M** | **96.92** | 99.96 | **7.52** | 0.63 |

Three observations (differences calculated): (1) ProKWS achieves the best AUC (96.92) and best EER (7.52) on the hard subset with 2.9M parameters, a relative reduction of 11.2% in EER compared to the previous best PLCL, while using 1.0M fewer parameters (-25.6%); (2) On the easy subset, it trades wins with PLCL (AUC 99.96 vs 99.97, EER 0.63 vs 0.57), indicating that the benefit of the prosody stream approaches zero in scenarios where content is already separable, and improvements are concentrated in hard scenarios; (3) It has a parameter advantage of over 500 times relative to Whisper-Large and significantly outperforms it, validating the dedicated small-architecture route.

**Prosodic Dimension (Table 3, baseline is MM-KWS)**:

| | AUC WPH | AUC WPE | AUC AC | AUC IT | EER WPH | EER WPE | EER AC | EER IT |
|---|---|---|---|---|---|---|---|---|
| BL (MM-KWS) | 85.84 | 99.15 | 52.39 | 61.35 | 22.06 | 4.26 | 47.23 | 37.67 |
| ProKWS | 84.82 | 99.81 | 71.45 | 86.42 | 23.33 | 1.84 | 27.92 | 18.10 |

The improvement in the prosodic dimension is overwhelming (differences calculated): Accent-KWS AUC +19.06, EER reduced from 47.23% to 27.92%; Intent-KWS AUC +25.07, EER reduced from 37.67% to 18.10%; WPE (Chinese easy) EER reduced from 4.26% to 1.84%. However, it must be honestly pointed out that **slight degradation occurs on WPH**: AUC 85.84 → 84.82 (-1.02), EER 22.06 → 23.33 (+1.27)—Chinese hard negatives inherently share rhyme structures (e.g., ting2yuan4 vs ning2yuan4), and the prosody path introduces slight interference for such adversarial samples where "content and style are both similar." The paper body does not discuss this degradation. Furthermore, the baseline's AUC on Accent-KWS is only 52.39% (close to random), indicating that traditional phoneme matching is indeed nearly ineffective against accent variations.

### Findings from Ablation Experiments

**Component Ablation (Table 4, LibriPhrase)**:

| Method | AUC LPH | AUC LPE | EER LPH | EER LPE |
|---|---|---|---|---|
| ProKWS Full | 96.92 | 99.96 | 7.52 | 0.63 |
| w/o Prosody Adaptation Module | 94.22 | 99.67 | 12.29 | 1.86 |
| w/o auxiliary $L_{pro}$ | 92.76 | 99.34 | 13.47 | 3.24 |
| w/o Prosody Stream | 88.71 | 98.99 | 15.34 | 4.67 |

The ablation ranking reveals the contribution structure of components: Removing the entire prosody stream causes the greatest harm (LPH EER 7.52% → 15.34%, doubling; AUC drops by 8.21 points)—prosody cues are critical for capturing intent and speaker variations; Removing the auxiliary loss $L_{pro}$ is the next worst (EER 13.47%), proving that without unsupervised constraints, prosodic representations degrade, and the "minimal supervision" of $L_{pro}$ is not optional; Removing the FiLM modulation layer causes relatively less harm (EER 12.29%), indicating that the $s_{pro}$ explicit evidence path bears a significant portion of prosodic discrimination, and the gain from FiLM is additive. A notable detail: w/o $L_{pro}$ performs worse than w/o FiLM (13.47 vs 12.29), meaning "modulation exists but representation is poorly learned" is worse than "no modulation but representation is well learned"—prosodic representation quality is a prerequisite for modulation to take effect.

**t-SNE Visualization (Fig. 2)**: In Intent-KWS, prosodic signatures for the same keyword spontaneously form three clearly separated clusters corresponding to imperative, interrogative, and neutral intents; whereas signature discrimination for the accent dimension is weaker, with significant overlap. The paper attributes this to two points: (1) Existing prosodic features may be insufficient to capture the stable rhythm and intonation patterns required to characterize accents; (2) The CosyVoice synthesizer has limited ability to consistently and stably generate distinctive accents (a flaw in the evaluation data itself).

**Intent Interpolation Experiment (Fig. 3, Equation 8)**: Register imperative "Turn on light," take the prosody vectors $v_{pos}$ (positive query, imperative) and $v_{neg}$ (hard negative query, interrogative), linearly interpolate $v_{interp}(\alpha) = (1-\alpha)v_{pos} + \alpha v_{neg}$, $\alpha \in [0,1]$, and feed them one by one into the **frozen** fusion module to calculate scores $s(\alpha)$. The results show: ProKWS scores remain high when prosody is close to the registration sample and monotonically decrease as interpolation slides toward mismatched intents; the baseline without the prosody stream maintains high scores throughout, being insensitive to prosodic changes. Freezing the fusion module is the rigor of this analysis—it eliminates the confounding variable of retraining, directly attributing "score changes to prosody vectors," and confirms from a causal chain that the prosody path encodes fine-grained intent.

## Main Contributions

1. **First dual-stream framework to explicitly introduce prosody into UDKWS**: Proposes ProKWS, where the phoneme stream (contrastive learning, speaker-independent content discrimination) and prosody stream (personalized style signature) are encoded in parallel and collaboratively fused, advancing KWS from "content-only" to a dual-condition of "content + style." It is a pioneering work for intent-aware, user-adaptive KWS.
2. **Prosodic Signature Mechanism**: Compresses the user's vocal style into a 64-dimensional vector using 3-dimensional classic prosodic features + two-layer BiGRU + attention pooling. It emerges with intent/accent discriminative structures relying only on positive sample similarity loss (minimal supervision) and participates in decision-making via two pathways: FiLM modulation and explicit similarity.
3. **Two Prosody Evaluation Benchmarks**: Accent-KWS (2400 samples, 4 accents) and Intent-KWS (900 samples, 3 intents), filling the evaluation gap for prosodic sensitivity by controlling variables via TTS.
4. **Better Accuracy-Parameter Trade-off**: With 2.9M parameters, it achieves EER 7.52% and AUC 96.92% on LibriPhrase-hard, outperforming 3.9M PLCL/MM-KWS and 15M AdaKWS-Tiny, and far surpassing 1550M Whisper-Large.
5. **Transferable Engineering Judgments**: Ablation proves that "representation quality precedes fusion methods" (the absence of $L_{pro}$ is more fatal than the absence of FiLM); interpolation analysis provides a reusable paradigm for verifying the causal impact of conditional vectors (freezing the backbone + linear interpolation to sweep scores).

## Limitations and Future Work

### Technical Limitations of the Method

- **Accent discrimination is not truly solved**: The paper admits that accent clusters overlap significantly in t-SNE, attributing this to 3-dimensional prosodic features being insufficient to characterize stable rhythm/intonation patterns of accents, and to unstable TTS accent synthesis. EER on Accent-KWS is still as high as 27.92%, far from usable.
- **Slight degradation on Chinese hard negatives**: On WPH, AUC -1.02, EER +1.27, suggesting that the prosody path introduces interference for Chinese near-homophones where "rhyme structures are inherently similar." The paper does not analyze the mechanism nor provide mitigation strategies.
- **Tension between personalization and false rejection is not quantified**: Since prosodic similarity directly enters the decision score, style mismatch will lower the score—when a user is emotionally excited, has a voice change due to illness, or when far-field pickup causes F0/energy distortion, legitimate wake-ups may be rejected. The paper only shows that hard negatives (interrogatives) are correctly down-scored, but does not report the cost of false rejection under natural prosodic fluctuations. This is a question that the "personalization" route must answer but has not.
- **Prosody front-end dependency is not specified**: The extraction tool and computational cost for F0/AP/RMS are not reported (AP typically requires a WORLD-like analyzer). There is doubt about the tolerance of edge-side KWS for additional front-ends; latency, RTF, and FLOPS are all unreported (only parameter count is given).
- **Asymmetry of pooling on the registration side and sensitivity to the number of registration samples**: The paper does not report how many audio samples are needed for registration, nor the relationship between the number of registration samples and performance (only stating "a few enrollment samples"); the asymmetric design of attention pooling (registration) and average pooling (query) lacks ablation support.
- **Lack of speaker-dimension verification**: The title is "Personalized," but the experiments do not include a speaker-discrimination evaluation where "others saying the same keyword should be rejected"—whether the prosodic signature truly encodes speaker identity (rather than just intent) is not directly tested.

### Shortcomings in Experimental Design

- **Prosody evaluation relies entirely on a single TTS**: Both Accent-KWS and Intent-KWS are synthesized by CosyVoice2. The prosodic dynamic range of synthesized speech differs from that of real humans (the paper itself points out that CosyVoice's accent generation is unstable), casting doubt on the extrapolation of conclusions to real accent/intent speech; moreover, Accent-KWS has only 3 speakers × 2 sentences per accent, which is a small scale.
- **Key hyperparameters and training details are not reported**: $\lambda$, $\tau$, batch size, total epochs, training data split, $D_p$, and Conformer layer count are missing, limiting reproducibility.
- **Asymmetric baseline coverage**: Table 3 only compares against one baseline, MM-KWS; strong baselines like PLCL and AdaKWS are not evaluated on the prosodic dimension; the control baseline in Fig. 3 is only described as "baseline model without prosody stream," without specifying the corresponding ablation configuration.
- **Single dimension of metrics**: Only EER/AUC are provided, lacking fixed false alarm rate operating points and latency/compute metrics, making it difficult to assess the value for edge-side deployment.

### Possible Directions for Future Improvement

1. **Strengthen accent representation**: Expand the prosodic feature set (e.g., phoneme duration distribution, stress position, prosodic boundary features) or introduce accent pre-training in the prosody stream to address the self-admitted weakness in accent discrimination; simultaneously, replace/supplement TTS benchmarks with real multi-accent recordings.
2. **Joint prosody-voiceprint signature**: Model speaker identity (voiceprint) and style (prosody) jointly, closing the personalization loop for "rejecting impersonators" and providing voiceprint-level protection for security scenarios.
3. **Intent-aware two-stage gating**: Decouple prosodic intent discrimination from the primary score into a secondary gate (judge intent after content matching passes), mitigating false rejections caused by WPH degradation and natural prosodic fluctuations, with costs and benefits adjustable via thresholds.
4. **Adaptive prosodic signature**: Online update of the prosodic signature (tracking slow changes in the user's voice) or adaptive pooling confidence based on the number of registration samples, enhancing robustness for long-term use.
5. **Edge-side deployment evaluation**: Report FLOPS/RTF/memory, explore lightweight prosody front-ends (e.g., end-to-end F0 prediction replacing WORLD analysis), and verify the feasibility of the 2.9M parameter model with a prosody path on MCU-level hardware.
6. **Prosody learning with negative sample contrast**: Currently, $L_{pro}$ only has positive pair constraints; introducing hard negative pairs anchored to intent/accents (e.g., question-command pairs) may further sharpen the discriminative structure of the signature space.

---

**Terminology Quick Reference**: UDKWS (User-Defined Keyword Spotting); QbyT/QbyE (Query-by-Text/Query-by-Example registration paradigms); prosody (suprasegmental information including intonation, stress, rhythm, etc.); prosodic signature (64-dimensional fixed vector of user style); FiLM (Feature-wise Linear Modulation, affine conditioning $\gamma \odot x + \beta$); MFA (Montreal Forced Aligner, phoneme-frame forced alignment tool); InfoNCE (contrastive learning loss, softmax cross-entropy of positive vs. negative pairs); EER (Equal Error Rate, FAR=FRR operating point); AUC (Area Under the ROC Curve); Conformer (Convolution-augmented Transformer speech encoder); G2P (Grapheme-to-Phoneme conversion); t-SNE (2D visualization of high-dimensional vectors).
