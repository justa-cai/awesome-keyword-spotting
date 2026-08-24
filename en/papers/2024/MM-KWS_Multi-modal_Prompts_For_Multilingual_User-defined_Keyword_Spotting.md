# MM-KWS: Multi-modal Prompts for Multilingual User-defined Keyword Spotting

- **Authors/Affiliations**: Zhiqi Ai, Zhiyong Chen, Shugong Xu (School of Communication and Information Engineering, Shanghai University)
- **Date**: June 2024 (arXiv:2406.07310v1, submitted 2024-06-11)
- **Link**: https://arxiv.org/abs/2406.07310
- **Keywords**: user-defined keyword spotting, multi-modal prompts, multilingual pre-trained models, cross-modal matching, hard case mining, zero-shot learning, data augmentation

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) is a resident voice entry point for devices such as smart speakers, wireless earbuds, in-car voice systems, and mobile voice assistants. The traditional KWS system works by collecting large-scale wake-word voice data for each predefined wake word (e.g., "Ok Google", "Hey Siri") and training a detection model with a fixed vocabulary. This workflow operates well for factory-default wake words, but the costs become immediately apparent when dealing with user-defined keywords (user-defined keyword spotting, UDKWS)—switching to a new word requires re-collecting data, re-training, and re-deploying. For manufacturers, this is an unscalable operational cost; for users, it means "wake words cannot be customized." The goal of UDKWS is to enable an already deployed model to accurately detect any new keyword using only a minimal amount of registration information (a few audio clips, or even just text).

The essential contradiction of this task lies in the fact that the model does not know what keywords users will register during training, so it cannot rely on fixed output nodes for classification. Instead, the information "what is the keyword" must be moved from the model parameters to the input at inference time—this is the so-called prompt (registration template). The modality used to carry the keyword information in the registration template differentiates various technical routes, which is the entry point of this paper.

### Specific Deficiencies of Existing Methods

The paper categorizes existing UDKWS solutions into four generations based on the registration modality and points out their defects one by one:

- **LVCSR Lattice Retrieval**: Early schemes used large vocabulary continuous speech recognition (LVCSR) to transcribe audio into lattices, performing keyword retrieval on the lattices, achieving high precision. However, performance drops significantly for out-of-vocabulary (OOV) words due to the fixed vocabulary of the recognizer, and the entire LVCSR system is too heavy for edge-side deployment.
- **Hot-word Enhanced ASR**: Injecting hot-word lists into ASR decoding (e.g., the hot-word mechanism in WeNet, FunASR) can improve the recall and accuracy of target words, but the inference cost remains high. The paper provides quantitative evidence from measurements on WenetPhrase (Table 2b): FunASR with a hot-word list (220M parameters) has a single inference latency of 300ms, and Whisper-Large (1550M) has a latency of 316ms—completely unacceptable for always-on listening scenarios.
- **QbyA (Registration by Audio)**: Users record the target word several times as voice templates, and the system performs audio-to-audio matching. The performance of such methods (frame-level embedding + DTW, acoustic word embeddings) heavily depends on the consistency between the registration recording and the usage scenario (channel, distance, noise, speaker state), and the registration process is cumbersome—users must read the word several times. The paper's Table 3 provides quantitative evidence: when the QbyA baseline has only 1 registration audio clip, the SPC multi-class Acc(close) is only 69.0%, and Acc(open) is only 66.0%; performance collapses when registration information is too scarce.
- **QbyT (Registration by Text)**: Users only input text, and the system performs text-to-speech cross-modal matching. This approach offers a good registration experience and strong reliability, becoming mainstream in recent years: CMCD leverages text-speech correspondence, EMKWS and PhonMatchNet optimize the matching structure and loss function respectively, and AdaKWS achieved the then-lowest EER by relying on larger-scale pre-training and hard example mining. However, a single text modality has an inherent weakness—for users with accents, mispronunciation significantly harms performance (the paper cites [16] pointing out the mispronunciation problem). A text template only tells the model "how the target word should be pronounced in standard form," but cannot tell it "how this specific user will actually pronounce it."

To summarize: QbyA information is close to real pronunciation but is expensive to register and sensitive to recording consistency; QbyT is cheap and stable to register but loses the speaker's actual pronunciation characteristics. Each route removes the other's strengths.

### Key Challenges to be Solved by This Paper

The paper aims to answer four questions simultaneously: First, how to fuse text templates and voice templates into a unified multi-modal registration mechanism, allowing text to provide stable vocabulary anchors and voice to provide supplementary information about the speaker's pronunciation, rather than simply choosing one; Second, how to enable the same system to cover multiple languages without re-training for a single language (the paper validates English and Mandarin); Third, how to specifically improve the discrimination ability for easily confused words (similar pronunciation, similar semantics)—which is precisely the main source of errors in UDKWS; Fourth, how to keep the online inference overhead at an acceptable level for always-on listening while fusing multi-modalities and introducing multiple pre-trained large models (the paper finally achieves 6ms, Table 2b).

## Methodology

### Overall Architecture Design and Design Motivation

MM-KWS consists of three sub-modules (Figure 1): a feature extractor, a pattern extractor, and a pattern discriminator.

**Feature Extractor**: Divided into a query branch and a support branch:

- The **query branch** processes the speech to be detected. It uses Conformer as the audio encoder (inspired by [6][16]) to convert the query speech into speech embeddings. The motivation for choosing Conformer over pure CNN/Transformer is that Conformer's convolutional branch is good at capturing local pronunciation details (consonant transients, formant trajectories), while its self-attention branch is good at global alignment across time steps; both are necessary for the judgment of "whether this phoneme sequence exists in the segment." The actual configuration is Tiny Conformer: 6 encoder layers, encoding dimension 128, kernel size 3, and 4 attention heads.
- The **support branch** is the physical carrier of the paper's "multi-modal prompts," containing a dual-branch text feature extractor and a high-performance speech encoder, producing three types of template representations: phoneme embeddings, text embeddings, and speech embeddings. Specific configuration: Multilingual DistilBERT produces 768-dimensional text embeddings (subword-level, length varies with text); a multilingual G2P model converts registered text into 64-dimensional phoneme embeddings; an 18-layer XLS-R (0.3B) extracts 1024-dimensional speech embeddings from registered voice templates. These three heterogeneous dimensions are then each passed through a lightweight mapper to unify them to 128 dimensions, aligning with the output dimension of the query branch.

Here is a very critical engineering design: **The support branch consists entirely of frozen-parameter pre-trained models, and registration embeddings are pre-computed only once during registration and directly looked up and reused during inference.** Why is it designed this way? Because registration templates (text and voice) remain fixed after the user sets them, so the cost of running XLS-R and DistilBERT for them can be completely amortized to zero—online inference only needs to run the query branch's Tiny Conformer and the subsequent two small attention modules. This explains why a system using a 0.3B parameter speech encoder has an online latency of only 6ms (Table 2b): the large model is not on the critical path. This is the entire secret behind how "multi-modal, large model, low latency" can coexist.

The **multilingual strategy** is also hidden in the selection of the support branch: DistilBERT, G2P, and XLS-R are all multilingual pre-trained models. The paper did not train any component specifically for Chinese or English; switching languages only requires changing a vocabulary and pronunciation dictionary, validating the route that "multilingual capability is inherited from pre-trained models rather than learned from KWS data."

**Pattern Extractor**: Built on self-attention mechanisms, it includes the Query-Text Attention Module (QTAM) and the Query-Audio Attention Module (QAAM). The paper cites the conclusion of [14] (PhonMatchNet) for choosing self-attention over DTW or Siamese networks for matching: self-attention performs well and is computationally efficient in KWS cross-modal matching. The deeper reason is: DTW can only perform monotonic alignment and cannot model "global phoneme order matching but local pronunciation deformation"; Siamese networks compress both paths into fixed-length vectors for comparison, losing frame-level correspondence; whereas attention naturally performs soft alignment between arbitrary positions within the concatenated sequence, allowing the model to learn "which frame of the query speech should focus on which phoneme of the template."

**Pattern Discriminator**: Uses GRU to derive sentence-level posterior probabilities from the joint embeddings of QTAM and QAAM respectively, and then fuses them. The decision "mainly relies on the more stable support text output, with support voice providing supplementary information" (Section 2.1 of the original text)—this sentence highlights the primary-secondary relationship in multi-modal fusion: text is the anchor, voice is the correction term.

### Mathematical Principles of Core Algorithms

**Embedding Tokenization and Input Transformation**. Let the query speech embedding be $E_a^q \in \mathbb{R}^{T_a^q \times d}$, support phoneme embedding be $E_p^s \in \mathbb{R}^{T_p^s \times d}$, support text embedding be $E_t^s \in \mathbb{R}^{T_t^s \times d}$, and support speech embedding be $E_a^s \in \mathbb{R}^{T_a^s \times d}$, where $T_a^q$ is the query speech frame length, $T_p^s$ and $T_t^s$ are the number of phonemes and subwords of the registered text respectively, $T_a^s$ is the registered speech frame length, and $d$ is the frame dimension (unified to 128).

To allow the attention module to distinguish "which modality this vector comes from and where it is in the sequence," each input embedding is superimposed with sinusoidal positional encoding $e_{pos}$ and learnable modality type encoding $e_{type}$ (Equation 1):

$$E = E + e_{pos} + e_{type}$$

In plain language: Give each frame an "ID badge + seat number." The badge specifies whether it comes from query speech, phoneme template, text template, or voice template; the seat number specifies its temporal position in its respective sequence. Without type encoding, attention would mix phonemes and subwords as the same token for alignment; without positional encoding, the permutation invariance of self-attention would make the phoneme strings "ni hao" and "hao ni" indistinguishable.

**QTAM (Equations 2, 3)**. Concatenate the transformed $E_a^q$, $E_p^s$, $E_t^s$ along the time dimension to form $\tilde{E}_{ta} = (E_a^{qc}; E_p^{sc}; E_t^{sc}) \in \mathbb{R}^{(T_a^q+T_p^s+T_t^s) \times d}$, then perform self-attention $E_{ta}^j = \text{Attention}(\tilde{E}_{ta}, \tilde{E}_{ta}, \tilde{E}_{ta})$. Four paths of information enter one attention field, allowing query frames to directly "query" phoneme tokens and subword tokens, forming cross-modal soft alignment.

**QAAM (Equations 4, 5)**. Same mechanism, but the input has only two paths: $E_{aa}^j = \text{Attention}(\tilde{E}_{aa}, \tilde{E}_{aa}, \tilde{E}_{aa})$, where $\tilde{E}_{aa} = (E_a^{qc}; E_a^{sc})$. It answers "how similar is the query speech to the registered speech," independent of the text path.

**Decision Fusion (Equation 6)**. GRU derives sentence-level posteriors $P_{utt}^t$ and $P_{utt}^a$ from the joint embeddings of QTAM and QAAM respectively, fused as:

$$P_{utt} = \sigma(W_u \cdot (P_{utt}^t + P_{utt}^a) + b_u)$$

Note that this is the most primitive additive fusion followed by a linear sigmoid—the paper does not learn a set of adaptive weights for the two paths, but lets the gradients decide how $W_u$ calibrates the scale of the sum of the two paths. The simplicity of the fusion formula contrasts interestingly with its stability (ablation in Table 4).

**Auxiliary Supervision (Equation 7)**. The total loss is the sum of three binary cross-entropy losses:

$$L_{total} = L_{utt} + L_{phon} + L_{text}$$

$L_{utt}$ is the main loss, judging "whether the query speech as a whole is the target keyword"; $L_{phon}$ and $L_{text}$ are auxiliary losses, cutting out phoneme segments ($(T_a^q, T_a^q+T_p^s]$) and subword segments ($(T_a^q+T_p^s, T_a^q+T_p^s+T_t^s]$) from the QTAM joint embedding by frame index respectively, judging "whether the target phoneme/target word appears in the query speech." Why are these two auxiliary heads needed? Because with only sentence-level labels, how alignment happens inside the attention is free and uninterpretable; adding unit-level supervision forces the phoneme and word positions in the joint embedding to encode "whether I have been matched by the query speech," effectively providing dense intermediate supervision signals for attention learning, directly suppressing error rates in easily confused word scenarios (Table 4: removing auxiliary losses causes LH's EER to worsen from 9.30% to 12.95%, the largest drop among the three ablation items).

### Key Technical Innovation 1: Multi-modal Prompt Registration with Text and Voice Dual Templates

This is the title-level contribution of the paper. Previous methods either used only text (QbyT) or only voice (QbyA). MM-KWS is the first to make the registration end a "text + voice" dual template, and both modalities can be omitted: when only text is present, QTAM works alone; when a voice template is added, QAAM joins to provide pronunciation correction. This design directly resolves the contradiction mentioned in the introduction—the text template provides a stable anchor independent of recording (solving QbyA's expensive registration and sensitivity to channel changes), while the voice template carries the speaker's real pronunciation (solving QbyT's sensitivity to accents). Zero-shot experiments (Table 3) verify flexibility: MM-KWS with pure text registration (0 audio clips) achieves Acc(close) 94.4% / Acc(open) 88.4%, surpassing the QbyA baseline's 90.5% / 80.6% with 5 audio clips; adding 1 audio clip further improves it to 95.2% / 90.6%. Multi-modality here is not "icing on the cake," but a robustness structure that "runs without either, and is more accurate with both."

### Key Technical Innovation 2: Multilingual Pre-trained Model Combination + Inference Economics of One-time Pre-fetching

The support branch uses three multilingual pre-trained models (DistilBERT, multilingual G2P, XLS-R 0.3B) + three lightweight mappers, all frozen. The innovation lies not in individual models, but in the combination method and cost accounting: registration embeddings are calculated only once, and online inference is spent only on the query branch and pattern extractor, so the entire system reports only 3.9M parameters and 6ms latency (Table 2b). This provides a transferable pattern for "wanting to use large model capabilities on the edge side"—large models run offline, small models run online, decoupled by pre-computed embeddings. The paper also honestly lists "edge-side deployment" as future work (Conclusion section), indicating that the authors are aware that XLS-R on the registration side remains a deployment threshold.

### Key Technical Innovation 3: Two-stage Data Augmentation for Hard Example Mining

Targeting easily confused words as the main error source in UDKWS, the paper designs two-stage augmentation (Section 2.3):

**Stage 1: Easily Confused Word Generation**. The goal is to systematically produce negative sample words that are "easily misdetected as the target word." Four complementary generation paths: (1) Use pre-trained G2P to convert the target language's common word corpus into phoneme sequences, calculate the phoneme edit distance with the target word, and filter out words with similar pronunciation (e.g., "Young Man" vs. "Youth", "Youth League"); (2) Use DistilBERT to extract semantic embeddings, calculate cosine similarity with the target word's semantic embedding, and filter out semantically similar words; (3) Word order permutation generates supplementary negatives (scrambling the order of multi-word fragments); (4) Use Large Language Models to batch generate negatives that may appear in real scenarios (the paper does not report which specific LLM was used). Why all four paths? Pronunciation-similar negatives attack acoustic confusion, semantic-similar negatives attack semantic confusion, permutation negatives teach the model "wrong order counts as wrong even if words are correct," and LLM negatives supplement the real language distribution—four types cover different error mechanisms.

**Stage 2: Speech Synthesis**. The generated easily confused words lack corresponding audio, so a multilingual zero-shot text-to-speech model (ZS-TTS, the authors' previous work [22]) is used to synthesize speech, totaling 1.5 million English and 2.4 million Mandarin training data. Using synthesis instead of real human recording reduces costs by an order of magnitude and allows on-demand control of pronunciation difficulty distribution; the cost is the domain gap between synthesized and real speech, which the paper does not report evaluating.

This augmentation is the source of the asterisked MM-KWS* in Table 1 and Table 2b, and the main contributor to reducing EER on LH (Hard Set) from 12.45% to 9.30% (Table 1, Table 4).

### Key Technical Innovation 4: WenetPhrase—Mandarin Easily Confused Word Evaluation Benchmark

The paper found that existing Mandarin KWS datasets lacked an "easily confused word evaluation" category, so they built WenetPhrase themselves: using forced alignment algorithms to segment the M/S subset of WenetSpeech (approx. 1000 hours), Jieba word segmentation to obtain the target word list, filtering fragments of 0.5–2 seconds containing 2–6 words, finally obtaining approx. 122k training classes, 54k testing classes, and a total of 2.9 million samples, divided into Easy (WE) / Hard (WH) subsets in the manner of LibriPhrase. The examples in Table 2a are intuitive: for the anchor word "Ningyuan" (Willing), the hard negatives are "Tingyuan" (Courtyard), "Xingyuan" (Row Courtyard), "Qingyuan" (Willing) (phonetically similar), and easy negatives are "Shamo" (Desert), "Dezhi" (Learn), "Gongwu" (Official Business). This benchmark fills the gap of "no public evaluation for Mandarin easily confused word UDKWS," constituting an independent contribution in itself.

### Technical Differences with Existing Methods

- **Difference from CMCD / EMKWS / PhonMatchNet (QbyT series)**: They only perform text-to-speech cross-modal matching. MM-KWS runs a parallel voice registration path (QAAM) alongside the text path. Registration information changes from single-modal to dual-modal, and fusion prioritizes text with voice as auxiliary. On Table 1, PhonMatchNet (0.7M) LH AUC is 88.52%, while MM-KWS without augmentation reaches 94.02%, and with augmentation 96.25%.
- **Difference from QbyA series**: QbyA relies on multiple registration audio clips to support performance (Table 3: 1 clip 69.0% → 5 clips 90.5%). MM-KWS downgrades the voice template to an optional supplement, with the text template bearing the main registration function, offering better registration cost and robustness.
- **Difference from AdaKWS**: AdaKWS boosts metrics with larger parameters (Small version 109M) and larger-scale pre-training data. MM-KWS surpasses AdaKWS-Small on LH with only 3.9M parameters (AUC 96.25% vs 95.09%, EER 9.30% vs 11.48%, Table 1)—the gap comes from multi-modal registration and hard example augmentation, not parameter stacking.
- **Difference from Hot-word ASR (FunASR, Whisper)**: These systems treat keyword detection as a byproduct of recognition, resulting in large models and high latency (FunASR 300ms, Whisper-Large 316ms, Table 2b), and are nearly ineffective on the Mandarin hard negative set WH (Whisper-Large AUC only 56.46%, FunASR only 58.31%); MM-KWS is designed specifically for detection, achieving AUC 85.84% and 6ms latency in the same scenario.
- **Horizontal comparison of training configurations**: 4 NVIDIA 4090s, PyTorch, Adam optimizer, 40k training steps (Section 3.2)—the training budget is quite restrained, indicating that the method's demand for computing power is mainly in data synthesis (ZS-TTS) rather than model training.

## Experimental Results

### Datasets Used and Their Scales

- **LibriPhrase (English, following the construction of [5][14])**: Training set comes from LibriSpeech train-clean-100/360, test set from train-others-500, divided into Easy (LE) / Hard (LH) subsets for testing. The paper does not report the total number of samples in this dataset.
- **WenetPhrase (Mandarin, newly built in this paper)**: Based on forced alignment segmentation of approx. 1000 hours of WenetSpeech M/S, filtering 0.5–2 second, 2–6 word fragments, approx. 122k training classes, 54k testing classes, total 2.9 million samples, divided into WE/WH subsets (Section 3.1).
- **SPC (Speech Commands, zero-shot evaluation)**: Follows the setup of [13] for 30-word multi-classification—randomly select 10 as target words, the remaining 20 as unknown classes.
- **Synthetic Augmentation Data**: ZS-TTS synthesizes 1.5 million English and 2.4 million Mandarin clips (Section 2.3).

The three datasets each have their role: LibriPhrase verifies English home-field performance, WenetPhrase verifies Mandarin and easily confused words, SPC verifies zero-shot cross-dataset generalization—especially the "train on LibriPhrase, test on SPC" zero-shot path, which tests whether multilingual pre-trained features truly decouple the task from specific data distributions.

### Definition and Rationale for Evaluation Metrics

- **EER (Equal Error Rate)**: The error rate when the false alarm rate equals the miss rate. It is a standard metric for detection/verification tasks, threshold-independent, suitable for comparing models with different operating system selection points.
- **AUC (Area Under ROC Curve)**: Average discrimination power across all thresholds, more sensitive to "overall separability." UDKWS actual deployment requires selecting thresholds based on product needs; the EER+AUC combination looks at both the optimal point and the global view, following the convention of QbyT series works (CMCD, PhonMatchNet, AdaKWS) to ensure comparability.
- **Acc(close) / Acc(open) (SPC Zero-shot)**: Multi-class accuracy for close-set (excluding unknown classes) and open-set (including unknown classes). In zero-shot scenarios where no registration audio is available to tune thresholds, classification accuracy is more direct; open-set tests the ability to "reject non-target words" more rigorously than close-set.
- **Latency (ms, Table 2b)**: The paper does not report the specific measurement conditions for latency (input duration, batch size, hardware), which is a口径 (caliber) issue to note during horizontal comparison.

### Detailed Comparison with Baseline and SOTA Methods

**English LibriPhrase (Table 1, metric order: LH AUC / LE AUC / LH EER / LE EER)**:

| Method | Params | LH AUC | LE AUC | LH EER | LE EER |
|---|---|---|---|---|---|
| Whisper-Tiny | 39M | 73.37 | 89.19 | 33.04 | 17.31 |
| Whisper-Small | 224M | 82.90 | 95.92 | 21.45 | 8.14 |
| Whisper-Large | 1550M | 85.80 | 97.54 | 19.57 | 5.33 |
| Triplet | N/A | 54.88 | 63.53 | 44.36 | 32.75 |
| CMCD | 0.7M | 73.58 | 96.70 | 32.90 | 8.42 |
| EMKWS | 3.7M | 84.21 | 97.83 | 23.36 | 7.36 |
| PhonMatchNet | 0.7M | 88.52 | 99.29 | 18.82 | 2.80 |
| CED | 3.6M | 92.70 | 99.84 | 14.40 | 1.70 |
| AdaKWS-Tiny | 15M | 93.75 | 99.80 | 13.47 | 1.61 |
| AdaKWS-Small | 109M | 95.09 | 99.82 | 11.48 | 1.21 |
| MM-KWS | 3.9M | 94.02 | 99.98 | 12.45 | 0.41 |
| MM-KWS* | 3.9M | 96.25 | 99.95 | 9.30 | 0.68 |

Three readings are worth expanding on. First, MM-KWS without augmentation already surpasses CED and AdaKWS-Tiny on the hard set LH (94.02% vs 92.70%/93.75%), trailing only slightly behind the 109M AdaKWS-Small; MM-KWS* with hard example augmentation tops the leaderboard with 96.25%, achieved with only 3.9M parameters and smaller training data scale (the paper explicitly states AdaKWS uses larger pre-training data). Second, on LE, MM-KWS's EER is as low as 0.41%, AUC 99.98%, basically breaking through the easy set—the hard set is where the real frontier of this task lies. Third, all Whisper versions struggle significantly on LH (best Whisper-Large EER 19.57%), indicating that "good recognition" does not equal "ability to distinguish phonetically similar words"; the difference between detection-style modeling and recognition-style modeling is amplified on the hard set.

**Mandarin WenetPhrase (Table 2b, WH AUC / WE AUC / WH EER / WE EER / Latency)**:

| Method | Params | WH AUC | WE AUC | WH EER | WE EER | Latency(ms) |
|---|---|---|---|---|---|---|
| Whisper-Tiny | 39M | 56.53 | 60.67 | 44.66 | 45.38 | 102 |
| Whisper-Small | 244M | 57.31 | 72.20 | 44.53 | 35.56 | 183 |
| Whisper-Large | 1550M | 56.46 | 88.77 | 48.76 | 15.51 | 316 |
| FunASR† (Hot-word) | 220M | 58.31 | 99.02 | 45.03 | 3.62 | 300 |
| MM-KWS | 3.9M | 83.73 | 99.79 | 23.88 | 1.95 | 6 |
| MM-KWS* | 3.9M | 85.84 | 99.15 | 22.06 | 4.25 | 6 |

This table contains more information than the English table. First, the paper observes that WE difficulty is comparable to LE, but all ASR systems collectively fail on WH—AUCs all stick to the random line (56–58%), and even FunASR with hot-words only reaches 58.31%, indicating that Mandarin phonetically similar words (same tone different characters, similar initials/finals combinations) are a catastrophic challenge for recognition-style systems. Second, MM-KWS ties with hot-word FunASR on WE (99.79% vs 99.02%), and pulls ahead by more than 27 points on WH (83.73% vs 58.31%), with latency only 1/50th of the other (6ms vs 300ms). Third, an honest detail: MM-KWS* slightly degrades compared to MM-KWS on WE (AUC 99.79%→99.15%, EER 1.95%→4.25%), and similarly on LibriPhrase's LE (0.41%→0.68%)—hard example augmentation essentially pushes the decision boundary towards the confused zone, paying a small cost of false positives on the easy set in exchange for significant gains on the hard set (WH EER 23.88%→22.06%, LH EER 12.45%→9.30%). Whether to enable augmentation during deployment depends on the product's relative tolerance for false alarms and misses. Fourth, Whisper-Small's parameter count is listed as 224M in Table 1 and 244M in Table 2b; the paper has internal inconsistency in caliber, which should be noted when citing.

**Zero-shot SPC (Table 3)**:

| Method | Reg. Audio | Acc(close) | Acc(open) |
|---|---|---|---|
| QbyA-baseline | 1 | 69.0±1.67 | 66.0±1.03 |
| QbyA-baseline | 5 | 90.5±0.53 | 80.6±0.44 |
| MM-KWS | 0 (Pure Text) | 94.4±0.18 | 88.4±0.28 |
| MM-KWS | 1 | 95.2±0.17 | 90.6±0.19 |

Zero-shot means the model is trained on LibriPhrase and tested directly on SPC. Two conclusions: (1) Pure text registered MM-KWS (94.4%/88.4%) surpasses QbyA baseline registered with 5 audio clips (90.5%/80.6%); the multi-modal system is still stronger than the single-modal system's "sufficient input mode" under its own "worst input mode"; (2) MM-KWS's variance (±0.18) is an order of magnitude smaller than the QbyA baseline (±1.67), indicating that the text anchor brings not only mean improvement but also stability. Adding 1 audio clip increases performance by another 0.8–2.2 points; marginal returns exist but diminish.

**Attention Visualization (Figure 2)**: Target word "good boy", positive example is speech containing the word, negative example is "in the United States". Subplot (a) is the attention response of phoneme and text embeddings to query speech in QTAM, (b) is the response of registered speech to query speech in QAAM, (c)(d) are negative example cases. Observation conclusion: Under positive examples, attention for all three modes presents a clear monotonic diagonal structure (query frames sequentially focus on corresponding template units as time progresses); under negative examples, this monotonicity disappears. This explains mechanistically what the model is doing—it is not a black-box similarity, but has learned frame-to-unit sequential alignment, converging with DTW's monotonic alignment assumption but allowing soft deformation.

### Findings from Ablation Experiments

Table 4 dissects three components on LibriPhrase (the full version is MM-KWS* with augmentation):

| Configuration | LH AUC | LE AUC | LH EER | LE EER |
|---|---|---|---|---|
| MM-KWS (Full) | 96.25 | 99.95 | 9.30 | 0.68 |
| Remove Easily Confused Word Generation | 94.02 | 99.98 | 12.45 | 0.41 |
| Remove Support Voice Branch | 95.36 | 99.94 | 10.41 | 0.82 |
| Remove Auxiliary Loss | 93.48 | 99.89 | 12.95 | 1.35 |

Three findings: (1) **Auxiliary loss contributes the most**—removing it causes LH AUC to drop 2.77 points and EER to worsen 3.65 points, the largest single degradation among the three, verifying the judgment that "unit-level dense supervision is a rigid demand for hard example discrimination"; (2) **Easily confused word generation is second**—removing it causes LH EER to worsen from 9.30% to 12.45%, but LE EER actually improves from 0.68% to 0.41%, again confirming that augmentation is a trade-off of hard set for easy set; (3) **Voice branch contribution is smallest but stably positive**—removing it worsens LH EER by 1.11 points (9.30%→10.41%) and LE EER by 0.14 points, indicating that on LibriPhrase, a reading dataset with no obvious accents, the text path bears most of the discrimination, and the marginal value of the voice path can only be fully realized in accent/real pronunciation scenarios—which the paper恰恰 (precisely) did not evaluate specifically for accents, a boundary to note when extrapolating ablation conclusions. Also worth affirming is that the row "Remove Easily Confused Word Generation" in Table 4 is exactly consistent with the MM-KWS row in Table 1 (94.02/99.98/12.45/0.41), ensuring data caliber consistency.

## Main Contributions

1. **Proposed MM-KWS**: The first UDKWS framework using "text + voice dual templates" for multi-modal registration prompts. The fusion structure of text as primary and voice as secondary achieves both the registration convenience of QbyT and the pronunciation adaptability of QbyA, and both modalities can be omitted (zero-shot evidence in Table 3).
2. **Efficient Introduction of Multilingual Pre-trained Models**: Using frozen DistilBERT + multilingual G2P + XLS-R(0.3B) to form the support branch, pre-fetched once during registration, verified high performance on both English and Mandarin, while compressing online inference to 3.9M parameters and 6ms latency (Table 1, Table 2b).
3. **Two-stage Hard Example Mining Augmentation**: Four paths of negative example mining via phoneme edit distance + semantic cosine + word order permutation + LLM generation, combined with ZS-TTS synthesis of 3.9 million bilingual data, reducing EER in easily confused word scenarios (LH) from 12.45% to 9.30% (Table 1, Table 4).
4. **Released WenetPhrase Benchmark**: Filled the gap in Mandarin easily confused word UDKWS evaluation (approx. 1000 hours source, 2.9 million samples, WE/WH division), and open-sourced code and data (project page in paper footnote 1).
5. **Methodological Level**: Demonstrated the feasibility of the "offline large model features + online small model matching" decoupled mode in edge-side speech tasks, and the value of unit-level auxiliary supervision for cross-modal alignment learning.

## Limitations and Future Work

### Technical Limitations of the Method

- **Absolute performance on hard sets is still not high**: The best WH result is only AUC 85.84%, EER 22.06% (Table 2b), and best LH EER 9.30% (Table 1). Mandarin phonetically similar words (micro-differences in initials/finals/tones, same tone near-homophones) are still far from practical use; multi-modal registration and hard example augmentation only partially alleviate this.
- **Augmentation causes degradation on easy sets**: MM-KWS* is systematically slightly worse than MM-KWS on LE/WE (LE EER 0.41%→0.68%, WE EER 1.95%→4.25%, Table 1, Table 2b), indicating that synthesized hard negatives push the decision boundary overall to the conservative side, lacking an adaptive difficulty curriculum or example-weighting mechanism.
- **Fusion mechanism is too primitive**: Equation 6 performs simple addition of the two posteriors followed by linear mapping, without considering that the confidence of the two paths changes dynamically with samples (e.g., when registration voice quality is poor, QAAM weight should be automatically reduced). Learnable gating or quality-based weighting might yield free gains.
- **Heavy reliance on offline processing**: Although the online path has only 3.9M parameters, the registration phase requires running XLS-R(0.3B), multilingual DistilBERT, G2P, and ZS-TTS (if synthesizing augmentation data on-site); the computational threshold on the registration side is not quantified by the paper; the authors themselves admit in the conclusion that edge-side deployment and model lightweighting are unfinished tasks, and a unified multilingual single-model framework is also left for future work.
- **Synthesis domain gap not evaluated**: All 3.9 million training negatives come from ZS-TTS; the paper does not report targeted analysis on whether the acoustic difference between synthesized and real speech causes the model to learn "synthetic acoustic feature" shortcuts.
- **Streaming capability not discussed**: The query branch's Conformer and two attention modules process whole sentences; the paper does not report streaming/frame-by-frame detection schemes, whereas always-on wake-up scenarios exactly require streaming low-power operation; the measurement conditions for the 6ms latency (input length, hardware) are also not reported.

### Deficiencies in Experimental Design

- **Accent hypothesis not directly tested**: The introduction cites "accent users' mispronunciation harming QbyT performance" as one of the core motivations, but there is no evaluation set with accent annotations (e.g., non-native English, dialect Mandarin) in the experiments; the voice branch's role in correcting accents is only indirectly hinted by the "adding 1 audio clip boosts points" in Table 3.
- **Incomplete ablation coverage**: Table 4 only dissects three components, without dissecting the individual contributions of phoneme embeddings vs. text embeddings (lack of pairwise ablation for the three inputs in QTAM), without dissecting the respective proportions of LLM negatives vs. rule-based negatives, and without reporting curves for non-zero-shot tasks with different numbers of registration audio clips (2, 5 clips).
- **Thin zero-shot comparison**: Table 3 only compares one QbyA baseline; same-domain methods like PhonMatchNet and AdaKWS are not included in zero-shot comparison, making it difficult to judge how much of the zero-shot advantage comes from multi-modality vs. how much from the pre-trained features themselves.
- **Incomplete details on Chinese training data source and hyperparameters**: The sample split ratio for WenetPhrase training/testing, class overlap situations, and training hyperparameters such as learning rate and batch size are not reported by the paper (only Adam, 40k steps, 4 4090s are reported); Whisper-Small parameter count also appears inconsistently as 224M/244M in the two tables.
- **Missing latency measurement caliber**: The latency in Table 2b does not specify whether it is registration-side or query-side, input length, or hardware used for measurement, allowing only relative comparison.

### Possible Directions for Future Improvement

- Follow the paper's self-described direction: Unified single-model multilingual framework (removing dependency on assembling multiple pre-trained components), model lightweighting for edge-side, and verification in real scenarios.
- Upgrade the additive fusion of Equation 6 to confidence-adaptive gating, and introduce registration voice quality estimation, allowing the voice path's weight to change dynamically with registration quality and speaker match degree.
- Introduce explicit tone modeling for Mandarin hard sets (adding tone embeddings in addition to phoneme embeddings) or contrastive tone hard example mining; the 22.06% EER on WH indicates that current phoneme representations are insufficient for tone discrimination.
- Introduce real recording re-injection or synthetic-real mixed training for augmentation data, with difficulty-graded curriculum, to alleviate easy-set degradation.
- Supplement specialized evaluations for accent/long-range/noise scenarios, putting the core selling point of "voice template correcting accents" directly to the data for verification; simultaneously explore streaming query branches (e.g., downsampled Conformer or cache-based attention) to enter always-on wake-up deployment.
- In negative example generation, expand the role of LLMs from "batch word generation" to "targeted hard example generation based on actual user false-trigger logs," forming a deployment-in-the-loop hard example closed loop.

## Appendix: Terminology Quick Reference

- **UDKWS (user-defined keyword spotting)**: User-defined keyword spotting, where the vocabulary is not limited during training, and new words are detected according to registration templates at inference time.
- **QbyE / QbyA / QbyT**: Three paradigms of registration by Example / by Audio / by Text. E is the general term, A uses only voice templates, T uses only text templates.
- **Conformer**: Convolution-augmented Transformer, a hybrid encoder where convolution captures local features and self-attention captures global dependencies.
- **XLS-R**: Multilingual self-supervised speech pre-trained model based on wav2vec 2.0; this paper uses the 18-layer version with 0.3B parameters.
- **G2P (grapheme-to-phoneme)**: Grapheme-to-phoneme conversion, converting spelled text into pronunciation phoneme sequences.
- **DTW (dynamic time warping)**: Dynamic Time Warping, a monotonic sequence alignment matching algorithm commonly used in the QbyA series.
- **EER / AUC**: Equal Error Rate / Area Under ROC Curve, a pair of standard metrics for detection tasks; the former looks at the optimal operating point, the latter looks at global separability.
- **ZS-TTS**: Zero-shot Text-to-Speech, capable of synthesizing any text speech without target speaker training data.
