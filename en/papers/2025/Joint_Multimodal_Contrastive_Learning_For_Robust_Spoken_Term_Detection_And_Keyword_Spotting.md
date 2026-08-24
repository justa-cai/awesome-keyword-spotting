# Joint Multimodal Contrastive Learning for Robust Spoken Term Detection and Keyword Spotting

- **Authors/Affiliations**: Ramesh Gundluru, Shubham Gupta, Sri Rama Murty K (Indian Institute of Technology Hyderabad; Intel)
- **Date**: December 2025 (arXiv 2512.14115v1, submitted 2025-12-16)
- **Link**: https://arxiv.org/abs/2512.14115
- **Keywords**: Acoustic Word Embeddings, Multimodal Contrastive Learning, CLAP-style Audio-Text Alignment, Deep Word Discrimination, Spoken Term Detection, Keyword Spotting, Cross-modal Retrieval

## Problem Statement

### Problem Background and Domain Pain Points

The problem domain addressed by this paper is **the retrieval of spoken content from large-scale unlabeled audio archives**. Audio assets such as podcasts and multimedia libraries are expanding continuously, yet they generally lack transcribed text or rich metadata. In such scenarios, the traditional cascaded pipeline of "ASR followed by text retrieval" either suffers from amplified ASR errors at the retrieval stage or fails entirely due to resource constraints (e.g., low-resource languages, domain mismatch). Consequently, bypassing ASR and performing matching directly at the acoustic level has become a mainstream approach.

Two core tasks within this approach are explicitly distinguished by the paper:
- **KWS (Keyword Spotting)**: Matching spoken utterances in an audio corpus against a **written keyword (text query)**, where the query modality is text;
- **QbE-STD (Query-by-Example Spoken Term Detection)**: Directly comparing a **spoken query audio** against audio content, where the query modality is speech.

It is industry norm to have separate dedicated systems for each task. However, the paper’s stance is that generalized spoken content retrieval truly requires **a single model that ingests both acoustic and lexical cues**, serving both types of queries in a unified embedding space, and capable of training with minimal resources. This requirement for a "single model, dual modality, dual task" serves as the overarching starting point for all design decisions in the paper.

The classic tool for direct acoustic matching is DTW (Dynamic Time Warping): aligning query and retrieval signals frame-by-frame to compute similarity. However, DTW is highly fragile in practice—it assumes inputs are clean, tightly cropped, and endpoint-aligned segments, is highly sensitive to noise and silence, exhibits **quadratic complexity** with respect to sequence length, possesses no trainable parameters (thus learning nothing from data), and lacks principled methods for selecting matching thresholds. These issues persist to varying degrees in improved variants such as Segmental DTW, Subsequence DTW, and Non-Segmental DTW.

To overcome the limitations of DTW, research focus has shifted toward **Acoustic Word Embeddings (AWE)**: using a neural encoder to map variable-length audio segments into fixed-length vectors, with similarity computed via simple metrics like cosine distance. Representative works on this line include: Levin et al.’s template-based embeddings (using DTW distance from word to a set of reference templates as embeddings); Settle et al.’s Siamese LSTM + triplet loss (the beginning of discriminative AWE, with lexical discrimination capabilities far exceeding DTW); He et al.’s multi-view RNN (fusing acoustic and character perspectives); Chung et al.’s Audio Word2Vec (unsupervised seq2seq autoencoder, significantly outperforming DTW at much lower computational cost on QbE-STD); Kamper et al.’s Correspondence Autoencoder CAE (reconstructing another instance of the same word rather than its own input, combined with unsupervised word discovery to achieve unsupervised discriminative embeddings); and multilingual methods for transferring from high-resource to low-resource languages.

### Specific Shortcomings of Existing Methods

The paper’s critique of existing AWE methods focuses on four points, each supported by a clear chain of evidence:

- **Unimodal Supervision**: The vast majority of models are trained using only audio-audio pairs (Siamese-RNN, Contrastive-RNN, A2E-DWD, CAE-RNN) or only audio-text pairs, failing to leverage complementary information from cross-modal inputs. The consequences are intuitive in the experiments: in Table I, under the acoustic perspective, Multiview-RNN achieves only 55.47% IV AP, and Contrastive-RNN 74.94%, both inferior to schemes that incorporate cross-modal + class-structure constraints.
- **Disjoint Optimization of audio-audio and audio-text objectives**: Even models like Multiview-RNN that use both perspectives do not jointly balance "class-structure constraints within the audio space" (intra-class compactness, inter-class separation) with "cross-modal alignment" in a unified objective. The extent to which each objective is satisfied is left to chance.
- **Task-Specific Models**: Unimodal models often excel at only one task—models trained purely on audio-audio are good at QbE but natively do not support text queries for text-STD; conversely, models trained purely on cross-modal data have poor structural performance in pure acoustic discrimination (the core operation of QbE).
- **Chaotic Evaluation Protocols**: Existing literature often fails to specify dataset splits, trial generation methods, trial counts, or IV (in-vocabulary) / OOV (out-of-vocabulary) distinctions, making it impossible to directly compare numbers across different papers. While this may seem like a non-algorithmic issue, the paper elevates it to a status equal to algorithmic contributions and addresses it specifically as the third contribution.

### Key Challenges to Be Solved by This Paper

In summary, the key challenge the paper aims to solve is: **How to use a shared multimodal embedding space and a set of joint optimization objectives to simultaneously acquire cross-modal alignment capabilities (supporting text-query KWS) and discriminative structure within the audio space (supporting speech-query QbE-STD), and to find the correct loss weighting when these two objectives may constrain each other, while standardizing the evaluation protocol to a level of reproducibility and horizontal comparability.** The notion that "the two objectives constrain each other" is not merely an assumption—Table II (KWS section) and Table III (weight ablation) empirically demonstrate the existence of this constraint relationship; choosing the wrong balance will cause performance drops in both areas.

## Methodology

### Overall Architecture Design and Design Motivation

The overall framework draws inspiration from the dual-tower approach of CLAP (Contrastive Language-Audio Pretraining) / CLIP, consisting of three components:

1. **Audio Encoder** $f_a(\cdot)$: A 3-layer bidirectional LSTM with 256 hidden units, followed by a fully connected layer projecting to a 512-dimensional embedding space;
2. **Text Encoder** $f_t(\cdot)$: Similarly, a 3-layer bidirectional LSTM + fully connected projection to 512 dimensions, used to encode text keywords (phoneme sequences);
3. **Two Learnable Linear Projection Layers** $g_a, g_t$: Projecting the outputs of the dual towers into the **same** $d$-dimensional shared embedding space ($E_a, E_t \in \mathbb{R}^{N \times D}$), followed by **unit norm normalization** of all representations.

Input side: Audio consists of Mel-filterbank energies $X_a \in \mathbb{R}^{T \times F}$ ($T$ is the number of frames, $F$ is the number of filterbank coefficients, taken as 128 in experiments); Text keywords are represented as $X_t \in \mathbb{R}^{P \times V}$ ($P$ is the number of phonemes in the keyword, $V$ is the phoneme embedding dimension). Each training batch contains $N$ unique audio-text query pairs. Two notable representation choices: **the text side uses phoneme sequences rather than character sequences**—phonemes are direct symbolizations of speech, with a tighter correspondence to acoustic events than characters (in English, one letter corresponds to multiple phonemes). This allows the audio-text alignment to learn the mapping from "acoustic content to pronunciation symbols" rather than an indirect mapping from spelling to pronunciation; Fig. 2’s text embeddings are explicitly labeled as "phoneme queries"; **the audio side uses a 25 ms window, 10 ms frame shift, and 128 Mel filterbanks**—this is the standard frontend configuration in speech recognition. A frame shift of 10 ms ensures that short words (even 0.5 seconds yields about 50 frames) have sufficient temporal resolution for the BiLSTM to model.

Design motivations broken down item by item:

- **Why use dual towers + shared space instead of single-tower concatenation**: The dual-tower structure allows audio and text to be **encoded independently and built into an offline index** during inference—millions of audio entries in the retrieval corpus only need to pass through the audio tower once, and text queries only need to pass through the text tower once; comparison is a cheap cosine operation. This is the engineering prerequisite for KWS and STD sharing the same foundation.
- **Why all models uniformly use the BiLSTM architecture**: The paper deliberately ensures that five baselines and the proposed model use the exact same encoder (3-layer BiLSTM-256 + FC-512), ensuring that comparative results reflect differences in **training objectives** rather than capacity differences. This is a very clean aspect of the experimental design.
- **Why add the DWD branch**: The authors explicitly point out that the audio-text contrastive loss is only responsible for "pulling matching audio-text pairs closer," **and does not explicitly constrain the structure of the audio embedding space itself**—lacking intra-class compactness and inter-class separation. During QbE-STD inference, both the query and corpus content pass only through the audio encoder, so the quality of the audio space structure directly determines QbE performance. Therefore, a pure audio-domain discriminative objective (DWD) must be added; this is the actual meaning of "joint."

### Mathematical Principles of Core Algorithms

**CLAP-style Audio-Text Contrastive Loss**. After encoding audio and text, linear projections yield $E_a, E_t$. After unit normalization, an $N \times N$ cosine similarity matrix is computed:

$$C = \exp(\tau) \cdot (E_t \cdot E_a^\top)$$

where the temperature parameter $\tau$ is **learned during training in the form of a log-parameterized scalar**. The authors’ rationale is that this eliminates the need to manually tune it as a hyperparameter (this practice is directly inherited from CLIP’s learnable logit scale). The diagonal elements of $C$ correspond to positive sample pairs, while off-diagonal elements are negative sample pairs within the batch. The symmetric contrastive cross-entropy loss is:

$$L_{audio} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(C_{i,i})}{\sum_{j=1}^{N}\exp(C_{i,j})}, \qquad L_{text} = -\frac{1}{N}\sum_{j=1}^{N}\log\frac{\exp(C_{j,j})}{\sum_{i=1}^{N}\exp(C_{i,j})}$$

$$L_{at} = \frac{1}{2}(L_{audio} + L_{text})$$

$L_{audio}$ uses row softmax to map each audio to the correct text, while $L_{text}$ uses column softmax to map each text to the correct audio. **Why make it symmetric**: The paper states that this symmetric objective ensures that cross-modal alignment is "balanced" rather than biased toward one modality—a unidirectional loss only optimizes retrieval in one direction, while the nearest-neighbor structure in the other direction may be degenerate.

**DWD-style Audio-Audio Contrastive Loss**. Within the batch, there are $N$ different spoken query words, with $M$ positive instance samples for each word. The cleverness of this construction lies in the fact that the $M$ instances of $N$ words naturally provide an embedding combination of magnitude $(N \times M)$. Positive examples come from different instances of the same word (covering speaker, speed, and channel variations), while negative examples come from all other words in the batch—**negative examples are free**, requiring no specialized negative mining pipeline. The larger the batch, the harder and richer the negative examples, which is the core efficiency advantage of in-batch InfoNCE-style methods relative to triplet methods. The centroid of the $j$-th word class is calculated using a **leave-one-out** method (excluding the current embedding $e_{ji}$ itself):

$$c_j = \frac{1}{M-1}\sum_{m=1, m\neq i}^{M} e_{jm}$$

**Why leave-one-out**: If the centroid includes the current embedding itself, the loss is diluted by its own self-similarity, distorting the gradient signal; leaving one out ensures the centroid is a pure external reference. For each embedding $e_{ji}$, cosine similarity $S_{ji,k} = \cos(e_{ji}, c_k)$ is computed against all class centroids. The total loss consists of two parts:

$$L_{sm} = -S_{ji,j} + \log\sum_{k=1}^{N}\exp(S_{ji,k})$$

$$L_{cc} = \sum_{j=1}^{N}\sum_{i=1}^{M}\left[(1 - S_{ji,j}) + \max_{k\neq j} S_{ji,k}\right]$$

$$L_{aa} = L_{sm} + L_{cc}$$

$L_{sm}$ is a softmax-style contrastive term (pulling embeddings toward their own class centroid, pushing away from other class centroids), while $L_{cc}$ is a centroid contrastive term: the first term $(1-S_{ji,j})$ is an intra-class compactness term (penalty decreases as similarity approaches 1), and the second term takes the **hardest negative class centroid** in hinge form, specifically targeting the most easily confused word pairs. **Why both are needed**: The softmax term provides soft normalization gradients across all classes, while the hinge term focuses on the hardest negatives; the two complement each other.

**Total Loss**. Since each word has $M$ positive examples, $L_{at}$ is computed $M$ times on the $N \times N$ audio-text pairs, and the average of these $M$ loss matrices is taken; $L_{aa}$ is computed once on the $N \times M$ audio embeddings. The total loss is:

$$L_{total} = \alpha_1 \cdot \frac{1}{M}\sum_{m=1}^{M} L_{at}^{(m)} + \alpha_2 \cdot L_{aa}$$

The paper finally sets $\alpha_1 = 0.1, \alpha_2 = 1$, meaning **the weight of the cross-modal term is only one-tenth that of the audio-audio term**. Why suppress the CLAP term so much—the basis is in the ablation in Table III (see Experimental Results section): audio-side discrimination is the performance bottleneck, and too high a weight for the cross-modal term squeezes the structural quality of the audio space.

### Key Technical Innovation 1: Joint Optimization of Cross-Modal Alignment and Audio Discrimination

Putting CLAP-style InfoNCE alignment and DWD-style class centroid discrimination into **the same pair of encoders** for joint training is the core innovation of the paper. The key lies in the fact that the two operate on **different geometric structures**: $L_{at}$ shapes the **cross-modal relative position** between the audio manifold and the text manifold, while $L_{aa}$ shapes the **clustering topology of the audio manifold itself**. The two are theoretically not orthogonal—in extreme cases, collapsing all audio into their respective text anchors could satisfy $L_{at}$, but the internal audio space would be a mess; conversely, optimizing only $L_{aa}$ would lose cross-modal retrievability. Joint optimization with weight balancing allows a single embedding space to possess both properties. The numbers in Table I directly verify this: when trained alone, CLAP achieves 74.83% IV / 91.68% OOV AP in the acoustic perspective; after adding DWD, this improves to 85.05% / 94.06%. **Both perspectives (acoustic, cross-modal) improve by about 10 percentage points**, while the cross-modal perspective’s score of 98.66% / 99.46% is not sacrificed by adding DWD (Table I).

### Key Technical Innovation 2: Reproducible Standardized Evaluation Framework

The paper’s third contribution is a complete evaluation recipe: standardized trial generation rules, explicit IV/OOV splits, clear trial counts (approx. 34.3 million IV pairs, 300,000 OOV pairs), released alongside the codebase (https://github.com/SIPLab-IITH/JMCL). The motivation is that existing literature often fails to specify these steps, making numbers incomparable. This protocol itself produces one of the most informative analyses in the paper—the mechanistic explanation for the counter-intuitive phenomenon where OOV AP is higher than IV AP (see Experimental Results section).

### Technical Differences from Existing Methods

Compared one-by-one with five baselines (baseline formulas are all given in the paper’s Section II):

- **vs Siamese-RNN**: The latter uses a triplet cosine hinge loss $L_{hinge} = \max\{0, m + d_{cos}(f(a),f(p)) - d_{cos}(f(a),f(n))\}$, comparing only one triplet at a time; JMCL uses in-batch softmax to normalize contrast against all $N$ class centroids, utilizing negative examples much more efficiently, and additionally carries cross-modal supervision.
- **vs CAE-RNN**: The latter uses an unsupervised reconstruction objective (reconstructing instance $b$ of the same word using the embedding of word $a$, $\sum_t \|x_t^{(b)} - f_t(X^{(a)})\|^2$), with absolutely no discriminative supervision. The IV AP of 7.24% (Table I) indicates that pure reconstruction is almost unusable in this setting; JMCL is fully supervised discriminative.
- **vs Multiview-RNN**: This is the closest baseline—also a dual-tower (audio BiLSTM + character BiLSTM) + contrastive framework, also supporting cross-view and same-view error pairing. The differences are threefold: JMCL’s audio-audio branch uses a **class centroid structured loss** (leave-one-out centroid + softmax + hardest negative hinge) rather than simple pairing distance; the temperature is learnable; and the loss weights $\alpha_1, \alpha_2$ are explicitly ablated (Table III). The performance difference is huge: in the pure acoustic perspective, JMCL achieves 85.05% IV AP compared to Multiview-RNN’s 55.47% (Table I).
- **vs Contrastive-RNN**: The latter is audio-audio InfoNCE (temperature-scaled cross-entropy, multiple negatives), lacking cross-modal capabilities; JMCL is equivalent to merging it with the CLAP objective and adding centroid constraints, outperforming it by about 10/17 percentage points in IV/OOV acoustic perspectives (Table I).
- **vs A2E-DWD**: The original source of the DWD loss, pure audio-domain supervision; JMCL adds a cross-modal shared space on top of it, achieving 85.05% vs 72.15% IV and 94.06% vs 79.83% OOV in the acoustic perspective (Table I), indicating that cross-modal supervision also helps pure audio discrimination.

## Experimental Results

### Datasets Used and Their Scale

- **Training**: LibriSpeech train-clean-100 subset, approx. 100 hours;
- **Evaluation**: test-clean (approx. 5.4 hours) and test-other (approx. 5 hours, containing more noise and speaker variation);
- **Word-level Segmentation**: Precise word boundaries obtained using Montreal Forced Aligner (MFA); subsequently filtered by duration, retaining only word instances of **0.5–2.0 seconds**—the rationale is that this is the common range in literature, excluding keywords that are too short/long to provide information for embedding learning. After filtering, **26,200 unique anchor words and 154,000 instances** are obtained.
- **Features**: Resampled to 16 kHz, Mel spectrograms use a 25 ms window length, 10 ms frame shift, and 128 Mel filterbanks, with a Hann window applied frame-by-frame to minimize spectral leakage.
- **Training Configuration**: PyTorch, batch size 128, AdamW (learning rate 1e-3, weight decay 1e-4), gradient clipping max norm 1.0, OneCycleLR cosine annealing + warmup covering 20% of total steps, 30 epochs in total, random sampling of positive and negative examples, all experimental settings consistent.

### Definition and Rationale for Evaluation Metrics

- **Lexical Discrimination Task (Intrinsic Quality)**: After embedding each word segment in the test set, cosine distances are calculated pairwise, thresholds are swept to generate PR curves, and **AP (Average Precision)** is used as the metric. The reason for choosing AP is that it is insensitive to thresholds, summarizes the entire PR curve, and is suitable for situations where different models operate at different points.
- **Two Perspectives**: Acoustic perspective (both sides pass through the audio encoder, simulating QbE) and Cross-modal perspective (audio embeddings directly compared with text embeddings, simulating text-KWS).
- **IV/OOV Split**: Words appearing in the training set are IV in the test set, otherwise OOV. Trial generation rules: $N$ instances of each anchor word are paired two-by-two as positive pairs; each anchor word instance is paired with all non-anchor word instances as negative pairs. This yields approx. **34.3 million IV word pairs and 300,000 OOV word pairs**.
- **STD/KWS Tasks (Extrinsic Tasks)**: Continuous test audio is windowed at fixed window lengths (0.2/0.3/0.4/0.6 seconds) with fixed steps, and **EER (Equal Error Rate)** is reported. The paper explicitly explains why task-level EER is generally worse than lexical discrimination AP: fixed window boundaries may cut off the target word, miss part of the word, or pack multiple words into one window.
- **Qualitative Analysis**: t-SNE (Fig. 2, audio embeddings of 10 spoken instances for each of 10 keywords + text embeddings of corresponding phoneme queries) and KDE similarity distributions (Fig. 3, MFA precise segmentation vs. fixed segmentation of 0.3 second window / 0.15 second step).

### Detailed Comparison with Baseline Methods and SOTA

**Lexical Discrimination (Table I, test-clean, AP%)**. Acoustic perspective: CAE-RNN 7.24 IV / 43.64 OOV; Siamese-RNN 57.12 / 82.84; Contrastive-RNN 74.94 / 77.22; A2E-DWD 72.15 / 79.83; Multiview-RNN 55.47 / 81.72; CLAP (ablation version of this paper without DWD) 74.83 / 91.68; **CLAP+DWD (complete model) 85.05 / 94.06**. Cross-modal perspective: Multiview-RNN, CLAP, and CLAP+DWD all achieve 98.66 / 99.46. Three key readings:

1. DWD brings a stable improvement of about 10 percentage points to the acoustic perspective (IV 74.83→85.05, OOV 91.68→94.06), confirming the value of joint optimization;
2. Adding DWD to the cross-modal perspective yields **zero gain and zero cost** (98.66/99.46 remains unchanged)—the paper explains itself: when both training and evaluation objectives are audio-text alignment, the audio-audio discrimination of DWD is redundant, but it has no side effects;
3. The counter-intuitive phenomenon of **OOV being comprehensively higher than IV** is given a mechanistic explanation by the paper: the IV set has approx. 34 million word pairs, while OOV has only approx. 300,000; the IV evaluation grid is much denser. Additionally, IV anchor words have an average of 5.64 test instances (having seen at least 100 instances during training), while OOV anchor words have an average of only approx. 1.4 instances—a smaller, sparser OOV evaluation set provides fewer difficult positive and negative examples for each anchor word, systematically inflating AP; the denser sampling of the IV set constitutes a harder discrimination task. This analysis is a byproduct of the standardized trial protocol and is the most honest and informative part of the entire paper.

The discrete patterns within the baselines are also worth reading separately: CAE-RNN’s IV crashing to 7.24% is not accidental—the unsupervised reconstruction objective only requires the encoder to retain decodable information, placing no direct pressure on discriminating "same word, different speakers," and it is prone to memorizing surface details on the densely seen vocabulary in training; yet its OOV is 43.64%, indicating that models lacking discriminative pressure behave completely unpredictably across the two distributions. Contrastive-RNN is the only model in the field where IV and OOV are almost flattened (74.94 vs 77.22, a gap of only 2.3 percentage points), indicating that the discriminative surface learned by pure audio-audio InfoNCE generalizes very evenly across the vocabulary, but its ceiling is also low; CLAP pulls OOV up to 91.68%, indicating significant generalization dividends from cross-modal supervision, while CLAP+DWD achieves the highest scores on both sides, indicating that the combination of these two supervisions happens to be complementary—cross-modal supervision provides generalized semantic anchors, while audio-audio discrimination provides class structure.

**QbE-STD (Table II, EER%)**. On test-clean, the complete model achieves the lowest EER in most window lengths: the best overall for IV queries is **15.71% at the 0.3 second window** (CLAP alone is 16.34%); for OOV queries, CLAP achieves the lowest 18.00% (0.4 second window), with this paper’s 18.35% closely following, while all other baselines are significantly worse (e.g., at the 0.4 second window, Siamese-RNN OOV 22.88%, Contrastive-RNN 34.57%, A2E-DWD 32.34%, Multiview-RNN 30.49%). On test-other, EER worsens for all methods: at the 0.4 second window, CLAP is 23.40% IV / 18.21% OOV, while CLAP+DWD improves IV to **22.20%**, with only marginal change in OOV (18.56%); the complete model is consistently lower across all window lengths, and the performance gap between test-clean and test-other is narrower—the paper uses this to argue that the joint objective is more robust under mismatch conditions.

**KWS (Table II lower part, only three cross-modal systems are comparable)**. This is the least pretty result of the paper, reported honestly: on test-clean, CLAP itself outperforms Multiview-RNN (e.g., 36.21% vs 44.32% IV at the 0.3 second window), but **adding DWD causes degradation**—CLAP+DWD reaches as high as 51.40% IV at the 0.3 second window, and 43.36% at the 0.6 second window, with no improvement in OOV (36.00%–40.02% range). On test-other KWS, the two have mixed results (e.g., at the 0.6 second window OOV, CLAP+DWD 53.45% is slightly better than CLAP 54.48%, but at the 0.3 second window IV, 46.30% is worse than CLAP 41.65%). The paper attributes the degradation to: DWD acts only on audio embeddings and does not enhance cross-modal alignment, instead squeezing cross-modal discriminative power. This negative result, combined with the positive result on the STD side, perfectly exposes the boundaries of "joint optimization": **gains occur in audio-side dominant tasks (QbE-STD), while costs occur in cross-modal dominant tasks (text-KWS)**.

**Qualitative Results**. The t-SNE in Fig. 2 shows that audio embeddings (dots) for each keyword cluster tightly around the corresponding phoneme query’s text embedding (crosses), with clear separation between clusters of different keywords. The KDE analysis in Fig. 3 quantifies the impact of segmentation quality: with MFA precise alignment, the mean similarity of positive pairs is approx. **0.89**, and negative pairs approx. **0.07**, with a clear gap; switching to fixed 0.3 second window segmentation causes the positive pair mean to drop to approx. **0.45** with obvious distribution divergence—confirming that fixed windows cutting off keywords is the root cause of STD/KWS EER being far worse than lexical discrimination AP.

### Findings from Ablation Experiments

**Loss Weight Ablation (Table III, IV/OOV Lexical Discrimination AP%)**:

| $\alpha_1$ (CLAP term) | $\alpha_2$ (DWD term) | IV | OOV |
|---|---|---|---|
| 1.0 | 0.1 | 78.40 | 72.88 |
| 1.0 | 0.5 | 80.45 | 68.86 |
| 1.0 | 1.0 | 82.37 | 87.21 |
| **0.1** | **1.0** | **85.05** | **94.06** |
| 0.5 | 0.1 | 84.48 | 91.41 |

Three findings: First, the optimal configuration is **significantly down-weighting the cross-modal term** ($\alpha_1=0.1$), indicating that in word-level discrimination, audio-domain structural constraints are the dominant bottleneck, and cross-modal supervision acts more like a regularization signal than a primary objective; Second, OOV is far more sensitive to weight combinations than IV—when $\alpha_1=1.0, \alpha_2=0.5$, OOV crashes to 68.86% (25.2 percentage points lower than optimal), while IV only drops from 85.05 to 80.45; Third, in the direction of increasing $\alpha_2$ from 0.1 to 1.0, performance improves monotonically, with no inflection point where "audio-audio term too strong causes collapse" (at least on word discrimination metrics). Additionally, the ablation of the DWD term itself is handled by the two rows of CLAP and CLAP+DWD in Table I, showing the aforementioned approx. 10 percentage point improvement. The paper does not report model parameter count, computational cost, or inference latency.

## Main Contributions

1. **The first joint multimodal contrastive learning framework that unifies CLAP-style audio-text contrastive alignment and DWD-style audio-audio discrimination into a single shared embedding space** (authors claim "to our knowledge, this is the first comprehensive approach of its kind"), with one set of encoders serving both QbE-STD and text-KWS query modalities;
2. **Flexible cross-modal capability**: The same model supports two inference paths: acoustic perspective (pure audio comparison) and cross-modal perspective (audio-text comparison), achieving 85.05% IV / 94.06% OOV AP in lexical discrimination acoustic perspective, 98.66% / 99.46% in cross-modal perspective (Table I), and 15.71% EER at 0.3 second window in STD task (Table II);
3. **Reproducible standardized evaluation framework**: Unified trial generation rules, explicit IV/OOV splits, trial counts of 34.3M/0.3M, and a recipe released with the codebase, directly addressing the long-standing problem of incomparable evaluations in AWE literature;
4. **Mechanistic explanation of evaluation bias**: Reveals and explains that "OOV AP higher than IV" is a statistical illusion caused by imbalance in trial density and instance counts, providing practical reference value for evaluation design in subsequent work.

## Limitations and Future Work

### Technical Limitations of the Method

- **Negative Transfer on Cross-Modal Tasks**: Joint optimization yields clear gains on QbE-STD, but systematically harms performance on text-KWS (Table II, test-clean KWS IV EER worsens from CLAP’s 36.21% to 51.40% @0.3 second window). This means that "unified model" actually requires task-aware adjustment of loss weights, while the paper only provides a single configuration oriented toward word discrimination/QbE ($\alpha_1=0.1, \alpha_2=1$), without ablation of weights separately on STD/KWS.
- **Dependence on Fixed Window Segmentation, Endpoint Sensitivity Unresolved**: Fig. 3 shows that fixed windows cause the mean similarity of positive pairs to drop from 0.89 to 0.45, which is the root cause of persistently high task-level EER. The paper only diagnoses the problem without proposing remedies like boundary detection or multi-scale fusion—this is precisely a weakness criticized since the DTW era and still present in the AWE era.
- **Supervision Dependency**: The method requires word-level boundaries (MFA forced alignment) and phoneme transcriptions for training supervision, essentially making it a supervised method; the zero-labeling advantage of the unsupervised CAE-RNN route is abandoned, and the paper does not discuss usability degradation in zero-resource scenarios.
- **Capacity and Structure Not Explored**: The encoder is fixed at 3-layer BiLSTM-256, with no comparison to conformer/transformer or wav2vec2-style self-supervised pretraining features; in-batch negative examples are randomly sampled, with no hard negative mining strategy; the paper does not report the value chosen for the number of positive examples $M$ per anchor word.
- **Missing Efficiency Metrics**: As a work in the AWE route主打 "efficiency," parameter count, FLOPs, embedding extraction latency, and retrieval index structure (e.g., ANN acceleration) are all unreported.

### Shortcomings in Experimental Design

- **Single Corpus, Single Language**: All experiments are conducted on LibriSpeech (English audiobook reading speech), and test-other only adds noise to the reading style, lacking real far-field/noisy/multi-speaker training or evaluation. The boundary of the "robust" conclusion only covers laboratory noise;
- **Self-Contradiction in Evaluation Protocol**: While pointing out that OOV’s high AP is artificially inflated due to sparse trials (average 1.4 instances/anchor word vs. IV’s 5.64), the paper still uses the same protocol to report 94.06% OOV AP as a selling point. The comparability issue between the two numbers is not further addressed (e.g., normalization by trial density or resampling for balance);
- **Incomplete KWS Baseline Coverage**: The KWS section of Table II only includes three systems: Multiview-RNN, CLAP, and CLAP+DWD. The other four baselines (Siamese, CAE, Contrastive, A2E-DWD) are absent due to lack of cross-modal capability. The horizontal comparison on the cross-modal side actually has only one true external opponent: Multiview-RNN;
- **Window Length Only Swept to 0.6 Seconds**: For training words filtered to 0.5–2.0 seconds, there is no data on how 2-second words perform at the 0.6 second window (worst-case scenario).

### Possible Directions for Future Improvement

- **Task-Adaptive Loss Scheduling**: Given the differing weight sensitivities of text-KWS and QbE-STD, one could fine-tune $\alpha_1/\alpha_2$ per task, or conditionally branch after training (e.g., two lightweight heads sharing the encoder), preserving the unified model while eliminating negative transfer on KWS;
- **Endpoint/Segmentation Frontend**: Introducing a lightweight word boundary detector or multi-window fusion to directly attack the problem of fixed windows causing 0.89→0.45 similarity collapse is the most direct lever to push EER down from the 15–20% range;
- **Self-Supervised Pretraining Foundation**: Replacing handcrafted Mel features with wav2vec2/HuBERT-style features, or performing audio-text contrastive pretraining on large-scale unlabeled audio before fine-tuning on 100-hour-level data, could simultaneously improve OOV generalization and noise robustness;
- **Zero-Resource and Multilingual Transfer**: Baseline literature already includes multilingual transfer and unsupervised CAE routes. Combining the joint multimodal objective with pseudo-positives mined via unsupervised word discovery is a natural extension toward zero-resource scenarios;
- **Hard Negative Mining and Larger Batches**: Replacing in-batch random negatives with confusion-word-aware hard negatives (especially homophones and co-occurring words), combined with large batches, could further widen inter-class boundaries;
- **Edge Deployment Evaluation**: AWE retrieval is essentially embedding comparison, naturally suited for quantization + ANN index deployment to edge devices. Completing reports on parameter count/latency and对接 quantization (e.g., binarization schemes like BiFSMN) would give this technical route engineering persuasiveness for embedded KWS products.
