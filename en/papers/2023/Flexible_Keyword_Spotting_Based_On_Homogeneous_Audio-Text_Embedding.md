# Flexible Keyword Spotting based on Homogeneous Audio-Text Embedding

- **Authors/Affiliations**: Kumari Nishu, Minsik Cho, Paul Dixon, Devang Naik (Apple)
- **Date**: August 2023 (arXiv:2308.06472v1, submitted August 12, 2023)
- **Link**: https://arxiv.org/abs/2308.06472
- **Keywords**: flexible keyword spotting, homogeneous audio-text embedding, phoneme-to-vector, grapheme-to-phoneme, confusable keyword generation, conformer, edge-side lightweight deployment

## Problem Statement

### Problem Background and Domain Pain Points

The task of Keyword Spotting (KWS) is to detect user-specified wake-up words from a continuous speech stream. Based on how keywords are determined, the field divides them into two main categories: fixed KWS, where the keyword set is locked during training and the model is essentially a closed-set classifier; and the more difficult user-defined/flexible KWS, where keywords are arbitrarily specified by the user after deployment. The model must handle words it has never seen during training, determining whether a segment of audio contains that specific word (Section 1). Flexible KWS is inherently more difficult: arbitrary keywords cannot be enumerated into fixed classes, and a large number of user words are simply not in the training set. The model has no direct supervision on what these words "sound like," yet it must provide real-time judgments on streaming audio.

Typical carriers for flexible KWS are always-on listening devices such as smart speakers, smartphones, and earbuds, which impose rigid engineering constraints: the model must be low-power and always-on, process streaming frames sequentially, and have a sensitive installation package size. Under these constraints, "how keywords are registered into the system" becomes a design watershed. Previous work falls into two camps: one camp requires users to read out the keyword and register it as an audio signal, i.e., query-by-example (QBE) methods [4, 3, 7]; the other camp [5, 6] accepts keywords registered in text form. Section 1 explicitly points out the advantages of text registration: the user interface is more friendly (typing is easier than reading aloud and is not affected by registration environment noise), and it is most compatible with practical low-power streaming detection applications—text registration only requires converting text to a cacheable embedding at registration time, while the inference path only computes audio-side operations frame-by-frame; whereas speech registration methods (e.g., [8, 9, 10] using DTW to measure similarity between registered and query audio) require continuous sequence alignment calculations on sliding windows, imposing high edge-side overhead, and the quality of the registered audio directly caps the system's upper limit.

However, the text registration camp (audio-text embedding methods) carries two heavy burdens of its own, which is exactly what this paper aims to address: embedding mismatch caused by modality heterogeneity, and the sheer size of the text encoder itself.

### Specific Shortcomings of Existing Methods

**Shortcomings of Speech Registration Methods (query-by-example) [4, 3, 7, 8, 9, 10]**: The registration process is cumbersome, requiring users to read out keywords personally, resulting in poor registration quality in noisy environments; [4] relies on an ASR model to extract embeddings for both registered and query audio, making the model bulky; [8, 9, 10] use Dynamic Time Warping (DTW) to compare registered and query embeddings pair-wise, incurring high overhead for sequence alignment calculations on streaming edge devices. Although these methods naturally avoid cross-modal mismatch issues (both sides are audio embeddings), the user experience and computational cost are suboptimal.

**Shortcomings of Text Registration Dual-Encoder Methods [5, 6]**: The paper dissects three points in Section 1 and Fig. 1:

- **Heterogeneous modality representations lead to large embedding mismatch**. Audio encoders are trained on speech data, and text encoders are trained on text corpora; the embeddings output by two independently trained encoders reside in completely different spaces. The abstract explicitly names this as *heterogeneous modality representation* (i.e., large mismatch) and points out that it leads to decreased accuracy. Fig. 1(a) further illustrates that even with projection layers added for alignment, projection methods may still create large mismatches between the two embedding spaces—projection is a learned post-hoc remedy, and residual mismatch cannot be eradicated.
- **Alignment mechanisms introduce extra parameters and package size**. Jointly processing embedding vectors from two independently trained modality encoders (and thus in different spaces) requires a transformation mechanism (such as projection) to map audio and text embeddings into a joint space. The paper notes that this increases parameters and package size at deployment (Section 1). This is a tangible cost for edge-side products.
- **The text encoder itself is expensive**. Specifically, [6] uses DistilBERT [19] as the text encoder, with 66M parameters (Section 3.2)—more than an order of magnitude larger than many edge-side KWS entire models.

**No specific mechanism for false alarms**. The core source of false alarms is phonetic confusability: user-defined keywords collide with a non-target word with a similar pronunciation, causing the system to wake up falsely (Section 2.4). Fixed KWS, having fixed classes, naturally obtains discriminative boundaries between classes during training; whereas flexible KWS models have no fixed classes internally, and many user words are not even in the training set, making it impossible to discuss the model's immunity to confusion for these words. Existing text registration methods [5, 6] lack specific training mechanisms for real-world phonetic confusability.

### Key Challenges to be Solved by This Paper

Condensing the above pain points into three buttons that must be untied simultaneously: First, how to make text embeddings and audio embeddings **naturally homogeneous** (homogeneous), eliminating projection layers and cross-modal mismatch from the root, rather than relying on post-hoc projection to patch; Second, the text branch must be **extremely lightweight**, ideally with zero additional parameters, to match the package size and memory constraints of edge-side deployment; Third, in a setting with no fixed classes and a large number of unseen keywords, how to make the model **gain discriminative power against phonetic confusability** to suppress false alarms. The common solution to these three challenges is the combination of "audio-compliant text encoder + confusable keyword generation" proposed in this paper.

## Methodology

### Overall Architecture Design and Design Motivation

The proposed model is called CED (Common Embedding based Detector), with the overall architecture shown in Fig. 2: it takes an audio-text pair as input and outputs a verification score to determine if the audio content matches the text. CED consists of three modules—an audio encoder, an audio-compliant text encoder, and a verifier. The input sample is denoted as $(a, t, l)$: audio $a = (a_1, a_2, ..., a_{n'})$ is a sequence of audio frames, text $t = (t_1, t_2, ..., t_{m'})$ is a sequence of words, and $l$ is a binary label, where $l = 1$ indicates a positive sample pair (Section 2).

Training proceeds in three steps (Section 2): Step 1 trains the audio encoder's phoneme prediction task using CTC loss [14]; Step 2 builds a Phoneme-to-Vector (P2V) database using the trained audio encoder; Step 3 performs end-to-end discriminative training on CED. The design motivation for these three steps is worth expanding:

- **Why choose conformer for the audio encoder and train it with a phoneme prediction task** (Section 2.1): This is the hub of the entire homogeneous design. The small conformer [12] combines self-attention layers [13] and convolutional layers to capture both global and local audio context; the phoneme prediction training objective forces the encoder's embedding space to be organized by phonemes—the embedding of each frame has predictive power for a certain phoneme. Only when the embedding space is "phonemized" is the subsequent step of "looking up phoneme vectors from the audio space" possible. In other words, the phoneme prediction task is the bridge that pulls the text side into the audio space.
- **Why freeze the audio encoder in the end-to-end stage** (Section 3.2: audio encoder frozen when training CED, using only cross-entropy loss to train the verifier): The P2V table is statistically derived from the embedding space of this specific audio encoder; once encoder weights are updated, the entire table becomes obsolete, and homogeneity collapses. Freezing is an engineering trade-off that sacrifices joint fine-tuning on the audio side to ensure the continued validity of the P2V table. This is a typical "trade freezing for stability" engineering compromise, with all compensation responsibilities placed on the verifier.
- **Why the text encoder can be parameter-free**: The text link is split into two segments—text to phoneme uses a pre-trained G2P model [11] (0.83M parameters), and phoneme to vector uses P2V lookup (a lookup table of 74 d-dimensional vectors, with no trainable parameters). The text embedding is calculated once during registration and cached; there is no computation of the text encoder in the frame-by-frame inference path.

### Mathematical Principles of Core Algorithms

**Audio Encoder Side** (Sections 2.1, 3.2): Input features are 80-channel filterbanks, with a 25ms window length and 10ms frame shift; conformer hyperparameters are 6 encoder layers, encoding dimension $d = 144$, kernel size 3, and 4 attention heads. The conformer contains subsampling layers, so the embedding sequence length $n <$ input frame count $n'$. Audio embeddings are denoted as $e = (e_1, e_2, ..., e_n)$, where $e_i \in \mathbb{R}^d$. Training uses the Adam optimizer [18] with transformer learning rate scheduling [13], 5k steps warm-up, for a total of 150 epochs.

**Mathematics of P2V Database Construction** (Section 2.2): Run the trained audio encoder in evaluation mode on the Libriphrase training set. For sample $(a, t)$, the audio embeddings $e$ output by the conformer pass through the final linear layer of the audio encoder, obtaining prediction scores $s = (s_1, s_2, ..., s_n)$ for all phonemes per frame, where $s_i \in [0,1]^{|P|}$, and the size of the phoneme set $|P| = 74$ (G2P vocabulary). Perform greedy decoding: take the phoneme with the highest probability for each frame, then remove consecutive duplicate phonemes, obtaining the predicted phoneme sequence $\hat{p} = \hat{p}_{j_1}, \hat{p}_{j_2}, ..., \hat{p}_{j_{m'}}$, where frame indices satisfy $1 = j_1 < j_2 < ... < j_{m'} \le n$. Use CER (Character Error Rate) to measure the quality of $\hat{p}$ against the ground truth phoneme sequence. **The sampling rule is to keep only samples with CER = 0, then randomly sample about 50K entries, denoted as dataset D**. For each phoneme $\hat{p}_{j_i}$ in the predicted sequence, backtrack to the corresponding frame interval $[l, r]$ in the audio embeddings $e$, defining the local vector for this occurrence as:

$$LV(\hat{p}_{j_i}) = \frac{1}{r-l} \sum_{k=l}^{r} e_k$$

That is, the average of all audio frame embeddings covered by this phoneme (transcribed from the original formula; strictly speaking, there are $r-l+1$ terms in the sum while the denominator is $r-l$, which is suspected to be a typo, but does not affect the semantics of "interval average"). Then define a global vector $GV(p) \in \mathbb{R}^d$ for each phoneme $p \in P$: $GV(p)$ is the average of all local vectors for $p$ in D across samples. Finally, store all 74 $GV(p)$s in the P2V database.

**Mathematics of the Verifier** (Section 2.3): Text embedding $f$ is obtained by looking up the P2V table for each phoneme in the text's phoneme sequence and concatenating them. The verifier receives audio embeddings $e$ and text embeddings $f$, first calculating the cosine similarity matrix between the two. The key observation is that positive sample pairs should exhibit a **monotonic staircase alignment pattern** in the cosine matrix—one phoneme is associated with one or more consecutive audio frames (this is precisely the structural property of CTC frame-level prediction). The paper borrows the DSP (Dynamic Sequence Partitioning) algorithm proposed in [6] to find this alignment path, then keeps only the similarity weights on the alignment path and masks out the rest of the cosine matrix, forcing order matching between audio and text. The dot product of the masked cosine similarity matrix ($m \times n$) and the audio embeddings ($n \times d$) yields an $m \times d$ audio-text consistency matrix ($m$ is the phoneme sequence length, $d$ is the embedding dimension)—each phoneme position gets an audio frame vector aggregated and weighted by alignment similarity. This output connects to a single-layer GRU, then to a feed-forward layer, outputting the final matching score for this input pair.

Why is masking necessary? Because keyword detection is essentially a temporal alignment problem. Without order constraints, matching phonemes in a "bag-of-words" manner, words with the same set of phonemes but different orders (anagrams) cannot be distinguished. Restricting matching to a monotonic alignment path injects the prior that "speech is spoken in order" into the verifier.

### Key Technical Innovation 1: Parameter-Free Audio-Compliant Text Encoder (G2P + P2V Lookup)

This is the core innovation of the paper (Contribution 1, Section 1). The deep logic can be deconstructed from three levels:

- **Source of Homogeneity**. The root cause of heterogeneous mismatch is that two encoders are trained independently, each possessing its own space. The solution of this method is not to "train a better projection," but to have text embeddings directly **derived from the phoneme embedding space learned by the paired audio encoder**—the vectors in the P2V table are statistical averages of the embeddings produced by the audio encoder itself at these word positions. Text and audio thus naturally share the same metric space, cosine similarity is directly comparable, and no transformation layer is needed. Fig. 1(b) illustrates this clearly: P2V reduces the mismatch between the two embedding spaces and requires no additional text encoder.
- **Phonemes as Shared Atomic Units**. Text and audio have no intersection at the symbol level, but G2P converts orthography (spelling) into a pronunciation sequence, bringing text into the same discrete symbol space (74 phonemes) as the audio encoder's prediction target; P2V then maps this discrete space to the audio encoder's continuous embedding space. Two steps relay, with zero learned parameters.
- **Why the sampling rule is CER = 0**. Only when decoding is completely correct does the frame-to-phoneme backtrack interval truly correspond to frames of correct pronunciation, making the calculated local vector a clean, uncontaminated estimate; samples with decoding errors average in misaligned frames, contaminating the global vector. The cross-sample average of about 50K zero-error samples serves to smooth out context differences—the realization of the same phoneme varies greatly in different speech contexts (coarticulation), and after averaging, it yields what the paper calls the "most general representation" of each phoneme (Section 1).

**Qualitative Evidence of Quality** (Fig. 3): The authors plot 100 randomly sampled local vectors colored by phoneme in t-SNE, with each subplot showing 10 phonemes. The local vectors exhibit high intra-class compactness and inter-class separation; phonemes with the same vowel symbol but different stress marks (OW0, OW1, OW2, top-left subplot) have closer inter-class distances than other phonemes, but are still separable. This shows that global averaging has not completely erased stress differences (retaining separability) but has indeed flattened some details—this point will be revisited in the Limitations section.

**Scale Accounting**: 74 phonemes $\times$ 144 dimensions = 10,656 floating-point numbers, estimated at about 42KB for the lookup table in fp32, with lookup complexity $O(m)$. The text branch adds zero parameters besides G2P (0.83M); compared to DistilBERT's 66M in [6], the text branch parameter count is reduced by about 80 times (estimated). CED total model parameters are 3.8M (Section 3.2; it is not explicitly stated whether 3.8M includes G2P).

### Key Technical Innovation 2: Confusable Keyword Automatic Generation (Confusable Keyword Generation)

Addressing the false alarm challenge (Contribution 1, Section 1; Section 2.4, Fig. 4 right side, using keyword "stop" as an example). The generation algorithm takes a keyword as input and outputs its confusable variants, in four steps:

1. Select edit distance $\delta$, representing the number of phoneme edits between the generated word and the original word; the paper suggests $\delta \in \{1, 2, 3\}$ to construct hard negative samples;
2. Randomly select $\delta$ positions $u_1, u_2, ..., u_\delta$ in the keyword's phoneme sequence;
3. For each position, randomly choose one of two transformations: "replace" or "insert";
4. For each position $u_i$, randomly select a phoneme different from the current phonemes at $u_{i-1}$, $u_i$, and $u_{i+1}$, and execute the transformation.

Two "whys" are worth chewing on. **Why constrain the new phoneme to be different from the three adjacent positions**: If replaced with a phoneme identical to $u_i$, the edit is a no-op, and the negative sample degenerates into a positive sample; if identical to adjacent phonemes, this edit is likely swallowed by pronunciation (analogous to CTC's consecutive repetition folding), making the generated "hard negative sample" acoustically almost indistinguishable from the original word, losing training value. **Why cap $\delta$ at 3**: The larger the edit distance, the greater the pronunciation difference between the generated word and the original word, and the "easier" the negative sample becomes; taking small values for $\delta$ (1 to 3) ensures the negative sample falls exactly near the decision boundary, providing information for discriminative training.

The value of this mechanism lies in its **automatic construction within the training stream**, requiring no collection of real confusable word data: flexible KWS has no fixed classes, and a large number of user words are unseen, making enumeration of real confusable pairs infeasible; synthesizing by phoneme edit distance exactly simulates the distribution of "non-target words with similar pronunciation," the main source of false alarms.

### Key Technical Innovation 3: Discriminative Batch Construction and End-to-End Training (Discriminative Setting)

Contribution 3 in Section 1 and Fig. 4 left side describe the training batch construction: take keywords from the Libriphrase training set to form a training batch of size 32; for each keyword in the batch, configure three mini-batches, each of size 11—the positive sample set (11 audio clips of that keyword paired with correct text), the random negative sample set (11 audio clips of other words), and the confusable negative sample set (**the exact same 11 audio clips as the positive sample**, but paired with automatically generated confusable keyword text). By this calculation, a single batch contains $32 \times 3 \times 11 = 1056$ audio-text pairs.

The key design lies in the third group: confusable negative samples share the same audio as positive samples. The same audio faces true text (positive label) on one side and fake text differing by only one or two phonemes (negative label) on the other, forming the most direct contrast structure—the model is forced to draw boundaries between "stop" and its phonetic neighbors, rather than passing easily with the coarse-grained differences of random negatives. This is the end-to-end discriminative training setting mentioned in the paper: the audio encoder (frozen) + audio-compliant text encoder + verifier trio are all exposed to this discriminative objective (the verifier is trained with cross-entropy loss, Section 3.2).

### Technical Differences from Existing Methods

| Dimension | Speech Registration query-by-example [4, 8, 9, 10] | Text Registration Dual-Encoder [5, 6] | This Paper CED |
|---|---|---|---|
| Registration Form | Read-out registered audio | Text | Text |
| Text Representation | No text encoder | General text encoder (DistilBERT 66M) | G2P (0.83M) + P2V Lookup (Parameter-Free) |
| Spatial Relationship | Both audio embeddings, but require DTW/ASR embedding comparison | Heterogeneous spaces, require projection layer alignment, residual mismatch | Homogeneous space (P2V derived from audio encoder itself), zero projection |
| Confusion/False Alarm Handling | No specific mechanism | No specific mechanism | Phoneme edit confusable keyword generation, same-audio hard negative contrast |
| Model Parameters | Not reported in paper | Text branch 66M | CED total 3.8M |

The essence of the difference: [5, 6] follows the route of "admitting two spaces and aligning post-hoc," while this paper follows the route of "having only one space from the start"—text embeddings are not guests projected into the audio space, but native residents growing directly from the audio space.

## Experimental Results

### Datasets Used and Their Scale

- **LibriSpeech [16] and its derivative Libriphrase** (Section 3.1): Following the construction process of [5, 6], the Libriphrase training set is built from train-clean-100/360, and the test set from train-others-500; the test set is divided into LE (Libriphrase Easy) and LH (Libriphrase Hard) parts, the difference between the two lies in the difficulty of constructing negative sample pairs (details in references [5, 6], not restated in this paper; LH negatives are hard samples with similar pronunciation). The exact number of samples in Libriphrase is not reported in the paper and requires checking the original literature.
- **Speech Commands V1** [17]: Take 10 short commands from the test set, **without any fine-tuning** (without any fine-tuning) for direct evaluation, used to measure generalization across speech characteristics, and compared with baselines [5] evaluated under the same setting.
- **P2V Construction Subset D**: Randomly sample about 50K entries with CER = 0 from the Libriphrase training set (Section 2.2).
- **Training Process** (Section 3.1): First train the audio encoder on longer audio from train-clean-100/360, then fine-tune on shorter audio from Libriphrase, and finally train CED end-to-end on the Libriphrase training set.
- **Experimental Environment**: PyTorch, x86 Linux machine with NVIDIA V100 GPU (number of GPUs not reported in the paper).

### Definition and Rationale for Evaluation Metrics

- **AUC (Area Under the ROC Curve, %)**: A threshold-independent overall discriminative power metric. Flexible KWS is modeled as an audio-text verification task, outputting a matching score rather than a closed-set class probability, so there is no "accuracy" to report; AUC measures the global ranking quality of positive and negative pair scores, making it the natural choice for open-set verification tasks.
- **EER (Equal Error Rate, %)**: The error rate when the false acceptance rate and false rejection rate are equal, corresponding to a typical operating point where costs of the two types of errors are balanced, and is a standard metric in the KWS and speaker verification communities, directly reflecting the false wake-up level at deployment.
- **Reason for reporting both metrics**: AUC looks at overall ranking, EER looks at specific operating points; viewing them together avoids selective optimism from a single metric.

### Detailed Comparison with Baseline Methods and SOTA

Main results are all in Table 1 (LH = Libriphrase Hard, G = Speech Commands V1, LE = Libriphrase Easy; † denotes model without confusable module, * denotes model with confusable module):

| Method | AUC LH ↑ | AUC G ↑ | AUC LE ↑ | EER LH ↓ | EER G ↓ | EER LE ↓ |
|---|---|---|---|---|---|---|
| [5] | 73.58 | 81.06 | 96.7 | 32.9 | 27.25 | 8.42 |
| [6] | 84.21 | - | 97.83 | 23.36 | - | 7.36 |
| Ours† | 89.2 | 93.16 | 99.94 | 18.4 | 14.05 | 0.8 |
| Ours+conf* | 92.7 | 93.94 | 99.84 | 14.4 | 13.45 | 1.7 |

([6] did not report results on the G dataset, marked as "-" in the table.)

Relative improvements at the end of Table 1 (§ notes explicitly: * model calculated relative to [6] (LE/LH) and [5] (G)):

- **LH (Core Battlefield)**: AUC relative improvement 10.1%, EER relative improvement 38.3%. In absolute numbers (Abstract): AUC increased from 84.21% to 92.7% (+8.49 percentage points), EER decreased from 23.36% to 14.4% (−8.96 percentage points).
- **LE**: AUC relative improvement 2.05%, EER relative improvement 76.9% (7.36 → 1.7).
- **G (Zero-Fine-Tuning Cross-Domain Generalization)**: AUC relative improvement 15.9% (81.06 → 93.94), EER relative improvement 50.6% (27.25 → 13.45). This was achieved on command word data with completely different speech characteristics, without any fine-tuning, showing that the phoneme structure learned by homogeneous embeddings has cross-domain robustness—neither the P2V table nor the audio encoder was adapted for SC-V1.

**Efficiency Comparison** (Section 3.2): CED total parameters 3.8M; text encoder adds zero parameters besides G2P's 0.83M, compared to DistilBERT's 66M used in [6]. Inference latency, RTF, runtime memory, and package size increment are not reported in the paper—parameter count is the only direct evidence of lightweighting.

### Findings from Ablation Studies

The only systematic ablation in the paper is the confusable keyword module (Table 1, Ours† trained without confusable negative mini-batch vs Ours+conf*):

- **LH**: AUC 89.2 → 92.7 (+3.5), EER 18.4 → 14.4 (−4.0), largest gain; the paper explicitly points out that removing the confusable module causes degradation on both LH and G.
- **G**: AUC 93.16 → 93.94 (+0.78), EER 14.05 → 13.45 (−0.60), small gain.
- **LE**: AUC 99.94 → 99.84 (−0.10), EER 0.8 → 1.7 (+0.9), **slight degradation**.

Three interpretations (first and third points are author's analysis):

1. **Why LH gains the most**: LH's negative sample pairs are constructed as hard samples with similar pronunciation (Libriphrase's Hard definition), sharing the same distribution as that simulated by confusable keyword generation; training distribution aligns with test distribution, so gains are naturally maximized. The confusable module is essentially a vaccine customized for phonetic confusion scenarios like LH.
2. **Why G also gains**: There are also pairs of words with similar pronunciation among command words (e.g., "stop", "up"), so the discriminative power of confusable training can transfer.
3. **Why LE suffers slightly**: LE's negative samples are easy-to-distinguish random pairings, where the model was already approaching full marks (Ours†'s EER was already 0.8); introducing synthetic hard negatives changes the training distribution, pushing the decision boundary from an "easy-to-separate position" to a more conservative one, paying a cost of about 0.9 points EER on the easy test set. This is a trade-off of precision for robustness, which the paper does not discuss.

Additionally, **Fig. 3's t-SNE can be viewed as a qualitative ablation of P2V construction quality**: 100 random local vectors exhibit high intra-class compactness and inter-class separation, supporting the path of "statistically phoneme vectors from audio embedding space"; OW0/OW1/OW2 stress variants are closer to each other but separable, showing that global averaging flattens some details without destroying separability.

**Ablations Not Done by the Paper** (all noted as not reported in the paper): Audio encoder layers and dimensions (6 layers / d=144), phoneme table size (74), sensitivity of confusable edit distance $\delta$, P2V construction sample size (50K), comparison of DSP alignment with other alignment methods, frozen audio encoder vs joint fine-tuning, proportion of confusable mini-batch in the batch, etc.

## Main Contributions

1. **Parameter-Free Audio-Compliant Text Encoder** (Contribution 1, Section 1): Using G2P + P2V lookup, text embeddings are directly derived from the phoneme embedding space learned by the paired audio encoder. Text and audio embeddings naturally share the same space, eliminating projection layers and cross-modal mismatch, with the text branch having zero parameters besides G2P (0.83M)—compared to DistilBERT's 66M in [6], the edge-side friendliness is an order-of-magnitude difference.
2. **Confusable Keyword Automatic Generation Mechanism** (Contribution 2): Automatically constructs hard negative samples with similar pronunciation using phoneme edit distance $\delta \in \{1,2,3\}$, enabling flexible KWS models with no fixed classes and a large number of unseen keywords to gain discriminative understanding of real-word phonetic confusability, directly addressing the false alarm pain point.
3. **Discriminative End-to-End Audio-Text KWS Model** (Contribution 3): CED composed of audio encoder (frozen) + audio-compliant text encoder + verifier, combined with three-way batch construction ("positive sample / random negative / same-audio confusable negative") for end-to-end training.
4. **Result-Level Contributions**: Pushed SOTA on Libriphrase Hard from AUC 84.21% / EER 23.36% to 92.7% / 14.4% (Table 1, Abstract), reduced EER by 76.9% relative on LE, and increased AUC by 15.9% relative / reduced EER by 50.6% relative on SC-V1 zero-fine-tuning—while compressing the text branch from 66M parameters to 0.83M.

## Limitations and Future Work

### Technical Limitations of the Method

- **G2P Dependency and Pronunciation Error Propagation**. G2P (0.83M, [11]) is the sole entry point for the text link, and there is no error correction opportunity during inference: mispronunciations (heteronyms), proper nouns, abbreviations, and non-standard spellings will directly become incorrect phoneme sequences, causing text embeddings to be wrong from the source. The coverage range of the 74-phoneme vocabulary also determines the upper limit of the expressible pronunciation space. In multilingual scenarios, the entire phoneme system and G2P need to be rebuilt.
- **Context-Independence of Global Vectors**. GV is an average across about 50k samples and across contexts, smoothing out coarticulation information; Fig. 3 shows OW0/OW1/OW2 stress variants are closer inter-class, indicating that averaging indeed compresses details. For minimal pairs with strong context dependence (difference only in stress or subtle variants), the discriminative power of a single vector representation is insufficient, relying on the verifier to cover the gap.
- **Frozen Audio Encoder Closes Joint Optimization Space**. The audio side not updating in the end-to-end stage is a prerequisite for P2V validity, but the cost is that audio representations cannot be task-adapted for the verification objective; all shortcomings on the audio side (e.g., phoneme prediction bias for specific accents) are transmitted directly into P2V and final scores.
- **Limited Transformation Space for Confusion Generation**. Only two transformations, replace and insert, are available, no deletion; replaced phonemes are sampled uniformly at random, not reflecting real-world phoneme confusability probabilities (some phoneme pairs are naturally more prone to confusion); perturbation is only at the phoneme symbol level, not simulating acoustic deformations at the channel level (noise, accent, speaker differences).
- **Alignment Path Dependency on DSP (Borrowed from [6])**. Alignment errors are transmitted directly into the $m \times d$ consistency matrix, and the choice of DSP itself is not compared or justified in the paper.

### Shortcomings in Experimental Design

- **Single Data Face**: Training, fine-tuning, and P2V construction are all on LibriSpeech derivative data; SC-V1 only underwent zero-fine-tuning evaluation. Lacks evaluation in landing scenarios such as real far-field, noise, and multi-speaker registration.
- **Limited Baseline Coverage**: Only compared with two baselines [5] and [6]; and [6] did not report results on G (Table 1 is "-"), so cross-domain comparison actually only holds for [5].
- **Lack of Edge-Side System Metrics**: Latency, RTF, runtime memory, and package size increment are not reported in the paper—lightweighting is the paper's core selling point, but the supporting evidence is only parameter count (3.8M / 0.83M vs 66M), without falling into deployment-perceptible metrics.
- **Thin Ablations**: Only one ablation for the confusable module; key hyperparameters such as three-way batch structure (11×3), batch size 32, $\delta$ values, 50K sample size, and 74 phonemes have no sensitivity analysis.
- **No Statistical Robustness Report**: Variance, confidence intervals, or significance tests over multiple runs are not reported; the statistical significance of some gaps (e.g., EER 0.8 vs 1.7 on LE) cannot be judged.
- **Training Cost Undisclosed**: Number of GPUs and training duration are not reported in the paper.

### Possible Directions for Future Improvement

- **Context-Dependent P2V**: Upgrade lookup from "one global vector per phoneme" to lookup conditioned on left/right phonemes (similar to triphones), or cluster local vectors for each phoneme to retain multiple prototypes, storing coarticulation and stress information back into text representations.
- **Learnable Codebook and Joint Fine-Tuning**: Change P2V from a statistical table to a learnable codebook (e.g., vector quantization ideas), training jointly with the audio encoder and verifier; or adopt a "train encoder → rebuild P2V → train verifier" iterative loop, gradually breaking the freezing limit while maintaining homogeneity of the two spaces.
- **More Complete Confusion Generation**: Add deletion transformation; sample replaced phonemes according to real phoneme confusability matrices rather than uniform random; further augment hard negatives at the acoustic level using TTS variants perturbed with accents/noise.
- **Lighter G2P or Pre-computation on Registration Side**: Use a smaller character-to-phoneme model, or complete text-to-phoneme sequence conversion on the server side, leaving only lookup on the edge side, further compressing the device-side text link.
- **Edge-Side Measurement Make-up**: Provide system-level comparisons of RTF, memory, power consumption, and projection-based dual-encoder schemes, turning "lightweight" from a parameter claim into deployment evidence.
- **Multilingual and Cross-Speaker Extension**: Rebuild non-English phoneme systems and G2P, verifying the generalization boundaries of homogeneous embedding methods on multilingual keywords and heavy-accent speakers.
