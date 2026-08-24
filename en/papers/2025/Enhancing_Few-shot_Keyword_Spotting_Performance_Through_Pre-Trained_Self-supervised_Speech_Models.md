# Enhancing Few-shot Keyword Spotting Performance through Pre-Trained Self-supervised Speech Models

- **Authors/Affiliations**: Alican Gok (Department of Electrical Engineering, Bogazici University / Analog Devices, Istanbul); Oguzhan Buyuksolak, Osman Erman Okman (Analog Devices, Istanbul); Murat Saraclar (Department of Electrical Engineering, Bogazici University, IEEE Senior Member). Institutional information confirmed from the paper’s first-page footnotes: The first author has a dual affiliation with Bogazici University and Analog Devices, the latter two authors are affiliated with Analog Devices, and the corresponding author is affiliated with Bogazici University.
- **Date**: June 2025 (arXiv v1: 2506.17686), v2 updated on October 8, 2025; The manuscript header indicates submission to IEEE Signal Processing Letters (no formal acceptance information yet).
- **Link**: https://arxiv.org/abs/2506.17686
- **Keywords**: few-shot keyword spotting, self-supervised learning, Wav2Vec 2.0, Sub-center ArcFace, knowledge distillation, edge deployment, metric learning

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) is a core technology for hands-free voice interaction on battery-powered edge devices: smart speakers, wearable devices, wireless earbuds, and battery-powered smart sensors all rely on KWS to detect wake words or voice commands in continuous audio streams. The paper identifies the pain points of traditional KWS in two dimensions in the introduction:

The first is **adaptability and scalability**. Traditional KWS systems are trained for a fixed vocabulary, requiring thousands of training samples per keyword, with the vocabulary frozen at factory settings. When users want to customize wake words (their own names, product brand names, or low-resource language words), they lack thousands of samples and cannot retrain a complete model on the device. Few-Shot Keyword Spotting (FS-KWS) addresses this by registering new keywords using only a few examples (K-shot, typically K=1/5/10), with the model itself having only hundreds of thousands of parameters, theoretically allowing it to run on the edge.

The second is **resource constraints**. The computational power and memory requirements of traditional high-accuracy models (especially Transformer-based self-supervised pre-trained speech models) far exceed the capabilities of embedded devices. The paper provides a very intuitive quantitative comparison (Section II): The Wav2Vec 2.0 teacher model has 218 million (218M) parameters and requires 63.3 billion (63.3G) multiply-accumulate operations for a single forward pass; whereas the edge student model ResNet15 has only 480 thousand (480k) parameters and 235 million (235M) multiply-accumulate operations—a difference of three orders of magnitude in parameters and more than two orders of magnitude in compute power. Battery-powered smart sensors simply cannot run the former directly.

The third pain point, often overlooked but most fatal in engineering, is that **existing FS-KWS methods lack sufficient accuracy at product-usable low false alarm rate (FAR) operating points**. Wake word scenarios are extremely sensitive to false awakenings (a false awakening means double losses in power consumption and user experience), so the truly meaningful operating points are FARs of 1% or lower, rather than just reporting arbitrary Accuracy. The abstract directly presents the dire situation of baseline methods: On the GSC dataset at 10-shot and 1% FAR operating point, the leading FS-KWS method (Rusci et al.'s IEEE Micro 2023 work, i.e., the baseline [5] of this paper) achieves a classification accuracy of only 33.6% (Table I)—two-thirds of the time it is wrong, making it completely unusable.

### Specific Deficiencies of Existing Methods

The paper outlines four layers of deficiencies in existing FS-KWS solutions from a technical perspective:

- **Metric learning-based edge solutions collapse at low FAR operating points**. Mainstream FS-KWS adopts the prototypical network framework: it trains a representation model to map speech segments to fixed-dimensional embedding vectors. During enrollment, the mean of the embeddings of K samples is taken to obtain the class prototype. During inference, the distance between the test sample's embedding and each prototype is used for decision-making. The loss function is typically Triplet or Prototypical series. The problem is that these small edge models (e.g., ResNet15 + Triplet) perform reasonably well within the training domain but suffer from poor cross-domain generalization and a cliff-like drop in accuracy at low FAR. The data in Table I speaks for itself: The baseline Triplet model achieves a DET1% (detection rate at 1% FAR) of 72.3 and AUROC of 97.5 on the MSWC test set at 1-shot, which seems okay; but when switched to the GSC cross-domain test, the same configuration yields an ACC1% of only 23.7 and AUC of 71.1—halved again. Fig. 3(c) further shows that the baseline ranks second at FAR=5%, but drops sharply when FAR decreases to 1%. The paper attributes the main cause to the presence of the silence class.
- **Strong representations from self-supervised models but the curse of dimensionality**. Self-supervised (SSL) models like Wav2Vec 2.0 and HuBERT can produce frame-level features with immense information content, which supervised small models lack. However, SSL models output a 1024-dimensional vector per frame. Calculating with a frame shift of approximately 20ms, 1 second of speech produces a feature matrix of 49×1024. The paper explicitly states: processing this volume of data on low-power edge processors is infeasible. That is, there is a direct contradiction between the "goodness" of SSL features and the "inability to fit" on the edge.
- **Existing SSL feature aggregation methods are not optimized for the open-set protocol of FS-KWS**. Literature already includes dimensionality reduction methods such as mean-pooling (SUPERB benchmark [9]), convolutional encoders (Wav2KWS [10]), and attention extraction [11]. However, these either serve fixed-vocabulary classification transfer or are general discriminative tasks, and are not jointly designed with the FS-KWS-specific deployment protocol of "few-shot enrollment + open-set rejection + low FAR operating point." In particular, simple average pooling treats all frames equally, failing to highlight discriminative time segments in keyword pronunciation.
- **Engineering defects of the Triplet loss itself**. Triplet requires the online construction of massive triplets (the teacher training in this paper used 3000 batches, with 512 triplets per batch, Section III-C); sampling strategies and hard example mining heavily influence performance. Moreover, Triplet only constrains that "the positive example is closer than the negative example by a margin," lacking explicit inter-class angular separation and intra-class multi-mode modeling. For the reality that the same keyword has multiple pronunciation variants (different speakers, speeds, accents), the constraint of a single class center flattens intra-class diversity.

### Key Challenges to be Solved

The paper aims to solve a transfer problem: **How to compress the representation capability of an SSL model with 218M parameters and 63.3G MACs into a ResNet15 with 480k parameters and 235M MACs, while satisfying three hard constraints**—(1) Significantly boost usable accuracy at product-level low FAR operating points like 1% (Abstract: GSC 10-shot improves from 33.6% to 76.9%); (2) Cross-domain robustness, i.e., data outside the training domain (GSC, different speakers and recording conditions) does not collapse; (3) Unchanged deployment form, remaining the "few-shot enrollment prototype + cosine distance threshold rejection" open-set protocol, without modifying the edge inference flow. The essence of the challenge lies in the fact that direct embedding distillation preserves geometry but lacks discriminability, while directly adding metric loss preserves discriminability but easily overfits the training domain—all experimental designs in the paper seek a balance between these two.

## Methodology

### Overall Architecture Design and Design Motivation

The overall framework (Fig. 1) is a three-stage pipeline: **Pre-trained SSL model → Dimensionality Reduction (DR) model → Teacher model (trained with supervised metric learning) → Knowledge Distillation → Edge model**. Note the semantic division of labor in this chain:

- **Pre-trained SSL model**: Directly uses the pre-trained Wav2Vec 2.0, frozen throughout, with no fine-tuning. Why use pre-trained SSL instead of a feature extractor trained with supervision? Because FS-KWS classes appear only at test time (unseen classes), and it is impossible to have seen the test keywords during training. What is needed is a general speech representation with cross-class generalization; the representations learned by SSL pre-training on massive unlabeled speech happen to provide this generalization. The paper's conclusion attributes the robustness under unseen conditions to "knowledge distillation from pre-trained SSL models."
- **Selection of feature layer**: Outputs are taken from the 16th Transformer layer of Wav2Vec 2.0. Why the 16th layer? The paper explicitly states that this choice is based on Sanabria et al. [16] (ICASSP 2023), a systematic empirical analysis of acoustic word embeddings from pre-trained self-supervised speech models—middle-layer Transformer outputs are most effective for word-level discriminative tasks, rather than the last layer. 1 second of 16kHz speech (1×16000 samples) encoded by Wav2Vec 2.0 with a ~20ms frame shift results in 49×1024 frame-level features.
- **DR model (Teacher-exclusive)**: Compresses 49×1024 to a 64-dimensional embedding. Why must it be compressed? A 64-dimensional vector is a lightweight operation for prototype enrollment and cosine distance calculation on the edge, whereas a 49×1024 matrix is difficult to store even. The paper compares two DR architectures (see Innovation 1).
- **Student model ResNet15**: Why choose ResNet15? The paper gives two reasons: ResNet series are the de facto standard benchmarks for edge KWS (citing [19][20][21], including Tang & Lin ICASSP 2018's pioneering work and Rusci et al. 2025's ultra-low-power audio sensor work); and in the comparison of the baseline paper [5], ResNet15 outperforms other edge-friendly candidates. At the same time, using the same skeleton as the baseline ensures fair comparison. Why use MFCC instead of waveform or log-mel? The paper states: Further compress data volume, memory, and computation, suitable for edge hardware. Specific frontend configuration (Section III-D): First 10 dimensions of MFCC, frame length 40ms, frame shift 20ms, Hamming window, resulting in a 49×10 feature map for a 1-second sample.
- **Quantitative statement of resource comparison** (Section II-B): Teacher 218M parameters / 63.3G MACs, Student 480k parameters / 235M MACs—parameters reduced by three orders of magnitude. This is precisely the reason for the existence of the distillation architecture where "the teacher exists only during training, and the student goes to the edge."

### Mathematical Principles of Core Algorithms

**Inference Protocol (Prototype + Cosine + Threshold)**: During the enrollment phase, K samples pass through the frozen representation model to obtain K embeddings, and their arithmetic mean is taken as the class prototype for that keyword $c_k = \frac{1}{K}\sum_{i=1}^{K} f(x_i)$. During inference, the test sample's embedding calculates the cosine distance with each class prototype. If the nearest distance is less than the threshold $T$, it is assigned to the nearest class; otherwise, it is judged as "others" (rejection). Scanning $T$ yields the accuracy-FAR trade-off curve, from which the accuracy at the 1%/5% FAR operating point can be read. This protocol is identical to the baseline [5], ensuring fair comparison.

**Distillation Loss (Paper's Eq. (1))**:

$$L = L_{KD} + \lambda L_T$$

where $L_{KD}$ is the Mean Squared Error (MSE) between the teacher and student model output embeddings, $L_T$ is the task loss (optional Triplet or SCAF), and $\lambda$ is the weighting coefficient. Why use MSE to align embeddings instead of classic KL divergence soft-label distillation? Because what is being transferred here is the geometric structure of the embedding space (which direction is close, which is far), not the classification probability distribution—the student's output is itself a 64-dimensional embedding vector rather than class logits. MSE performs regression directly in the target space, delivering geometric information in one step. The four student training strategies correspond to: KD only ($\lambda=0$), KD+Triplet ($\lambda=0.03$), KD+SCAF ($\lambda=0.0003$), and SCAF only (no distillation, only loss change). The two $\lambda$ values differ by two orders of magnitude; the paper only states they are "determined empirically." A reasonable technical explanation (author's inference, not stated in the paper) is that the SCAF loss contains a scale amplification factor of $s=32$, so its numerical scale is naturally larger than Triplet, requiring a smaller weight to balance with MSE.

**Metric Learning Losses**: Due to space constraints in the main text, the complete definitions of Triplet and SCAF are placed in Section II of the supplementary material; the main text only provides training configurations. Standard forms are given below based on their cited original literature for understanding:

- Triplet (citing baseline [5]'s usage): $\max(0, d(a,p) - d(a,n) + m)$, the anchor-positive sample distance must be compressed to be closer than the anchor-negative sample by a margin $m$. This paper's configuration (Section III-C): margin=0.5, distance uses squared normalized Euclidean distance, trained for 3000 batches, with 512 triplets per batch.
- ArcFace (Deng et al. CVPR 2019 [13]): For normalized features and normalized classification weights, the target class logit is $s\cos(\theta_y + m)$—adding an additive margin in the angular space to the target class, forcing samples of the same class to tighten towards the class center on the hypersphere, and different classes to open up by at least angle $m$.
- Sub-center ArcFace / SCAF (Deng et al. ECCV 2020 [12]): Each class is set with $K$ sub-centers. During training, only the sub-center with the highest cosine similarity to the sample participates in the margin penalty. The motivation is to handle intra-class multi-modes: various pronunciation styles and speaker variants of the same keyword can be distributed around different sub-centers, preventing them from being flattened by a single center.

This paper's SCAF training hyperparameters (Section III-C): 10 epochs, angular margin $m=28.6$, scale $s=32$, $K=3$ sub-centers per class.

### Key Technical Innovation 1: Attention Encoder + SCAF Loss for Teacher Representation Compression

This is the paper's most core innovation combination, solving the problem of "how to use SSL features."

**Two DR Architectures (Fig. 2)**:

- Simple Pooling Encoder (Fig. 2(a)): 49×1024 → Temporal Average Pooling → 1×1024 → Linear Layer → 1×64. This is the conventional practice in literature (SUPERB-style mean pooling).
- Attention Encoder (Fig. 2(b)): 49×1024 → Scaled Dot Product Attention Block → PReLU → Conv1D → 1×1024 → Linear Layer → 1×64.

Why is the Attention Encoder better? The paper's mechanistic explanation (Section II-A.2): The attention mechanism calculates temporal relationships on the input sequence, allowing the model to focus on significant temporal features—discriminative phonetic segments in keyword pronunciation (e.g., consonant burst segments) receive higher weights, whereas average pooling treats all 49 frames equally, diluting discriminative segments into silence and transition segments. The subsequent Conv1D summarizes the temporal dimension into a single value through weighted averaging, preserving key temporal information while compressing computation. PReLU provides learnable non-linearity (negative half-axis slope is learnable, citing He et al. [18]), which is smoother than the hard clipping of ReLU. Finally, the linear layer compresses to 64 dimensions, producing compact embeddings suitable for edge deployment.

**Why change the loss from Triplet to SCAF**: SCAF's angular margin explicitly separates inter-class distances, and sub-centers explicitly accommodate intra-class diversity. Moreover, it is a classification-style training (sampling directly by word labels per batch), eliminating the need for triplet mining. The paper claims this is the first time SCAF is used for audio discriminative tasks in FS-KWS ("To the best of our knowledge, our work is the first to employ SCAF in the context of audio discrimination for FS-KWS").

**Experimental Evidence (Section III-C, corresponding to Fig. 3(a))**: On the test set, the average inter-class cosine distance increased from 0.89 (Simple Pooling + Triplet) to 0.93 (switching to Attention Encoder), and further to 0.95 (switching to SCAF); the average intra-class distance correspondingly decreased from 0.27 to 0.26 and then to 0.25. That is, both architectural improvements and loss improvements contributed measurable enhancements in embedding space quality. The paper also verified that among the students trained with three different teacher configurations, the student of the "Attention + SCAF" teacher was the best—the teacher quality was transmitted downstream.

### Key Technical Innovation 2: Embedding Distillation + Task Loss for Edge Training

The student ResNet15's training loss is $L = L_{KD} + \lambda L_T$. The paper systematically evaluates three choices for $L_T$ (none / Triplet / SCAF) plus SCAF only without distillation, comparing four strategies against the baseline Triplet (re-testing using the baseline authors' public repository [25] training checkpoint to ensure consistent comparison criteria).

Breaking down the "why" of the design logic:

- **Why distilling embeddings is effective**: MSE directly copies the embedding geometry of the teacher (SSL features compressed by attention) to the student. The student does not need to rediscover the discriminative structure of speech from scratch in MFCCs; it only needs to imitate an already good target. In Table I, KD only achieves ACC1% 74.4 and AUC 90.2 on GSC 10-shot, an improvement of more than double compared to the baseline (33.6/87.5), indicating that pure geometric transfer carries the vast majority of the benefit.
- **Why add task loss**: Pure MSE preserves geometry but not discriminability. Task loss further shapes the embedding space within the training domain. Results on the same domain (MSWC) support this: KD+Triplet and KD+SCAF are slightly better than KD only (10-shot DET1% 95.5/96.1 vs 95.7, 1-shot 86.3/86.3 vs 83.7).
- **Why the cross-domain conclusion reverses**: On GSC, KD+Triplet is significantly worse than pure KD (1-shot ACC1% 29.5 vs 42.9; 10-shot 55.8 vs 74.4). The paper judges that Triplet introduced overfitting to training conditions. In contrast, KD+SCAF maintains its advantage in cross-domain settings (10-shot ACC1% 76.9, the highest among all GSC students in the table). Thus, the paper provides a practical recommendation: For practical applications requiring low false alarms, choose KD+SCAF, which is robust across all settings.

### Technical Differences with Existing Methods

- **With baseline [5] (Rusci & Tuytelaars, IEEE Micro 2023)**: The baseline is "ResNet15 + Triplet direct metric learning on MSWC," done in one step; this paper changes it to a two-stage scheme where "SSL teacher first shapes the embedding space, then distills to the student," and the teacher's metric loss is changed to SCAF. The inference protocol, student skeleton, and MFCC frontend remain identical. The differences focus solely on the training scheme—this is a clean controlled variable comparison.
- **With SSL transfer works like Wav2KWS [10]**: Those works transfer SSL representations to fixed-vocabulary classification KWS; this paper is open-set metric learning, where classes appear only at enrollment, and SSL features trained with SCAF must support prototype discrimination for any new word.
- **With ArcFace/SCAF usage in face recognition [12][13]**: The loss form is the same, but the point of application is different—this paper applies SCAF in two places (training loss of the teacher DR model + task loss of student distillation), and it acts on speech embeddings extracted from the middle layer of the SSL model rather than face image embeddings. The authors also cite precedents of ArcFace in speech emotion recognition [14] and far-field speaker verification [15], but this is the first time in FS-KWS audio discrimination.
- **With the alternative route of directly deploying SSL models**: This paper does not attempt to quantize or prune Wav2Vec 2.0 for edge deployment (218M parameters vs 480k). Instead, it utilizes it only during the training phase, and the edge deployment form is identical to the baseline—this is a typical engineering trade-off of "heavy during training, light during deployment."

## Experimental Results

### Datasets Used and Their Scales

(Section III-A)

- **Training Set**: English part of MSWC (Multilingual Spoken Words Corpus) train split—**5.5 million 1-second samples, 39,000 different words**. Used to train the teacher's DR model and student models.
- **Test Set 1**: MSWC test split—**700,000 samples, 8,900 words** (held-out words not overlapping with the training vocabulary, testing unseen class generalization).
- **Test Set 2**: Google Speech Commands (GSC)—**100,000 1-second audio clips, 35 short commands**. Different speakers and recording conditions from MSWC, used for cross-domain testing.
- **Test Protocol Details**: On MSWC, simulate wake word scenarios word-by-word—each mini-experiment randomly takes K samples of one word from the test set to build a prototype, comparing it with 1 positive sample of the same word and 1 negative sample of a random different word, repeated **100,000 times** (the paper verifies that results are stable at this count). On GSC, modify the KWS-12 setting: Classify 11 classes (target words {on, off, left, right, up, down, go, stop, yes, no} plus silence class), while the remaining 25 classes should be rejected as "others"; the silence class contains various background noise recordings. Enrollment uses K samples per word from the training set, evaluated on the **entire test set**; to reduce few-shot sampling variance, each experiment is repeated **100 times and averaged**. The paper specifically emphasizes that adding the silence class makes its protocol harder than the baseline [5]'s original protocol.

### Definition and Rationale for Evaluation Metrics

- **On MSWC**: DET1% / DET5% (detection rate at 1% / 5% false alarm rate) and AUROC. Since MSWC is a binary decision of 1 positive and 1 negative, detection rate is used.
- **On GSC**: ACC1% / ACC5% (11-class classification accuracy at 1% / 5% FAR) and AUC.
- **Rationale**: The real operating point of wake word products is at extremely low FAR (high cost of false awakening); reporting raw Accuracy alone would overestimate usability. DET/ACC@1%FAR directly gives the usable accuracy at the product operating point, while AUC provides a full-threshold overview. This set of metrics aligns with the baseline [5] protocol, ensuring comparability.
- **Stability Control**: The teacher DR model was retrained with 3 random seeds using the final hyperparameters, verifying that loss fluctuations were less than 0.5%; each student model was trained 3 times, and validation loss and AUC fluctuations in Table I were less than 0.6% in all cases.

### Detailed Comparison with Baseline and SOTA Methods

Table I provides a complete matrix of {1, 5, 10}-shot × 6 configurations × 6 metrics (3 on MSWC + 3 on GSC). Key numbers:

**Teacher (Attention + SCAF, Wav2Vec 2.0) Performance Upper Bound**:
- MSWC: 1-shot DET1%/DET5%/AUROC = 91.6 / 97.4 / 99.3; 5-shot = 96.8 / 99.2 / 99.8; 10-shot = 96.9 / 99.3 / 99.9.
- GSC: 1-shot ACC1%/ACC5%/AUC = 69.0 / 77.1 / 83.4; 5-shot = 81.6 / 84.8 / 89.9; 10-shot = 82.2 / 85.2 / 90.9.

**Student (480k parameter ResNet15) vs. Baseline (Same Skeleton Triplet, re-tested by authors' original checkpoint)**:

| Configuration | MSWC 1-shot (DET1%/DET5%/AUROC) | GSC 1-shot (ACC1%/ACC5%/AUC) |
|---|---|---|
| Baseline Triplet [5] | 72.3 / 88.3 / 97.5 | 23.7 / 52.3 / 71.1 |
| SCAF only | 76.8 / 88.5 / 92.5 | 19.9 / 37.3 / 56.3 |
| KD only | 83.7 / 92.7 / 98.1 | 42.9 / 59.2 / 73.0 |
| KD+Triplet | 86.3 / 95.4 / 98.9 | 29.5 / 47.6 / 66.5 |
| KD+SCAF | 86.3 / 93.9 / 98.3 | 44.0 / 59.0 / 73.7 |

| Configuration | MSWC 10-shot (DET1%/DET5%/AUROC) | GSC 10-shot (ACC1%/ACC5%/AUC) |
|---|---|---|
| Baseline Triplet [5] | 91.1 / 97.6 / 99.4 | 33.6 / 76.4 / 87.5 |
| SCAF only | 91.0 / 95.9 / 97.1 | 47.9 / 65.5 / 78.1 |
| KD only | 95.7 / 98.2 / 99.5 | 74.4 / 83.9 / 90.2 |
| KD+Triplet | 95.5 / 98.9 / 99.7 | 55.8 / 74.0 / 86.1 |
| KD+SCAF | 96.1 / 98.4 / 99.5 | **76.9 / 85.2 / 91.0** |

(Full 5-shot data see paper Table I: KD+SCAF on MSWC is 95.1/98.1/99.4, on GSC is 69.5/80.2/88.6; KD+Triplet on GSC is 53.5/71.1/83.6.)

Three levels of conclusions:

1. **Abstract-level headline**: GSC 10-shot, 11-class classification accuracy at 1% false alarm rate improved from baseline 33.6% to 76.9% (KD+SCAF, Table I)—an absolute improvement of 43.3 percentage points, a relative improvement of 2.29 times. This is the basis for the paper's claim of being "significantly more suitable for real-world usage scenarios."
2. **Student approaches Teacher**: At 10-shot, the student reaches 96.1 on MSWC (Teacher 96.9) and 76.9 on GSC (Teacher 82.2), achieving most of the teacher's performance with three orders of magnitude fewer parameters—the most direct evidence of distillation effectiveness.
3. **Curve Behavior (Fig. 3(b)(c))**: In the MSWC 1-shot student curves, KD+Triplet has the highest detection rate, KD is second, and the baseline is worst (especially at FAR=1%); in the GSC 10-shot curves, KD leads consistently at FAR=1% and 5%, while the baseline Triplet ranks second at FAR=5% but drops sharply towards lower FAR—the paper attributes this to the silence class (its keyword-level analysis is not shown in the main text), and thus proposes the hypothesis that knowledge distillation from pre-trained SSL models is key to robustness under unseen conditions.

### Findings from Ablation Studies

Although the paper is short (SPL format), it performs three layers of ablation:

1. **DR Architecture Ablation (Fig. 3(a) + Section III-C distance statistics)**: With Triplet training, the Attention Encoder outperforms Average Pooling (inter-class cosine distance 0.89 → 0.93); and after training students with three different teachers, the student of the Attention+SCAF teacher was the best—confirming that architectural gains are transmitted downstream.
2. **Teacher Loss Ablation**: Switching from Triplet to SCAF on the Attention architecture further increased inter-class distance from 0.93 to 0.95 and decreased intra-class from 0.26 to 0.25. The DET-FAR curves in Fig. 3(a) show Attention+SCAF dominating comprehensively.
3. **Student Training Strategy Ablation (Table I, most core)**, four findings:
   - **Distillation is the main source of benefit**: All KD configurations crush non-KD configurations on GSC (KD only 10-shot ACC1% 74.4 vs. baseline 33.6 vs. SCAF only 47.9).
   - **Adding task loss within the same domain is slightly beneficial**: On MSWC, KD+Triplet / KD+SCAF are slightly better than KD only.
   - **Adding Triplet cross-domain is harmful**: KD+Triplet is comprehensively worse than pure KD on GSC (1-shot ACC1% 29.5 vs. 42.9)—Triplet overfits the model to the training domain's speakers and recording conditions.
   - **KD+SCAF is harmless and beneficial cross-domain**: GSC 10-shot ACC1% 76.9 is the highest among all students in the table, and robust across all shot settings—the final recommended configuration.
   - **SCAF does not work without SSL Teacher**: Training the student directly with SCAF (no distillation) yields an AUROC of only 92.5 on MSWC 1-shot, which is actually lower than the Triplet baseline's 97.5, and generally worse on GSC—indicating that SCAF's benefits are built upon the strong representations provided by the SSL teacher; simply changing the loss cannot save a weak skeleton.

## Main Contributions

1. **Proposes a new FS-KWS training framework**: Pre-trained SSL model (Wav2Vec 2.0 16th layer) + Attention Dimensionality Reduction + SCAF Metric Learning Teacher + MSE Embedding Distillation to ResNet15 Student. While maintaining the edge deployment form (MFCC input, prototype enrollment, cosine threshold rejection), it significantly boosts usable accuracy at low false alarm operating points (GSC 10-shot 1% FAR: 33.6% → 76.9%, Table I).
2. **First introduction of Sub-center ArcFace to audio discriminative tasks**: Used as the training loss for the SSL embedding dimensionality reduction model (and as the task loss for student distillation). Angular margins separate inter-class distances, and K=3 sub-centers accommodate intra-class pronunciation diversity. Measured inter-class cosine distance increased from 0.89 to 0.95, and intra-class from 0.27 to 0.25 (Section III-C).
3. **Systematic benchmarking against leading methods**: Using the baseline authors' public checkpoint and a unified test protocol (and a harder protocol including the silence class than the baseline), complete {1,5,10}-shot matrix comparisons were performed on two public datasets, MSWC (5.5 million training samples) and GSC (Table I), providing actionable engineering recommendations (KD+SCAF).
4. **(Implicit Contribution) A transferable engineering judgment**: In cross-domain scenarios, "pure embedding distillation + careful selection of task loss" outperforms "direct metric learning." The choice of task loss depends on whether it will overfit to training conditions (Triplet does, SCAF does not).

## Limitations and Future Work

### Technical Limitations of the Method

- **Teacher bound to a single SSL architecture and single layer**: All experiments only use the 16th layer of Wav2Vec 2.0. The layer selection is based on external literature [16] rather than self-verification in this paper. Alternative architectures like HuBERT, other layer positions, and layer combinations were not ablated (the authors state in the conclusion that "exploring alternative SSL architectures" is future work).
- **Embedding dimension fixed at 64**: The paper does not report dimension sensitivity (what if 32/128/256?), and no quantitative argument is given for the choice of 64.
- **Lack of real deployment measurements**: The paper does not report real-time edge latency, memory usage, or power consumption—235M MACs are just theoretical compute power, not validated on specific MCUs/DSPs. The authors also list "deployment optimization on ultra-low-power hardware" as future work.
- **Evaluation limited to 1-second segments**: All FARs are numbers on a segmented protocol (MSWC 1 positive 1 negative, GSC per clip). Streaming detection performance on continuous audio streams, real hourly false wake-up counts, and detection capability when keywords are in sentences are not reported.
- **Hyperparameters are empirical**: $\lambda$ (0.03 and 0.0003), SCAF's $m=28.6$, $s=32$, $K=3$, and epoch count are all "determined empirically," with no sensitivity analysis. The two $\lambda$ values differ by two orders of magnitude without an explanatory criterion (the author infers it is related to the $s=32$ amplification contained in SCAF, which is not stated in the paper).
- **Teacher training computational cost not reported**: The wall-clock time and compute overhead for Wav2Vec 2.0 forward extraction of 5.5 million samples and teacher DR training are not reported by the paper—although the teacher exists only during training, this is a key cost item for teams wishing to reproduce the work.

### Deficiencies in Experimental Design

- **English only**: MSWC English part + GSC. Cross-lingual generalization is not verified (the authors state multi-language extension as future work); yet "few-shot KWS in any language" is precisely the declarative goal of this field [1].
- **Single baseline**: Only one baseline, Rusci & Tuytelaars IEEE Micro 2023 [5], is empirically compared. Another leading method mentioned in the introduction (the same group's Interspeech 2023 open-set work [6]) does not enter the experimental table, nor is there horizontal data with broader FS-KWS literature (e.g., [2][3][4]).
- **Key definitions externalized**: The mathematical definitions of the two metric learning losses and embedding distance distribution plots are placed in the supplementary material (mentioned in two places in the main text Sections II/III), compromising the self-consistency of the main text's information.
- **Protocol changes affect comparability**: The GSC protocol adds a silence class and explicitly states it is "harder than the baseline." The baseline numbers are results of checkpoints re-tested under the new protocol by the authors—these cannot be directly compared with the numbers self-reported in the baseline's original paper.
- **Frontend information gap not ablated separately**: The teacher consumes 16kHz waveforms (via Wav2Vec 2.0), while the student consumes 10-dimensional MFCCs—the difference in information volume between the two inputs is huge. However, "how much was lost due to the MFCC information bottleneck" has no separate experiment (e.g., a control group of the student consuming log-mel), making it impossible to distinguish the respective shares of distillation benefits and frontend losses.
- **Silence class analysis not shown**: The main cause of the baseline's low FAR collapse (silence class) is brushed over with a single sentence of "unshown keyword-level analysis," preventing readers from verifying it.

### Possible Directions for Future Improvement

- **Directions stated by authors** (Conclusion section): Extend to multi-language scenarios; explore alternative SSL architectures; deployment optimization for ultra-low-power hardware.
- **Author's additions**: (1) Systematic ablation of embedding dimensions, teacher layer positions, and number of sub-centers K, to complete the quantitative map of the design space; (2) Report delay-power-accuracy three-dimensional trade-offs on real MCUs/low-power DSPs, turning 235M MACs into measured numbers; (3) Streaming continuous audio evaluation and product-level false wake-up metrics (number of false awakenings per hour); (4) Direct comparison with new open-set FS-KWS methods [6] and subsequent SSL benchmarks (WavLM, HuBERT variants); (5) Robustness evaluation under noise/far-field/multi-speaker mixed conditions—these are precisely the real acoustic environments of wake word products; (6) Explore automated parameter tuning methods for the relationship between $\lambda$ and loss scales (e.g., normalization by loss dimensionality), replacing manual tuning with a two-order-of-magnitude gap.
