# Personalized Keyword Spotting through Multi-task Learning

- **Authors/Affiliations**: Seunghan Yang, Byeonggeun Kim, Inseop Chung, Simyung Chang (Qualcomm AI Research / Qualcomm Korea YH, Seoul; Third author Inseop Chung is affiliated with Seoul National University on a part-time basis; according to the footnote, part of the research was conducted during an internship at Qualcomm)
- **Date**: June 2022 (arXiv:2206.13708v1, submitted June 28, 2022)
- **Link**: https://arxiv.org/abs/2206.13708
- **Keywords**: personalized keyword spotting, multi-task learning, speaker verification, task-specific scoring function, metric learning, false alarm reduction, on-device speech interface

## Problem Statement

### Problem Background and Domain Pain Points

Always-on lightweight keyword spotting (KWS) systems serve as the wake-up entry point for smart audio devices: only after the system detects a wake-word is the subsequent audio stream uploaded to the speech recognition system [1][2][3][4][5][6]. The power consumption logic of this cascaded chain dictates two hard metrics for KWS—high recall (few missed wake-ups) and low false alarm rate (few false wake-ups)—which jointly determine the total device power consumption. Conventional KWS (C-KWS) [2][7] aims to detect a small set of predefined speech signals (e.g., "Alexa", "OK Google") and constitutes the main body of current commercial always-on solutions.

However, C-KWS is inherently non-personalized by definition: it recognizes words, not speakers. Starting from actual product forms, the paper points out an overlooked fact—the vast majority of user interactions on devices come from the target user registered on the device. Therefore, the model should be biased toward the target user. More importantly, a system that ignores user information has a direct engineering consequence: it cannot reject "general negatives," i.e., background sounds containing the target keyword or other words with similar pronunciation to the target word—typical scenarios include TV streaming, online meetings, or conversations from people nearby. Once such audio hits the keyword, the system unnecessarily activates the subsequent recognition system, leading to additional power overhead. In other words, KWS false alarms are not just about "turning on a light more often," but about dragging the entire ASR chain to run, which causes perceptible battery life loss for users in scenarios like a living room with a TV always on.

### Specific Deficiencies of Existing Methods

- **C-KWS is completely user-agnostic**: ts-tk (target user saying target keyword) and nts-tk (non-target user saying target keyword) are treated equally in scoring. The paper directly quantifies this defect with experiments (Fig. 1b): evaluating the C-KWS system (BC-ResNet [7]) directly on personalized tasks significantly worsens its EER on both personalized tasks, with the TO-KWS (target user only) task worsening to the 20%+ magnitude (Table 1 exact value: BC-ResNet-3 Vanilla TO-KWS EER is 21.17%). The mechanism behind this number is straightforward: keyword scores can only judge "is this the word," having zero discrimination for speaker identity, while TO-KWS classifies nts-tk as negative samples, rendering keyword scores completely ineffective in this quadrant.
- **Query-by-example KWS adapts to words, not people**: QbE keyword spotting [8][9][10] allows users to register their own keywords, solving the flexibility problem of "changing words," but still does not explicitly model speaker identity—what is registered is "how this word is pronounced," not "who is pronouncing it."
- **Computational cost of naive dual-model schemes**: Assembling a KWS model with an independent speaker verification (SV) single-task model (Vanilla (+SV) in the paper) requires 22.5M multiplications and 82.4k parameters on the BC-ResNet-3 backbone (Table 1), which is significantly more expensive than the 16.7M of the single-model Vanilla. Moreover, there is no information exchange between the two models at the encoding and representation levels; speaker information does not "enter" the keyword system.
- **Misaligned objectives of existing KWS+SV multi-task works**: Previous joint training of KWS and SV [24][25] aimed to "mutually improve the performance of both tasks." No one answered how to adapt the shared representation for the decision objective of "personalized detection" (task-adaptation) after learning it. This missing link is precisely where the core contribution of this paper lies.

### Key Challenges to be Solved by This Paper

First, transforming "personalized KWS" from product language into an evaluable task definition. The core divergence point lies in a vague quadrant: what exactly is nts-tk (non-target user saying target keyword)? The paper explicitly defines two personalized tasks—TB-KWS (Target user Biased KWS, biased toward the target user but does not explicitly handle nts-tk, treating it as neutral) and TO-KWS (Target user Only KWS, accepting only ts-tk, classifying nts-tk as negative samples)—corresponding to two types of real device needs: the former suits bias-type products for "reducing TV false wake-ups," while the latter suits exclusive-type products for "only responding to the device owner."

Second, introducing speaker information within edge-side budgets. Always-on models are born for power saving; any personalized modification must satisfy the constraint that "parameters and computation only allow slight increases, and C-KWS performance cannot drop." Otherwise, it is gambling personalized gains against the main task's engineering budget.

Third, non-overlapping speaker sets in training and testing. Speakers cannot be treated as a closed-set classification problem (new faces unseen during training appear during testing); an open-set verification route via "embedding + registered reference audio" must be taken. This requires the scoring function to have a unified form between the keyword side (predefined, seen in the training set) and the speaker side (open-set).

Fourth, the learned representations must be "combined" by task to make decisions. TB and TO have different attitudes toward nts-tk, requiring task-specific scoring mechanisms to fuse keyword and speaker representations according to the semantics of each task, rather than using a single universal score.

## Methodology

### Overall Architecture Design and Design Motivation

PK-MTL (Personalized Keyword spotting through Multi-task Learning) is a two-stage system (Fig. 2): the first stage is multi-task learning, and the second stage is task-adaptive scoring.

The topology of the multi-task part adopts hard-parameter sharing [17][18]: the low-level layers of the backbone network $f_\theta(\cdot)$ ($\theta = \{\phi, \phi\}$) form a shared encoder $f_\phi(\cdot)$, while the high-level layers branch out into a keyword sub-network $f_{\phi k}(\cdot)$ and a speaker sub-network $f_{\phi s}(\cdot)$, outputting keyword features $z_i^k$ and speaker features $z_i^s$, respectively, each connected to cosine similarity classifiers $g^k(\cdot)$ and $g^s(\cdot)$. This "low-level sharing, high-level branching" topology is not chosen arbitrarily; its motivation stems from the feature properties of the two tasks:

- **Reason for low-level sharing**: Both KWS and SV require general acoustic encoding at the low level—spectral texture, phoneme-level acoustic patterns. These low-level statistics are shared by both tasks. The paper cites [19][20] that separated task designs are less efficient in memory and computation than shared designs; a shared encoder can absorb complementary information from both tasks at nearly zero extra cost.
- **Reason for high-level branching**: The two tasks are "antagonistic" at the high level [21]—keyword features require speaker invariance (different people saying the same word should fold into the same class), while speaker features require content invariance (the same person saying different things should fold into the same class). Forcing high-level sharing would cause the two objectives to drag each other down, so sub-networks are used to分流 (divert) at the high level, each learning its own characteristics.

Engineering split point implementation: The BC-ResNet backbone removes the last two convolutional layers as the shared encoder; Res15 and DS-ResNet remove the last convolutional block; sub-networks are composed of the remaining backbone layers plus an additional fully connected layer. The split point selected at "shallow sharing, deep branching" corresponds exactly to the two motivations above.

The choice of cosine similarity classifiers [22][23] instead of standard fully connected softmax is a key foreshadowing for the feasibility of the entire framework (see next section): the class weights of the cosine classifier are "class prototypes" in the normalized direction, allowing keyword prototypes to be directly reused at test time, forming a unified scoring form with the registered embeddings on the speaker side.

The task-adaptive part provides two scoring modules: SCM (Score Combination Module, training-free score combination) and TRM (Task Representation Module, trainable task representation module). During inference, C-KWS uses the keyword score $\psi^k$, while TB-/TO-KWS uses task-specific scores $\psi^{tb}, \psi^{to}$ (Fig. 2b).

### Mathematical Principles of Core Algorithms

**Multi-task training objective**. The cosine classifier is defined as (Eq. 1):

$g^k(z_i^k) = \text{softmax}(w \cdot \text{sim}(z_i^k, W^k) + b)$

where $W^k$ is the learnable keyword classification weight, $\text{sim}(a, b) = a \cdot b / (\|a\| \cdot \|b\|)$, and $w$ and $b$ are scale and bias. The keyword classification loss is the sum of negative log probabilities for true classes (Eq. 2):

$L^k = \sum_i -y_i^k \log g^k(z_i^k)$

The SV side is constructed similarly to obtain $L^s$, and the total loss is (Eq. 3):

$L_{mtl} = L^k + \lambda \cdot L^s, \lambda = 0.1$

The semantics of $\lambda$ is the importance of speaker information. The reason for taking 0.1 instead of 1.0: keyword detection is the main task, and the speaker is an auxiliary information source. The weight is lowered to ensure the main task is not misled—the results in Table 1 show that for Naive MTL, the C-KWS accuracy is 97.68% versus Vanilla's 97.57%, indicating that under this ratio, speaker supervision is "freely" injected into the network without perturbing keyword capability.

**Scoring and Decision**. Given a predefined target keyword, the C-KWS score is the cosine similarity between the test sample's keyword embedding and the classifier weight: $\psi^k_{i,ref} = \text{sim}(z_i^k, W^k_{ref})$, where $W^k_{ref}$ can be viewed as the most representative embedding of that predefined target word. Personalized tasks additionally utilize one registered voice $x_{ref}$ of the target user: $\psi^s_{i,ref} = \text{sim}(z_i^s, f_{\phi s}(f_\phi(x_{ref})))$. The system accepts the sample when $\psi_{i,ref} > \delta$. Note the asymmetry here: keywords are predefined and seen in the training set, using "learned prototypes" (classifier weights); speakers are open-set and appear only at test time, using "registered embeddings." Since training and testing speakers do not overlap, this is the only feasible route.

**SCM (Eq. 4)**. Linearly combine the two scores into a task score: $\psi = \alpha \cdot \psi^k + (1-\alpha) \cdot \psi^s$. The parameter $\alpha$ is obtained by solving:

$\alpha^* = \arg\min_\alpha \text{FRR}(\text{SCM}(\psi^k, \psi^s; \alpha)), \text{ s.t. FAR} = c$

on the validation set via grid search at each target operating point (e.g., FAR = 1%). Why $\alpha$ uses "calibration" instead of "learning": The engineering goal of KWS is to minimize miss rate given a false alarm budget. The optimal value of $\alpha$ naturally depends on the operating point. Binding it to the target FAR for point-by-point calibration is more aligned with deployment methods than learning a single fixed value globally. The advantage of SCM is zero training and plug-and-play.

**TRM (Eq. 5, Eq. 6)**. The limitation of SCM is that the two initial representations have not learned anything for the task; no matter how good the combination method is, it can only perform arithmetic on existing representations. TRM is a trainable module $\text{TRM}^{tb}(\cdot, \cdot) / \text{TRM}^{to}(\cdot, \cdot)$, taking keyword and speaker embeddings as input and outputting task-specific embeddings. The score is defined as the similarity to the prototype (Eq. 5):

$\psi^{tb}_{i,j} = \text{sim}(\text{TRM}^{tb}(z_i^k, z_i^s), \text{TRM}^{tb}(p_j^k, p_j^s))$

During training, prototypes $p_j^k, p_j^s$ take the classifier weights $W_j^k, W_j^s$; at test time, the speaker prototype is replaced by the embedding of the registered voice. The training loss is the angular prototypical loss [16] (Eq. 6):

$L^{tb} = -(1/N) \sum_i \log \left[ \frac{\exp(w \cdot \psi^{tb}_{i,i} + b)}{\sum_j \exp(w \cdot \psi^{tb}_{i,j} + b)} \right]$

Why choose prototype-based metric learning instead of standard triplet loss: The comparison object at test time is the prototype (keyword weight plus registered embedding). Training directly uses same-class prototypes as positive and different-class prototypes as negative, strictly aligning the training distribution with the inference distribution. More critically, batch construction: the definition of positive and negative samples deliberately mimics the four-quadrant test scenario in Fig. 1a—in the TB task, nts-tk is treated as neutral (neither positive nor negative), while in the TO task, nts-tk is explicitly selected as a negative sample. This is the only entry point for injecting task semantics into training and the essential difference between TRM and SCM: SCM only combines scores at the inference end, while TRM shapes representations at the representation end according to the task.

The structure of TRM is very lightweight: concatenating normalized keyword and speaker embeddings, using two fully connected layers (intermediate feature dimension is 2) to produce attention weights, performing SE-style attention re-weighting [27]. The meaning of the intermediate dimension being 2: the attention head only needs to answer one question—"what weight does the keyword dimension and speaker dimension each occupy?"—the capacity is extremely small, and the parameter overhead is negligible—Table 1 shows PK-MTL (82.0k parameters) increases by only about 1.8k relative to Naive MTL (80.2k). TRM is trained for 50 epochs, following the backbone's training strategy.

### Key Technical Innovation 1: Task Formalization of Personalized KWS (Four-Quadrant Classification + TB/TO Dual Tasks)

The paper divides the input space given a target speaker and target keyword into four quadrants (Fig. 1a): ts-tk, nts-tk, ts-ntk, nts-ntk. The ambiguous quadrant is nts-tk—it is the target keyword to be detected, but from a non-target user. Two different handling strategies for this quadrant give rise to two tasks: TB-KWS treats it as neutral (the model is biased toward the target user, no explicit handling), while TO-KWS treats it as a negative sample (only accepts target user). The value of this formalization lies in turning "personalization" from a marketing term into a protocol that can be repeatedly evaluated on fixed data—all subsequent experiments (including the construction of 16,000 pairs × 10 splits for testing) are built on this classification system. It is the premise for all numbers in this paper to be discussed.

### Key Technical Innovation 2: KWS+SV Hard-Parameter Sharing Multi-task Skeleton and Unified Cosine Prototype Scoring

Hard-parameter sharing is used to "transport" speaker supervision into the KWS network: the low-level shared encoder absorbs complementary acoustic information, the high-level twin sub-networks divert antagonistic objectives, and the total loss is hung on the main task with a weight of $\lambda=0.1$. The key accompanying design is the cosine similarity classifier: class weights are class prototypes, making the scoring on the keyword side (predefined words) using learned prototypes and the speaker side (open-set new speakers) using registered embeddings unified into the same cosine similarity form, aligning training objectives with inference criteria. The measured cost of this combination (BC-ResNet-3, Table 1): multiplications from 16.7M to 17.5M (approx. +4.8%), parameters from 63.5k to 80.2k (approx. +29.1%), while C-KWS accuracy slightly increased (97.57% → 97.68%). Compared to dual-model assembly (22.5M multiplications), the shared design saves about 22% computation.

### Key Technical Innovation 3: Two-Level Design of Task-Adaptive Scoring (SCM Training-Free / TRM Trainable)

Task adaptation is split into two levels from light to heavy. SCM is the training-free route: directly linearly combining two existing scores, with $\alpha$ calibrated on the validation set according to the target operating point, suitable for scenarios where the model cannot be retrained. TRM is the trainable route: using batch construction that mimics test scenarios plus angular prototypical loss, directly writing task semantics (TB tolerates nts-tk, TO rejects nts-tk) into representation learning. Ablation (Table 1) shows that each level of this ladder has real benefits, and the final state TRM is optimal in almost all metrics—indicating that "representation-end task shaping" is superior to "inference-end score arithmetic." The paper names the complete system with TRM as PK-MTL.

### Technical Differences with Existing Methods

Compared to C-KWS [2][7][12]: The same backbone, retaining C-KWS capability at the same accuracy level (Table 1 C-KWS EER changes within ±0.15 for each backbone), additionally gaining speaker dimension and availability for two personalized tasks. Compared to QbE [8][9][10]: QbE solves "changing words," while this paper solves "recognizing people." These are two orthogonal dimensions, theoretically stackable (the paper does not perform stacking experiments). Compared to existing KWS+SV multi-task works [24][25]: The objective changes from "mutually boosting points" to "transporting user information for personalized decisions," and adds a task-adaptation stage—how shared representations are combined into task scores is a question not answered by previous works. Compared to dual-model assembly: Parameter counts are similar (82.0k vs 82.4k) but computation is about 22% less (17.5M vs 22.5M, BC-ResNet-3, Table 1), because the encoder is shared rather than two separate models running independently.

## Experimental Results

### Datasets Used and Their Scale

The main dataset is Google Speech Commands v1 [11]: 64,727 audio clips, 30 words, 1,881 speakers. Following the standard 12-class setup of [11]: 10 target words (Yes, No, Up, Down, Left, Right, On, Off, Stop, Go) plus Unknown (the remaining 20 words) and Silence (no speech). Training/validation/testing splits follow [3][2][7][11][26]. Personalized task testing uses sample pair construction: randomly select an anchor, pair with four-quadrant samples to form positive/negative pairs; 10 test splits, 16,000 pairs each, reporting average performance. Silence and Unknown classes can be used as non-target keywords but not as target keywords. SV validation separately constructs 10 splits, totaling 160,000 speaker pairs.

Real-scenario evaluation introduces two external negative sample sets: WSJ-SI200 [13] (cut into 1-second segments from complete audio streams following [9]) and Librispeech [14] (cut into 1-second segments from the entire stream on the public noisy audio test set). One side of the test pair is a general negative sample, and the other is a target keyword sample spoken by a random speaker from GSC, with all 10 target words paired.

Features and augmentation: BC-ResNet uses 40-dim log-Mel spectrograms (30ms window length, 10ms frame shift), augmentation strategy follows [7]; Res15 and DS-ResNet use 40-dim MFCCs, adding noise and random shifts following [2][12]. To ensure fairness of the scoring function, the three Vanilla baselines also added an extra fully connected layer before the classifier and switched to cosine classifier $g^k(\cdot)$—the paper explicitly states that this change itself brings performance improvement (i.e., baselines are strengthened, not weakened).

### Definition and Rationale for Evaluation Metrics

FAR (False Alarm Rate): Proportion of negative samples incorrectly accepted; FRR (Miss Rate): Proportion of positive samples incorrectly rejected; EER: Point where FAR equals FRR; Top-1 accuracy for multi-word classification. The rationale for selection is directly linked to deployment methods: the operating point of an always-on system is determined by the false alarm budget (power consumption), so the main metrics are "FRR given FAR" (Table 1 reports two tiers: FAR 1% and 10%) and "FAR given FRR" (Table 2 reports two tiers: FRR 1% and 5%); EER serves as a summary metric independent of operating points for cross-model evaluation; Top-1 corresponds to the closed-set classification usage of C-KWS. All numbers in Table 1 are means of 5 independent trials (standard deviation).

### Detailed Comparison with Baseline Methods and SOTA

**C-KWS main task does not drop points (Table 1)**: On BC-ResNet-3, Vanilla accuracy 97.57%, EER 1.92%, PK-MTL accuracy 97.68%, EER 1.98%, FRR@1% 2.58%, FRR@10% 0.75% (Vanilla is 2.85% / 1.89%); On BC-ResNet-8, EER 1.89% vs 1.79% (improvement) but accuracy 98.01% vs 97.87% (slight drop); On Res15, accuracy 95.82% vs 96.30%, EER 2.38% vs 2.25% (multi-task acted as regularization). Overall, it maintains "comparable."

**Magnitude jump in TB-KWS (Table 1)**: On BC-ResNet-3, Vanilla FRR@1% is 96.50%, FRR@10% is 64.32%, Naive MTL is 96.44% / 64.26%—i.e., only learning multi-task without task adaptation is almost unusable at practical operating points; PK-MTL (TRM) achieves EER 2.02%, FRR@1% 6.56%, FRR@10% 1.52%. FRR@1% improves from 96.5% to 6.56%, an improvement of about 90 percentage points, indicating that on personalized tasks, "whether there is task-adaptive scoring" is far more important than "backbone strength."

**Consistent improvement in TO-KWS across all backbones (Table 1)**: Vanilla TO-KWS EER is stuck at 21.17–21.25% on all four backbones (a direct manifestation of keyword scores having zero discrimination for speaker identity); PK-MTL drops to 3.37% for BC-ResNet-3, 3.22% for BC-ResNet-8, 3.68% for DS-ResNet18, 4.44% for Res15. The model-agnostic nature of the framework is thus proven.

**Overhead (Table 1, #Mult / #Param)**: BC-ResNet-3 16.7M / 63.5k → 17.5M / 80.2k (Naive MTL) → 17.5M / 82.0k (PK-MTL); BC-ResNet-8 91.9M / 386.9k → 96.6M / 501.7k; Res15 966.3M / 241.1k → 1,040.6M / 262.3k; DS-ResNet18 305.6M / 79.9k → 326.1M / 89.9k. Compared to dual-model Vanilla (+SV) 22.5M / 82.4k, the shared design achieves similar parameter levels and equivalent SV capability (SV EER 3.36% vs 3.32%) with less computation. The claim of "slight increase" holds: multiplication increase is about 5% (calculated per backbone).

**Real-scenario (Table 2, FAR@FRR 1% / 5%)**: Thresholds are first selected based on the target FRR of ts-tk positive samples on GSC, then FAR on general negative samples is measured on WSJ / Librispeech. The four Vanilla baselines have FAR@FRR1% of 41.49% (Res15), 47.44% (DS-ResNet18), 46.61% (BC-ResNet-3), 45.00% (BC-ResNet-8) on GSC+WSJ; and 33.77%, 37.46%, 27.59%, 21.83% on GSC+Librispeech—i.e., in real noise, almost every 2–5 negative samples containing the word trigger a false wake-up. PK-MTL (TB) drops to 1.32% / 0.99% on BC-ResNet-8, PK-MTL (TO) drops to 0.20% / 0.15%, with the TO mode approaching zero even at the lowest operating point. The paper's explanation: TO learned to reject nts-tk, and general negatives all come from non-target speakers, falling exactly into the rejection surface trained by TO. Notably, Vanilla performs worse on WSJ than on Librispeech (BC-ResNet-3: 46.61% vs 27.59%); a reasonable inference is that WSJ-SI200 consists of studio-recorded speech with clearer pronunciation, leading to higher keyword scores and thus easier false penetration—this is the author's mechanistic interpretation, not elaborated in the paper.

**Score Distribution Analysis (Fig. 3)**: Vanilla keyword scores can separate target words from non-target words, but many general negatives also receive high scores; the scatter plot of Naive MTL (x-axis keyword score, y-axis speaker score) shows that even if the word is the same or similar, general negatives have very low speaker scores—proving that speaker information is indeed learned into the network, but is unusable without task-adaptive scoring; In PK-MTL (TB), ts-tk scores are generally higher than nts-tk (bias toward target user is visible); In PK-MTL (TO), ts-tk is completely separated from all other quadrants.

### Findings from Ablation Experiments

**Value of multi-task itself (Table 1 first three rows)**: Vanilla (+SV) exchanges 22.5M multiplications for SV EER 3.32%; Naive MTL achieves SV EER 3.36% with 17.5M multiplications and C-KWS accuracy increases by 0.11 points (97.57% → 97.68%). Conclusion: The shared encoder freely absorbed complementary information, saving about 22% computation, and the acquisition of speaker representation comes at no cost to the main task.

**Scoring function ladder (Table 1 middle section, BC-ResNet-3)**: SCM-M (manual $\alpha$ 0.5) pulls TB-KWS FRR@1% from 96.44% to 6.28% and TO-KWS EER from 21.12% to 3.89%, but the paper explicitly points out it "cannot consistently improve TB-KWS"—fixed $\alpha$ does not change with operating points, and SCM-M's TB EER 2.12% and C-KWS FRR@10% 2.99% are not as stable as subsequent schemes. SCM-GS (grid search $\alpha$ on validation set according to target FAR) significantly improves at target operating points (TB FRR@1% 6.24%, FRR@10% 1.97%, TO EER 3.76%), but the improvement is limited because the two initial representations have not explicitly learned task characteristics. TRM (complete PK-MTL) is optimal in almost all columns: TB EER 2.02%, FRR@10% 1.52%, TO EER 3.37%, C-KWS FRR@1% 2.58% / FRR@10% 0.75%. The conclusion from the three-level ladder: inference-end arithmetic combination < combination calibrated by operating point < representation-end task shaping.

**Backbone generalization (Table 1 last three groups)**: Three backbones with very different structures and scales—Res15, DS-ResNet18, BC-ResNet-8—all benefit (TB EER respectively 3.03%→2.14%, 2.41%→1.61%, 2.43%→1.33%). Meanwhile, SV performance varies significantly with the backbone (SV EER: Res15 4.43%, DS-ResNet18 3.29%, BC-ResNet-8 2.58%), indicating that speaker discrimination power is affected by backbone capacity and structure, but personalized gains do not depend on a specific backbone—this is also the evidence for the paper's claim that it "can be applied to any C-KWS architecture."

## Main Contributions

1. **Task Formalization**: For the first time, personalized KWS is defined as two tasks, TB-KWS and TO-KWS, and a four-quadrant input classification method (ts-tk / nts-tk / ts-ntk / nts-ntk) is provided (Fig. 1a), turning the product requirement of "recognizing people" into a reproducible evaluation protocol. The two handling strategies for the nts-tk quadrant (neutral / negative sample) cover bias-type and exclusive-type device needs.
2. **PK-MTL Two-Stage Framework**: A hard-parameter sharing KWS+SV multi-task skeleton (low-level sharing, high-level sub-network diversion of antagonistic objectives) plus task-adaptive scoring (SCM / TRM), consistently effective on three backbones, with a model-agnostic framework.
3. **Unified Prototype Scoring**: Cosine classifier weights serve as keyword prototypes, registered voice embeddings serve as speaker prototypes, unifying the scoring form on both sides, solving the open-set difficulty caused by non-overlapping training/testing speakers, and strictly aligning training loss (prototype metric learning in Eq. 6) with inference criteria.
4. **Magnitude-Level Empirical Benefits**: TB-KWS FRR@1% drops from 96.50% to 6.56%, TO-KWS EER drops from 21.17% to 3.37% (BC-ResNet-3, Table 1); Real-scenario FAR@FRR1% drops from 46.61% to 5.58% (TB) / 0.15% (TO) (Table 2); Cost is approx. +4.8% multiplications, +29% parameters (calculated by the author from Table 1), and the C-KWS main task does not drop points.

## Limitations and Future Work

### Technical Limitations of the Method

- **Single-utterance registration assumption**: The entire scoring is built on the speaker embedding of a single registered voice $x_{ref}$; the impact of the registered utterance's duration, content, and channel quality on embedding stability is not reported in the paper; in actual products, the variance of speaker embeddings from single-utterance registration under noisy channels may swallow the benefits of TB/TO.
- **Linear expansion of task count**: $\text{TRM}^{tb}$ and $\text{TRM}^{to}$ are two independent modules; adding one personalized task requires training one more TRM; although SCM is training-free, $\alpha$ must be re-grid-searched for each target operating point, and the cross-operating-point transferability of $\alpha$ is not reported.
- **Ceiling of speaker discrimination power**: The speaker sub-network is trained on GSC v1 (1,881 speakers, isolated words, close-talking speech); the cross-channel, far-field, and accent robustness of its speaker embeddings have not been tested; the residual errors of TO-KWS EER 3.22–4.44% likely come from penetration by speakers with similar voices, and how these errors change under larger speaker scales is unknown.
- **Lack of hyperparameter sensitivity**: $\lambda$ is fixed at 0.1; the paper does not report sensitivity analysis for $\lambda$; nor is there an ablation for the intermediate dimension of 2 in TRM attention. These choices appear to be engineering experience values rather than search results.
- **Personalization only covers the speaker dimension**: Keywords are still the predefined set of 10 words, not supporting QbE-style custom words; the stacking of "custom words + recognizing people" dual-dimension personalization is not touched upon in this paper.

### Deficiencies in Experimental Design

- **Data scale and scenario gap**: All main experiments are completed on GSC v1 (64,727 sentences, approx. 1-second isolated words), which is significantly different from real wake-word scenarios (continuous streaming, far-field, multiple noise sources); the negative samples in the "real-scenario" evaluation (Table 2) are also 1-second simulated segments cut from WSJ / Librispeech, not real TV or meeting recordings, remaining simulation rather than field testing.
- **Missing edge-side deployment metrics**: The paper reports #Mult and #Param, but does not report deployment metrics such as streaming inference latency, peak memory, power consumption, or RTF; whether +5% multiplications equate to +5% energy consumption on a real DSP cannot be deduced from this paper.
- **Opacity in baseline protocols**: How the Vanilla / Naive MTL rows use registered voice scoring on TB-/TO-KWS tasks is not fully explained in the protocol; moreover, the TB-KWS EER (2.47%) of Vanilla in Table 1 is difficult to simultaneously hold with its FRR@FAR1% (96.50%), FRR@FAR10% (64.32%) under a single scoring protocol (constrained by ROC monotonicity), suggesting that different columns may be based on different pairings or scoring protocols, and horizontal reading should be treated with caution.
- **Narrow comparison scope**: Only compared with self-built Vanilla, Vanilla (+SV), and Naive MTL ladders, without empirical comparison with QbE personalized methods [8][9][10] or existing multi-task KWS+SV works [24][25] (differences remain at the motivation discussion level); nor is there statistical significance testing, with some differences between adjacent numbers (e.g., SCM-GS and TRM TB EER 2.40% vs 2.02%) falling on the edge of standard deviation.
- **Unstudied registration-side variables**: Key variables for personalized systems such as the number of registered voices (single vs. multiple), interval between registration and testing, and speaker state changes (cold, speech speed) are not reported in the paper.

### Possible Directions for Future Improvement

- **Multi-utterance registration and registration quality adaptation**: Replace single-utterance embeddings with average or uncertainty-weighted speaker prototypes from multiple registered voices, and study online updates of registered embeddings to track speaker state drift.
- **Task-conditioned unified TRM**: Replace one TRM per task with a shared module plus task embedding, reducing task count expansion from linear to constant, and allowing mutual regularization between TB and TO tasks.
- **Stacking with QbE**: Put "changing words" (audio-registered keyword embeddings) and "recognizing people" (speaker embeddings) into the same TRM-style framework to achieve full-dimension personalized KWS.
- **Injection of stronger speaker priors**: Use large-scale speaker data or self-supervised pre-trained speaker encoders for distillation to improve the upper limit of speaker discrimination power for small backbones (e.g., BC-ResNet-3's 3.36% SV EER), while preserving edge-side budgets.
- **Real negative samples and edge-side measurement**: Re-do Table 2-style evaluation using real TV/meeting far-field audio, and supplement streaming latency and power data to verify the cost of "+5% multiplications" on real hardware.
- **Security dimension**: Personalized KWS naturally brings a replay attack surface (others triggering wake-up using the owner's recording). Joint modeling with anti-spoofing is a natural extension, which is not discussed in the paper.
