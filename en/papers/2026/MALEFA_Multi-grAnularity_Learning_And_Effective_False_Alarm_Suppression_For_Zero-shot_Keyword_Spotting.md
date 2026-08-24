# MALEFA: Multi-grAnularity Learning and Effective False Alarm Suppression for Zero-shot Keyword Spotting

- **Authors/Affiliations**: Lo-Ya Li, Tien-Hong Lo, Berlin Chen (National Taiwan Normal University); Jeih-Weih Hung (National Chi Nan University); Shih-Chieh Huang (Realtek Semiconductor Corp)
- **Date**: April 2026 (arXiv:2604.03689v1, submitted 2026-04-04)
- **Link**: https://arxiv.org/abs/2604.03689 (official implementation: https://github.com/Debbyyy10158/MALEFA)
- **Keywords**: zero-shot keyword spotting, multi-granularity contrastive learning, false alarm suppression, cross-attention alignment, CTC phoneme alignment, precision-constrained loss, lightweight on-device deployment

## Problem Statement

### Background and Pain Points

Keyword spotting (KWS) is the human-machine entry point of voice assistants and smart devices, and is especially critical in hands-busy scenarios such as driving and gaming. Traditional KWS follows the closed-set route: a dedicated detector is trained on large amounts of pre-annotated audio collected around fixed wake words like "Hey Siri" or "OK Google". This paradigm works under controlled conditions, but a fixed vocabulary means end users cannot customize their wake word and the system cannot generalize to keywords unseen during training—yet personalization is precisely a hard requirement in real products.

Zero-shot keyword spotting (ZSKWS) was born for this: to detect any keyword, only its **text representation** is needed; the decision is made by matching audio against text, with no keyword-specific audio required for training or fine-tuning. The representative route is the cross-modal framework proposed by CMCD, which aligns audio utterances and text queries in a shared embedding space, thereby eliminating the dependence on enrollment audio.

But the paper opens by pointing out three rocks still blocking ZSKWS deployment: first, **limited compute**—KWS runs always-on on edge devices with an extremely tight compute budget; second, **limited labeled data**—annotating for every new keyword is impossible; third, and the paper's core pain point: **the high false alarm rate (FAR) caused by acoustically similar keywords**. Figure 1 gives a precise example: "come on" (phoneme sequence K-AH1-M-AA1-N) and "call mom" (K-AO1-L-M-AA1-M) are semantically unrelated, yet their phoneme sequences overlap heavily (the orange highlights in the figure). A matcher that looks only at global similarity can easily mistake "come on" for the wake word "call mom"—on an always-listening device, this kind of false alarm (FA) is the user's most directly perceivable degradation, more annoying than missed detections.

### Specific Shortcomings of Existing Methods

The paper's critique of existing ZSKWS methods concentrates on three points, each backed by Table 1 numbers:

- **Coarse-grained global representations cannot distinguish phoneme-level similar pairs**. CMCD and its successors (U2-KWS, AdaIN, etc.) improved overall alignment accuracy and generalization, but they encode a whole utterance into a single global vector before matching. Global representations are good at capturing the keyword's overall semantic contour but are inherently deficient at fine-grained pronunciation differences—the global acoustic contours of "come on" and "call mom" are highly similar, with the difference hidden in individual phonemes (AH1 vs. AO1, the trailing N vs. M). The consequence shows directly in the data: on LibriPhrase Hard (the high phoneme-confusion set), CMCD achieves only 73.58% AUC with an EER as high as 32.90%, while on the Easy set its AUC is 96.70%—the gap comes entirely from confusable pairs.
- **Large pretrained audio encoders are too computationally expensive**. MM-KWS and CED use Conformer encoders to gain cross-lingual robustness; CED reaches 92.70% AUC on LibriPhrase Hard and was the strong baseline of its time—but CED has 4.6M parameters, a substantive obstacle to real-time deployment on resource-constrained devices. Accuracy and compute are locked together on this route.
- **Training objectives are disconnected from false alarms**. The conventional approach trains the matching head with binary cross-entropy (BCE). BCE maximizes overall classification accuracy and does not explicitly penalize false positives (FP); false alarm control can only be patched afterwards via post-hoc threshold tuning. Yet the real deployment constraint is exactly that FAR must be pushed extremely low (every extra false trigger on an always-listening device is a real cost). This misalignment between training objective and deployment metric is a structural defect of the BCE paradigm.

### Key Challenges This Paper Aims to Solve

The paper must satisfy four mutually constraining goals simultaneously: (1) separate phoneme-level similar confusable pairs without any keyword-specific audio—requiring fine-grained pronunciation discriminability; (2) push the false alarm rate from PhonMatchNet's 17.879% (AMI) to near zero—requiring a training objective aimed directly at FAR rather than overall accuracy; (3) keep the model light enough (the abstract reports 650K parameters and 93M FLOPs; Table 1 records 0.7M) for real-time deployment on resource-constrained devices; (4) train using only public speech data plus noise augmentation, without large-model distillation or multilingual pretraining. The essence of the challenge: **fine-grained discrimination usually demands a larger model, and explicit false alarm optimization usually demands more data for threshold tuning—both must be achieved within lightweight, zero-shot constraints**.

## Methodology

### Overall Architecture and Design Motivation

MALEFA chains three modules: Feature Extractor → Pattern Extractor → Pattern Discriminator, plus a multi-granularity contrastive learning objective applied throughout training. The overall setup follows PhonMatchNet's experimental configuration (the paper explicitly states "experimental settings are the same as [8]"); the core increments lie in the contrastive learning and the false alarm loss.

**Dual-stream audio encoder**. The audio side runs two streams in parallel: the first sends the whole utterance through a **pretrained** speech encoder (Google speech embeddings, Lin et al., ICASSP 2020) with a 775 ms window and 80 ms shift, yielding 96-dimensional features; the second converts the raw waveform to a log-Mel spectrogram (25 ms frames, 10 ms hop) and passes it through a lightweight **trainable** convolutional projection. The two streams are concatenated into the audio embedding $E_a \in \mathbb{R}^{T_a \times 128}$ ($T_a$ is the number of frames). Why this design: the pretrained encoder is used frozen—free-riding robust acoustic representations learned from large-scale data and avoiding the compute of doing one's own large-model pretraining; the trainable convolutional stream supplements task-adaptive spectral detail, compensating for the frozen features' insensitivity to the current contrastive objective. One stream stable, one adaptive—this is the lightweight-route compromise replacing the Conformer: CED spends 4.6M parameters of Conformer for robustness; MALEFA assembles comparable robustness from "frozen small encoder + trainable small convolution". (Note: the 775 ms/80 ms and 25 ms/10 ms streams have inconsistent temporal resolutions; the paper does not explain how they are aligned before concatenation—see the limitations section.)

**Text encoder**. The keyword is first converted by G2P (g2pE) into a phoneme sequence; each phoneme is embedded through a fully connected layer with ReLU, giving $E_t \in \mathbb{R}^{T_t \times 128}$ ($T_t$ is the sequence length). Choosing phonemes rather than characters/subwords as the text unit speaks directly to the paper's pain point: acoustic confusion happens at the pronunciation level, and only by decomposing the text side to phoneme granularity can fine-grained alignment with frame-level acoustic detail on the audio side be possible. Both sides add sinusoidal positional encodings to preserve temporal order and strengthen alignment robustness.

**Cross-attention pattern extractor**. Text embeddings serve as the Query, while audio embeddings provide both Key and Value: $E_{joint} = \text{CrossAttention}(Q=E_t,\ K=E_a,\ V=E_a)$. The direction choice (text queries audio, not vice versa) embodies a clear inductive bias: each phoneme actively "searches the audio for its corresponding frames", so the attention matrix naturally forms a phoneme-frame alignment map—producing the joint representation while also providing free, visualizable, interpretable alignment evidence (the heatmaps in the paper's Figure 4 are exactly this). Compared with global matching that pools both sides into single vectors and computes similarity, this preserves sequence-level structural correspondences—which is precisely what distinguishing "come on / call mom" requires.

**GRU discriminator with two heads**. $E_{joint}$ passes through a GRU into two classification heads: one outputs the utterance-level match probability $q_{utt}$ (does this audio match this keyword overall), the other outputs a phoneme-level alignment sequence over time $q_{phon}$ (is each position aligned). Two heads correspond to two granularities of supervision, providing separate prediction entries for the two-level contrastive learning below.

### Mathematical Principles of the Core Algorithm

The training objective is the equally-weighted sum of six losses:

$$\mathcal{L}_{total} = \mathcal{L}_{utt} + \mathcal{L}_{phon} + \mathcal{L}_{CTC} + \mathcal{L}_{PCL} + \mathcal{L}_{UCL} + \mathcal{L}_{FA}$$

Here $\mathcal{L}_{utt}$ and $\mathcal{L}_{phon}$ are BCE supervision on the two classification heads; $\mathcal{L}_{CTC}$ is the base supervision for phoneme-level alignment; $\mathcal{L}_{PCL}$ and $\mathcal{L}_{UCL}$ are the two-level contrastive losses; $\mathcal{L}_{FA}$ is the false-alarm-aware loss. The paper sets all six weights to 1 and explicitly states that weight-sensitivity exploration is beyond its scope (a candid simplification, and a criticizable one).

**CTC base alignment**. The audio encoder outputs frame-level CTC logits $z \in \mathbb{R}^{T_a \times V}$ ($V$ is the phoneme vocabulary size), supervised with the standard CTC loss:

$$\mathcal{L}_{CTC} = -\log q_{CTC}(y \mid z)$$

$q_{CTC}$ marginalizes over all legal frame-phoneme alignment paths. This item inherits PhonMatchNet's core mechanism: once CTC is trained, **Viterbi decoding** yields the optimal alignment path and its confidence—PCL takes exactly this alignment confidence as the supervision signal for contrastive learning. This is the key point where the "alignment" and "contrast" mechanisms mesh in the framework.

### Key Innovation 1: Phoneme-level Contrastive Learning (PCL)

**Motivation**: a correct utterance-level match score does not mean correct phoneme-frame alignment. A model that aligns "call" onto the frames of "come" but whose overall score happens to pass the threshold is a false alarm landmine buried in the system. PCL's goal is to turn alignment quality itself into an explicit training signal.

**Mechanism**: for each audio-text pair in the batch, Viterbi decoding gives the alignment confidence $s_i$, and a regression-style contrastive loss is constructed:

$$\mathcal{L}_{PCL} = \frac{1}{N}\sum_{i=1}^{N}\left[\, m_i(1-s_i)^2 + (1-m_i)s_i^2 \,\right]$$

where $m_i \in \{0,1\}$ indicates whether the $i$-th pair truly matches. Behavioral interpretation: matched pairs ($m_i=1$) keep only the $(1-s_i)^2$ term, pushing alignment confidence toward 1; mismatched pairs ($m_i=0$) keep only the $s_i^2$ term, crushing spurious overlap toward 0. The form is squared error rather than InfoNCE-style softmax contrast; the benefit is that each pair's confidence is constrained independently and pairwise (positive pairs full, negative pairs empty), instead of only requiring "the positive pair's score to beat the other negatives in the batch"—the latter still leaves false alarm room when negative pairs are all collectively high, whereas the $s_i^2$ term pushes down each negative pair's absolute score too, corresponding to what the paper calls "penalizes spurious overlaps".

**Evidence of effect** (Figure 4): on the keyword "hey android", without PCL the attention is diffuse and phoneme boundaries are blurry (exactly the kind of imprecise boundary that breeds false alarms); with PCL the alignment sharpens and localizes markedly, indicating the model learns more discriminative frame-level representations.

### Key Innovation 2: Utterance-level Bidirectional Contrastive Learning (UCL)

**Motivation**: PCL governs local alignment, but global inter-class separation needs utterance-level constraints—Figure 3 shows the baseline's cosine similarity on confusable pairs (e.g., "bed" vs. "three") remains stubbornly high: a problem of the global embedding space not being pulled apart.

**Mechanism**: within a mini-batch of $M$ pairs, compute the $M \times M$ similarity matrix $S_{utt}$; compute BCE losses in both directions—text-to-audio ($s^{text}_{v,r}$) and audio-to-text ($s^{audio}_{v,r}$)—and average:

$$\mathcal{L}_{UCL} = \frac{1}{2}\left(\ell^{text} + \ell^{audio}\right)$$

$$\ell^{*} = -\frac{1}{M^2}\sum_{v=1}^{M}\sum_{r=1}^{M}\left[\, m_{v,r}\log\sigma(s^{*}_{v,r}) + (1-m_{v,r})\log\left(1-\sigma(s^{*}_{v,r})\right) \,\right]$$

where $m_{v,r}=1$ means audio $v$ matches text $r$. The **bidirectional** design (one direction each for text→audio and audio→text) applies the constraint symmetrically to both sides of the embedding space, avoiding the unbalanced solution where only one side is pulled apart while the other remains crowded. The mini-batch size is $M=5$; the paper's stated rationale is "balancing stability and discrimination"—the number of in-batch negative pairs is $M^2-M=20$, too few weakens the discriminative signal, too many destabilizes training; 5 is the empirical compromise.

**Evidence of effect** (the triple comparison of Figure 3): the baseline (left) shows large bright off-diagonal similarity; adding UCL (middle) suppresses mismatched similarity wholesale and cleans up inter-class separation; adding PCL as well (right) further sharpens diagonal matches while mismatched scores approach zero. The division of labor between the two contrastive levels is very clear in the visualization: UCL handles global inter-class separation, PCL handles fine-grained alignment sharpening, and the two are complementary.

### Key Innovation 3: False Alarm-aware Loss

**Motivation**: this is the paper's blade aimed straight at FAR. BCE optimizes overall accuracy; under the real deployment distribution where "non-target audio vastly outnumbers target audio", the cost of FPs is systematically underestimated; engineering can only compensate by tuning thresholds post hoc, and once the dataset or deployment environment changes the threshold must be re-tuned. The paper's idea is to write "Precision" directly into the training objective—precision is TP/(TP+FP), naturally putting FP in the denominator: the mathematical incarnation of false alarms.

**Formula**: a margin-bearing precision-constrained objective (the idea originates from Rath & Hughes' minimum-precision-constrained work, AISTATS 2022):

$$\mathcal{L}_{FA} = -\log(\text{Precision}) + \lambda \cdot \max(0,\ \alpha - \text{Precision})$$

The first term is a smooth log barrier—the lower the precision, the greater the penalty; the second is a hinge term requiring precision to be no lower than the margin $\alpha$, else penalized weighted by $\lambda$. The paper takes $\alpha = 0.9$ and $\lambda = 10.0$.

**Differentiable treatment**: true TP/FP are non-differentiable counts; the paper approximates them with smooth sigmoid bounds ($x$ is the predicted match score, $x_{true}\in\{0,1\}$ the label):

$$\text{TP} = \sum (1+\gamma\delta)\,\sigma(\gamma x - \delta)\, x_{true}, \qquad \text{FP} = \sum (1+\gamma\delta)\,\sigma(\gamma x + \delta)\,(1 - x_{true})$$

Steepness $\gamma = 7.0$ controls how closely the soft gate approaches hard counting, offset $\delta = 0.035$ sets the decision bias, and $(1+\gamma\delta)$ is a compensation factor (the paper calls these smooth sigmoid bounds; the direction of the approximation's deviation from true counts is not further analyzed). Note the two gates' offsets point in opposite directions: the TP gate $\sigma(\gamma x - \delta)$ requires $x$ to exceed the bias before counting positive; the FP gate $\sigma(\gamma x + \delta)$ lets even slightly elevated scores on negative samples be counted as FP—a conservative approximation that prefers overestimating errors. $\mathcal{L}_{FA}$ is used alongside BCE rather than replacing it, effectively adding an extra gradient channel that "beats FP to death" while preserving overall accuracy.

**Magnitude of effect**: this loss is the largest single contributor among all ablations—removing it (w/o FA) explodes AMI FAR from 0.007% to 14.542%, nearly knocking MALEFA back into PhonMatchNet's original shape (17.879%); see the experiments section for details.

### Technical Differences from Existing Methods

- **Versus CMCD-family global matching**: CMCD aligns with a single global representation; MALEFA uses "utterance-level + phoneme-level" dual granularity and makes phoneme-frame alignment explicit (the cross-attention alignment map + Viterbi confidence). Direct evidence of the difference on LibriPhrase Hard: CMCD AUC 73.58 / EER 32.90, MALEFA 93.58 / 13.91.
- **Versus PhonMatchNet (direct predecessor, same 0.7M parameters and the same feature/experimental setup)**: MALEFA retains its CTC/Viterbi matching backbone and adds the cross-attention joint representation, the GRU dual discriminator heads, UCL, PCL, and the FA-aware loss. One could say MALEFA answers exactly the question "how much more water can be squeezed from PhonMatchNet's backbone"—at the same parameter count, Q-set ACC4 rises from 80.45% to 98.77%, and AMI FAR drops from 17.879% to 0.007%, an order-of-magnitude reduction.
- **Versus CED / MM-KWS (the heavyweight Conformer route)**: CED spends 4.6M parameters for LPH AUC 92.70; MALEFA achieves 93.58 with about 1/6 the parameters (0.7M) and also lower EER (13.91 vs. 14.70)—the paper argues from parameter count and four-dataset comprehensiveness that "you can win without relying on large pretrained models".
- **Versus CLAD / ADML (the metric-learning route)**: CLAD (contrastive + audio discrimination, 2.2M) and ADML (adversarial metric learning, 1.8M) are both surpassed by MALEFA on LibriPhrase (LPH: 76.15 and 88.71 vs. 93.58); MALEFA's differentiation is turning "false alarms" from a byproduct of training into the training objective itself.

## Experimental Results

### Datasets Used and Their Scale

- **Training**: the train-clean-100 and train-clean-360 subsets of LibriPhrase, augmented with MUSAN noise. The paper does not report the exact number of positive/negative training pairs.
- **Evaluation on four public benchmarks**: (1) LibriPhrase Easy / Hard ($L_E$/$L_H$), split from train-other-500 and constructed by low/high phoneme confusion—the Hard set is purpose-built for confusable pairs and is the main battlefield of this paper's story; (2) Google Speech Commands V2 (G), 35 command words under diverse recording conditions; (3) Qualcomm Keyword Speech (Q), accented, domain-specific keywords testing out-of-domain generalization; (4) AMI, 12 hours of meeting recordings cut into 2-second segments, dedicated to false alarm evaluation—a pure ocean of "non-target audio" testing whether the model mistakes everyday conversation for wake words (the keyword set and decision threshold used for FAR evaluation are not stated in the paper).

### Definitions and Rationale of Evaluation Metrics

- **AUC (↑)**: area under the ROC curve, threshold-free, measuring the overall ranking quality of match scores;
- **EER (↓)**: equal error rate, the FAR=FRR intersection, likewise threshold-free, the common currency of speaker verification/open-set detection;
- **ACC4 (↑, reported only on Q)**: the paper body gives no formal definition; following the convention of the Qualcomm/LibriPhrase line of work it is the Top-1 accuracy of a four-way choice (1 positive + 3 hard-negative texts), threshold-free—this explains why it complements FAR: one asks "can you pick the right one", the other asks "do you dare run always-on";
- **FAR (↓)**: the rate at which non-target audio falsely triggers the keyword—this paper's headline metric.
The logic of the metric combination: AUC/EER give threshold-free academic comparability, ACC4 and FAR give deployment-perspective hard constraints; together the four metrics close off the fake-optimization space of "pretty AUC but exploding false alarms".

### Detailed Comparison with Baseline Methods and SOTA

**Table 1 (AUC/EER/ACC4, four datasets)** key numbers:

| Method | AUC: G / Q / $L_E$ / $L_H$ | EER: G / Q / $L_E$ / $L_H$ | ACC4(Q) | Params |
|---|---|---|---|---|
| CMCD | 81.06 / 94.51 / 96.70 / 73.58 | 27.25 / 12.15 / 8.42 / 32.90 | – | – |
| PhonMatchNet* | 98.11 / 98.90 / 99.29 / 88.52 | 6.77 / 4.75 / 2.80 / 18.82 | 80.45 | 0.7M |
| CED | – / – / 99.84 / 92.70 | – / – / 1.70 / 14.70 | – | 4.6M |
| CLAD | – / – / 97.03 / 76.15 | – / – / 8.65 / 30.30 | – | 2.2M |
| ADML | – / – / 99.86 / 88.71 | – / – / 1.33 / 20.09 | – | 1.8M |
| **MALEFA** | **99.13 / 99.81 / 99.98 / 93.58** | **3.88 / 1.92 / 1.14 / 13.91** | **98.77** | 0.7M |

(PhonMatchNet carries an asterisk; the PDF-extracted text shows no footnote explaining it—by convention it should be the authors' reproduction.) Three readings worth expanding:

1. **Main battlefield $L_H$ (high confusion)**: MALEFA's AUC 93.58 leads everyone—5.06 points above the same-parameter-count PhonMatchNet, and even 0.88 points above the 4.6M-parameter CED with 0.79 points lower EER. Note ADML is the closest competitor on the Easy set: $L_E$ AUC 99.86 / EER 1.33 versus MALEFA's 99.98 / 1.14—MALEFA slightly better on both but the gap is small; the real watershed is the Hard set: ADML falls to 88.71 (EER 20.09) while MALEFA holds 93.58 (EER 13.91). **MALEFA's advantage is in essence an "anti-confusion advantage"**, dovetailing exactly with the design goal of multi-granularity learning.
2. **The leap in Q-set ACC4**: 80.45% → 98.77% (+18.32 percentage points), the most intuitive usability gain at equal parameter count.
3. **Lightweight**: 0.7M parameters achieving comprehensive best on four datasets (Table 1 caption: best overall performance with only 0.7M parameters); the abstract additionally gives 650K parameters / 93M FLOPs (the two figures are rounding of each other).

**Table 2 (FAR, %)**—the paper's most dramatic table:

| Method | AMI | G | Q |
|---|---|---|---|
| PhonMatchNet* | 17.879 | 7.438 | 5.743 |
| **MALEFA** | **0.007** | **0.002** | **0.000** |
| w/o PCL | 0.085 | 0.019 | 0.105 |
| w/o UCL | 1.334 | 3.580 | 0.029 |
| w/o FA | 14.542 | 6.710 | 0.690 |

Relative to PhonMatchNet, AMI FAR drops by roughly four orders of magnitude (17.879% → 0.007%), G from 7.438% → 0.002%, Q from 5.743% → 0.000% (zero at three decimals). The abstract summarizes this as "FAR below 0.01% across all benchmarks".

### Findings of the Ablation Study

The paper runs three ablations (all at the unchanged 0.7M parameters), and the findings are richer in layers than the headline totals:

- **Removing the FA-aware loss is the largest single collapse**: G-set EER 3.88 → 9.85, Q-set 1.92 → 8.16, $L_H$ EER 13.91 → 21.10, ACC4 98.77 → 84.19, AMI FAR 0.007% → 14.542%. Note the collapsed FAR (14.5%) almost hugs PhonMatchNet's original value (17.9%)—**this ablation practically proves that MALEFA's false alarm advantage is contributed mainly by the FA-aware loss alone; the architecture itself (without the FA loss) is not inherently low-false-alarm**.
- **Removing PCL concentrates the damage on the "hard" places**: $L_H$ AUC 93.58 → 87.64 (-5.94), EER 13.91 → 20.29, ACC4 98.77 → 91.80, AMI FAR 0.007% → 0.085%. But intriguingly, on the easy sets PCL is actually slightly negative: removing PCL raises G-set AUC from 99.13 to 99.41 and lowers EER from 3.88 to 3.82; Q-set AUC 99.81 → 99.91 and EER 1.92 → 1.22 also improve. This non-monotonic phenomenon shows PCL's role is not a universal score booster but **specifically supplying fine-grained discrimination**: on easy sets where global ranking is already good enough, forcibly sharpening alignment is a mild constraint conflict; its value cashes out only on confusable pairs. The paper's own wording—"removing PCL degrades fine-grained alignment on $L_H$"—skirts the slight degradation on easy sets; keep this in mind when reading the tables.
- **Removing UCL mainly hurts FAR and robustness**: AMI FAR 0.007% → 1.334%, G → 3.580% (worsening by more than two orders of magnitude), G-set EER 3.88 → 4.78; but ACC4 barely moves (98.76 vs. 98.77) and Q-set FAR only reaches 0.029%. Combined with the Figure 3 visualization (UCL's contribution is suppressing mismatched similarity and pulling classes apart), the division of labor reads: **four-way-choice tasks like ACC4 rely on the relative ranking of positives against hard negatives, which PCL+FA already supports; whereas FAR demands that negative pairs' absolute scores be pressed below the threshold—exactly what UCL's inter-class separation provides**. Why Q-set FAR stays low after removing UCL (0.029%) while G-set explodes (3.580%) is unexplained in the paper; it may relate to the two test sets' negative-sample composition.
- **Concluding statement**: the paper argues from this that PCL, UCL, and FA are complementary and jointly essential—the FA loss governs the false alarm fundamentals, UCL governs absolute inter-class separation, PCL governs fine-grained alignment on confusable pairs.

Additional experimental setup: Google speech embeddings as the pretrained audio encoder; 50 epochs, Adam, fixed learning rate $10^{-3}$, batch size $N=1000$, UCL mini-batch $M=5$; single NVIDIA RTX 4090, TensorFlow implementation.

## Main Contributions

1. **A unified multi-granularity contrastive learning framework**: the first (in the ZSKWS context) to unify utterance-level and phoneme-level contrastive objectives in one lightweight framework—utterance-level captures the keyword's global semantics, phoneme-level captures fine-grained pronunciation; the two levels complement each other, verified both experimentally and via the cosine-similarity visualization (Figure 3).
2. **The false alarm-aware loss**: brings a precision constraint ($-\log(\text{Precision})$ + hinge margin) into ZSKWS training after differentiating it via smooth sigmoid bounds, optimizing FAR directly at the gradient level and replacing the patch-style post-hoc threshold tuning; this is the component contributing most to the paper's FAR drop from 17.9% to 0.007%.
3. **Lightweight on-device feasibility**: 0.7M parameters (abstract's 650K / 93M FLOPs), SOTA on four benchmarks (99% AUC, ~1% EER, FAR 0.007%), taking both accuracy and false alarms without relying on a Conformer-class large encoder—the paper positions it as a feasible solution for resource-constrained deployment.
4. A methodological by-product: the cross-attention alignment map (Figure 4) gives ZSKWS rare interpretability—one can directly see which frames each phoneme aligns to and how the boundaries sharpen before and after PCL.

## Limitations and Future Work

### Technical Limitations of the Method

- **Streaming and latency entirely unaddressed**: the model processes whole utterances offline (evaluation cuts 2-second segments); the paper reports no streaming inference scheme, measured latency, RTF, or memory footprint—93M FLOPs is the sole efficiency evidence. Real wake-word devices need frame-by-frame streaming decisions and wake-moment localization; how the GRU discriminator and the whole-segment cross-attention can be streamed is a question that must be answered before deployment and that the paper does not answer.
- **A suspect point in the dual-stream temporal resolution**: the pretrained encoder's 775 ms window / 80 ms shift and the log-Mel 25 ms frame / 10 ms hop differ by 8× frame rate; the paper only says the two streams concatenate into $T_a \times 128$ without explaining how the two time grids are aligned (downsampling? replication? unreported)—reproducers must guess.
- **No hyperparameter sensitivity analysis**: $\mathcal{L}_{FA}$'s $\gamma=7.0$, $\delta=0.035$, $\alpha=0.9$, $\lambda=10.0$, and the equal-weight combination of six losses (all weights 1) are single-point choices; the paper explicitly admits weight exploration is out of scope. Given the FA loss's outsized contribution, its hyperparameters' robustness directly determines the method's transferability—whether the $\alpha=0.9$ precision margin still suits a different data distribution cannot be judged.
- **PCL depends on CTC/Viterbi quality**: the alignment confidence $s_i$ comes from Viterbi decoding after CTC converges; in early training the confidence itself is unreliable, and PCL using it as a regression target may then be amplifying noise (which may explain PCL's slight negative effect on easy sets); the paper does not discuss curriculum-style or delayed-enablement strategies for PCL.
- **Language binding**: G2P maps to an English phoneme set (ARPAbet style); the framework is inherently bound to English—cross-lingual extension is the paper's self-admitted unfinished item.

### Shortcomings of the Experimental Design

- **No statistical robustness**: all results are single runs—no multi-seed means/variances, no significance tests; whether ablation differences at the 0.5-point scale (e.g., G-set AUC 99.13 vs. 99.41) exceed random fluctuation cannot be determined.
- **Incomplete FAR evaluation protocol**: AMI's 12 hours of meetings are cut into 2-second segments for false alarm evaluation, but the keyword set used, how the decision threshold was set, and whether the threshold is fair and uniform across methods are all unreported—FAR is a threshold-sensitive metric; without this layer of information, the comparability of the headline 0.007% figure is discounted.
- **ACC4 reported only on Q**; the other three datasets lack this metric; CED/CLAD/ADML likewise report only the LibriPhrase subsets (marked "–" in the table)—the full four-benchmark comparison holds only against CMCD and PhonMatchNet, so parts of the SOTA comparison are stitched together.
- **No adversarial/noised robustness testing**: training used MUSAN noise augmentation, but the evaluation sets contain no dedicated noise/accent stress dimension (Q has accents but that alone); no dedicated evaluation of "non-speech sound false triggers" (TV audio, similar phoneme strings mixed into music)—a real false alarm source that AMI's meeting corpus, covering "people speaking without the keyword", does not cover.
- **Lightweight claims lack measurements**: the comparison with CED stops at parameter counts (0.7M vs. 4.6M), with no latency or power measurements on edge hardware (e.g., ARM/DSP); the "resource-constrained deployment" conclusion currently rests only indirectly on FLOPs.

### Possible Future Directions

- **Cross-lingual extension**: the direction the paper itself lists in its conclusion—generalizing G2P and the phoneme vocabulary to multiple languages and testing the transferability of multi-granularity contrast and the FA loss across languages (converging with MM-KWS's cross-lingual route is a natural conjecture).
- **Streaming conversion**: converting utterance-level decisions into sliding-window incremental decisions, restricting cross-attention to a causal variant, and supplementing RTF/latency/memory measurements—the unavoidable step from "offline matcher" to "deployable wake engine".
- **Automation of loss weights**: six equal weights are an obvious optimization point; one could explore gradient-magnitude-adaptive weighting (e.g., GradNorm-style schemes) or curriculum-style enabling of PCL by training stage (avoiding the unconverged-CTC period).
- **A head-on solution for thresholding**: the FA loss reduces but does not eliminate dependence on post-hoc thresholds (FAR evaluation still needs an operating point); combining temperature calibration or open-set score normalization so a single threshold stays stable across datasets is the key to turning 0.007% from "tuned into" to "guaranteed".
- **A more complete false alarm evaluation**: adding stress tests for three false alarm sources—non-speech events, TV/broadcast pass-through audio, cross-speaker phoneme variation—and reporting multi-seed variance; for a paper titled around false alarm suppression, this is the most valuable self-reinforcement.
