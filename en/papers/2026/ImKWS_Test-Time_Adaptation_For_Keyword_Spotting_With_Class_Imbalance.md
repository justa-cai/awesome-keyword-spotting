# ImKWS: Test-Time Adaptation for Keyword Spotting with Class Imbalance

- **Authors/Affiliations**: Hanyu Ding, Yang Xiao, Jiaheng Dong, Ting Dang - Jiangsu University & University of Melbourne (the first two authors contributed equally)
- **Date**: 2026.03 (arXiv:2603.05821v2, updated 2026-06-17)
- **Link**: https://arxiv.org/abs/2603.05821
- **Keywords**: keyword spotting, test-time adaptation (TTA), class imbalance, decoupled entropy minimization, multi-view consistency loss, macro F1, BC-ResNet

## Problem Statement

### Problem Background and Pain Points in the Field

Keyword spotting (KWS) is the voice entry point for smart device control, voice assistants, and voice search; its core engineering constraint is to be as accurate as possible under low power and low compute. Yet such systems fail frequently once they leave the training distribution: in real acoustic environments, sudden noise significantly changes audio patterns, and as soon as the test distribution drifts, detection performance drops. There are three traditional paths for repairing this kind of drift, each with a hard flaw in on-device deployment scenarios:

1. **Supervised fine-tuning** requires a small batch of labeled target-domain data. In dynamic deployment environments, noise is sporadic; it is impossible to anticipate the drift and collect and label data on the spot.
2. **Unsupervised domain adaptation (UDA)** needs no target labels, but requires access to the raw source-domain training data. For resource-constrained devices, continuously storing or transmitting source data both occupies memory and carries privacy risks.
3. **Do nothing** and accept the accuracy collapse — at −10 dB SNR, the unadapted model's macro F1 on a 1:8 imbalanced test stream is only 61.87 (ESC-50 noise, Table 1), nearly unusable.

Test-time adaptation (TTA) is proposed precisely for this constraint: using only the unlabeled test stream and a single forward pass, the model is updated while inferring, requiring neither source data nor labels. TTA already has mature recipes in vision (TBN, Tent, SAR, ETA) and in large-model ASR (SUTA updates the feature layers and normalization layers and applies temperature smoothing).

But KWS has a **second pain point that none of the above methods confronts: extreme class imbalance**. In a continuous speech stream, the wake word is a rare event and background segments dominate absolutely — the ratio of keyword to non-keyword samples is naturally and severely skewed. Lightweight KWS models, moreover, rely on short audio windows with limited contextual information, making them especially sensitive to error accumulation from unreliable high-entropy samples. The two combined produce a TTA-specific failure mode: **the update magnitude of entropy minimization is dominated by background samples**; during adaptation the model becomes ever more confident in the background class, the decision boundary keeps drifting toward the background class, and rare keyword events become ever harder to detect. This is not a hypothesis — measured at 1:8 imbalance and −10 dB, the macro F1 of the standard EM-family methods (Tent/SAR/ETA) is dragged down by the background class, with the gap to ImKWS widening from +0.67 all the way to +2.64 as imbalance intensifies (Table 2, MS-SNSD).

### Specific Shortcomings of Existing Methods

- **TBN (Test-time Normalization)**: only refreshes the BN statistics with target data, has no loss function, and has no capability for any scenario that requires actively correcting the decision boundary. In Table 1, TBN's micro F1 on 10 dB MS-SNSD (93.55) is even lower than the unadapted model (94.65) — purely re-estimating statistics is actually a negative gain at high SNR.
- **Tent / ETA / SAR**: this family of methods derived from visual TTA all center on (some form of) entropy minimization; the loss is **symmetric** over all classes and completely oblivious to class-prior shift. On a stream where more than 90% of samples are background, they execute the goal of "make the model confident" as "make the model certain that everything is background".
- **AdaKWS (Interspeech 2025)**: the most direct prior work in the KWS setting, which uses selective entropy minimization to address small models' sensitivity to high-entropy samples, and is effective in itself (in Table 1 it consistently beats Tent/SAR/ETA). But it **ignores a key reality**: the sample ratio of keyword to background is extremely imbalanced in continuous speech. During entropy minimization most updates are driven by background segments; the model gradually becomes overconfident in the background class, and wake-word detection sensitivity is sacrificed.
- **Supervised imbalance-handling methods do not transfer**: training-phase class imbalance has standard prescriptions — class-frequency reweighting and resampling. But TTA has no labels, so class frequencies cannot be counted and these methods all fail. This forces the problem to be solved at a different level: **act on the geometric shape of the loss function rather than on sample weights**.

### Key Challenges This Paper Addresses

The paper claims to be the first work to study test-time adaptation for KWS in "realistic imbalanced scenarios" (original text: "To our knowledge, this is the first study to explore TTA for KWS in realistic imbalanced scenarios"). It must simultaneously satisfy four mutually constraining requirements:

1. **No source data, no labels, single-pass streaming** — the baseline requirement of TTA; looking back over the data for a second pass is not allowed;
2. **Suppress majority-class (background) logit explosion without sacrificing minority-class (keyword) detection sensitivity** — i.e., break the seesaw of "raising recall means accepting background false-alarm inflation"; macro F1 and micro F1 must improve in the same direction;
3. **Gradient stability** — the gradient-norm fluctuations and spikes brought by the imbalanced stream must be suppressed, otherwise the adaptation trajectory oscillates and becomes unpredictable;
4. **On-device affordability** — the backbone is a lightweight network on the order of BC-ResNet-3, and the adaptation overhead must not spiral (the paper updates, and only updates, the BN affine parameters).

## Methodology

### Overall Architecture Design and Design Motivation

ImKWS's complete inference–adaptation pipeline (Figure 1) is as follows:

```
Test-stream sample x
   │
   ├─ Two-stage sample selection (Eq. 7): L_dem(x) < 0.4 and L_pkc(x, x') > 0.05
   │        ├─ Selection 1: decoupled-entropy threshold (low entropy = the model is confident about this sample)
   │        └─ Selection 2: pseudo-keyword consistency threshold (pseudo-label confidence before vs. after filtering)
   │
   ├─ Each selected sample x_s generates two augmented views x̃_s, x̂_s (time masking + frequency masking)
   │
   └─ Total loss (Eq. 8): L_total = w(x_s)·L_dem(x_s) + λ·L_consist(x_s, x̃_s, x̂_s)
            │
            └─ SGD updates only the BN-layer affine parameters (lr = 1e-4, batch = 128, single pass)
```

The "why" behind the three architectural decisions:

- **Updating only the BN affine parameters**: the same choice as Tent. The affine parameters are tiny in count (negligible relative to the whole network); this both confines the adaptation capacity to a safe low-dimensional subspace to keep the model from drifting away, and naturally requires storing no source data. For a direction that emphasizes on-device deployment, this is the trade-off point between compute and stability.
- **Single pass**: real deployment has no "run it again" option. All experiments are done under the single-pass setting, which is also the defining constraint that distinguishes TTA from unsupervised fine-tuning.
- **Select first, then adapt**: directly inherits AdaKWS's insight — short-window lightweight models are most afraid of being led astray by high-entropy noise samples, so perform a reliability selection first, then drive updates with reliable samples. ImKWS's delta: the first gate of selection replaces standard entropy with its own decoupled entropy L_dem (the paper's original text explicitly states "actually implemented via DEM here"), and after selection it adds a layer of reliability-based **continuous weighting** (Eq. 9) instead of a hard binary choice.

### Mathematical Principles of the Core Algorithm

**Natural decomposition of the entropy loss (Eq. 1)**. Let the model output logits $z = M_\theta(x) \in \mathbb{R}^C$, with softmax probabilities $p_i = e^{z_i} / \sum_{k=1}^{C} e^{z_k}$. Using $\log p_i = z_i - \log\sum_k e^{z_k}$, the conditional Shannon entropy can be split exactly into two terms:

$$L_{\text{ent}}(x) = -\sum_{i=1}^{C} p_i \log p_i = \underbrace{-\sum_{i=1}^{C} p_i z_i}_{T(z)\ \text{(reward)}} + \underbrace{\log \sum_{i=1}^{C} e^{z_i}}_{Q(z)\ \text{(penalty)}}$$

This decomposition is not an identity-manipulation game; the two terms have completely different gradient semantics: $T(z) = -\sum_i p_i z_i$ is the negative of the probability-weighted expectation of the logits, and minimizing it amounts to **raising the logits of the classes the model already believes in** (the reward branch, producing "confidence"); $Q(z) = \log\sum_i e^{z_i}$ is a log-sum-exp whose gradient with respect to each logit is exactly $p_j$, and minimizing it amounts to **pushing down the overall level of all logits** (the penalty branch). Standard EM ties these two forces to the same coefficient 1.0; the paper's core move is to pull them apart and give each its own knob.

**Reward branch (Eq. 2)**: introduce a temperature $\tau$ to control the sharpness of the distribution,

$$T_\tau(z) = -\sum_{i=1}^{C} p_i^\tau z_i, \quad p_i^\tau = \frac{e^{z_i/\tau}}{\sum_{k=1}^{C} e^{z_k/\tau}}$$

**Penalty branch (Eq. 3)**: introduce a scaling factor $\alpha$,

$$Q_\alpha(z) = \alpha \log \sum_{i=1}^{C} e^{z_i}$$

**Gradient analysis (Eq. 4)** — the single most critical equation in the paper. Differentiating $L_{\text{dem}} = T_{1.0}(z) + Q_\alpha(z)$ with respect to some logit $z_j$ (this author verified it step by step via the differentiation rules, and the result matches the paper: the $T$ term gives $p_j(\mathbb{E}_p[z] - z_j - 1)$, the $Q_\alpha$ term gives $\alpha p_j$; adding them yields):

$$\frac{\partial L_{\text{dem}}}{\partial z_j} = p_j(z)\left(\sum_{i=1}^{C} p_i(z)\, z_i - z_j - (1-\alpha)\right)$$

Translated into plain language: for a non-target class $j \neq \arg\max z_i$, its logit is relatively small, and the difference of the first two terms inside the brackets (the probability-weighted mean logit minus $z_j$) is usually positive. Under standard EM ($\alpha = 1.0$) the entire bracket is positive, so gradient descent **keeps pushing down** $z_j$, and round after round of updates keeps concentrating confidence mass on the dominant class. By contrast, $\alpha < 1$ amounts to **subtracting a fixed positive margin $(1-\alpha)$ from the gradient**: the push-down action happens only when $z_j$ is far enough from the probability-weighted mean (the gap exceeds the margin); once the gap narrows to within the margin, the gradient changes sign and the push-down stops. In other words, $\alpha$ sets an **equilibrium point** for the descent of non-target logits, preventing the network from driving non-target predictions all the way to $-\infty$ — precisely a regularization against the "one-hot-ization into the majority background class through overconfidence". Note that the gradient is also multiplied on the outside by $p_j(z)$, meaning the suppression strength is proportional to that class's own probability; the imbalance is accumulated repeatedly on the stream through the sheer **number** of background samples, and DEM's margin mechanism acts exactly on this accumulation pathway.

**Multi-view consistency (Eqs. 5, 6)**. Given the input $x$, construct two augmented views $\tilde{x}$ (time/frequency masking); the consistency loss is:

$$L_{\text{consist}}(x, \tilde{x}, \hat{x}) = L_{\text{sce}}(x, \tilde{x}) + L_{\text{sce}}(x, \hat{x})$$

$$L_{\text{sce}}(x, \tilde{x}) = -\frac{1}{2}\left[\sum_{i=1}^{C} p_i(z) \log p_i(\tilde{z}) + \sum_{i=1}^{C} p_i(\tilde{z}) \log p_i(z)\right]$$

This is the symmetric cross-entropy (SCE, Wang et al. 2019): the average of the cross-entropy in both directions. The paper gives two reasons for choosing SCE over plain CE or KL: SCE has strong tolerance to label noise — in TTA the pseudo-labels are the model's own (unreliable) predictions, inherently carrying noise, and SCE still provides a stable training signal when predictions are biased; additionally, SCE balances the learning of easy and hard classes.

**Two-stage sample selection (Eq. 7) and the total objective (Eqs. 8, 9)**:

$$x_s = \{x \mid L_{\text{dem}}(x) < \tau_{\text{dem}},\ L_{\text{pkc}}(x, x') > \tau_{\text{pkc}}\}, \quad L_{\text{pkc}}(x, x') = p(x)_c - p(x')_c$$

where $c$ is the model's pseudo-label on the original input and $x'$ is the transformed input. Samples passing the selection enter the total objective:

$$L_{\text{total}} = w(x_s) \cdot L_{\text{dem}}(x_s) + \lambda \cdot L_{\text{consist}}(x_s, \tilde{x}_s, \hat{x}_s)$$

$$w(x) = \frac{1}{\exp\{L_{\text{dem}}(x) - \sigma\}} + \frac{1}{\exp\{-L_{\text{pkc}}(x, x')\}}$$

The weight $w(x)$ is the sum of two exponential gates: the lower the decoupled entropy (the more it falls below the normalization factor $\sigma$), the larger the first term; the closer the pseudo-label confidence before and after the transformation (the smaller the $L_{\text{pkc}}$), the larger the second term. It can be read as: **selection guarantees the floor (high-entropy samples and samples with unstable pseudo-labels are kept out), weighting provides the ceiling (the more reliable a sample, the greater its say)**. A detail worth pondering: selection stage two requires $L_{\text{pkc}} > 0.05$ (the original view must be more confident than the transformed view), while the weighting prefers a small $L_{\text{pkc}}$ — together they compose a soft-resampling logic of "once past the stability threshold, the more stable, the higher the weight", a continuous version adapted from AdaKWS's resampling scheme.

### Key Technical Innovation 1: Decoupled Entropy Minimization (DEM)

**The problem it solves**: standard EM pushes the majority class toward pathological confidence on imbalanced streams.

**Core mechanism**: as in Eqs. (1)–(4) of the previous section, split the entropy into a reward branch and a penalty branch, with the temperature $\tau$ governing the reward branch's sharpness and the scaling factor $\alpha$ governing the penalty branch's strength. At the gradient level, the essence is to add a $(1-\alpha)$ margin to the push-down gradient on non-target logits, so that the majority class's logit explosion brakes at an equilibrium point.

**Why this rather than per-class reweighting**: TTA has no labels and cannot count class frequencies, so any explicit mechanism that "knows which class is the minority" is unavailable. DEM's cleverness lies in **not needing to know class identity** — the margin treats all classes equally, yet it naturally acts only on "the classes being repeatedly suppressed" (i.e., the competitors of the background class that dominates the stream and is being continuously pushed up by EM), because imbalance is itself a count-driven cumulative effect. This is a class-agnostic surgery on the loss geometry that nonetheless directionally mitigates majority-class collapse.

**A detail that must be honestly pointed out**: the final hyperparameters were set by grid search to $\tau = 1.0$ and $\alpha = 0.8$. $\tau = 1.0$ means the reward branch effectively degenerates to the standard form; the temperature mechanism is **not actually activated** in the final configuration — the working ingredient is the penalty-branch margin with $\alpha = 0.8$. The paper reports no sensitivity curves for $\tau$ and $\alpha$, so the reader cannot judge whether the $\tau$ knob is redundant design or was conservatively abandoned by the grid search.

### Key Technical Innovation 2: Multi-view Consistency Loss

**The problem it solves**: the side effects when DEM is used alone. The paper is candid about this: while DEM suppresses majority-class overconfidence and prevents minority-class gradient signals from being drowned out, it "inevitably amplifies the relative influence of individual noisy samples and increases gradient-norm fluctuations". This is intuitively easy to understand — once the penalty branch is weakened by the margin, the loss constraint loosens, and the aberrant gradients of a few unreliable samples gain relatively greater say.

**Core mechanism**: apply two SpecAugment-style augmentations to the same sample (two time masks of maximum length 20 plus two frequency masks of maximum length 5), and use SCE to constrain the predictions of the original view and the two augmented views to agree. When the model is highly uncertain and produces divergent logits for perturbed inputs, $L_{\text{consist}}$ acts as a **spatial regularizer**, suppressing the occasional high-amplitude gradient spikes introduced by DEM's penalty relaxation. The gradient-norm box plots in Figure 3 verify this directly: after adding consistency, the upper tail of the gradient distribution is visibly flattened, and a stable, monotonic adaptation trajectory is maintained even under 1:8 extreme imbalance.

**Why two views instead of one**: Eq. (5) takes one term each for $\tilde{x}$ and $\hat{x}$; the two mask combinations differ, so the consistency constraint covers a broader range of perturbation directions, at the cost of two extra forward passes per step (the paper does not report this computational overhead; see Limitations).

### Key Technical Innovation 3: Two-stage Sample Selection and Confidence Weighting (a modification built on the AdaKWS skeleton)

The paper explicitly states that the sample selection strategy is "based on AdaKWS" ("employs a robust two-stage sample selection strategy based on AdaKWS"), so it is not itself original; ImKWS's modifications are threefold: the first selection gate replaces standard entropy with the decoupled entropy $L_{\text{dem}}$ (same origin as the main loss, keeping the selection criterion internally consistent); on top of selection it adds the continuous sample weight of Eq. (9) (AdaKWS is selective resampling, whereas ImKWS is a two-level "selection + continuous weighting" scheme); and it plugs the multi-view consistency into the update objective applied to the selected samples. The ablations show that this inherited skeleton is not redundant — on MS-SNSD, removing selection is the heaviest-scoring drop among the three ablations (69.91 → 68.00), indicating that under single-source real noise, keeping bad samples out matters more than reshaping the loss.

### Technical Differences from Existing Methods

| Method | Adaptation mechanism | Handling of class imbalance | Essential difference from ImKWS |
|---|---|---|---|
| TBN | Re-estimates BN statistics only | None | No loss function, no decision-boundary correction capability |
| Tent | Standard entropy minimization (updates BN affine) | None | Symmetric loss; inevitably collapses when background dominates updates |
| ETA | High-entropy sample filtering + EM | None (filtering is by entropy, not by class) | Filtering addresses sample reliability, not class-prior tilt |
| SAR | Sharpness-aware optimization + reliable EM | None | Gradient stability comes from optimizer geometry; the loss itself remains symmetric |
| AdaKWS | Selective EM + pseudo-keyword-consistency resampling | None | Closest baseline; ImKWS directly builds on its skeleton, swapping EM for DEM and adding weighting and consistency |
| SUTA | Single-utterance TTA for ASR foundation models | Not aimed at KWS | Serves large models on long speech; does not handle imbalanced streams of lightweight short-window models |
| **ImKWS** | DEM + multi-view consistency + two-stage selection with weighting | **At the loss-geometry level (gradient margin)** | Directionally stops majority-class logit explosion without needing class labels |

One-sentence summary of the differences: vision-line TTA works on the "sample reliability" dimension (ETA/SAR), and AdaKWS brings that idea into KWS; ImKWS points out that KWS streams have a second dimension — **class-prior tilt** — and, under the constraint of being unable to count class frequencies, fixes this problem at the gradient-geometry level by decomposing the entropy loss and multiplying the penalty branch by $\alpha < 1$.

## Experimental Results

### Datasets Used and Their Scale

- **Core dataset**: Google Speech Commands v2 (GSC v2, 16 kHz), described in the paper as "12 classes", split 80% / 10% / 10% into train/validation/test. The exact number of utterances and total duration are not reported.
- **Task construction**: from the 12 classes, the three words "yes", "up", "stop" are chosen as the positive keyword classes, and the remaining nine classes are merged into a single non-keyword class, forming a 4-class classification task. Training and validation keep the source distribution; **the test set is resampled to construct class imbalance**, with the keyword:non-keyword ratio pushed stepwise from 1:4 down to 1:8.
- **Noise and SNR**: the test audio is corrupted with ESC-50 multi-source noise and five real single-source noises from the MS-SNSD test set, at three SNR levels: −10 / 0 / 10 dB. The exact noise-mixing procedure (the additive mixing formula, alignment by energy or by amplitude) is not reported.
- **Implementation configuration**: backbone BC-ResNet-3 (the lightweight on-device KWS network from Interspeech 2021); input is 40-dimensional MFCC, with the paper's original text reading "160 ms hop length" (the common KWS configuration is a 10 ms hop; this is suspected to be a typo, and the paper does not clarify); at test time batch 128, SGD with learning rate 1e-4, single-pass adaptation, updating only the BN affine parameters; hyperparameters set by grid search to $\tau = 1.0$, $\alpha = 0.8$, $\lambda = 1.0$, $\tau_{\text{dem}} = 0.4$, $\tau_{\text{pkc}} = 0.05$, $\sigma = 0.5$. Code is open-sourced on GitHub (github.com/dhyzy123/ImKWS).

### Definition and Rationale of the Evaluation Metrics

The paper reports **macro F1** (primary metric) and **micro F1**. Its stated rationale is direct: macro F1 averages over classes, preventing the dominant non-keyword class from masking the recognition performance of the minority keyword classes; micro F1 pools globally, and is used to monitor "whether gains in keyword sensitivity were bought with background false alarms". The data themselves prove that using micro alone would mislead: for the unadapted model under ESC-50 −10 dB, as imbalance rises from 1:4 to 1:8, micro F1 actually climbs from 85.97 to 91.32 (Table 2) — the higher the majority-class proportion, the better micro looks, while keyword detection has not actually improved. The fixed-operating-point false-alarm/miss rates (FAR/FRR) or DET curves commonly used in KWS engineering are not reported.

### Detailed Comparison with Baseline Methods and SOTA

**Baselines**: TBN, Tent, SAR, ETA, AdaKWS (the most direct predecessor).

**Table 1 (1:8 imbalance, three SNR levels, format macro/micro F1)**, ESC-50 noise:

| Method | −10 dB | 0 dB | 10 dB |
|---|---|---|---|
| Unadapted | 61.87 / 91.32 | 74.06 / 93.46 | 81.91 / 95.10 |
| TBN | 69.14 / 89.83 | 77.41 / 92.65 | 83.15 / 94.56 |
| Tent | 68.99 / 89.83 | 77.32 / 92.66 | 82.86 / 94.44 |
| SAR | 69.35 / 89.95 | 77.14 / 92.60 | 82.80 / 94.46 |
| ETA | 69.29 / 89.88 | 77.27 / 92.62 | 82.66 / 94.43 |
| AdaKWS | 69.68 / 90.25 | 77.55 / 92.72 | 82.89 / 94.47 |
| **ImKWS** | **70.91 / 91.20** | **78.98 / 93.57** | **84.51 / 95.23** |

MS-SNSD noise: ImKWS achieves 69.91 / 91.82, 76.49 / 93.13, 81.46 / 94.43 (−10/0/10 dB), and AdaKWS 66.95 / 89.95, 74.30 / 91.97, 79.96 / 93.63. The macro F1 gains over the strongest baseline AdaKWS are +1.23 / +1.43 / +1.62 on ESC-50 and **+2.96 / +2.19 / +1.50** on MS-SNSD — the harsher the noise (−10 dB), the larger the gain. Equally critical, micro F1 rises in the same direction (up to +1.87 on MS-SNSD −10 dB), showing that the penalty-branch margin did not buy keyword recall with inflated background false alarms — the precision-recall trade-off has genuinely been broken. Two additional observations: first, all EM-family baselines have micro F1 below the unadapted model at 10 dB MS-SNSD (e.g., TBN 93.55 vs 94.65), i.e., under mild drift adaptation actually hurts majority-class accuracy, whereas ImKWS at ESC-50 10 dB remains above the unadapted model on both metrics (84.51/95.23 vs 81.91/95.10), with only a slight micro F1 dip at MS-SNSD 10 dB (94.43 vs 94.65); second, the joint rise of macro and micro is itself direct evidence of the DEM mechanism's effectiveness.

**Table 2 (−10 dB, imbalance 1:4 → 1:8, mean±std over multiple runs; the number of runs is not reported)**: ImKWS is best across both noise sets and all five ratio levels. In macro F1 (ESC-50): 75.24±0.13, 74.31±0.18, 73.58±0.28, 72.00±0.08, 70.76±0.14 (from 1:4 to 1:8); the corresponding AdaKWS numbers are 74.58±0.18, 73.50±0.29, 72.47±0.14, 71.02±0.14, 69.31±0.37. On MS-SNSD, ImKWS achieves 71.33±0.05, 71.28±0.41, 71.27±0.19, 69.87±0.02, 69.73±0.16. The paper uses this data to support the core claim of "the regularization effect of decoupling the penalty branch": **the more extreme the imbalance, the more severely standard EM's gradient bias accumulates, and the larger ImKWS's lead** — on MS-SNSD the macro F1 advantage over AdaKWS widens from +0.67 at 1:4 to +2.64 at 1:8, and on ESC-50 the lead also holds from 1:4 all the way to 1:8. Another cross-cutting detail: at 1:4 on ESC-50, TBN (74.67) is still slightly above AdaKWS (74.58), showing that under mild imbalance the methods are not far apart; differentiation happens precisely after 1:6 — exactly the applicability domain the paper claims. Note also a small discrepancy between ImKWS's 1:8 numbers in Table 1 (70.91/91.20) and the means in Table 2 (70.76±0.14/91.07±0.12), which the paper does not explain (presumably Table 1 is a single run and Table 2 the multi-run mean).

**Figure 2 (MS-SNSD, −10 dB, per-class F1 as the ratio varies)**: the typical failure mode of standard EM is that the decision boundary tips toward the majority class — non-keyword F1 becomes inflated while keyword F1 collapses. In the figure, ImKWS strictly maintains (and often improves) non-keyword F1 while raising keyword F1. The paper argues on this basis that decoupling the penalty branch is **calibrating the majority-class logits** rather than blindly lowering the detection threshold, so the gain in keyword sensitivity does not introduce excess background false alarms.

### Findings from the Ablation Studies

**Table 3 (1:8, −10 dB; full-configuration ImKWS is 70.91/91.20 (ESC-50) and 69.91/91.82 (MS-SNSD))**:

| Variant | ESC-50 ma/mi | MS-SNSD ma/mi | macro drop |
|---|---|---|---|
| w/o DEM (reverts to standard EM) | 70.24 / 90.62 | 68.39 / 90.75 | −0.67 / −1.52 |
| w/o Consistency | 69.96 / 90.53 | 69.06 / 91.15 | −0.95 / −0.85 |
| w/o Selection | 70.19 / 90.62 | 68.00 / 90.56 | −0.72 / −1.91 |

Three findings:

1. **All three components are positive, but their importance ranks differently by noise type**. Under multi-source noise (ESC-50), removing the consistency loss costs the most (−0.95), indicating that gradient stability is the primary bottleneck in mixed-noise environments; under single-source real noise (MS-SNSD), removing sample selection costs the most (−1.91), indicating that bad samples from the five real noises are more lethal, and blocking samples matters more than reshaping the loss. DEM costs −0.67 / −1.52 on the two sets respectively. The lesson for practitioners: the returns of mechanism components do not follow a universal ranking but depend on the statistical structure of the noise.
2. **DEM is the primary mechanism preventing majority-class logit explosion** (the paper's own conclusion); removing it reverts to standard EM, and MS-SNSD macro F1 falls from 69.91 to 68.39, landing between AdaKWS (67.09) and the full configuration — roughly, DEM accounts for about sixty percent of the gain over AdaKWS.
3. **The benefit of the consistency loss is more visible at the gradient level than at the metric level**: the box plots in Figure 3 show that with DEM only, the gradient-norm distribution has a long tail and extreme outliers (axis range 0–30), and after adding $L_{\text{consist}}$, the upper tail is visibly flattened at both 1:4 and 1:8. This confirms the causal chain of the design narrative: DEM relaxes the penalty → noisy samples' say is relatively amplified → gradient spikes → the consistency constraint backstops as a spatial regularizer.

Dimensions the ablations do not cover: no separate ablation of the reward-branch temperature $\tau$ (e.g., the softening effect of τ>1), no sensitivity curve for $\alpha$, no ablation of the weighting term in Eq. (9), and no reported selection pass rate (what fraction of the test samples actually participated in the updates).

## Main Contributions

1. **Problem definition**: the first work to explicitly bring "class imbalance" into the problem space of test-time adaptation for KWS, identifying and empirically demonstrating the majority-class collapse failure mode of standard EM on imbalanced streams — a gap in the prior TBN/Tent/SAR/ETA/AdaKWS line of work.
2. **Method**: proposes decoupled entropy minimization (DEM), which splits the conditional entropy exactly into a reward branch and a penalty branch, configuring the temperature $\tau$ and the scaling factor $\alpha$ respectively, and uses the gradient margin $(1-\alpha)$ to stop the runaway inflation of majority-class logits without using any class labels — a reusable solution to the contradiction of "how to handle class imbalance without labels".
3. **Stability**: proposes the SCE-based multi-view consistency loss to specifically compensate for the gradient fluctuations introduced by DEM's penalty relaxation, verifying the mechanism directly via the gradient-norm distribution (Figure 3) rather than only looking at final metrics.
4. **Empirics**: on GSC v2 it constructs 1:4–1:8 imbalance at −10/0/10 dB across two noise sources, fifteen test conditions in total, with ImKWS best on all of them; the macro F1 advantage over the strongest baseline AdaKWS widens as imbalance intensifies (+0.67 → +2.64 on MS-SNSD), and the joint rise of macro/micro proves the precision-recall seesaw is broken. Code is open-sourced.

## Limitations and Future Work

### Technical Limitations of the Method

- **The gradient margin is a global scalar and does not distinguish classes**. $(1-\alpha)$ treats all logits of all classes equally; the "directional protection of the minority class" effect is indirect — it relies on the dynamical fact that "imbalance is a count-driven cumulative effect". When the imbalance becomes extreme enough (or the keyword class happens to be the model's current argmax class), whether the protection of a fixed margin still suffices receives no analysis or upper-bound argument in the paper.
- **The temperature mechanism is effectively disabled**. Grid search picks $\tau = 1.0$, the reward branch degenerates to the standard form, and half of the design space of Eq. (2) is dead code in the final system. Without sensitivity curves for $\tau$ and $\alpha$, the reader cannot judge the method's robustness bandwidth — whether $\alpha = 0.8$ is a global optimum or an arbitrary point in a flat region cannot be determined.
- **Computational overhead is not reported**. Each adaptation step requires forward passes for the original view + two augmented views + the PKC-transformed view (about four forwards in total), plus the backpropagation for the SCE consistency. For a direction aimed at lightweight on-device models (BC-ResNet-3), the paper does not report this extra compute, the per-step latency, or the energy consumption — yet it is exactly what determines "whether single-pass TTA is feasible on MCU-class devices".
- **Test-time batch = 128 conflicts with true streaming**. Both the BN statistics and the within-batch gradients depend on a batch size of 128, while the real wake-word scenario is a single stream arriving frame by frame / segment by segment; where the batch comes from is a question that deployment must answer and that the paper does not discuss.
- **A fairly wide hyperparameter surface**. Six hyperparameters — $\tau$, $\alpha$, $\lambda$, $\tau_{\text{dem}}$, $\tau_{\text{pkc}}$, $\sigma$ — are determined by grid search, and all rely on a labeled validation environment to be chosen — strictly speaking this already brushes the boundary of the TTA "label-free" setting (hyperparameter selection uses supervised signal), and the paper does not discuss how these thresholds transfer across noises and ratios.

### Shortcomings of the Experimental Design

- **A single data surface**: only one core dataset (GSC v2), three keywords, one backbone. The gap between utterance-level 4-class classification and real continuous streaming KWS detection (sliding windows, false alarms counted per hour) is large; the paper's "stream" is simulated by resampling the test set, not a genuine continuous audio stream.
- **The imbalance intensity cap is too mild**: the experiments go at most to 1:8, while the reality the paper itself argues for in the introduction is that "the keyword-to-background ratio is extremely imbalanced" — the imbalance of a real wake-word stream can reach hundreds of times and beyond. Whether conclusions holding at 1:8 still hold at 1:100 is precisely the key test point of the method's claimed value, and the paper does not touch it.
- **The evaluation metrics lack the KWS industry standard**: no false-alarm/miss rates at fixed operating points, no FAR/FRR, no DET-style curves, and no per-keyword F1 breakdown (yes/up/stop individually); macro F1 is the average over the three keywords and could mask an individual word's failure.
- **Questionable implementation details**: the paper's original "160 ms hop length" contradicts the common KWS configuration (10 ms hop), suspected to be a typo but affecting reproduction; there is an unexplained discrepancy of about 0.1 between the full-configuration numbers in Table 1 and Table 2; the number of repeated runs is not reported; the sample-selection pass rate (how many samples actually drive the updates) is not reported — and this directly concerns the method's actual behavior on imbalanced streams.
- **SUTA missing from the baselines**: the paper's introduction discusses SUTA as a representative speech TTA work, yet the experiments do not include it in the comparison (although it targets ASR foundation models and cross-backbone comparison is difficult, the trade-off should at least have been stated).

### Possible Future Directions

- **Direction stated by the paper**: extend the decoupled framework to memory-constrained on-device learning scenarios — also the natural way to make up for this paper's "unreported computational overhead" shortcoming.
- **Class-aware or adaptive margin scheduling**: upgrade the fixed $\alpha$ to an $\alpha_t$ that adjusts dynamically to the confidence distribution on the stream (e.g., a soft prior from a sliding estimate of pseudo-label frequencies), approaching class-aware reweighting without introducing labels; or apply a finer per-class decomposition to the reward and penalty branches.
- **More extreme imbalance and true-streaming evaluation**: test the boundaries of the margin mechanism at 1:50 and 1:100 magnitudes and on continuous audio streams (FAR at a fixed recall) — the touchstone of the method's claims.
- **Small-batch / single-sample settings**: combine with BN alternatives (e.g., group norm or anchor-free statistics) to remove the dependence on batch 128, making the method genuinely deployable to wake streams arriving segment by segment.
- **Cost-benefit curves**: report the Pareto relationship among per-step forward count, latency, and macro F1, answering "how many milliwatts the stability bought by multi-view consistency is worth".
- **Combination with personalized KWS**: the "background-dominated updates" problem in the paper's motivation only becomes sharper in user-defined wake-word scenarios (where samples are even rarer); combining the decoupled-entropy framework with enrollment-style methods is an intersection with real promise.
