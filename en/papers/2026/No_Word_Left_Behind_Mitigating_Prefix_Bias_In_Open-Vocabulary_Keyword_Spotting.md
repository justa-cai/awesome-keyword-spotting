# No Word Left Behind: Mitigating Prefix Bias in Open-Vocabulary Keyword Spotting

- **Authors/Affiliations**: Yi Liu (Department of Electrical and Computer Engineering, University of California, San Diego); Chuan-Che (Jeff) Huang, Xiao Quan (Bose Corporation) — This work was completed during the first author's internship at Bose.
- **Date**: 2026.02 (arXiv v3 submitted on 2026-02-11; officially published at ICASSP 2026, May 3-8, 2026, Barcelona, Spain)
- **Link**: https://arxiv.org/abs/2602.08930
- **Keywords**: open-vocabulary keyword spotting, prefix bias, partial overlap benchmark, equal-weighting position scoring, audio-text joint embedding, synthetic hard negatives, edge deployment

## Problem Statement

### Problem Background and Domain Pain Points

Open-Vocabulary Keyword Spotting (OV-KWS) allows users to register device control commands using arbitrary phrases, rather than selecting from a fixed command list. The typical scenarios presented in the paper stem from real product requirements of headphone/speaker manufacturers like Bose: customizing voice shortcuts for devices ("turn the light on", "open the garage") or triggering skills via voice on portable gaming consoles (shouting "thunderbolt" to unleash a move). Compared to closed-set KWS with predefined vocabularies, OV-KWS grants registration freedom to users, but at the cost that the system must determine whether the "query audio" matches the "registered text" during inference, rather than relying on a fixed classification head to handle all words.

The current mainstream paradigm is a three-stage neural architecture (Fig. 1 in the paper): (1) Text and audio encoders project registered text and query audio into a shared joint embedding space, yielding $E_t \in \mathbb{R}^{m\times D}$ and $E_a \in \mathbb{R}^{n\times D}$; (2) An alignment/matching module (cosine similarity, cross-attention, or affinity matrix) computes aligned representations between the two modalities; (3) A scoring layer (pooling / GRU / FC) aggregates the aligned representations into a matching score $z$. During training, auxiliary heads such as CTC phoneme heads can be attached for training-only purposes. To align the two modalities, previous works have extensively used phoneme-level auxiliary objectives, contrastive learning, and adversarial metric learning.

The pain point lies in the distribution of real-world usage: multi-word commands registered by users often have similar semantics and share prefixes. An example given in the abstract is "turn the volume up" and "turn the volume down", which share the prefix "turn the volume". If the scoring layer places significant discriminative weight on the initial phonemes, a query for "turn the volume down" may receive a high score because it matches the complete prefix of the registered "turn the volume up", leading to false triggers. While this is merely annoying for volume control, it constitutes a safety incident for game skill triggers or security commands. Furthermore, this problem is rarely detectable on standard benchmarks like LibriPhrase or Google Speech Commands (GSC)—these benchmarks themselves contain very few prefix-overlapping samples.

### Specific Deficiencies of Existing Methods

The paper attributes the defects of existing solutions to two mutually amplifying lines of reasoning:

(1) **Distribution skew in evaluation and training data**. Prefix-overlapping phrases are rare in LibriPhrase and GSC, and the overall dataset is biased towards phrases of two words or fewer. The paper analyzes LibriPhrase using the "first-different phoneme index" (formally defined later, see Fig. 3): the vast majority of positive and negative sample pairs diverge within the first 0-4 phonemes, with extremely low proportions falling into long-prefix intervals such as [10,14], [15,19], or [20,24]. Consequently, models have not seen negative samples with long shared prefixes during training, nor are they penalized for such errors during evaluation—the scenario of multi-word registration (more than 3 words, i.e., more than 10 phonemes) is systematically ignored.

(2) **Structural positional bias in the scoring layer**. Two sub-1M parameter SOTA models for edge deployment treat "position" as a learnable degree of freedom: SLiCK (ICASSP 2025) uses a flattened fully connected layer with softmax for utterance-level scoring; PhonMatchNet (Interspeech 2023) uses a GRU followed by a fully connected layer with sigmoid scoring. Through weight visualization (Fig. 5), the paper directly observes that SLiCK's positional contribution is highly concentrated in the beginning of the sequence—after training, the model prioritizes matching the beginning of audio-text pairs, ignoring mismatches in the latter half. A notable "why" here: SLiCK's subsequence matching auxiliary loss is its core contribution, designed specifically to distinguish pairs with shared phoneme components like "blue" vs. "glue". However, the paper's empirical findings reveal that even with this loss, prefix bias persists—indicating that constraints at the loss level do not propagate to the positional allocation of scoring weights, and the structural bias is independent of the training objective.

(3) **The line of hard negative sample synthesis ignores the "position" dimension**. Works such as GE2E-KWS, GraphemeAug, and LLM-Synth4KWS use generative models to synthesize negative samples that are "phonetically similar but semantically different". These methods control content similarity but do not control the positional distribution of mismatches, thereby failing to systematically evaluate and repair prefix robustness.

### Key Challenges Addressed by This Paper

First, how to transform "partial overlap" from a manually described failure case into a measurable, controllable evaluation object—requiring formal definitions, metrics, and dedicated benchmark data. Second, how to eliminate the prefix bias in the scoring layer without retraining encoders, increasing parameters, or modifying the alignment module—this determines whether the repair solution can be directly applied to deployed models. Third, how to avoid sacrificing performance on original benchmarks and single-word commands (GSC) after introducing prefix-overlapping data into training—the paper honestly confirms that this trade-off exists and has not been fully resolved, establishing it as an open problem.

## Methodology

### Overall Architecture Design and Design Motivation

The methodological stance of this paper is very restrained: it proposes no new encoders, no new alignment modules, and narrows the research scope to the final stage of the three-stage architecture—the scoring layer. The design motivation is variable isolation: only the final scoring layer is replaced, while the encoders and alignment modules remain completely unchanged (the paper explicitly emphasizes "Only the final scoring layer is changed; the rest of the pipeline is unchanged"). This ensures that all performance differences in the experiments can be cleanly attributed to the scoring design itself, rather than changes in the overall architectural capability. The experimental carrier chosen is SLiCK—the strongest OV-KWS model in the sub-1M parameter category at the time of publication, whose cross-attention alignment layer has a 25-phoneme length constraint to reduce computational cost. The variant with only the scoring layer changed is named SLiCK-EPS. There is an implicit motivation for choosing a sub-1M model for experiments: such models are inherently designed for edge hardware like headphones and speakers, so the repair solution must be lightweight.

### Mathematical Principles of Core Algorithms

**Partial Overlap (Definition 2.1)**: Let the registered phrase be the token sequence $x = (x_1, \dots, x_p)$, and the token sequence corresponding to the query audio be $y = (y_1, \dots, y_q)$. If

$$y_i = x_i \quad \forall i = 1, \dots, q \ \wedge\ (p > q)$$

then $(x, y)$ is said to exhibit partial overlap—the query and the registered phrase's prefix completely coincide, but the registered phrase is longer and diverges from the query content at subsequent positions.

**First-Different Phoneme Index**: For any two words or phrases, $i_{\text{diff}}$ is defined as the phoneme position where the two first differ. This serves as a ruler to quantify the "degree of prefix overlap" in a dataset: the distribution of $i_{\text{diff}}$ in LibriPhrase is concentrated in the first 0-4 phonemes, indicating that it cannot measure long-prefix robustness.

**Prefix Bias (Definition 3.1)**: Let the hidden variables output by the alignment/matching module be $X = [X_1, \dots, X_m] \in \mathbb{R}^{m \times n}$, where $X_i$ encodes the degree of match between the $i$-th phoneme position of the registered text and the query audio; the scoring layer assigns position-dependent weights $A = [a_1, \dots, a_m]$ to each position. The expected contribution of position $i$ is defined as

$$C_i = \mathbb{E}_{X_i}\left[|a_i^\top X_i|\right]$$

Furthermore, the **Prefix Concentration Score** is defined as:

$$\rho(k) = \frac{\sum_{i=1}^{k}\|a_i\|_2}{\sum_{i=1}^{m}\|a_i\|_2}$$

This measures the proportion of scoring weights occupied by the first $k$ positions. If weights are uniform, $\rho(k)$ should follow the diagonal $\rho(k) = k/m$; if the curve is significantly higher than the diagonal, prefix bias exists. The value of this set of definitions lies in transforming a vague qualitative observation ("the model favors the beginning") into a structural diagnostic quantity that can be read directly from trained weights, without requiring any inference experiments.

**Equal-Weighting Position Scoring (EPS)**: Replace "flatten + position-dependent FC" with "shared linear mapping + average pooling":

$$z_i = w^\top X_i, \qquad z = \frac{1}{m}\sum_{i=1}^{m} z_i + b$$

where $w: \mathbb{R}^n \to \mathbb{R}$ is a single weight vector shared across all positions, and $b$ is the bias. All positions contribute with equal weight, eliminating prefix bias structurally. Mechanistically, why this repairs false triggers: in a biased scorer, early positions dominate the total score; once the prefix matches, the score is elevated, and tail mismatches have no "veto power". With equal weighting, the total score is the mean of match degrees at each position, so any segment mismatch directly lowers the total score.

### Key Technical Innovation 1: Partial Overlap Benchmark (POB-LP and POB-Spark Dual Datasets)

POB consists of two complementary datasets with different design motivations:

**POB-LP (POB-LibriPhrase)**: Derived from LibriPhrase, the approach appends a word taken from the 10,000 most common English word list to the registered text sequence to simulate prefix overlap, while deliberately maintaining the same phrase length distribution as the original LibriPhrase. Why construct it this way: it reuses real human recordings from LibriSpeech, modifying only the text side, so that negative sample pairs strictly satisfy Definition 2.1 (query = complete prefix of registration), making it a clean test for "pure prefix negative samples". Adhering to the original length distribution ensures comparability with LibriPhrase results.

**POB-Spark**: Synthesized using Spark-TTS (a single-stream decoupled speech token text-to-speech model based on LLMs), providing controlled overlap patterns across speaker characteristics, with balanced phrase length distributions (unlike POB-LP, which is constrained by the LibriPhrase distribution). Why TTS synthesis is necessary: to sample the $i_{\text{diff}}$ of negative sample pairs according to a target distribution (approximately uniform), one needs complete control over "which word to replace and with what", which real corpora cannot provide. The construction process involves six steps:

1. **Word-Phoneme Mapping**: Map each word $w$ in the common word list $W$ to the CMU phoneme sequence $P(w)$, with length denoted as $L(w) = |P(w)|$;
2. **Nearest Neighbor**: For each $w$, use the Levenshtein distance $\delta(\cdot, \cdot)$ to measure phonetic similarity and find its phonetic neighbors;
3. **Phrase Construction**: Randomly sample a word sequence $\{w_i\}$ satisfying $\sum_i L(w_i) < L_{\max}$ to form a phrase;
4. **Pair Construction**: Randomly select a word $w_j$ from the phrase and replace it with its phonetic neighbor $w_j'$ to obtain a negative sample phrase pair;
5. **Sampling**: Calculate the actual pronunciation and $i_{\text{diff}}$ for all candidate pairs using G2P, and sample according to an approximately uniform distribution of $i_{\text{diff}}$;
6. **Table Formation**: Each phrase pair $(a, b)$ is expanded into three "query-anchor-label" triplets: $(a, b, \text{False})$, $(b, a, \text{False})$, and $(a, a, \text{True})$, adding bidirectional negative samples and positive samples.

Step 2 uses "phonetic neighbor replacement" rather than random word replacement to concentrate the difficulty of negative samples on the positional dimension rather than the content dimension—the replaced words sound very similar to the original words, so the only discriminative clue the model can utilize is "at which position the mismatch occurs". The uniform sampling in Step 5 is the methodological core of the entire benchmark: it transforms the "mismatch position" from a confounding variable into a controlled independent variable. The dataset is open-sourced (github.com/cijinsama/Partial-Overlap-Benchmark).

### Key Technical Innovation 2: Formal Definition and Diagnostic Method for Prefix Bias

The pair of definitions $C_i$ and $\rho(k)$ (see the Mathematical Principles section) is itself a contribution. Why is it needed: observing only a 40-point drop in EER cannot distinguish whether the issue lies in training data or model structure; $\rho(k)$ provides a structural diagnostic readable directly from weights—Fig. 5 shows empirical results at $m=25$ positions, where the per-position contribution $C_i$ of the SLiCK baseline exhibits a left-heavy distribution, and the $\rho(k)$ curve rises steeply, significantly above the diagonal; after switching to EPS, $C_i$ becomes nearly uniform, and $\rho(k)$ basically hugs the diagonal. This diagnostic chain (change scoring layer → change weight distribution → change benchmark performance) is a key link in the paper's attribution argument.

### Key Technical Innovation 3: EPS (Equal-weighting Position Scoring) Module

EPS is a surgical replacement of the final scoring layer (formula see Mathematical Principles section), bringing three engineering properties, each with a clear "why":

(1) **Parameter count decreases rather than increases**: From 580K in SLiCK to 557K in SLiCK-EPS, a net reduction of 23K. The reason is that the weight matrix scale of the position-dependent FC is proportional to $m \times n$ (each position and each feature dimension has independent weights), whereas the shared weight vector has only $n$ parameters—replacing "position-specific parameters" with "weight sharing" exactly removes the degrees of freedom that produced the bias.

(2) **Decoupling from fixed input length**: SLiCK's flattening operation requires the sequence length before the matcher to be fixed (hence the 25-phoneme constraint in the cross-attention layer), whereas EPS's per-position shared mapping plus mean pooling is computable for any $m$. This leaves structural room for future relaxation of length constraints and support for longer registered phrases.

(3) **Interpretability**: Each position contributes equally to the total score. Once an error occurs, one can directly locate the match vector at the mismatch position, without needing to perform attribution within a flattened FC layer with hundreds of thousands of parameters.

The paper's self-positioning of EPS is clear: it is a minimal, interpretable baseline, not the optimal solution—uniform weights discard all positional discriminative information. Constrained dynamic attention or positional weight regularization might find a better point between "not re-introducing bias" and "improving discriminative power", leaving these for future work.

### Technical Differences with Existing Methods

Compared to two direct baselines: PhonMatchNet uses self-attention for phoneme-level and utterance-level alignment, with GRU+FC+sigmoid scoring; SLiCK uses cross-attention alignment (25-phoneme length constraint), flatten+FC+softmax scoring, and a subsequence matching auxiliary loss. This paper does not touch the first two stages, only replacing the third with EPS, which structurally lacks positional degrees of freedom, and pairs it with a new benchmark that exposes the defects of the third stage. Compared to the line of hard negative sample synthesis (GE2E-KWS, GraphemeAug, LLM-Synth4KWS): they control the content similarity of negative samples, whereas POB controls the positional distribution of mismatches, and POB-Spark turns position into a controlled experimental variable through $i_{\text{diff}}$ uniform sampling—the two are orthogonal and can be stacked. Compared to alignment structure innovations such as CTC forced alignment and dynamic sequence partitioning: this paper explicitly does not touch the alignment module, advocating for fixing the cheapest lesion first—the scoring layer.

## Experimental Results

### Datasets Used and Their Scales

Two training data configurations: LibriPhrase only (182,570 samples), or LibriPhrase + POB training set (90,808 samples, totaling approximately 273k). Five evaluation sets: LibriPhrase-easy, LibriPhrase-hard, Google Speech Commands (GSC), POB-Spark, and POB-LP. The paper does not report the number of samples in the POB-Spark and POB-LP test sets; the value of $L_{\max}$ in POB-Spark construction and the number of synthesized speakers are not reported; all data is in English. Training configuration: Adam optimizer (2500 steps warm-up), batch size 1024, trained for 50k steps, phonemes encoded using the CMU phoneme dictionary (73 tokens), implemented in PyTorch, trained on 4 RTX 4090 GPUs on x86 Linux. Baseline reproduction methods: PhonMatchNet uses the official PyTorch implementation (github.com/ncsoft/PhonMatchNet), SLiCK is reproduced according to the original paper's configuration, and the reproduction numbers on LibriPhrase/GSC are close to the values reported in the original papers—this ensures the fairness of subsequent comparisons.

### Definition and Rationale for Evaluation Metrics

LibriPhrase, GSC, and POB-Spark use EER (Equal Error Rate, lower is better) and AUC (Area Under the ROC Curve, higher is better); POB-LP uses ACC (Accuracy). The paper does not explicitly state the rationale for metric selection, but EER/AUC are standard threshold-independent metrics for verification binary classification tasks, avoiding beautification caused by working point selection; the triplet structure of POB-LP (bidirectional negative pairs + positive pairs) naturally forms a balanced binary classification, allowing accuracy to be read directly. A detail to note regarding the paper's terminology: the "35.1% EER reduction" mentioned in the abstract refers to the absolute percentage point difference (64.4% − 29.3%), not a relative reduction.

### Detailed Comparison with Baseline and SOTA Methods

Table 1 presents a 2×3 experimental matrix: two training mechanisms (LibriPhrase only / with POB data augmentation) × three models. The former isolates architectural effects (EPS itself), while the latter isolates training data composition effects.

**Mechanism 1: Training on LibriPhrase Only** (Parameter counts: PhonMatchNet 655K, SLiCK 580K, SLiCK-EPS 557K) —

- PhonMatchNet: LP-easy EER 4.49% / AUC 99.02%; LP-hard 23.18% / 84.55%; GSC 10.28% / 95.95%; POB-Spark 34.98% / 70.14%; POB-LP ACC 64.30%
- SLiCK: LP-easy 2.14% / 99.76%; LP-hard 14.30% / 91.77%; GSC 8.00% / 97.52%; POB-Spark 64.41% / 31.34%; POB-LP 87.62%
- SLiCK-EPS: LP-easy 1.82% / 99.80%; LP-hard 13.70% / 92.66%; GSC 8.87% / 97.19%; POB-Spark 29.28% / 77.47%; POB-LP 96.82%

Interpretation at three levels:

(1) **Partial overlap is indeed the main failure mode of baselines** (Section 5.1 of the paper). SLiCK achieves an EER of 64.41% and AUC of 31.34% on POB-Spark—an AUC below 50% means the scoring ranking is worse than random guessing, indicating that the model systematically assigns higher scores to negative samples sharing prefixes. False triggers are not accidental but a structural directional error. PhonMatchNet is relatively better (EER 34.98%) but still unusable. The two "strong" baselines look impressive on traditional benchmarks (LP-easy 2-4% EER) but collapse under partial overlap conditions, validating the judgment in the problem statement that "benchmarks cannot detect real failures".

(2) **Effect of using EPS alone** (Section 5.2): POB-Spark EER drops from 64.4% to 29.3%, and AUC rises from 31.3% to 77.5% (the paper's terminology "+46.2%", also absolute percentage points); POB-LP ACC rises from 87.6% to 96.8% (relative to SLiCK), and the improvement relative to PhonMatchNet's 64.3% is as high as 32.5 percentage points. Meanwhile, original benchmarks did not drop but slightly increased: LP-easy 1.82% (SLiCK 2.14%, PhonMatchNet 4.49%), LP-hard 13.70% (14.30% and 23.18% respectively). A detail that must be honestly pointed out: on GSC, SLiCK-EPS's EER of 8.87% is slightly higher than SLiCK's 8.00% (AUC 97.19% vs 97.52%), although it is still better than PhonMatchNet's 10.28%—the paper's statement of "not sacrificing original metrics" is approximately true for GSC, not strictly true. Equal-weighting scoring imposes a slight cost on single-word commands, foreshadowing a larger GSC regression in Mechanism 2.

(3) **Cost-effectiveness of changing the scoring layer vs. changing training data**: When facing prefix negative samples, SLiCK-EPS cuts the POB-Spark EER by 35 percentage points with zero training changes, whereas in Mechanism 2, SLiCK+POB (retrained) only drops to 29.23%, which is worse than the un-retrained EPS.

**Mechanism 2: Adding POB Training Data** —

- PhonMatchNet + POB: LP-easy 11.72% / 95.25%; LP-hard 29.81% / 76.77%; GSC 27.73% / 80.52%; POB-Spark 18.68% / 89.25%; POB-LP 99.87%
- SLiCK + POB: LP-easy 5.26% / 98.69%; LP-hard 25.46% / 81.52%; GSC 25.92% / 80.92%; POB-Spark 29.23% / 77.88%; POB-LP 98.70%
- SLiCK-EPS + POB: LP-easy 3.24% / 99.49%; LP-hard 17.75% / 89.41%; GSC 18.75% / 89.31%; POB-Spark 16.15% / 91.14%; POB-LP 99.42%

Interpretation: POB data causes a surge in POB metrics for all models and a全线 regression in LibriPhrase and GSC, confirming that "naively adding data" is not the optimal solution (Section 5.3). The optimal combination is SLiCK-EPS + POB: POB-Spark EER 16.15% / AUC 91.14% are the best in the field for both metrics, while the regression of original metrics is minimal—LP-easy increases by only 1.42 percentage points relative to its own LibriPhrase-only training (SLiCK increases by 3.12, PhonMatchNet by 7.23), and GSC increases by only 9.88 percentage points (SLiCK increases by 17.92, PhonMatchNet by 17.45). The mechanistic hypothesis given by the paper for the GSC regression (Section 5.4): GSC consists entirely of single-word commands, and the long-overlap prior introduced by POB causes the model to weaken its utilization of limited phonetic information in short audio, belonging to a data composition conflict. Another phenomenon evident in the data but not deeply explored by the paper: PhonMatchNet + POB achieves the highest POB-LP score of 99.87% in the field, but pays the heaviest GSC cost (EER 27.73%)—the GRU scoring structure combined with overlapping data can achieve near-perfect results for "pure prefix negative samples", but the generalization loss is the largest; whereas SLiCK+POB shows almost no improvement on POB-Spark (64.41% → 29.23%, still comparable to its own EPS-only 29.28%), indicating that even when fed targeted data, the flattened FC scoring layer cannot learn positional robustness, and the architectural lesion is more stubborn than the data lesion.

### Findings from Ablation Studies

Due to the 5-page limit of ICASSP, the paper does not have an independent ablation section; ablation functions are borne by three parts of evidence:

(1) **Architectural ablation** (within Mechanism 1, SLiCK vs SLiCK-EPS): The only variable is the scoring layer, and performance differences are entirely attributed to scoring design—this is the cleanest control in the paper.

(2) **Data ablation** (Mechanism 2 vs Mechanism 1, cross-mechanism comparison of the same model): Quantifies the gain of "adding POB data" on POB metrics and the cost on LP/GSC, finding that neither is free, and the cost size is related to the scoring structure (the EPS group has the smallest cost).

(3) **Mechanism ablation** (Fig. 5 diagnostic): After switching to EPS, per-position contribution $C_i$ changes from left-concentrated to uniform, and $\rho(k)$ changes from steep rise to hugging the diagonal, confirming that "positional allocation of scoring weights" is the mediating variable for performance changes—this is causal chain evidence more persuasive than performance numbers.

Explicitly missing ablations: No EPS variant comparisons (e.g., partially learnable positional weights, constrained dynamic attention—the paper lists these as future work); No fine-grained evaluation results binned by $i_{\text{diff}}$ (although POB-Spark construction specifically performed uniform sampling, evaluation only reported summary metrics, wasting the opportunity to analyze controlled variables); No parameter count-performance curves; Inference latency, FLOPS, and memory usage are not reported; the only quantitative support for the edge efficiency claim is the parameter count reduction from 580K to 557K.

## Main Contributions

1. **Discovery and empirical evidence of a previously unnamed failure mode**: Sub-1M parameter SOTA models experience a sharp performance collapse when negative sample pairs share more than two prefix words. SLiCK's AUC of 31.34% on POB-Spark (below random) proves this is a structural directional error rather than a loss of precision.
2. **A set of formal tools**: Definition of partial overlap, first-different phoneme index $i_{\text{diff}}$, definition of prefix bias (expected contribution $C_i$), and prefix concentration score $\rho(k)$—the latter transforms "model favors the beginning" into a diagnostic quantity readable directly from weights.
3. **Partial Overlap Benchmark**: Dual datasets POB-LP (real audio, strict prefix negative samples, length distribution aligned with LibriPhrase) and POB-Spark (TTS synthesized, $i_{\text{diff}}$ uniformly controlled, balanced length), open-sourced to fill the distribution blind spots of standard benchmarks.
4. **EPS Repair Solution**: Zero new parameters (net reduction of 23K), decoupling from fixed length dependency, plug-and-play by only touching the scoring layer. POB-Spark EER drops from 64.4% to 29.3%, POB-LP ACC rises from 87.6% to 96.8%, and LibriPhrase metrics do not drop but slightly increase.
5. **Revealing and publicly stating the data composition trade-off**: Adding POB training brings the strongest POB results (EER 16.15%) but causes a 9.88 percentage point regression in GSC single-word commands. The paper does not hide this but establishes "data balance between long and short commands" as an open problem.

## Limitations and Future Work

### Technical Limitations of the Method

- **EPS is a minimal solution, not the optimal one**: Forcing equal weights discards all positional discriminative information. For negative samples of the "prefix mismatch, suffix match" type (user mispronounces the beginning), the penalty of equal-weight scoring is the same as for mismatches at any other position, potentially under-penalizing; for scenarios where effective discriminative information is concentrated in specific positions (e.g., phrases with uneven phonetic density), equal weighting is suboptimal. The paper itself positions it as a baseline, with limited upper bounds on discriminative power.
- **GSC regression is unresolved**: SLiCK-EPS + POB's GSC EER is 18.75%, still a regression of more than 10 percentage points relative to SLiCK's original 8.00%. Repair for single-word command scenarios remains open.
- **EPS is only validated on one scoring structure, SLiCK**: Table 1 has no row for PhonMatchNet-EPS (neither in Mechanism 1 nor 2). How the GRU scoring layer should be replaced with EPS, and whether it is equally effective after replacement, is not addressed by the paper. Implicit positional weighting in GRU and explicit positional weights in flattened FC are two different bias mechanisms; the adaptability of EPS to the former is unknown.
- **Tied to language and phoneme set**: All experiments are based on English and the CMU phoneme dictionary (73 tokens). Multilingual and accent robustness are not reported.

### Deficiencies in Experimental Design

- The sample sizes of the POB-Spark and POB-LP test sets are not reported by the paper, making it impossible to judge the statistical confidence of various metrics; the entire paper lacks confidence intervals or significance tests.
- The two self-built datasets use different metrics (POB-Spark uses EER/AUC, POB-LP uses ACC), reducing cross-comparability, and the paper does not explicitly state the reasons for the change.
- The relationship between the POB training set (90,808 samples) and the POB-Spark/POB-LP test sets, and whether they are generated from the same source, is not explained by the paper—if the training and evaluation synthetic negative samples come from the same TTS and the same construction process, there is a risk of synthetic domain distribution leakage, which cannot be ruled out from the paper.
- Evidence for efficiency dimensions is thin: latency, FLOPS, and runtime memory are not reported; the only proxy indicator for edge feasibility is parameter count.
- POB-Spark participates in both training (POB training set) and evaluation. The optimistic bias of TTS synthetic domain shift (synthetic data training, synthetic data evaluation) is not discussed.
- Training-side hyperparameters (e.g., mixing ratio of POB and LibriPhrase, sampling strategy) are not reported; reproducing the data ratio requires checking the open-source repository.

### Possible Directions for Future Improvement

The paper lists three directions: constrained dynamic attention (applying constraints to positional weights, such as "total concentration does not exceed $\rho$ upper limit", balancing discriminative power and unbiasedness), positional weight regularization (transforming $\rho(k)$ from a diagnostic quantity into a regularization term in the training objective, which is a natural extension of its formal definition), and optimizing data composition to balance long and short commands (e.g., resampling by phrase length, upsampling single-word commands, or curriculum learning). Based on experimental data, one can also infer (note viewpoint, not from the original paper): using POB-Spark's $i_{\text{diff}}$ controlled sampling for binned evaluation reports can precisely locate how long a shared prefix each model can tolerate; validating the portability of EPS on PhonMatchNet's GRU scoring layer; extending the prefix bias diagnostic to other cross-modal verification tasks (speaker verification, audio-text retrieval) where positional weighted scoring is also present. Overall, the value ranking of this paper is: problem discovery and measurement tools > benchmark data > repair module—it transforms a product-level failure mode into a reproducible, diagnosable, and iteratable research object, and EPS is merely the first control object proving that "the scoring layer is the lesion".
