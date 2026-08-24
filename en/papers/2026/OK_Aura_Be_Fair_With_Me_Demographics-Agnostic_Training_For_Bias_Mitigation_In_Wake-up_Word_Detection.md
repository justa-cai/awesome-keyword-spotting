# "OK Aura, Be Fair With Me": Demographics-Agnostic Training for Bias Mitigation in Wake-up Word Detection

- **Authors/Affiliations**: Fernando López, Paula Delgado-Santos, Pablo Gómez, David Solans, Jordi Luque - Telefónica Innovación Digital (Telefónica Digital Innovation, Madrid); Universidad Autónoma de Madrid (Autonomous University of Madrid)
- **Date**: 2026.04
- **Link**: https://arxiv.org/abs/2604.05830
- **Keywords**: wake-up word detection, demographic bias, fairness, demographics-agnostic training, data augmentation, knowledge distillation, self-supervised learning

## Problem Statement

### Problem Background and Domain Pain Points

The Wake-up Word (WuW) is the entry point for the vast majority of voice interaction systems: a predefined trigger phrase continuously detected by a lightweight acoustic model running in the background, which activates the device for full interaction (e.g., smart speakers, mobile assistants) upon a hit. This determines two engineering attributes of WuW detection: first, the model must be extremely small, fast, and always-on, typically running entirely on the edge device; second, the decision relies on extremely short speech segments (fixed at a 1.5-second window in this paper) with very limited contextual information, amplifying speaker-specific acoustic variability. This is the technical root cause of why children, elderly speakers, regional accents, and non-native speakers are most likely to fail to "wake up" the device.

The evidence chain cited at the beginning of the paper covers various sub-tasks in speech technology: in Automatic Speech Recognition (ASR), word error rates (WER) for speakers with regional/non-native accents are systematically higher, with additional disparities introduced by gender, age, and intersectional factors (Garg 2018; Zolnoori 2024, etc.); evaluations of foundation models like Whisper further confirm persistent biases across race, gender, and dialect dimensions (Fuckner 2023; Slaughter 2023); poor performance of systems towards children, elderly speakers, and non-standard accents is also documented in Keyword Spotting (KWS) and emotion recognition. Benchmark works such as Fair-Speech (Veliche 2024) and FaiST (Jahan 2025) systematically record performance gaps across multiple demographic attributes. This paper specifically emphasizes a pain point unique to always-on scenarios: performance differences are not just manifested as gaps in aggregate error rates, but more critically as **unequal interactional burdens**—some users must repeat instructions more frequently or deliberately alter their speech style to achieve the same functionality. For a commercially deployed wake-up word system (Telefónica's Spanish assistant "Aura"), this experience disparity of "some wake it up on the first try, while others have to shout three times" directly damages product fairness.

### Specific Shortcomings of Existing Methods

The paper categorizes existing mitigation paths into three types and points out their implementation barriers one by one:

1.  **Mitigation methods relying on explicit demographic labels** (e.g., group-based reweighting, CLUES using subgroup discovery results to guide contrastive learning): These require knowing the gender/age/accent label for each sample during training. However, in real-world deployments, these labels are often unavailable, incomplete, or involve privacy-sensitive information (Barocas & Selbst 2016; Dheram 2022). Diagnostic tools (DivExplorer, SpEAT) can locate biases, but locating does not equate to mitigating.
2.  **Data rebalancing**: Collecting additional data for underrepresented groups faces high costs and practical barriers (privacy restrictions, difficulty in reaching target users). More importantly, the paper points out that even if the data is balanced, bias may still persist due to design choices (Hutiri 2023) or feature selection (Bailey & Plumbley 2021)—meaning that "adding data" is not a sufficient condition.
3.  **Personalization routes** (Labrador 2025, conditioning the KWS model with speaker embeddings): Effective but requires additional user data and an enrollment process, which is not feasible for a WuW detector that must be compact, always-on, and run purely on the edge.

Furthermore, Slaughter et al. (2023) demonstrated that the embedding spaces of pre-trained speech models such as Whisper, wav2vec 2.0, WavLM, and HuBERT themselves encode and amplify social biases—this is both a threat (using large models directly is not a guarantee) and an opportunity (the second family of methods in this paper is built on the robustness assumption of large-scale SSL representations).

### Key Challenges to be Solved by This Paper

The core challenge can be summarized in one sentence: **Significantly narrow the performance gap of WuW detection across gender, age, and accent without touching any demographic labels (demographics-agnostic throughout training, with labels retained only for post-hoc evaluation), while not sacrificing overall detection performance and not breaking edge-deployability.** Hard constraints include: the student model must maintain approximately 145.6k parameters, with single inference taking about 25ms on a Pixel XL; no enrollment process can be introduced; and there is no assumption that more data for underrepresented groups can be collected. The quantitative goal is to reduce Predictive Disparity (PD, the maximum F1 difference between groups) relative to the baseline as much as possible, without compromising any of the three attributes.

## Methodology

### Overall Architecture Design and Design Motivation

The entire solution is a three-part set of "Teacher-Student + Training-time Perturbation":

**Edge-side student model device-sgru**. A single-layer GRU (200 hidden units) + fully connected binary classification layer (WuW vs. unknown), totaling 145.6k parameters, with single inference taking about 25ms on a Pixel XL. Input features are 13-dimensional MFCCs, with an analysis window of 100ms and a frame shift of 50ms; a 1.5s audio window corresponds to 29 frames; the 0th dimension MFCC is replaced with log-energy to better capture overall signal intensity (following the configuration of the author team's López 2023). The choice of GRU over larger architectures has a clear motivation: a trade-off between "real-time efficiency and detection accuracy"—fairness modifications are not allowed to come at the cost of model bloat, which is the anchor of the constraints in this paper.

**SSL Teacher w2v-BERT2-kws**. Based on w2v-BERT 2.0 (Barrault 2023, part of the Seamless series), a multilingual speech encoder backbone pre-trained on 4.5 million hours of unlabeled audio. The teacher structure is: raw audio → 80-channel Mel filterbank → convolutional subsampling + linear projection (feature projector) → 24-layer Conformer encoder (**frozen**) → **learnable weighted sum** of the 24-layer hidden states (motivation: different transformer layers encode complementary information, Pasad 2023) → Multi-Head Factorized Attention (MHFA, an efficient variant that decomposes attention projections into low-dimensional subspaces, Peng 2025) → attention pooling → linear classification head. The teacher is trained only with cross-entropy, and the paper explicitly states that it is **not used, nor intended to be used, for edge-side real-time inference**—fitting a 4.5M-hour model onto an edge device is unrealistic; its sole responsibility is to serve as a distillation teacher.

**Training Process**. Up to 700 epochs, batch size 128, Adam initial learning rate 0.001, validation plateau learning rate multiplied by 0.1, stopping if learning rate is reduced 4 times consecutively without improvement. Additive noise and reverberation (RIR convolution) are applied to the validation set to simulate diverse acoustic conditions; due to the random variance introduced by validation augmentation, the final checkpoint is not taken as the single best point, but as the model corresponding to the **minimum mean of the three epochs with the lowest validation loss**—a small trick for stable model selection in random validation environments. Augmentation is applied with probability p=0.2 during training, with the motivation of "preserving strong baseline characteristics while injecting robustness perturbations"—too frequent augmentation would harm overall performance.

### Mathematical Principles of Core Algorithms

Given a waveform $x \in \mathbb{R}^N$, the spectrogram is obtained by taking the squared magnitude of the Short-Time Fourier Transform (STFT):

$$X = \text{Spectrogram}(x) = |\text{STFT}(x)|^2 \tag{1}$$

where $X \in \mathbb{R}^{T \times F}$, $T$ is the number of time frames, and $F$ is the number of frequency bins. Time-frequency domain augmentation only modifies the magnitude and preserves the original phase, then reconstructs the augmented waveform $x'$ using ISTFT—phase fidelity ensures the physical reasonableness of the augmented audio.

**FreqMixStyle** (Frequency-domain statistical mixing): Normalize the spectrogram along the frequency axis, then rescale using the statistics of another random sample $X_j$:

$$\mu_{new} = \lambda\mu_i + (1-\lambda)\mu_j, \quad \sigma_{new} = \lambda\sigma_i + (1-\lambda)\sigma_j \tag{2}$$

where $\lambda \sim \text{Beta}(\alpha, \alpha)$ controls the interpolation intensity (in this paper $\alpha=0.4$), and mixing is restricted to **same-label sample pairs**.

**FilterAugment** (Filter simulation): Perform element-wise multiplication using a smooth frequency-dependent gain mask:

$$X' = X \odot W_{FA} \tag{3}$$

$W_{FA} \in \mathbb{R}^{T \times F}$, in this paper a linear variant is used (gain interpolates linearly along frequency) to avoid abrupt changes at band boundaries.

**Frequency Masking** (Strong baseline augmentation for SpecAugment): Mask width $f \sim U(0, W_F)$, start point $f_0 \sim U(0, \nu - f)$ ($\nu$ is the number of Mel channels), zeroing out the frequency band $[f_0, f_0+f)$:

$$X' = \text{FreqMask}(X) \tag{4}$$

**Device Impulse Response DIR** (Time domain): Convolve the training utterance with the sampled device impulse response, then truncate to the original length to maintain input dimensions:

$$x' = x * h_{dir} \tag{5}$$

**Knowledge Distillation Loss**: The student minimizes a weighted sum of CE and temperature-scaled KL:

$$L_{KD} = \delta\, L_{CE}(p_{student}, y_{true}) + (1-\delta)\,\tau^2\, D_{KL}\left(p^{\tau}_{teacher} \,\|\, p^{\tau}_{student}\right) \tag{6}$$

where the cross-entropy

$$L_{CE}(p_{student}, y_{true}) = -\sum_i y_{true,i} \log p_{student,i} \tag{7}$$

Probabilities are given by softmax $p_{student,i} = e^{z_{student,i}} / \sum_j e^{z_{student,j}}$, temperature-scaled distributions are $p^{\tau}_{\cdot,i} = e^{z_{\cdot,i}/\tau} / \sum_j e^{z_{\cdot,j}/\tau}$, and the KL term $D_{KL}(p^{\tau}_{teacher} \| p^{\tau}_{student}) = \sum_i p^{\tau}_{teacher,i} \log \frac{p^{\tau}_{teacher,i}}{p^{\tau}_{student,i}}$. In this paper, $\delta=0.2$ and $\tau=2$ are taken: the heavily weighted KL component (0.8) allows the student to mainly imitate the teacher's class confidence structure rather than just hard labels; $\tau=2$ softens the targets, transmitting relative confidence. The distillation phase switches to SGD (momentum 0.9, weight decay $10^{-4}$, initial LR $10^{-4}$, plateau scheduling), with the motivation that SGD can produce flatter minima and better generalization; the student is initialized with pre-trained baseline weights to accelerate convergence.

### Key Technical Innovation 1: Frequency-domain Augmentation Family Targeting "Disruption of Demographic Acoustic Cues"

The theoretical basis for this family of methods is three phonetic associations: **gender** is strongly correlated with fundamental frequency F0 and formant structure (Vorperian 2019), **age** is correlated with spectral envelope (Harnsberger 2008), and **accent** is correlated with prosody (Piat 2008). The inference is: a model that relies too heavily on specific frequency bands such as F0/low formant regions for decision-making will naturally learn better for populations strongly correlated with these bands. Therefore, **modulating or partially deleting frequency information** during training forces the model not to concentrate evidence on any single demographic-related frequency band, but to distribute discriminative evidence across a wider spectral range—this is the core assumption of "perturbation as debiasing" (the invariant representation idea of Vandenberghe 2023).

The four augmentations each have specific roles and configurations: FilterAugment uniformly samples the number of frequency bands in [3, 9], minimum bandwidth 187 Hz, gain ±6 dB (smooth energy perturbation, simulating vocal tract/channel filtering); FreqMixStyle takes $\alpha=0.4$ and **only mixes same-label pairs**—this restriction is a necessary engineering decision: mixing spectra of positive and negative samples would directly pollute the discriminative boundary of "OK Aura", introducing label noise; FreqMasking takes $W_F=30$, $\nu=128$ (hard deletion of an entire frequency band, forcing learning from partial spectra); DIR is a time-domain device generalization augmentation (Morocutti 2023), included in the paper for comparison to test "device diversity ≠ demographic diversity". All augmentations are applied with probability p=0.2 to avoid excessive perturbation harming baseline performance.

The key point is: these four augmentations were originally designed in the literature for **robustness/generalization**; this paper is the first to reposition them as **fairness mechanisms** and systematically evaluate them using inter-group F1 differences (PD/RRPD)—this is the methodological claim of "augmentation design is debiasing design".

### Key Technical Innovation 2: Distilling "Demographically Neutral" Knowledge from Large-scale SSL Foundation Models

The assumption chain for the second route: the upper layers of SSL models suppress speaker identity information (Mohamed 2022); the training data scale has reached 4.5 million hours, covering population diversity far exceeding any annotated corpus, so its representations should be demographically robust. However, directly deploying such models conflicts with edge-side constraints, so a "large teacher → edge-side small student" distillation topology is adopted, with the student still being the 145.6k parameter device-sgru.

This design directly inherits the "Fairness without demographics through knowledge distillation" paradigm proposed by Chai et al. (2022) in the visual domain and migrates it to the WuW scenario. It simultaneously bypasses all defects of the aforementioned three existing paths: no demographic labels are needed (the teacher only outputs soft labels), no enrollment process is needed (the student is a general model), and the edge-side budget is not broken (the student architecture remains unchanged). The two structural choices within the teacher also have clear motivations: the learnable weighted sum of the 24-layer hidden states (rather than using only the last layer) stems from evidence that "different layers encode complementary information"; MHFA obtains sequence modeling capabilities at a minimal parameter cost—the teacher is not deployed on the edge, but an overly large classification head would also waste annotated data.

### Technical Differences with Existing Methods

Compared to demographics-aware methods such as group-based reweighting/CLUES: the training pipeline in this paper is completely blind to demographic labels, with labels used only for post-hoc evaluation—zero privacy cost, and applicable to scenarios where labels are fundamentally uncollectible. Compared to personalized KWS (Labrador 2025): no enrollment, no speaker embedding conditioning, maintaining a general model form. Compared to "directly using large SSL models": the edge-side student with 145.6k parameters is retained, and fairness gains are injected indirectly through soft labels. Compared to traditional SpecAugment-like augmentations designed for accuracy: the objective function of the augmentation changes from "test set accuracy" to "inter-group PD", and experiments prove that different augmentations have vastly different effects on fairness—augmentation selection itself is a fairness design variable.

## Experimental Results

### Datasets Used and Their Scale

**OK Aura (Internal In-domain Corpus)**: Approximately 5.8k audio clips, about 4.5 hours, 546 anonymous speakers, all in Spanish. Includes speech and non-speech materials (background noise). Positive samples cover everything from isolated wake-up words ("OK Aura") to wake-up words embedded in contextual sentences (e.g., "Perfecto, voy a mirar qué dan hoy. OK Aura"); negative samples are deliberately designed with three levels of difficulty: partial matches (containing only "Aura", e.g., "Hay un aura de paz y tranquilidad."; containing only "OK", e.g., "OK, a ver qué ponen en la tele."), phonetically similar distractors ("Hola Laura", "Prefiero el hockey al baloncesto"), and combinations of both ("Porque Laura, ¿qué te pareció la película?"). Recordings span from quiet rooms to natural noise environments, across various devices, and include speech event time annotations (obtained using CTC iterative pseudo-forced alignment from López & Luque 2022). Some data has been made public with the Albayzin 2024 Wake-up Word Detection Challenge.

Demographic annotations are divided into three attributes: gender (female/male binary), age (five tiers: 0–20, 21–30, 31–40, 41–50, 51+), and Spanish accent (the complete annotation set includes 15 categories: unknown, Central-Southwestern, Southern, Caribbean, Northern, Northwestern, Chilean, Eastern and Balearic, Non-native, La Plata, Canary, Central American, Andean-Pacific, Mexican, Filipino). Training/validation set distribution: Female 2131 clips (41.74%) vs. Male 2974 clips (58.26%); average speaker age 37 years, samples concentrated in 20–50 years, scarce under 20 and over 51; accent highly skewed, with Central-Southwestern Spanish dominant. Test set: 575 clips, 47 speakers: Female 254 clips (44.88%, 20 speakers), Male 321 clips (55.12%, 27 speakers); age distribution: 21–30 is 135 clips/11 speakers, 31–40 is 138 clips/11 speakers, 41–50 is 295 clips/24 speakers, 51+ is only 7 clips/1 speaker, **0–20 is 0 clips**; accents have samples in only 6 categories (Central-Southwestern 313/26, Eastern and Balearic 15/1, Non-native 49/4, Northern 90/7, Southern 84/7, Unknown 12/2).

**Public Out-of-domain Resources** (used only for training/validation augmentation and robustness, not for fairness evaluation, due to lack of demographic metadata, insufficient granularity, or mismatched annotations): Spanish Common Voice v7.1, M-AILabs Spanish, OpenSLR SLR28 real and simulated room impulse responses and noise, DEMAND environmental noise, MicIRP and multi-angle multi-distance microphone impulse response libraries (the latter two specifically for device robustness).

### Definition and Rationale for Evaluation Metrics

**Data Bias measured by Disparate Impact (DI)**: $DI = P(Y=1|G=d) / P(Y=1|G=a)$, i.e., the ratio of positive sample rates between the disadvantaged group and the advantaged group; for multi-valued attributes, the maximum ratio among all group pairs is reported. It is chosen because it is a standard ratio-type metric in algorithmic fairness, directly characterizing "which group is given more opportunities for positive examples in the data".

**Predictive Bias measured by F1 and Predictive Disparity (PD)**: F1 is calculated per group under a fixed 1.5s window and fixed decision threshold of 0.5 (harmonic mean of precision and recall), then:

$$PD = \max_{i,j \in G} |F1(g_i) - F1(g_j)| \tag{16}$$

The maximum inter-group F1 difference is adopted, following the pairwise group comparison protocol of Singh et al. (2023), with the motivation that the "worst group pair" is the upper bound characterization of fairness risk. **Inter-technology comparison uses Relative Reduction in PD (RRPD)**:

$$RRPD = 100 \times \frac{PD_{baseline} - PD_{technique}}{PD_{baseline}} \tag{17}$$

Positive values represent bias reduction, negative values represent bias expansion. To ensure reliable subgroup estimation, **demographic groups with fewer than 20 test samples are excluded**; and the paper reminds that the number of speakers within retained groups is also limited (e.g., only 20 females), so fairness metrics should be viewed as trend evidence rather than generalization guarantees.

### Detailed Comparison with Baseline Methods and SOTA

**Data-layer Bias (Table 5)**: Training/validation set DI—Gender 0.7170 (Male advantaged/Female disadvantaged), Age 0.6804 (41–50 advantaged/21–30 disadvantaged), Accent 0.1692 (Central-Southwestern advantaged/Northern disadvantaged). The accent DI is as low as 0.17, the most severe data skew among the three attributes.

**Baseline Model Bias (Table 6)**: device-sgru trained from scratch (uniform initialization). Gender: Male 0.9863 (support 296) vs. Female 0.9825 (204), PD=0.0038—small but measurable. Age: 21–30 best 0.9956 (115), 41–50 worst 0.9827 (265), 31–40 is 0.9828 (118), PD=0.0129, the largest gap among the three attributes, with middle-aged adults forming the key vulnerable group. Accent: Central-Southwestern highest 0.9873 (278), Northern lowest 0.9781 (70), Southern 0.9818 (84), Non-native 0.9870 (39), PD=0.0092. The overall pattern is consistent with data DI: data skew and predictive bias co-occur, forming the motivation for debiasing.

**SSL Teacher (Table 7)**: The RRPD of w2v-BERT2-kws relative to the baseline is 79.64% for gender, 85.35% for age, and 41.05% for accent—large-scale SSL pre-training **can significantly narrow but not eliminate** inter-group gaps. This result hits two birds with one stone: it validates the assumption that "SSL representations are demographically robust" (supporting the distillation route), and also shows that pre-training alone is not enough (supporting the need for continued augmentation).

**Panorama of RRPD for Six Training Configurations (Table 8)**:

| Configuration | Gender RRPD | Age RRPD | Accent RRPD |
|---|---|---|---|
| DIR (Device Impulse Response) | 67.35% | 0.00% | −20.13% |
| FreqMixStyle | −21.42% | 34.12% | 40.48% |
| FilterAugment | 88.26% | 30.14% | −40.19% |
| FreqMasking | **39.94%** | **83.65%** | **40.48%** |
| KD | 67.35% | 15.10% | −20.13% |
| KD + FreqMasking | 21.24% | 15.10% | −40.19% |

The conclusion is very clear: **only Frequency Masking achieves positive gains across all three attributes** (Gender 39.94%, Age 83.65%, Accent 40.48%, i.e., the numbers cited in the abstract); FilterAugment has the largest gender gain (88.26%) but accent worsens by 40.19%; FreqMixStyle improves accent/age but worsens gender by 21.42%; DIR and KD are effective for gender but both worsen accent by 20.13%; their combination (KD+FreqMasking) is comprehensively weaker than their respective best single techniques.

**Detailed Breakdown of FreqMasking by Group (Table 9)**: Male 0.9828 (296) / Female 0.9851 (204), PD Gender 0.0023; Age 21–30 is 0.9880 (115), 31–40 is 0.9828 (118), 41–50 is 0.9847 (265), PD Age 0.0052; Accent Southern 0.9818 (84), Central-Southwestern 0.9835 (278), Northern 0.9781 (70), Non-native 0.9870 (39), PD Accent 0.0089. Comparing with Table 6 reveals a detail worth warning about: the improvement in age fairness comes half from the best group 21–30 dropping from 0.9956 to 0.9880, and half from the worst group 41–50 rising from 0.9827 to 0.9847; the reduction in accent PD comes almost entirely from the advantaged group Central-Southwestern dropping from 0.9873 to 0.9835 (Northern 0.9781 and Southern 0.9818 remain unchanged, the maximum gap pair changes from "Central-Southwestern–Northern" to "Non-native–Northern")—i.e., some fairness gains are obtained via "peaks cutting" (lowering advantaged groups) rather than "valleys lifting" (improving disadvantaged groups); Gender is the exception, with females indeed rising from 0.9825 to 0.9851 and surpassing males.

### Findings from Ablation Experiments

This paper does not ablate single technologies module by module, but treats the horizontal comparison of the six configurations as a structured ablation, discovering four patterns:

1.  **Fairness gains are attribute-dependent, with no universal augmentation**. Each augmentation is effective or even harmful for only some attributes. The paper's mechanistic explanation: FreqMixStyle and FilterAugment perform poorly for some populations, possibly because their modification of frequency statistics is too aggressive, destroying critical formant/prosody cues, exchanging mixed fairness gains at a higher error cost; while the acoustic cues corresponding to gender, age, and accent (F0 and formants, spectral envelope, prosody) have different sensitivities to frequency perturbations.
2.  **DIR only saves gender (67.35%)**, with zero gain for age and negative gain for accent—indicating that device-level acoustic diversity cannot capture heterogeneous acoustic changes related to demographic attributes, "device robustness ≠ population fairness".
3.  **Hard deletion is superior to soft perturbation**. Frequency masking hard-zeroes a certain frequency band, forcing the model to distribute evidence across multiple regions of the spectrum, avoiding overfitting to a single demographic-related frequency band (such as F0 or low formant regions); this "distributed attention" simultaneously improves overall robustness and fairness; whereas the ±6 dB smooth gain of FilterAugment and the statistical mixing of FreqMixStyle retain in-band information, resulting in less debiasing power than hard masking.
4.  **Distillation and random spectral perturbation are difficult to stack**. KD+FreqMasking is worse than the best single technique; the paper speculates that there is an interaction between random spectral corruption and logit matching that a small-capacity student cannot optimize simultaneously; the reason KD itself fails in the accent dimension is attributed to the insufficient accent diversity of in-domain annotated data, meaning the teacher cannot provide "accent-neutral" soft targets.

## Main Contributions

1.  **Completed an end-to-end bias quantification loop on a real Spanish wake-up word corpus**: From data-layer DI (Gender 0.7170, Age 0.6804, Accent 0.1692) to prediction-layer inter-group F1/PD (baseline age gap 0.0129 is the largest), quantifying "data skew and predictive bias co-occurrence" using the same attribute system, establishing a reusable evaluation protocol for WuW fairness (1.5s window, threshold 0.5, exclusion of groups with <20 samples, PD/RRPD).
2.  **Proposed and validated two families of demographics-agnostic debiasing methods**: Frequency-domain augmentation (FreqMixStyle/FilterAugment/FreqMasking/DIR) and SSL foundation model distillation (w2v-BERT2-kws teacher → device-sgru student), consuming no demographic labels throughout the process, with labels retained only for post-hoc evaluation—a feasible route under privacy and label availability constraints.
3.  **Identified Frequency Masking as the most robust single fairness augmentation**: Improving all three attributes simultaneously (39.94%/83.65%/40.48%), and its mechanistic explanation (forcing evidence to distribute across the spectrum, cutting off overfitting to demographic-related bands like F0) is transferable to other speech classification tasks.
4.  **Falsified two intuitions**: SSL pre-training can only narrow, not eliminate, bias (teacher RRPD 41%–85% but not zero); fairness gains from augmentation are not free nor universal (FilterAugment accent worsens by 40.19%, combined techniques are inferior to single techniques)—negative evidence for "augmentation design is fairness design" is equally valuable.

## Limitations and Future Work

### Technical Limitations of the Method

-   **Attribute-dependent and non-composable**: The only FreqMasking that is positive across all three attributes still has a lower gain in the gender dimension (39.94%) than FilterAugment (88.26%) and DIR/KD (67.35%), but combined techniques degrade comprehensively in experiments, making it impossible to approach optimal dimensions by stacking in practice.
-   **"Peaks-cutting" fairness**: From the comparison of Table 6 and Table 9, the reduction in PD for age and accent comes significantly from the performance decline of advantaged groups (21–30 drops from 0.9956 to 0.9880, Central-Southwestern drops from 0.9873 to 0.9835), rather than purely improving disadvantaged groups—this path to fairness may be unacceptable in products.
-   **Accent ceiling for distillation**: The accent RRPD for KD is −20.13%, constrained by the accent diversity of in-domain annotated data; the teacher's soft targets are themselves not accent-neutral; the 4.5M hours of pre-training also failed to solve accent skew after transfer to small in-domain data.
-   **Aggressive spectral perturbation has costs**: The paper admits that FreqMixStyle/FilterAugment may destroy critical formant and prosody cues, performing worse at higher error costs—the boundary between augmentation intensity and information retention is not characterized.

### Shortcomings in Experimental Design

The paper lists five points itself: (i) single-variable analysis, not covering intersectional fairness (e.g., "elderly women with regional accents"); (ii) out-of-domain audio introduced in training/validation, bringing distribution mismatch and metadata inconsistency; (iii) multiple test subgroups are excluded (<20 samples) and retained groups have very few speakers (only 20 females, only 1 person/7 clips for 51+, 0–20 completely absent), so related conclusions can only be interpreted cautiously; (iv) F1 cannot decompose false accepts (false alarm) and false rejects (miss), while the costs of the two are extremely asymmetric in the WuW scenario; (v) conclusions under a fixed threshold of 0.5 may not generalize to other operating points. Additionally, a few problems not self-confessed by the paper can be supplemented from the tables: **the overall (aggregated over the entire test set) detection performance of each training configuration is not reported**, "overall performance maintained" can only be inferred indirectly from subgroup F1, with no quantitative evidence; **random seed repetitions and statistical significance tests are not reported**, and since PD takes the maximum difference between group pairs, it is sensitive to extremes; differences under small support groups (39–296 windows) may be unstable; the correspondence between the evaluation support numbers in Table 6/9 (Male 296 + Female 204 = 500 windows) and the 575 sample test set is not explained; only 4/15 accent categories are retained, with Mexican, Filipino, Caribbean, etc., annotation categories absent in the test set; key hyperparameters such as $\delta=0.2$, $\tau=2$, $p=0.2$ are not subjected to sensitivity analysis; nor is there a comparison with demographics-aware methods (such as group-based reweighting), making it impossible to quantify how much fairness upper bound is sacrificed by "abandoning labels".

### Possible Directions for Future Improvement

Directions given by the paper: intersectional fairness analysis (intersectional groups of Age × Gender × Accent); curating more balanced data, focusing on collecting data for underrepresented age and accent groups; replacing isolated single-metric evaluation with a multi-objective evaluation framework that simultaneously optimizes overall accuracy, per-group F1, and cross-attribute fairness. Combined with the empirical gaps in this paper, worth doing also includes: decomposing F1 into false accepts/false rejects and evaluating them weighted by business costs (the FA/FR costs for wake-up words are naturally asymmetric); re-verifying fairness conclusions across multiple thresholds/DET curve families to exclude the contingency of a single operating point; explicitly testing the two fairness paths of "valley lifting" and "peak cutting" (e.g., applying targeted augmentation only to disadvantaged groups without touching advantaged groups); supplementing the distillation teacher with in-domain data balanced for accents to verify if the failure in the accent dimension can be repaired; and conducting a head-to-head comparison with demographics-aware upper bounds to quantify the theoretical loss of the label-free route.
