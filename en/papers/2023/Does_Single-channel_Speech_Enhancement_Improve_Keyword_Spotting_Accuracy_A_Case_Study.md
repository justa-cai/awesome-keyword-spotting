# Does Single-channel Speech Enhancement Improve Keyword Spotting Accuracy? A Case Study

- **Authors/Affiliations**: Avamarie Brueggeman (University of Texas at Dallas, completed during an internship at Apple); Takuya Higuchi, Masood Delfarah, Stephen Shum, Vineet Garg (Apple, USA)
- **Date**: September 2023 (arXiv v1); v2 updated on February 21, 2024 (eess.AS, ICASSP short paper format)
- **Link**: https://arxiv.org/abs/2309.16060
- **Keywords**: keyword spotting, speech enhancement, audio injection, soft switching, joint training, noise robustness, single-channel processing

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) is the entry technology for voice interaction on consumer electronic devices: users speak a wake word to activate the device, which must listen continuously before that moment. Compared to general speech tasks, the deployment environment for KWS has two harsh, inherent conditions. First, listening is all-day, all-scenario—TV noise in the living room, ambient noise in restaurants, and traffic noise on the street all enter the microphone. Background noise directly degrades detection accuracy, making noise robustness the lifeline for the commercialization of KWS. Second, mainstream wake word applications require streaming processing, with strict constraints on latency and computational budgets; the front-end cannot arbitrarily stack computations.

Speech Enhancement (SE) is a classic method for suppressing background noise and interfering speakers. The academic community has conducted extensive research on the cascaded paradigm of "SE front-end + downstream task back-end." However, as pointed out in the opening of the paper, almost all these studies use Automatic Speech Recognition (ASR) as the downstream task. ASR input consists of complete sentences lasting several seconds, whereas KWS input is often just a keyword pronunciation of about one second. ASR can be processed in offline batches, while KWS must run in streaming mode. These two differences mean that conclusions drawn from ASR cannot be directly transferred to KWS.

More specific pain points lie in channel constraints: multi-channel SE (such as microphone array beamforming) has had several successful reports in KWS, but many real-world devices have only one microphone, or although they have multiple microphones, only single-channel signals are available. Single-channel SE is fundamentally more difficult than multi-channel SE—it lacks spatial information to rely on and can only separate speech from noise based on learned statistical differences, and such non-linear processing introduces processing artifacts. The paper's introduction highlights this core contradiction: although single-channel SE models have made surprising progress in Signal-to-Noise Ratio (SNR) metrics in recent years, improvements in SNR/SDR do not necessarily translate to improvements in downstream task metrics. The culprit is likely these non-linear processing artifacts. In other words, there is an unverified assumption between "the signal becoming cleaner" and "the classifier becoming more accurate."

### Specific Shortcomings of Existing Methods

Section 2 (Prior Work) of the paper divides existing work into three parts and points out the problems left by each:

- **Research on multi-channel SE + KWS cannot answer the single-channel question.** Literature [5-7] uses microphone array signals for explicit enhancement before feeding them into KWS, reporting accuracy improvements. However, a significant portion of these improvements comes from spatial information (beamforming aligning with the target direction). Single-channel devices completely lack this dividend, so multi-channel conclusions cannot be extrapolated.
- **Research on noise-robust KWS training lacks explicit SE.** Literature [18] (ConvMixer, curriculum learning to train small models for far-field noise KWS) and [19] (MatchboxNet, 1D temporal separable convolution) train and evaluate KWS models on noisy data, following a pure data augmentation route, without any SE module from start to finish. The superiority of this route versus the SE front-end route has not been directly compared.
- **Early conclusions on single-channel SE + KWS may be outdated.** Literature [20] (Yu et al. 2018, text-dependent SE) and [21] (Gu et al. 2019, monaural SE) did use single-channel SE to improve KWS. However, both front-ends and back-ends have undergone generational updates since then: front-ends have evolved from time-frequency domain methods to time-domain methods (e.g., Conv-TasNet [10]), and strong models like BC-ResNet [17] have emerged in the back-end. Whether old conclusions hold under new combinations must be re-verified.
- **SE research oriented towards ASR has two non-transferable points.** First, ASR evaluation uses complete sentences, while keyword pronunciations are much shorter than sentences, meaning the context available to the SE model is completely different. Second, streaming wake word applications have hard constraints on front-end causality and latency, making non-causal large model schemes from ASR research impossible to copy directly.

In summary, the literature map is exactly missing a piece: under the combination of a time-domain SOTA front-end (Conv-TasNet) and a SOTA back-end (BC-ResNet-8), is single-channel SE useful for KWS? This paper aims to fill this gap.

### Key Challenges to be Solved by This Paper

The core question the paper aims to answer is a controlled experimental question: **Can single-channel speech enhancement improve KWS accuracy under noisy conditions, and under what training conditions can it, and under what conditions can it not?** Broken down, there are four challenges:

1. How to construct a fair testbed—noise types, SNR ranges, and data splits must be controllable and reproducible, otherwise the conclusions are not credible;
2. How to distinguish the key variable of "back-end training distribution"—whether the back-end is trained on clean data or noisy data may directly determine whether SE is useful. The paper explicitly separates this variable using two models, M1 and M2;
3. If cascaded use (SE first, then KWS) doesn't work, are joint training (end-to-end backpropagation penetrating front-end and back-end) and audio injection (weighted mixing of enhanced and original signals) effective remedial measures?
4. How to explain failure—why signals that are "cleaner" in terms of SNR do not necessarily make the classifier more accurate. This requires returning to mechanisms such as context length, data volume, artifacts, and training-testing mismatch to find reasons.

This is a "negative result + mechanism analysis" type of study. Its value lies not in proposing a new model, but in providing the engineering community with a translatable judgment: when is it worth hanging an SE front-end in front of KWS, and when is it better to spend the budget on noisy training data.

## Methodology

### Overall Architecture Design and Design Motivation

The overall pipeline is a cascaded link that can be switched on or off arbitrarily:

```
Noisy speech x (1 second, 16 kHz single-channel)
   → Conv-TasNet (causal, time-domain SE front-end) → Enhanced signal x′
   → [Optional] Audio Injection: x″ = α·x′ + (1−α)·x
   → BC-ResNet-8 (KWS back-end) → 12-class output (10 keywords + unknown + silence)
```

Three key design decisions and their motivations:

- **Front-end selects the causal version of Conv-TasNet.** The paper trained both causal and non-causal versions (SDRi of 5.35 dB and 5.56 dB, respectively, Table 1 in the original text), but ultimately used the causal version for all subsequent experiments. The reason is streaming application constraints—wake word detection cannot wait for future frames. This choice itself reflects the orientation of "conclusions should guide real-world deployment."
- **Back-end selects BC-ResNet-8 (the official largest model).** The motivation is to "give SE the best chance": if even the strongest back-end cannot save the distortion introduced by SE, then the conclusion (SE is ineffective) is not caused by the back-end being too weak. This is a very clean comparative design—excluding alternative explanations such as "SE is useless because the back-end is too poor."
- **Dual back-ends M1/M2 separate the training distribution variable.** M1 is trained only on clean GSC v2 (clean test set 98.7%, reaching SOTA levels, Table 2 in the original text). M2 is trained on WHAM! noise-enhanced GSC v2 (noisy test set 96.0%). The two back-ends have identical architectures, with the only difference being whether the training data contains noise—so any difference in accuracy can be attributed to the interaction effect of "training distribution × whether SE is used."

Front-end training details (Section 3.2 of the original text): Implemented using the DNS ICASSP22 recipe from the ESPnet toolkit, with encoder convolution kernel and stride reduced to 320 and 160, respectively, to maintain equivalent window length/frame shift of 20 ms and 10 ms for 16 kHz signals; trained for 200 epochs on noisy GSC v2 with SDR as the loss function; learning rate linearly increased from 0 to 0.1 in the first 5 epochs, then cosine annealed to 0; optimizer is SGD with momentum 0.9 and weight decay 0.001. The back-end is trained according to the official open-source implementation of BC-ResNet (Qualcomm AI Research repository).

### Mathematical Principles of Core Algorithms

**Conv-TasNet Encoder-Separator-Decoder Framework.** Given a noisy time-domain signal $x \in \mathbb{R}^T$, a 1D convolutional encoder maps it to a 2D representation $\mathbf{E}(x)$ (which can be understood as a learnable time-frequency decomposition, replacing fixed STFT); the separator consists of a stack of dilated convolution towers, learning the statistical characteristics of speech and noise over a long receptive field, and predicting a set of masks $m$; the enhanced representation is $m \odot \mathbf{E}(x)$ (element-wise multiplication, retaining speech-dominated components); finally, a transposed convolutional decoder restores the enhanced representation back to a 1D time-domain signal $x'$. The entire process is end-to-end differentiable, which is the prerequisite for joint training. The training objective uses Signal Distortion Ratio (SDR): measuring the waveform-level similarity between the enhanced signal and the clean reference. SDRi (SDR improvement) is the difference in SDR before and after enhancement, in dB.

**Joint Training Loss (Equation (1) in the original text):**

$$L = L_{CE} + \beta \cdot L_{SDR}$$

where $L_{CE}$ is the keyword classification cross-entropy, $L_{SDR}$ is the signal-level SDR loss, and $\beta = 0.01$, tuned on the noisy validation set. Both pure CE joint training (front-end and back-end each trained for 100 epochs) and mixed loss configurations were tested. The deep motivation behind this design: when the SE model is trained separately with signal-level loss (SDR), it is suboptimal for KWS (it optimizes for waveform fidelity rather than class separability); but completely discarding the signal-level constraint and training only with CE causes the front-end to degrade (the paper's actual test shows SDRi dropping to negative values, see ablation section), because the classification loss only requires "class separability" and does not require "clean signals." The front-end can completely distort the waveform to fit class boundaries. The mixed loss essentially uses the SDR term as a regularizer, tethering the front-end to the manifold where "the signal still looks like speech."

**Audio Injection (Equation (2) in the original text):**

$$x'' = \alpha x' + (1 - \alpha) x$$

where $x'$ is the enhanced signal, $x$ is the original noisy signal, and $\alpha \in [0, 1]$. $\alpha = 1$ is equivalent to directly using the SE output, and $\alpha = 0$ is equivalent to not doing SE at all. A global fixed $\alpha$ is a hyperparameter (shared by all utterances); predicting $\alpha$ per utterance is done by the soft switching model. The physical intuition behind audio injection: the non-linear processing artifacts and residual noise of SE can be diluted by linear mixing with the original signal—equivalent to finding a decent compromise between two types of damage: "noise pollution" and "processing distortion."

**Soft Switching Model.** Inputs the original signal $x$ and the enhanced signal $x'$, independently predicting the optimal $\alpha$ for each utterance. Structure: 256-dimensional log-Mel filter bank features → three-layer bidirectional LSTM (128 hidden units per layer) → attention pooling → two fully connected layers with 128 hidden units → softmax output. Trained for 20 epochs with KWS cross-entropy loss, learning rate 0.01. Design motivation: different utterances are contaminated by noise to different degrees; SE helps some utterances but hinders others. Theoretically, per-utterance gating should be superior to global fixed $\alpha$—this assumption has been verified in ASR (Literature [3,4]), and the paper brings it to KWS for verification.

**BC-ResNet-8 Back-end:** Dual-path structure, first using 2D convolution to jointly model time-frequency patterns, then converting to a 1D per-channel convolution path via average pooling subsampling, fusing the two paths using broadcasted residual learning, balancing accuracy and efficiency (details in Literature [17]). This paper does not modify the back-end structure, only changing its training data and whether it is fine-tuned for enhanced signals.

### Key Technical Innovation 1: Controlled "Training Distribution × Enhancement Method" 2D Evaluation Protocol

The paper's biggest methodological contribution is not a specific module, but the experimental design itself: creating a 2D matrix of "back-end training data (clean M1 / noisy M2)" and "enhancement method (no SE / cascaded SE / joint training fine-tuning front-end / fine-tuning back-end / dual-end fine-tuning / audio injection / soft switching)," providing clean, noisy, and average accuracy for each cell (Tables 2-4 in the original text). This design breaks down the vague question of "is SE useful" into "how much benefit under what conditions for what metrics," avoiding the generalization from single-point comparisons common in other studies. The direct value for engineering teams: they can directly check the table to decide whether to add a front-end based on their own training data conditions (whether there is noisy labeled data).

### Key Technical Innovation 2: Joint Training with Mixed Loss

Penetrating the SE front-end and KWS back-end with backpropagation for joint training, and explicitly comparing three loss configurations: pure SDR (cascaded baseline), pure CE, and mixed loss of CE plus $\beta$ times SDR. The magnitude of $\beta=0.01$ in the mixed loss is itself a significant finding—the scale difference between classification gradients and signal gradients is huge; signal-level regularization only needs to account for one percent of the weight to pull the front-end back from the edge of "waveform collapse" (detailed in the ablation section).

### Key Technical Innovation 3: Migration of Audio Injection and Soft Switching from ASR to KWS

Audio injection and soft switching are already existing technologies in the ASR field (Literature [3,4]). The contribution of this paper is the first systematic verification of their effectiveness in KWS under the combination of SOTA time-domain SE + SOTA KWS back-end, including fixed $\alpha$ scanning (Figures 1, 2 in the original text) and per-utterance prediction of $\alpha$, providing negative results: soft switching did not bring improvements in KWS (last row of Table 4).

### Technical Differences with Existing Methods

Compared to multi-channel SE+KWS (Literature [5-7]): does not rely on spatial information, conclusions apply to single-microphone devices; compared to the noise-robust training route (Literature [18,19]): explicitly introduces an SE module and compares it directly, rather than choosing one or the other; compared to early single-channel SE+KWS (Literature [20,21]): front-end changed to time-domain Conv-TasNet, back-end changed to BC-ResNet-8, noise changed to WHAM! urban environment noise, representing a generational update and redo; compared to ASR-oriented joint optimization research (Literature [2-4]): incorporates three KWS-specific variables: "short input (1 second) + streaming causal constraints + strong data augmentation back-end," and uses a dual-back-end design to separate the training distribution effect. Essentially, this paper conducts a domain-level controlled experiment, not a model innovation.

## Experimental Results

### Datasets Used and Their Scale

- **GSC v2 (Google Speech Commands v2)**: 105,829 one-second utterances, 35 words, over 2,600 speakers, all single-channel 16 kHz (Section 3.1 of the original text). According to the standard setting, 10 keywords (yes, no, up, down, left, right, on, off, stop, go) are taken, with the remaining words serving as negative samples, resulting in 12 classes (10 keywords + unknown + silence); the unknown and silence classes are rebalanced according to the average number of utterances in the other classes (following the common setting in Literature [15]); training/validation/testing splits follow the official split in Literature [15].
- **WHAM! Noise Set**: 28,000 urban environment noise files (bars, restaurants, etc., no intelligible speech in the dataset), dual-channel 16 kHz, average duration 10 seconds. A one-second segment is randomly cropped from the first channel and added to GSC utterances, with SNR randomly sampled between 0 and 15 dB; online augmentation is performed during training (SNR and noise segments are randomly resampled for each utterance), and only 80% of the utterances are augmented (following the implementation in Literature [17]), ensuring that clean speech is also seen during training. The noise-enhanced data is denoted as "noisy GSC v2," and the original data as "clean GSC v2."
- Scale hint: Approximately 106,000 one-second utterances, total duration of about 29 hours, which is a typical small-to-medium scale dataset—this volume was later identified by the paper itself as a suspected cause of joint training failure (see discussion in Section 5.2).

### Definition and Rationale for Evaluation Metrics

- **KWS Accuracy**: Top-1 accuracy for 12-class classification, reported separately on clean and noisy test sets, and an average is given (Tables 2-4 in the original text). Rationale: This is the true target metric for the task; any SNR-like proxy metric cannot replace it—the paper's thesis is precisely that "proxy metric improvement does not equal task metric improvement."
- **SDRi (SDR Improvement, dB)**: Quantifies front-end enhancement quality. The causal Conv-TasNet achieves 5.35 dB on noisy GSC, and 5.56 dB for the non-causal version (Table 1). Rationale: First, to verify that the front-end is indeed working (excluding broken links); second, to monitor whether the front-end undergoes "waveform collapse" during joint training (SDRi dropping to negative values is an alarm).
- **$\alpha$ (Injection Weight)**: The core hyperparameter for audio injection, ranging from 0 to 1. Fixed $\alpha$ is optimized based on validation set accuracy and then reported (Table 4), while soft switching predicts per utterance.

Metrics not reported by the paper: computational overhead metrics such as parameter count, FLOPS, and inference latency for the SE front-end and back-end; statistical significance tests for accuracy differences; per-class error analysis; statistical distribution of $\alpha$ predicted by soft switching. These omissions have a certain impact on judging whether "conclusions are confounded by compute and significance."

### Detailed Comparison with Baseline Methods and SOTA

**Baseline (Table 2 in the original text, no SE):**

| Model | Training Data | Clean Accuracy | Noisy Accuracy | Average |
|---|---|---|---|---|
| M1 | Clean | 98.7% | 94.3% | 96.5% |
| M2 | Noisy | 98.4% | 96.0% | 97.2% |

M1 achieves 98.7% on the clean test set, which is SOTA level (BC-ResNet reported level), but drops 4.4 percentage points on the noisy test set; M2, trained with noisy data, raises noisy accuracy to 96.0%, with only a slight regression on the clean set to 98.4%. Looking at this table alone, the benefits of the data augmentation route are already clear.

**Cascaded and Joint Training (Table 3 in the original text):**

| Back-end | Configuration | Learning Rate | SDRi [dB] | Clean | Noisy | Average |
|---|---|---|---|---|---|---|
| M1 | Freeze front-end and back-end (direct cascade) | — | 5.35 | 98.1% | 92.1% | 95.1% |
| M2 | Freeze front-end and back-end (direct cascade) | — | 5.35 | 98.1% | 93.0% | 95.6% |
| M1 | Fine-tune front-end | 1e-4 | −3.66 | 97.5% | 94.0% | 95.8% |
| M1 | Fine-tune back-end | 1e-3 | 5.35 | 98.0% | **95.4%** | 96.7% |
| M1 | Dual-end fine-tuning | 1e-4 | −1.30 | 97.5% | 95.2% | 96.4% |
| M2 | Fine-tune front-end | 1e-5 | 1.36 | 97.6% | 94.9% | 96.3% |
| M2 | Fine-tune back-end | 1e-3 | 5.35 | 98.0% | 95.1% | 96.6% |
| M2 | Dual-end fine-tuning | 1e-5 | 2.07 | 97.8% | 95.5% | 96.7% |
| M2 | Dual-end fine-tuning + Mixed Loss ($\beta$=0.01) | 1e-5 | 4.96 | 97.8% | 95.8% | 96.8% |

Four main line conclusions:

1. **Direct cascaded SE leads to comprehensive regression.** M1 noisy drops from 94.3% to 92.1%, M2 from 96.0% to 93.0% (first two rows of Table 3). SE raises SNR by 5.35 dB, but task accuracy drops—this is the most direct evidence of "decoupling between proxy metrics and task metrics"; the harm of processing artifacts to the classifier exceeds the harm of noise itself.
2. **M1 (clean-trained back-end) + fine-tuning back-end is the only successful combination:** Noisy 94.3% → 95.4% (fourth row of Table 3), Average 96.5% → 96.7%. Mechanism explanation: M1 has never seen noise, so noisy input is inherently out-of-distribution; the SE front-end here acts as a "domain adapter," pulling noisy input back near the clean training distribution, and then allowing the back-end to fine-tune and adapt to residual artifacts on the enhanced signal. The two steps cooperate to achieve a 1.1 percentage point gain.
3. **SE is ineffective no matter how it is saved on M2 (noisy-trained back-end).** The best cell is mixed loss at 95.8%, still lower than the no-SE baseline of 96.0% (Table 2). Mechanism explanation: M2 has experienced 0-15 dB WHAM! noise during training, learning built-in noise-robust representations; while SE-enhanced signals contain artifact patterns never seen during training, constituting new out-of-distribution inputs for M2. Data augmentation and SE front-ends are substitute relationships here, and the former is free while the latter requires front-end compute costs.
4. **Pure CE joint training collapses the front-end:** SDRi drops from 5.35 to −3.66 (M1 fine-tune front-end row), a net deterioration at the signal level; adding $\beta=0.01$ SDR regularization brings SDRi back to 4.96, and noisy accuracy rises from 95.5% to 95.8% (comparison of the last two rows of M2 dual-end fine-tuning).

**Audio Injection and Soft Switching (Table 4 in the original text, all using M2 back-end):**

| Configuration | Learning Rate | Freeze Front-end | Freeze Back-end | $\alpha$ | SDRi [dB] | Clean | Noisy | Average |
|---|---|---|---|---|---|---|---|---|
| Direct Injection | — | Yes | Yes | 0.1 | 0.59 | 98.6% | 96.0% | 97.3% |
| Injection + Fine-tune Front-end | 1e-5 | No | Yes | 0.6 | 0.67 | 97.4% | 95.2% | 96.3% |
| Injection + Fine-tune Back-end | 1e-5 | Yes | No | 0.1 | 0.59 | 98.5% | 95.8% | 97.2% |
| Injection + Dual-end Fine-tuning | 1e-5 | No | No | 0.05 | 0.28 | 98.4% | 95.7% | 97.1% |
| Soft Switching (Predict $\alpha$) | 1e-2 | Yes | Yes | Predicted | 9.19 | 97.8% | 95.0% | 96.4% |

Key points of audio injection: Without fine-tuning, $\alpha=0.1$ (i.e., 10% enhanced signal + 90% original signal), clean set 98.6% slightly exceeds M2 baseline 98.4%, noisy持平 96.0%, average 97.3% vs 97.2%—barely considered not losing, essentially because $\alpha$ is very small, the signal is almost direct, and the harm of SE is diluted. Soft switching is the most ironic row: the per-utterance $\alpha$ it predicts results in an SDRi as high as 9.19 dB (signal quality far better than the fixed small $\alpha$ of 0.59 dB, and even higher than pure SE's 5.35 dB), but KWS noisy accuracy is only 95.0%, the worst in Table 4. The背离 between signal metrics and task metrics reaches its peak here—soft switching learned to pick the mixing ratio that "sounds cleaner," but not the one that the "classifier prefers."

### Findings from Ablation Experiments

- **Causality Ablation (Table 1):** The SDRi of the non-causal Conv-TasNet is only 0.21 dB higher than the causal version (5.56 vs 5.35). The paper uses this to infer the context bottleneck: each GSC utterance is only one second long, so the non-causal model sees very few future frames, and the long receptive field of dilated convolutions cannot be fully utilized. That is, evaluating SE on one-second inputs inherently underestimates the potential of SE in real-world long-context streaming scenarios—this is an important self-limitation on the "scope of applicability of conclusions." The paper also mentions that in pre-experiments, further compressing the receptive field by reducing the encoder kernel size resulted in no change in SDR, further corroborating that context is not the dominant variable in the current setting.
- **Freeze Matrix Ablation (Table 3):** Systematic comparison of three configurations: fixed back-end fine-tuning front-end (SDRi collapses to negative values), fixed front-end fine-tuning back-end (SDRi remains 5.35, M1 noisy best 95.4%), and dual-end fine-tuning (in the middle). The results indicate: **the main source of benefit is allowing the back-end to adapt to the front-end's output distribution, rather than allowing the front-end to adapt to the back-end's classification target.** The former is a distribution alignment problem, while the latter causes the front-end to lose signal fidelity.
- **Loss Configuration Ablation (last two rows of Table 3):** Pure CE dual-end fine-tuning SDRi=2.07, noisy 95.5%; adding 0.01 times SDR results in SDRi=4.96, noisy 95.8%. Mixed loss improves both signal quality and task accuracy, indicating that signal-level constraints as regularizers are a necessary component in KWS joint training—but even so, it failed to catch up to the no-SE baseline.
- **$\alpha$ Scanning and Validation/Test Split (Figures 1, 2):** Without fine-tuning, accuracy recovers as $\alpha$ is lowered from 1, but never exceeds the baseline of $\alpha=0$ (no SE); after fine-tuning, there are significant improvements in the high $\alpha$ interval on the noisy validation set, exceeding the baseline (Figure 1), but these improvements disappear entirely on the noisy test set (Figure 2). This is a typical sign of overfitting: the tuned $\alpha$ and fine-tuned parameters are only effective for the validation distribution. The paper infers from this that the GSC data volume cannot support the parameter scale of joint training (Section 5.2).
- **Fixed $\alpha$ vs Predicted $\alpha$ Ablation (Table 4):** Fixed small $\alpha$ (0.05-0.1) is harmless and slightly beneficial; predicted $\alpha$ has the best signal quality but the worst task performance—per-utterance gating fails in KWS, and the direction of failure is precisely "over-trusting the enhanced signal."

## Main Contributions

1. **Provided the first systematic controlled study on the effectiveness of single-channel SE for KWS:** Under the combination of SOTA time-domain front-end (Conv-TasNet) + SOTA back-end (BC-ResNet-8) + standard noise protocol (GSC v2 + WHAM!, 0-15 dB), covering all mainstream usage methods of cascaded, joint training, audio injection, and soft switching, the conclusion is not a single point but a condition matrix.
2. **Separated the decisive variable of "back-end training distribution":** M1/M2 dual-back-end comparison proves that the benefit of SE completely depends on whether the back-end has seen noise—clean-trained back-ends can borrow SE for domain adaptation to gain a 1.1 percentage point improvement (Table 3, 94.3%→95.4%), while noisy-trained back-ends gain no benefit from any SE configuration (best 95.8% vs baseline 96.0%).
3. **Quantified the decoupling between signal metrics and task metrics:** SDRi improvement of 5.35 dB leads to cascaded point drops; soft switching SDRi 9.19 dB results in the worst task performance; pure CE training SDRi −3.66 dB—these three sets of data jointly show that waveform-level metrics cannot predict KWS benefits, and evaluation must fall to task metrics.
4. **Provided loss design experience for KWS joint training:** Classification loss must be paired with a small-weight (β=0.01) signal-level regularizer, otherwise the front-end will collapse at the waveform level.
5. **Pointed out hard constraints for future research:** Data scale (GSC is too small, causing validation/test split), context length (1-second input limits SE performance), simulation vs. real noise mismatch, untested speech noise—four directions all come with mechanism-level reason analysis, not just generic future work.

## Limitations and Future Work

### Technical Limitations of the Method

- **One-second input cuts off SE's context.** The dilated convolution design of Conv-TasNet intends to use a long receptive field to learn long-term characteristics of speech and noise, but GSC utterances are only one second long, and the SDRi difference between causal and non-causal versions is only 0.21 dB (Table 1), indicating that the receptive field dividend was not realized. All conclusions of "SE is ineffective" derived from GSC are strictly limited to the one-second short input condition and cannot be extrapolated to long-context listening scenarios in continuous streaming on real devices.
- **Artifacts from single-channel non-linear processing are an endogenous defect.** Cascading causes both M1 and M2 to drop points (first two rows of Table 3); soft switching has the best signal but worst task performance (last row of Table 4). The root cause is that the processing distortion of the enhanced signal constitutes a new out-of-distribution input. As long as the back-end training data does not contain homogeneous artifacts, this problem is unsolvable—unless joint training is used, but joint training is constrained by data volume.
- **Soft switching mechanism fails in KWS.** Soft switching is trained with KWS cross-entropy, with the goal of predicting the $\alpha$ optimal for accuracy, but the noisy result of 95.0% is lower than the fixed $\alpha=0.1$ of 96.0% (Table 4). The paper does not report the distribution of predicted $\alpha$, making it impossible to judge whether it is generally biased towards large $\alpha$ or per-utterance judgment is inaccurate, leaving an unsolved mystery at the mechanism level.
- **Computational overhead is completely unreported.** There is not a single number for the parameter count, FLOPS, or streaming latency of the SE front-end, while KWS is one of the tasks with the tightest compute budgets. Even if a configuration's accuracy is flat, the feasibility of a front-end at the Conv-TasNet scale for wake word devices is questionable—the paper does not report this, making it impossible to perform cost-benefit analysis on the engineering side.

### Shortcomings in Experimental Design

- **Dual limitations of data scale and authenticity.** Approximately 106,000 one-second utterances (about 29 hours) cannot support dual-end joint training; the validation/test split in Figures 1 and 2 is the most direct evidence; noise comes entirely from simulated mixing (clean speech added with WHAM! noise segments), which is fundamentally different from far-field reverberant noise collected by real devices.
- **SE training is forced to use mismatched data.** Front-end training requires clean-noisy signal pairs, so real noisy recordings with keyword labels cannot be used to train the front-end; only simulated data can be used—the back-end can use real data, but the front-end cannot, creating a training distribution mismatch between the two (Section 5.3).
- **Only non-speech noise was tested.** WHAM! contains no intelligible speech, while speech noise (background voices, TV talk shows) is exactly the interference most commonly faced by voice assistants. The paper itself admits in Section 5.4: unlike ASR, KWS can theoretically "directly ignore" non-keyword speech, and the necessity of SE under speech noise is another unverified question.
- **Lack of significance and error analysis.** Most key gaps are in the 0.1-0.4 percentage point range (e.g., 95.8% vs 96.0%); the paper does not provide confidence intervals or significance tests, nor per-class error analysis, making it impossible for readers to judge the statistical strength of these gaps.

### Possible Directions for Future Improvement

- **Larger, more authentic KWS datasets:** The paper's conclusion section explicitly calls for this; the failure of joint training and soft switching may simply be a data volume issue, and re-running on million-level real labeled data is the first step to verify the extrapolation of conclusions.
- **SE training free from clean-noisy pairing dependency:** Section 5.3 points out three routes—MixIT (Literature [23]), RemixIT (Literature [24]), and unsupervised domain adaptive SE (CHiME-7 UDASE, Literature [25])—which allow the front-end to directly consume real noisy recordings, eliminating the training distribution mismatch between front-end and back-end, but their effectiveness in practical KWS scenarios remains to be verified.
- **Long-context streaming SE:** Use continuous streaming audio (rather than one-second slices) for training and evaluation, allowing the dilated convolution receptive field to truly play a role, and verify whether the conclusion of SE failure reverses under real context conditions.
- **Speech noise and personalization directions:** Expand noise types to include background voices, and consider the value of target speaker separation (VoiceFilter-like methods, Literature [3]) in the wake word scenario—if wake words are usually spoken by a fixed user, target speaker extraction may be more tailored to the task than general enhancement.
- **Task-oriented redesign of soft switching:** The existing soft switching fails because it predicts $\alpha$ based on signal quality tendency; future work can explore gating directly based on the classification confidence of a frozen back-end, or predicting $\alpha$ based on noise type rather than per utterance.

**One-sentence summary:** Under strict comparison with SOTA front-end plus SOTA back-end, single-channel SE is only effective for "back-ends that have never seen noise" (Table 3, 94.3%→95.4%), and is not only useless but harmful to "back-ends trained with noisy data" (best 95.8% vs baseline 96.0%); the comprehensive decoupling of signal quality metrics and task accuracy (SDRi 9.19 dB换来 worst accuracy) reminds all engineers building cascaded systems—the value of enhancement modules must be accepted on task metrics, not self-comforted on SDR. For KWS, spending the budget on noisy training data is more cost-effective than spending it on SE front-ends, at least under conditions of one-second short input and small-to-medium data scale.
