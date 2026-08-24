# Masked Self-distilled Transducer-based Keyword Spotting with Semi-autoregressive Decoding

- **Authors/Affiliations**: Yu Xi, Xiaoyu Gu, Haoyu Li, Jun Song, Bo Zheng, Kai Yu (Key Laboratory of Artificial Intelligence, Ministry of Education of China / X-LANCE Lab, Shanghai Jiao Tong University; Alibaba Group)
- **Date**: May 2025 (arXiv:2505.24820v1, submitted 2025-05-30)
- **Link**: https://arxiv.org/abs/2505.24820
- **Keywords**: keyword spotting, RNN-T Transducer, masked self-distillation, semi-autoregressive decoding, overfitting suppression, streaming inference, false alarm

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) serves as the always-on voice entry point for devices such as smart speakers and headphones, requiring 7×24-hour streaming operation within the computational budget of edge devices. Unlike general classification tasks, the product acceptance criteria for KWS involve suppressing errors in two directions simultaneously: missed detections (recall drop) and false alarms (FA). FA is particularly fatal for edge devices—each false wake-up lights up the screen and initiates the interaction chain, directly consuming battery power and rapidly destroying user trust in whether "Hey Snips" will wake up randomly at any time. Therefore, the community's standard evaluation method fixes the number of FAs and compares recall (the main tables in this paper all use Recall@FA=4). This approach is closer to real-world deployment constraints ("allowing only a few false wake-ups per day") than fixing recall and comparing FA.

The Transducer (RNN-T), due to its natural streaming capability and strong accuracy, has seen widespread adoption in KWS in recent years: adding bias modules to enhance keyword detection [14], improving robustness with large-scale synthetic and real keyword data [15][16], suppressing FA using multi-level detection or multi-stage decoding [17][18], and TDT-KWS proposing a streaming decoding algorithm for KWS to accelerate inference [19].

The pain point addressed in this paper is more subtle than "adding another module"; it is a **structural problem determined by data morphology**: in KWS training data, the transcription of positive samples contains only the keyword itself. This means that the prediction network (predictor, the module in the Transducer trio responsible for modeling text context) faces inputs and prediction targets that are almost **fixed** during training—every positive sample fed to the predictor is the same keyword sequence, and the target to predict is also the same sequence. The predictor has no "text diversity" to learn; it can only learn the text patterns of the keyword itself. When large-scale keyword-specific data is introduced to improve robustness, this structurally simple module will **overfit to the text patterns of the keyword**: the model tends to "blindly output the keyword" in complex acoustic scenarios, reducing the discriminative power between positive and negative samples and increasing FA. The paper explicitly points out (citing [16]) that this problem is particularly pronounced in RNN-T KWS because the predictor structure is generally very simple.

On the environmental side, there is an amplifier: real far-field data like MobvoiHotwords—collected at different distances from the speaker, mixed with home background noises such as music/TV, with uneven SNR. Edge small models not only suffer from predictor overfitting on such data but also tend to overfit to the noise distribution as a whole, causing FA to further spiral out of control.

### Specific Shortcomings of Existing Methods

The paper categorizes existing countermeasures into two types and points out the costs of each:

- **Hybrid Encoders / Multi-stage Strategies**. [17] performs multi-level detection based on the posterior probabilities output by the Transducer acoustic model, reducing FA level by level; CaTT-KWS [18] cascades a Transducer keyword detector + a frame-level forced alignment module + a Transformer decoder to form multi-stage decoding; U2-KWS [20] uses a CTC branch and a decoder branch as the first and second-level models, respectively. These methods can indeed alleviate overfitting and reduce FA, but they **pack complexity into both the training process and the decoding pipeline**—there is a need for extensive manual operations and prior knowledge between stages (alignment, cascading thresholds, coordination of two branches), and these manual designs may cause keyword detection to stall at a sub-optimal point.
- **ASR-style Decoding + Keyword Matching**. Most works [15]–[18] directly adopt ASR greedy/beam search to generate hypotheses and then perform keyword string matching. Such searches are oriented towards the entire vocabulary space, considering "what was said" rather than "whether the keyword is present," which is suboptimal for KWS.
- **Existing KWS-specific Decoding is More Fragile under Noise**. This is a counter-intuitive phenomenon revealed by data in the paper (Table III): dedicated AR streaming decoding [19] that restricts the search space to the keyword lattice outperforms ASR-style decoding comprehensively on clean data, but collapses more severely than full-space search in noisy scenarios (on Xiaowen, AR is only 26.30, worse than greedy search's 40.28, while Beam Search is 83.93). The mechanistic explanation is: ASR decoding searches in the full phoneme space, while streaming KWS is trapped in a compact keyword phoneme lattice; when acoustic conditions are poor and the model is overfitted, the restricted search amplifies the bias of "tending to output the keyword," increasing FA. Under the hard constraint of FA=4, recall is forced to be lowered. The advantage of a small search space flips into a disadvantage in the face of overfitting.

### Key Challenges to be Solved by This Paper

How to transform only the training framework and decoding algorithm of Transducer KWS itself, **without increasing model structure or introducing multi-stage pipelines**, while achieving three things simultaneously: (1) Ensure that the compact edge-side Transducer remains trainable and converges even when "predictor information is partially deprived"—early experiments in the paper found that small models struggle to converge with direct masking, as the learning capacity cannot support it; (2) Completely remove the predictor during inference (NAR decoding), physically isolating the carrier of overfitting, while maintaining performance in clean scenarios; (3) Both modes coexist in the same model and parameter set, allowing runtime switching or fusion based on scenarios. In one sentence: **Achieve AR accuracy and NAR anti-overfitting capabilities within a single model and single stage.**

## Methodology

### Overall Architecture Design and Design Motivation

The overall structure remains the standard Transducer trio: encoder + predictor + joint network (joiner). The overview of training and inference is shown in the paper's Fig. 1. The key designs are deconstructed by asking "why" one by one:

- **Encoder uses 8-layer DFSMN** (Deep Feedforward Sequential Memory Network), with input/hidden/output dimensions of 440/768/320, taking 20 and 8 left/right context frames per layer (Section IV-B). Why choose DFSMN over Conformer/Transformer: This is for edge-side KWS, where computational power and latency budgets are tight. DFSMN is a structure of pure feedforward plus memory blocks, without self-attention or recursion, making parameters and latency controllable—it is also the direct descendant of the cFSMN from Issue 01 of this series and has always been the main backbone for Alibaba's edge-side KWS. To add a plain-language anchor: The memory block of DFSMN replaces the state propagation of RNN with convolution of a fixed-length left/right context, turning time modeling into parallelizable feedforward operations. The left/right context of 20/8 frames at this layer determines how far back and forward each layer can look; after stacking 8 layers, the receptive field is sufficient to cover the duration of a wake-up word, while maintaining frame-by-frame streaming output.
- **Predictor uses stateless embedding** (context length 2, embedding dimension 320, implemented in NeMo [22]). Why dare to use such a weak predictor: ASR-side research has already proven that a single-layer LSTM [22], CNN [23], or stateless embedding predictor [24] is sufficient to model context dependencies; complex predictors are not necessarily better. Under edge-side resource constraints, choosing the simplest structure is logical. However, the data characteristic of KWS "fixed positive sample transcripts" happens to turn this simplest module into a high-risk area for overfitting—this is exactly the object the paper intends to operate on.
- **Modeling unit uses 70 monophones + 1 blank token** (converted using G2P tool [33]). Why use phonemes instead of characters/subwords: KWS has the requirement for "arbitrary keywords" (the EN Arbitrary scenario in this paper). The modeling unit must be fine-grained enough; phoneme-level output allows the same model to spell out the pronunciation of any word.
- **Input asymmetry between training and inference** (marked in Fig. 1): During training, the predictor is fed transcripts (following the ground truth transcriptions of the data); during inference, it is fed a fixed keyword sequence—KWS decoding is pinned inside the keyword lattice.
- The **joiner** fuses the 320-dimensional hidden vectors from the encoder and predictor into a 256-dimensional hidden representation, outputting token posterior probabilities $p^{token} \in \mathbb{R}^{T \times U \times (V+1)}$, where $V+1$ is the vocabulary size plus the RNN-T specific blank token $\phi$.

One set of weights, two modes of use, is the engineering selling point of this framework: Fig. 1 explicitly states that conventional and masked RNN-T training share the same architecture and parameters. The trained single model simultaneously supports AR / NAR / SAR three decoding modes, eliminating the need to maintain two models on the deployment side.

### Mathematical Principles of Core Algorithms

The joint scoring base formula of Transducer (Equation 1): For the $t$-th acoustic frame and the $u$-th text position,

$$p^{token}_{t,u} = \mathrm{Joiner}(h^{audio}_t, h^{text}_u)$$

The RNN-T loss takes the negative log-likelihood of the probabilities of all paths that can align to $y$ (Equation 5): $\mathcal{L}_{RNN\text{-}T}(x,y) = -\log p(y \mid h^{audio}, h^{text})$. The modifications in this paper all revolve around $h^{text}$, divided into three steps:

**Step 1: Token-level random zero masking (Equation 2)**. Zero out the hidden representations output by the predictor token by token independently:

$$h^{mask}_u = \mathrm{RandomMask}(h^{text}_u, \gamma_{mask})$$

Note the point of action: This is **dropout-style zeroing of the predictor's output hidden vector, not masking the input text tokens**—the "product" of semantic modeling is randomly erased, and the semantic pathway is partially cut off. The joiner score after zeroing is $p^{mtoken}_{t,u} = \mathrm{Joiner}(h^{audio}_t, h^{mask}_u)$ (Equation 4), existing in parallel with the unmasked Equation 3.

**Step 2: Dual forward pass per sample + Self-distillation**. Each training sample performs a forward pass twice, obtaining unmasked logits $p^{token}$ (teacher) and masked logits $p^{mtoken}$ (student), aligning the two output distributions using KL divergence (Equation 7):

$$\mathcal{L}_{MSD} = \sum_{t,u} D_{KL}\left(p^{mtoken}_{t,u} \,\|\, p^{token}_{t,u}\right)$$

That is, using the output distribution without masking as a soft target, pulling the masked side towards the complete distribution. The meaning of "self-distilled" lies here: the teacher and student are **two forward passes of the same network**, introducing no additional teacher model, and the training cost increases by only one forward pass (the paper does not report the specific increase in training time).

**Step 3: Total Loss (Equation 8)**:

$$\mathcal{L} = \mathcal{L}_{RNN\text{-}T} + \lambda_{mask}\,\mathcal{L}^{mask}_{RNN\text{-}T} + \lambda_{MSD}\,\mathcal{L}_{MSD}$$

The three items have their own roles: The first item (standard RNN-T loss on unmasked logits) preserves the AR capability of conventional Transducer, which is the foundation for not dropping points in clean scenarios; The second item (RNN-T loss on masked logits) allows the model to learn alignment even on inputs where semantics are randomly erased, with $\lambda_{mask}=1$ equal weight to the main loss; The third item, KL consistency, specifically reduces the convergence difficulty of masked learning, with $\lambda_{MSD}=0.003$ being very small—it is the "rope leading the masked branch," not optimizing the main target.

**Why not just mask without distillation**: Early experiments in the paper explicitly state that edge-side KWS models have limited capacity, and Transducer struggles to converge after directly removing part of the semantic information. The role of self-distillation is to tell the masked branch the "output distribution the network should have when possessing complete predictor information" as a soft target, forcing the encoder and joiner to approximate the complete distribution even under missing semantics—Table IV shows that NAR without MSD achieves only 84.05/68.97 on test-clean/test-other, but reaches 97.31/83.13 with MSD added. This is quantitative evidence of "unable to learn without this rope."

The paper also provides two mechanistic benefits of masking: (1) For the joiner—when the text input is unchanged, the predictor output should be nearly deterministic; random masking injects variability into the joint layer, equivalent to decoupling two co-adapted modules, improving generalization (the classic logic of dropout applied to the Transducer's text pathway); (2) For the encoder—keyword semantics are fixed, but recording conditions, speakers, and accents are naturally diverse. Masking part of the semantic input forces the model to refocus attention on acoustic changes themselves, weakening dependence on fixed keyword information and making acoustic modeling more robust.

### Key Technical Innovation 1: Masked Self-distillation Training (MSD)

The complete logic chain of MSD is: "The KWS predictor is destined to overfit (determined by data morphology) → Intentionally randomly disable part of its output during training (Equation 2) → Small models cannot train this way, so use self-teaching to assist convergence (Equation 7) → Train two RNN-T losses in parallel to preserve dual-mode capabilities (Equation 8)." The masking rate $\gamma_{mask}$ is not arbitrary: Table II scans from 0 to 0.5 for SAR decoding on LibriKWS-20 (macro-recall under FA=4). The sweet spot is at 0.35 (test-clean 98.74 / test-other 93.04), while $\gamma=0$ (degenerating to standard RNN-T) achieves only 98.08/88.20—masked training itself yields positive gains, with almost all non-zero masking rates outperforming no masking.

### Key Technical Innovation 2: NAR Decoding and SAR Semi-autoregressive Decoding

MSD training buys a degree of freedom on the inference side: the predictor can be completely removed.

- **NAR (Non-autoregressive) Decoding**: During inference, the predictor output is pre-masked to zero, leaving only the encoder and joiner to participate in scoring. The model degenerates into a stateless acoustic model. One can understand its feasibility this way: KWS inference is originally pinned inside the keyword lattice, and the lattice structure itself externally encodes the prior of the keyword phoneme sequence; the predictor's semantic modeling is redundant information for scoring. And it happens to be the carrier of overfitting that has memorized the keyword text patterns—NAR physically removes this carrier, cutting off the problem of FA out of control in noisy scenarios from the root.
- **SAR (Semi-autoregressive) Decoding (Equation 9)**: Fuses activation scores from two paths frame by frame,

$$Score^{SAR}_t = \alpha \cdot Score^{AR}_t \oplus (1-\alpha) \cdot Score^{NAR}_t$$

The three scores are all in the $(0,1)$ interval, $\oplus$ is the fusion function, and $\alpha$ is the balance coefficient. The specific implementation is seen in Algorithm 1: Run AR and NAR Viterbi-style recursions **simultaneously** on the keyword lattice. At initialization, insert a blank before the keyword phoneme sequence: $y = [\phi_{RNN\text{-}T}, y_1, \dots, y_U]$, with boundary conditions $\delta(0,u)=\delta^{mask}(0,u)=1$ and $\varphi(0,u)=\varphi^{mask}(0,u)=0$. For each frame $t$, first set $\delta(t,0)=\delta^{mask}(t,0)=1$ (allowing new paths to start at any time, which is key for streaming), then perform lattice recursion for $u=1 \dots U$:

$$\delta(t,u) = \max\big(\delta(t,u-1)\cdot p_{t,u-1}(y_u),\;\; \delta(t-1,u)\cdot p_{t-1,u}(\phi_{RNN\text{-}T})\big)$$

That is, choose the larger of "text progression" (emitting keyword phoneme $y_u$ in the current frame) and "blank stay" (staying in place since the previous frame). $\delta^{mask}$ performs the exact same recursion using the masked posterior $p^{mtoken}$. The frame-level SAR score is the weighted sum of the scores of the two optimal paths (specific implementation of $\oplus$):

$$S[t] = \alpha \cdot \delta(t,U)\cdot \varphi(t,U) + (1-\alpha)\cdot \delta^{mask}(t,U)\cdot \varphi^{mask}(t,U)$$

where $\varphi(t,U)$ is the completion probability term at the end lattice point (the paper body does not give an explicit analytical formula for $\varphi$, only appearing in boundary condition form in Algorithm 1). Then perform path length normalization $S[t] = \mathrm{pow}(S_{Bonus} \cdot S[t],\, 1/\ell(t))$, and directly zero out paths exceeding the timeout (when $\ell(t) > T_{out}$, $S[t]=0$) to prevent stale hypotheses from hanging around for too long; the specific values of $S_{Bonus}$ and $T_{out}$ are not reported in the paper.

The value of $\alpha$ is a "scenario knob": The three English test sets (EN Fixed + EN Arbitrary) use $\alpha=0.5$ (equal weight for AR/NAR), and the two Chinese noisy sets (Wenwen/Xiaowen of ZH Noisy) use $\alpha=0.3$ (giving dominance to the NAR score). Why set it this way: In clean scenarios, the predictor's context semantics still provide positive contributions (Table III: NAR 94.07 < AR 98.55 on test-clean, NAR 83.13 < AR 92.17 on test-other), so equal-weight fusion is sufficient; in noisy scenarios, AR is the side polluted by overfitting, so its weight must be lowered to let NAR dominate. The search skeleton of both modes (keyword lattice + streaming activation scoring + length normalization) is inherited from the SOTA streaming algorithm of TDT-KWS [19]. From an engineering perspective, emphasize again the source of streaming nature: all quantities in the recursion depend only on the current frame and the $\delta$ of the previous frame; paths can start anew at any frame ($\delta(t,0)=1$), combined with the timeout zeroing mechanism, the entire decoding does not need to wait for the sentence to end nor look back at future frames, outputting activation scores frame by frame—this is exactly the form needed by edge-side wake-up engines: "calculate one frame at a time, report scores anytime"; dual-path recursion simply runs this process in parallel twice and weights them, without affecting the streaming nature.

### Technical Differences with Existing Methods

- **With HAT [21]** (hybrid-autoregressive Transducer, ICLR 2025): The training side "randomly zeroing the predictor output" is directly inspired by HAT (HAT randomly sets the predictor output to zero during training, turning the Transducer into a non-autoregressive mode). There are three differences: HAT is oriented towards ASR, while this paper brings this idea into KWS and reorganizes it into a self-distillation framework targeting the KWS-specific ailment of "fixed transcript overfitting"; this paper adds task-specific inference that HAT lacks—SAR streaming decoding and dual-path score fusion on the keyword lattice; this paper systematically verifies the overfitting suppression effect using three sets of datasets.
- **With Multi-stage Methods** (CaTT-KWS [18], U2-KWS [20], multi-level detection [17]): No need for cascaded second-level models, forced alignment modules, or dual-branch structures. Single model, single stage, without adding structural complexity to the training and decoding pipelines; the cost is replaced by dual forward passes per sample and dual lattice recursions during inference.
- **With TDT-KWS [19]** (previous work by the same authors): This paper directly adopts its keyword lattice and streaming search algorithm as the skeleton for the AR path and SAR, adding a NAR recursion on top and performing frame-by-frame score fusion; the AR baseline in Table III is the method from [19].
- **With Dropout [25]**: Both involve random zeroing, but dropout is an unstructured general regularization, whereas MSD's masking has clear task semantics (cutting off the semantic modeling pathway to combat fixed transcript overfitting), has a配套 distillation target (Equation 7), and is used extremely on the inference side (NAR decoding with full zeroing)—these three points are not possessed by dropout.

## Experimental Results

### Datasets Used and Their Scales

The paper organizes evaluation according to three scenarios (IV-A), covering fixed/arbitrary words, clean/noisy, English/Chinese:

- **EN Fixed (English Fixed Keywords)**: Snips dataset [26], keyword "Hey Snips". The official data lacks negative sample transcriptions (essential for training RNN-T). To avoid wasting data, the authors merged all negative samples from train/dev/test into a **97-hour negative sample test set**, specifically used to measure FA (how negative samples for training obtained transcriptions and were used is not reported in the paper).
- **EN Arbitrary (English Arbitrary Keywords)**: LibriSpeech **960 hours** of training data for English RNN-T models, killing two birds with one stone—serving as the seed model for Snips KWS training and also directly performing arbitrary keyword evaluation on LibriKWS-20. LibriKWS-20 selects 20 arbitrary keywords from LibriSpeech test-clean/test-other (Table I: almost, anything, behind, captain, children, company, continued, country, everything, hardly, himself, husband, moment, morning, necessary, perhaps, silent, something, therefore, together). The two subsets use the same set of words.
- **ZH Noisy (Chinese Noisy Fixed Words)**: AISHELL-2, approximately **1000 hours** of open-source Chinese ASR data, used to train the Chinese seed model; MobvoiHotwords contains two keywords "nihao wenwen" (Wenwen) and "hi xiaowen" (Xiaowen). Unlike Snips/LibriKWS-20, MobvoiHotwords contains both keyword and non-keyword data, collected at different distances from smart speakers in a less controlled environment, with typical home noises (music, TV) and varying SNR—this is the data most likely to lead edge-side models into the quagmire of overfitting.

Feature and Training Configuration (IV-B): 40-dim FBank (25 ms window, 10 ms shift), concatenating 5 frames before and after to get 11 frames, resulting in a 440-dim encoder input; training uses online speed perturbation (random ratios taken from {0.9, 1.0, 1.1}) and SpecAugment (2 random masks each in time/frequency domains, fmax=10, tmax=50); AdamW optimizer with initial learning rate 1e-3, betas (0.9, 0.999), ReduceLROnPlateau scheduler; 8 NVIDIA V100 GPUs; batch upper bound takes the stricter of 12,288 frames and 64 sentences; loss coefficients $\lambda_{mask}=1$, $\lambda_{MSD}=0.003$; unless otherwise specified, masking rate is uniformly $\gamma_{mask}=0.35$. During evaluation, $\alpha=0.5$ for the three English sets and $\alpha=0.3$ for the two Chinese sets.

### Definition and Rationale for Evaluation Metrics

The main metric is **Recall@#FA**: Recall rate under a fixed number of false alarms. The rationale is analyzed in the Problem Statement section—the product constraint for edge-side KWS is the "number of false alarms allowed per unit time/unit data volume." Fixing FA is closer to acceptance criteria than fixing recall; the main tables in this paper uniformly use FA=4. LibriKWS-20 contains 20 keywords, reporting **macro-recall** (average of 20 words) to avoid any single word dominating the mean. Table V further provides a low-FA analysis for FA=1/2/3/4/5/6/12/24—the low FA region is precisely the ECG of the overfitting problem: the manifestation of an overfitted model is that recall collapses as soon as the threshold is tightened slightly.

### Detailed Comparison with Baseline Methods and SOTA

Table III (Recall@#FA=4, five test sets, five decoding strategies):

| Decoding Strategy | Snips | test-clean | test-other | Xiaowen | Wenwen |
|---|---|---|---|---|---|
| Greedy Search | 82.13 | 87.65 | 51.69 | 40.28 | 94.56 |
| Beam Search | 89.44 | 87.82 | 54.09 | 83.93 | 95.63 |
| AR [19] (Streaming KWS) | 97.31 | 98.55 | 92.17 | 26.30 | 90.44 |
| NAR | 97.31 | 94.07 | 83.13 | 95.33 | 96.72 |
| SAR | **97.35** | **98.74** | **93.04** | **95.68** | **96.83** |

Four conclusions:

1. **AR streaming decoding leads ASR-style decoding comprehensively on clean data**: On Snips, 97.31 vs Beam's 89.44, Greedy's 82.13; similarly leading on LibriKWS-20. This confirms the conclusion of [19]—restricted search oriented towards keywords is more suitable for KWS under clean conditions, where overfitting is not an issue.
2. **The situation reverses on noisy data**: AR collapses to 26.30 on Xiaowen (Table III data; the paper body has a typo writing 25.30, refer to the table), even worse than Greedy's 40.28, and Wenwen's 90.44 is also lower than Beam's 95.63 (the paper underlines these two AR results to mark "overfitted to noise"). The reason is the superposition and amplification of "restricted search + overfitting" as mentioned above.
3. **NAR's capability boundary is clear**: On EN, it "beats ASR-style decoding but cannot beat AR" (test-clean 94.07 vs AR 98.55), indicating that in clean scenarios, the predictor's context semantics still provide effective information; on ZH noise, it makes a big turnaround—Xiaowen 95.33 vs AR 26.30 (an improvement of nearly 69 points), Wenwen 96.72 vs 90.44, and both exceed Beam Search (83.93 / 95.63). This directly verifies the hypothesis that "the overfitting carrier is in the predictor, and removing it solves the problem."
4. **SAR is optimal in all five scenarios**: In clean scenarios, it benefits from AR's strength (test-clean 98.74 is even higher than pure AR's 98.55); in noisy scenarios, it benefits from NAR's robustness (Xiaowen 95.68). Fusion does not fail in any scenario—this is direct evidence that "a single model obtains benefits from both modes simultaneously."

### Findings from Ablation Experiments

**Masking Rate Scan (Table II, LibriKWS-20, SAR Decoding, macro-recall under FA=4)**: $\gamma_{mask}=0$ (standard RNN-T) 98.08/88.20; at 0.1, 98.06/87.31 (masking too little is almost ineffective or even slightly lower); at 0.15, 98.55/93.67; at 0.2, 98.70/91.52; at 0.3, 98.46/92.90; **at 0.35, 98.74/93.04 (selected value)**; at 0.4, 98.55/90.84; at 0.45, 98.74/92.02; at 0.5, 97.50/91.85. The pattern: Moderate masking generally outperforms no masking, and performance slowly declines after the sweet spot—the predictor is not useless; masking too much loses useful semantics as well. Notably, 0.15 actually achieved the highest score on test-other (93.67), but the paper selected 0.35 based on overall performance and applied it uniformly to all subsequent experiments; the selection criterion is not formally explained; and this ablation was only completed in the EN Arbitrary scenario, not re-tuned for Snips/Mobvoi scenarios (which can also be read as a bonus for cross-scenario robustness).

**MSD Loss Switch (Table IV, Recall@#FA=4, Average Absolute Improvement Avg. Imp. across five sets)**:

| Decoding | MSD | Snips | test-clean | test-other | Xiaowen | Wenwen | Avg. Imp. |
|---|---|---|---|---|---|---|---|
| AR | No | 96.56 | 98.85 | 93.51 | 4.40 | 9.12 | — |
| AR | Yes | 97.31 | 98.55 | 92.17 | 26.30 | 60.44 | +14.47 |
| NAR | No | 96.48 | 84.05 | 68.97 | 28.49 | 98.40 | — |
| NAR | Yes | 97.31 | 94.07 | 83.13 | 95.33 | 96.72 | +18.03 |
| SAR | No | 96.32 | 98.55 | 85.11 | 26.91 | 91.53 | — |
| SAR | Yes | 97.35 | 98.74 | 93.04 | 95.68 | 96.83 | +16.64 |

Three important observations: (1) **MSD saves the AR route too**—when AR has no MSD, Xiaowen is only 4.40 and Wenwen 9.12; after adding MSD, they reach 26.30/60.44; the masked branch learning well will feed back into the anti-overfitting capability of the conventional Transducer (random masking in the training distribution itself is regularization). But even with MSD added, AR is only 26.30 on Xiaowen, indicating that merely "learning to adapt to masking" in training is not enough; the predictor must truly be removed during inference (NAR/SAR). (2) **NAR without MSD cannot learn**: test-clean 84.05, test-other 68.97, significantly worse than AR, quantitatively confirming the early experiment judgment that "direct masking is hard to converge"; after adding MSD, it reaches 97.31/83.13. (3) **MSD is not a free lunch**: Places where the model originally performed well will drop slightly (AR test-clean 98.85→98.55, test-other 93.51→92.17; NAR Wenwen 98.40→96.72). The paper frankly admits this—it is a "charity in snow" type of regularization, not a "icing on the cake" type; all benefits come from pulling collapsed scenarios back.

**Low FA Analysis (Table V, Xiaowen, the test set with the most severe overfitting)**:

| FA Count | 1 | 2 | 3 | 4 | 5 | 6 | 12 | 24 |
|---|---|---|---|---|---|---|---|---|
| AR | 5.90 | 10.81 | 11.53 | 26.30 | 70.55 | 80.13 | 97.72 | 99.49 |
| NAR | 82.10 | 93.81 | 93.99 | 95.33 | 96.23 | 96.36 | 98.37 | 99.46 |
| SAR | 89.49 | 94.45 | 95.22 | 95.68 | 96.98 | 97.18 | 98.59 | 99.60 |

AR's root cause is fully exposed: When FA≤6, it is almost unusable (only 5.90 at FA=1); only when FA is relaxed to 12/24 does it catch up with NAR (97.72 vs 98.37, 99.49 vs 99.46)—this is a typical overfitting form of "false alarms flying everywhere, forced to suppress FA by raising the threshold, sacrificing recall." SAR is optimal at all levels; at FA=1, 89.49 is 7.39 points higher than NAR, indicating that AR scores still contribute effective information in the extremely low FA region. The NAR/SAR curves are extremely flat (82→99 vs AR's 6→99), proving that masked training also cured the "threshold sensitivity," which is the most headache-inducing property for deployment.

## Main Contributions

1. **Precisely locate the KWS overfitting problem to the predictor's fixed transcript memory, and provide a minimal intervention solution**: Without changing the architecture, adding stages, or introducing external teachers, solve it within the training framework using "output random zeroing + self-distillation" (MSD). Hyperparameters require only three: $\lambda_{mask}=1$, $\lambda_{MSD}=0.003$, $\gamma_{mask}=0.35$.
2. **Make NAR decoding practically usable on KWS for the first time**: MSD training directly unlocks the mode of "discarding the predictor during inference." The model degenerates into a stateless acoustic model, scoring on the keyword lattice, physically removing the overfitting carrier (Xiaowen improved from AR's 26.30 to NAR's 95.33, Table III).
3. **SAR Semi-autoregressive Decoding**: Run AR/NAR recursions in parallel on the keyword lattice, fusing frame-by-frame weighted scores (Equation 9 + Algorithm 1). A single $\alpha$ knob adapts to scenario cleanliness (EN 0.5 / ZH Noise 0.3), achieving optimal results on all five test sets (Table III), with the greatest advantage in the low FA region (Table V).
4. **Complete chain of mechanistic evidence**: Table II (existence of masking rate sweet spot), Table IV (bidirectional effect of distillation switch, average improvements +14.47/+18.03/+16.64), Table V (low FA analysis). These three ablations round out the explanation of "overfitting comes from the predictor, why masked training is effective, and where the benefits and costs lie."

## Limitations and Future Work

### Technical Limitations of the Method

- **Inference Overhead Not Reported**: SAR requires calculating both AR and NAR posteriors simultaneously and maintaining dual lattice recursions—the joiner must run twice for each $(t,u)$ lattice point (once with predictor, once with full zeroing), and decoding states also double. The paper does not report comparisons of parameters, FLOPs, RTF, memory, or power consumption (not reported in the paper). Edge-side KWS is extremely sensitive to these metrics; whether the cost of "double joiner" is acceptable needs to be quantified before drawing conclusions.
- **$\alpha$ is a Manual Scenario Knob**: English uses 0.5, Chinese noise uses 0.3, meaning the deployer needs prior knowledge of the scenario's cleanliness to select $\alpha$; the paper does not report a systematic scan curve for $\alpha$, nor does it provide an adaptive scheme (e.g., dynamically weighting based on online SNR estimation or confidence).
- **Training Cost**: Dual forward passes per sample bring approximately double the forward computation. For compact models with stateless predictors, this might not be expensive, but the paper does not report training time comparisons.
- **Training-Inference Masking Inconsistency**: Training uses 0.35 partial zeroing, while NAR inference uses overall zeroing; there is a distribution gap between the two. The paper does not report experiments on "partial masking during inference" or a gradual curriculum from partial to full zero, and there is no ablation support for what bridges this gap (presumably the distribution smoothness forced by KL distillation).

### Shortcomings in Experimental Design

- **Lack of Direct Comparison with External SOTA**: The comparison objects in Table III are different decoding strategies of the same model plus the AR decoding of their own previous work [19]; multi-stage methods CaTT-KWS [18] and U2-KWS [20], cited as motivations, are not included in the table for head-to-head comparison, and the implication of "outperforming multi-stage solutions" lacks direct data.
- **Masking Rate Ablation Covers Only One Scenario** (Table II only LibriKWS-20); 0.35 is applied universally to all experiments; and 0.15 is higher on test-other (93.67 vs 93.04) but is abandoned, with no explanation of the selection criterion.
- **Imbalance between Chinese and English Scenarios**: EN Arbitrary has 20 arbitrary keywords, while Chinese has only 2 fixed keywords; the combination scenario of "arbitrary keywords + noise," where MSD should theoretically exert the most force, was not tested.
- **Attribution of Xiaowen AR Collapse is Slightly Single-Faceted**: The Chinese seed model comes from AISHELL-2 (approx. 1000 hours, biased towards clean reading style). The collapse of AR may be affected by both insufficient noise coverage in the seed data and predictor overfitting; the paper does not conduct a control experiment on "whether stronger noise augmentation can save AR alone," making it impossible to completely exclude data factors.
- **Missing Details on Snips Negative Sample Usage**: The 97-hour negative sample is merged from train/dev/test negative samples and used for FA evaluation. If training set negative samples also appear on the test side, there is an optimistic risk in the FA numbers; how negative samples for training obtained transcriptions and entered the loss is not reported.

### Possible Future Improvement Directions

- **$\alpha$ Adaptivity**: Upgrade from a manual knob to online adaptive quantification—dynamically weight AR and NAR based on real-time SNR estimation, divergence of dual-path scores, or a lightweight gating network, eliminating the need for manual scenario judgment before deployment.
- **Quantify and Compress SAR's Edge-Side Cost**: Measure the latency/power consumption of dual joiners + dual lattices, and explore distilling NAR robustness back into single-path inference (e.g., solidifying the masked branch's capability into the encoder post-training), preserving robustness while cutting inference doubling.
- **Supplement Comparisons and Scenarios**: Directly compare with U2-KWS, CaTT-KWS, and TDT variants; verify MSD on the "arbitrary keywords + far-field noise" combination; report parameter counts and RTF.
- **Upgrade Masking Strategy**: From independent token zeroing to structured masking (by phoneme segments, by predictor layers), or training-inference consistent gradual masking curricula, narrowing the gap between 0.35 partial masking and full-zero inference.
- **Paradigm Extrapolation**: The "overfitting of the prediction network due to highly fixed target sequences" targeted by MSD is not unique to KWS—tasks such as custom wake-up words, fixed command words, and TTS fixed prompts have isomorphic ailments. The masked self-distillation training paradigm can be directly migrated and verified.
