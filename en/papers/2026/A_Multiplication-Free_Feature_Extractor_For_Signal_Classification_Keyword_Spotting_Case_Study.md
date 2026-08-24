# A Multiplication-Free Feature Extractor for Signal Classification: Keyword Spotting Case Study

- **Authors/Affiliations**: Radu Dogaru (Member, IEEE), Ioana Dogaru - Department of Applied Electronics and Information Engineering, National University of Science and Technology POLITEHNICA Bucharest, Romania; The first author is also affiliated with the Technical Sciences Academy of Romania.
- **Date**: 2026.08
- **Link**: https://arxiv.org/abs/2608.17108
- **Keywords**: multiplication-free feature extractor, iRDT (improved Reaction-Diffusion Transform), keyword spotting, TinyML, 1D Laplacian, INT8 quantization, low-power edge devices

## Problem Statement

### Problem Background and Domain Pain Points

In a typical signal classification pipeline (feature extractor → classifier), the bulk of computational power is often consumed not by the neural network, but by the feature extraction step. The paper provides a set of illustrative empirical data (cited from Reference [7], a resource-constrained MCU animal sound recognition system published in *Sensors* in 2026): the MFCC frontend latency is 300 ms, while the classifier takes only 2 ms—the frontend consumes over 99% of the time. Keyword Spotting (KWS), as a resident feature in portable assistants and smart speakers, must be deployed on TinyML platforms where energy consumption and latency are hard constraints. Mainstream TinyML development frameworks (such as EdgeImpulse [6]) basically only provide spectral feature extractors like MFCC, leaving developers with no "cheap feature" alternatives.

Why is MFCC expensive? Its computational complexity concentrates on multiplications and cosine operators within the FFT, weighted multiplications in the Mel filter bank, logarithmic operations, and cosine multiplications in the DCT—three out of these four stages rely on multipliers and transcendental functions. The ledger of underlying hardware (the paper cites Horowitz’s classic ISSCC 2014 energy analysis [12]) is clear: the energy cost of a single multiplication/float operation is an order of magnitude higher than integer addition/subtraction; on MCUs without an FPU, a single float multiplication burns dozens of clock cycles. In other words, "multiplications in the feature extractor" is the original sin in TinyML scenarios.

### Specific Deficiencies of Existing Methods

The paper categorizes existing solutions into two types and points out their deficiencies one by one:

- **MFCC-like spectral feature extractors**: Mature in accuracy and widest in ecosystem, but computationally intensive (multiplication + cosine + logarithm), unfriendly to FPGA and low-power MCU implementations; FFT windows, Mel filter banks, etc., require DSP units or multi-bit multipliers in hardware.
- **CNN Autoencoder Feature Extractors** (Baseline [8], Vitolo et al., IEEE Signal Processing Letters 2024): Trains a CNN autoencoder on Google KWS data, extracting 2D feature embeddings from hidden layers, reported to outperform spectral features like MFCC. However, it has three structural defects: high training overhead; relies on complex convolution operators containing multiplications (the complexity problem remains unchanged); and requires complete retraining when switching to a different signal class (e.g., from speech to ultrasound or EEG), lacking cross-task transferability.

Neither method addresses the same fundamental question: **Can the feature extraction step completely abandon multiplication, using only integer-friendly simple operations, without losing discriminative power?**

### Key Challenges to be Solved by This Paper

Under the Google 12-class KWS dataset and the exact same experimental setup as Baseline [8], verify whether a feature extractor iRDT, composed solely of add/subtract, absolute value, and shift (multiply by 2) operations, can: (1) achieve classification accuracy comparable to MFCC and CNN-FE (within 1 percentage point or even surpassing them); (2) reduce the computational time of feature extraction by at least one order of magnitude; (3) reduce the hardware footprint to "requiring only a very small number of logic gates (FPGA LUT/FF)". The core challenge lies in: without FFT, without Mel filter banks, and without any learnable parameters, how can a purely deterministic integer operation sequence construct a "pseudo-spectrogram" with sufficient information to feed a CNN classifier?

## Methodology

### Overall Architecture Design and Design Motivation

The entire system pipeline (Paper Fig. 2): 1 second @16 kHz input signal → iRDTv feature extractor (outputs an M×m int32 "pseudo-spectrogram") → Scaler (normalization scaling) → CNN classifier (DS-CNN or VRES-CNN) → 12-class decision.

The "why" behind the design of each component:

- **Why focus optimization efforts on the feature extractor rather than the classifier**: The 300 ms vs 2 ms data from Reference [7] already indicates that the classifier side (in this paper, the complexity of DS-CNN is only in the range of 2.1-4.7 K-MAC, Table II) is small enough; the constant overhead of the frontend is the main contradiction. Continuing to compress the classifier yields diminishing returns, while compressing the frontend offers massive gains.
- **Why retain the M×m 2D output format**: Deliberately aligned with the shape of the MFCC spectrogram (M frames × m coefficients), allowing existing CNN classifier ecosystems to be used with zero modifications—changing features without changing the classifier, ensuring fair comparative experiments.
- **Why choose iRDTv (variable M version) instead of iRDTf (fixed M version)**: KWS data is fixed-length 1 second at 16 kHz sampling (N=16000; "N=16" in the paper is a typo). The number of frames is most naturally determined by the signal length; iRDTf is reserved for variable-length signal scenarios (classification tasks with signals of different lengths across classes).
- **Why place a Scaler in the middle**: The iRDT output is the sum of absolute values of unsigned integers, and its numerical dimension does not match the normalized input expected by the CNN. The Scaler runs through all samples of the training set after feature extraction, statistics the extrema xmi/xma, and stores them for reuse during inference (Fig. 2 example parameters: scale=8, xmi=0.0, xma=296.377). This step is a pure lookup-table-based linear scaling, not introducing a substantial burden of multiplication overhead (can be approximated by shifts during inference).
- **Methodological Lineage**: iRDT was not invented out of thin air—the 1D Laplacian descriptor was first proposed in 2007 [13] under the name RDT (reaction-diffusion transform), inspired by reaction-diffusion cellular neural networks (previously used for classification of emergent dynamics in cellular automata [14]), paired with SVM for speech classification at the time; in 2020 [9], it was extended to 2D spectrogram-style features and used for RD-CNN; this paper's version is a "heavily optimized and improved" (nearly 60 times faster than [9]), hence the name improved RDT (iRDT). The team has already validated this approach in environmental sound/emotional speech recognition [15] and EEG classification on TinyML [16]; this paper is the systematic implementation in the KWS scenario.

### Mathematical Principles of the Core Algorithm

The basic computational unit of iRDT is the **symmetric second-order difference on scale d** (discrete Laplacian):

$$L_{d_k}(t) = s(t-d_k) + s(t+d_k) - 2\,s(t)$$

For each sliding window of length w and each delay channel $d_k$, accumulate absolute values in the central region of the window:

$$\text{spectrum}[i,k] = \sum_{t=\lim}^{w-1-\lim} \big| s(\text{base}+t-d_k) + s(\text{base}+t+d_k) - 2\,s(\text{base}+t) \big|, \quad \lim=\lfloor w/4 \rfloor$$

Finally, **intra-frame average** the spectral lines of every win_per_segm adjacent sliding windows into one row of the pseudo-spectrogram (Algorithm lines 17-26):

$$\text{Feat\_spec}[k,\text{col}] = \sum_{i=\text{base}}^{\text{base}+\text{win\_per\_segm}-1} \text{spectrum}[i,\text{col}]$$

**Frequency Domain Interpretation (Derived by the Note-Taker, Not Provided in the Paper, for Understanding Design Intent)**: The transfer function of $L_d$ is

$$H(e^{j\omega}) = e^{-j\omega d} + e^{j\omega d} - 2 = 2\cos(\omega d) - 2 = -4\sin^2\!\left(\frac{\omega d}{2}\right)$$

It is constantly zero at DC (suppressing DC drift), with the main peak located at $f = f_s/(2d)$. At 16 kHz sampling, the default delay list chan=[1,2,4,8,16,32] corresponds exactly to six octave channels: 8 kHz / 4 kHz / 2 kHz / 1 kHz / 500 Hz / 250 Hz—**"delay takes powers of 2" this tuning rule is equivalent to constructing a logarithmically spaced (Mel-like spaced) filter bank**. Accumulating absolute values within the window approximates the short-term energy of that frequency band (similar to the Teager energy operator idea, but deliberately using |·| instead of squaring to avoid multiplication). This interpretation also explains the phenomenon in the ablation study that "the densified Class B delay list performs better with the same feature size" (denser frequency axis sampling) and "w=128 performs poorly" (coarsest time granularity becomes coarser).

**Complexity Model (Paper Equation (1))**: The innermost loop (Algorithm line 10) has exactly 9 simple operations per iteration—4 additions, 1 shift (multiply by 2), 1 absolute value, 3 SRAM reads; due to the clipping of lim=w/4, the inner loop only traverses the central w/2 samples of each window, so there are 4.5w basic operations per window per channel. For the entire signal with N/w windows, the upper bound of total operations is:

$$\text{Ops}(\text{iRDTv}) \cong 5mN \qquad (1)$$

Substituting the typical configuration of this paper N=16000, m=6, we get 480 kilo-ops (actually int32 integer operations). For comparison, according to the estimation process in Reference [19], MFCC (FFT window 2048, hop 512, 13 Mel coefficients) under the same signal requires approximately 3 Mega-flops, and among them are mixed multiplications and harmonic functions—each operation consumes more expensive clock cycles. That is, the operation count differs by about 6 times, and adding the single-operation cost difference between "integer add/sub vs float multiply-add" constitutes the underlying explanation for the Fig. 2 measured 0.2 ms vs 6.7 ms (about 33 times) latency gap. The operations in the intra-frame average part (lines 17-26) are negligible compared to the main loop, so Equation (1) is a good estimate for the entire transform.

When deploying with quantization, append logarithmic compression at the output (similar to the log level of MFCC):

$$\text{Feat}' = \log_2\!\left(1 + \text{Feat\_spec}\right)$$

### Key Technical Innovation 1: Multiplication-Free Multi-Scale Laplacian Pseudo-Spectrogram

**Essence of the Innovation**: "Delay is Channel"—without performing any explicit transform domain calculations, using symmetric second-order differences + absolute value accumulation to approximate the short-term energy of various frequency bands at multiple time scales, directly stretching out an M×m pseudo-spectrogram. The basic computational units contain only three types of operations: add/subtract, absolute value, and shift (multiply by 2), which is the literal source of the paper's title "multiplication-free".

**Why it is feasible**: The discriminative information of speech commands is mainly hidden in the time-frequency distribution of the energy envelope (which frequency bands have bursts at which moments), not in the precise complex spectrum phase. The long chain of transformations in the MFCC pipeline (FFT + Mel + DCT) ultimately leaves the classifier with only a coarse-grained distribution of the logarithmic energy spectrum; iRDT approximates the same information using energy estimates from 6-9 octave channels, pushing the trade-off of "accuracy for compute" to the limit of zero multiplications.

**Engineering Details "Why"**: lim=w/4 cuts off 1/4 of the samples on both sides of each window, half the reason is to cut half the operations (4.5w instead of 9w), the other half is to cooperate with global indexing (signal[base+t±delay], out-of-window sampling is allowed) to complete the symmetric difference of the maximum delay d_m=w/2 within the signal, avoiding zero-padding (Algorithm comment: ignore the first and last spectral lines, base starts from 1); the sample reads of three delay points (3 SRAM loads) are explicitly counted in the 9-operation complexity ledger, indicating that the authors are budgeting at the MCU cycle-level granularity.

### Key Technical Innovation 2: "Complexity-Accuracy" Knob System Constituted by Three Hyperparameters

iRDT has no learnable parameters; all expressive power comes from three manual hyperparameters. The paper provides clear tuning rules (Section II.C):

| Hyperparameter | Value Rule | Mechanism of Action |
|---|---|---|
| w (Sliding Window Length) | Take powers of 2; KWS selects 64 | Determines the finest time granularity and delay upper limit (d_m ≤ w/2); w=128 experiment proves it is too large (Table I Class C systematic bias) |
| chan (Delay List) | Powers of 2 list, default [1,2,4,8,16,32] (m=6) | Determines the frequency axis channel distribution; intermediate delays (e.g., 6, 12, 24) can be inserted to densify frequency sampling, at the cost of increased m and linear increase in operations according to 5mN |
| win_per_segm (Number of Sliding Windows per Frame) | The most critical parameter | Directly determines the number of frames M and feature size M×m, thereby linearly determining the K-MAC complexity of the downstream classifier |

The value of this knob system lies in: **The complexity of the feature extractor (linear in m in Equation (1)) and the complexity of the classifier (linear in M×m) can be continuously adjusted by the same knob**. Engineers can slide along the accuracy-complexity curve to select a point according to the budget of the target platform, rather than jumping between several discrete pre-made features (MFCC-13/MFCC-20/...).

### Key Technical Innovation 3: log2 Dynamic Range Compression Revives 8-bit Quantization

This is the most engineering-valuable section of the entire paper. The problem chain: TinyML deployment usually requires INT8 quantization → the paper empirically finds that **8-bit quantization of the raw iRDT output drops accuracy by 2-2.5 percentage points**, while MFCC drops by less than 0.6—because the MFCC pipeline comes with logarithmic compression, flattening the dynamic range, while the iRDT output is an integer sum of linear energy, with a large numerical span, causing the resolution of 8-bit uniform quantization to be largely wasted.

Fix scheme: Append $\log_2(1+x)$ after the iRDT output. Effect (italic numbers in Table II): Quantization drop is compressed to within 0.1%, and the verification accuracy of most configurations does not decrease but increases. The key is that **this log does not break the multiplication-free property**—the paper explicitly points out that it can be implemented using a lookup table pre-calculated in Flash on the MCU, with zero multiplication overhead; even the most aggressive shift approximation (taking the position of the integer most significant bit, equivalent to $\lfloor\log_2(1+x)\rfloor$) outputs an integer spectrogram in the range 0-9, with an accuracy loss of less than 3%. This provides a complete degradation curve for three levels of quantization granularity (float / LUT-log / shift-log).

### Key Technical Innovation 4: Implementation Engineering for Deployment

- **NUMBA JIT Compilation** optimizes the Python implementation, making the overall speed nearly 60 times faster than the 2020 RD-CNN version [9]—the authors treat "implementation efficiency" itself as a design variable rather than a post-hoc optimization.
- **Full int32 integer path**, no float dependency.
- The division in Algorithm lines 1-2 (calculation of windows, M) can be **pre-calculated** in MCU implementation (Algorithm comment explicitly marks it), with zero division at runtime.
- **Open Source + Reproducible**: Code and Google Colab demo are public (GitHub: radu-dogaru/rdt_transform_for_tiny_ml_signal_classifiers/tree/main/Keyword_Spotting, Reference [18]), the demo includes multiple pre-trained .tflite classifiers, supporting random sampling testing and new signal input beyond KWS.

### Technical Differences with Existing Methods

| Dimension | MFCC | CNN Autoencoder FE (Baseline [8]) | iRDT (This Paper) |
|---|---|---|---|
| Core Operators | FFT, Mel filtering, log, DCT (multiplication + trigonometric functions) | Convolution (multiplication-intensive) | Add/Subtract, Absolute Value, Shift |
| Training Requirement | None (but MFCC has no adaptive capability) | Retrain autoencoder | None |
| Cross-Signal Transfer | Directly usable | Must retrain when switching signal class | Directly usable (only need to retune 3 hyperparameters) |
| Feature Size | Fixed steps (e.g., 416 = 32×13) | 800 (hidden layer size) | Continuously adjustable (measured range 246-750) |
| Hardware Friendliness | Requires Multiplier/DSP | Requires Multiplier | Only LUT/FF level logic (qualitative inference in paper) |
| CPU Measured Latency | 6-7 ms (Fig.3) | 60 ms (Fig.2 cited) | 0.2-0.3 ms (Fig.3) |

The essence of the difference: MFCC and CNN-FE invest in "transform accuracy", while iRDT invests all in "operator complexity", and then uses hyperparameter knobs to make up for the lost expressive power.

## Experimental Results

### Dataset Used and Its Scale

Based on the MLCommons MLPerf Tiny benchmark suite [17] (its data originates from Google Speech Commands [11], i.e., the paper's所称 Google's KWS 12-classes dataset). The construction method is consistent with Baseline [8]: In addition to 10 speech commands (0:Yes, 1:No, 2:Down, 3:Up, 4:Left, 5:Right, 6:Off, 7:On, 8:Go, 9:Stop), two additional classes aligned with the number of samples of each class (3772 items per class) are added—10:Silent (1-second background noise segment randomly selected from the original set), 11:Unknown (balanced mixture of other commands). Total 12 classes, 3772 items per class (total 45,264 items estimated by the note-taker as 12×3772, total not directly given in the paper), all fixed-length 1 second @16 kHz. Split: 80% training / 20% validation. Hyperparameter tuning uses a reduced set of only 200 items per class to accelerate search. Training uniformly 200 epochs, batch size 48. Two classifiers: DS-CNN (Hello Edge [20], MLPerf reference implementation [21], same model as Baseline [8]); VRES-CNN (author's own compact classifier from 2024 [10][22], configuration flat=0, fil=[40,100,60,30], nl=[2,1,1,0], hid=[], convolution kernel 5×3 instead of the original 3×3).

### Definition and Rationale for Evaluation Metrics

- **12-class validation accuracy**: Exactly the same setup as Baseline [8] (same data, same split, same classifier), ensuring horizontal comparability—this is the "fair reference" principle deliberately adhered to by the paper.
- **Classifier K-MAC (thousand multiply-accumulate)**: Estimated using TensorFlow profiler (tf.profiler), used as a proxy metric for deployment complexity. The reason for choosing it is that changes in feature size M×m will directly transmit to the classifier MAC count, capturing "whether what is saved at the frontend is eaten back by the backend".
- **CPU-side feature extraction latency** (Fig.1/2/3): 0.2-0.3 ms (iRDT) vs 6-7 ms (MFCC).
- **Accuracy after 8-bit quantization** (italic row in Table II): Simulates the real input conditions of TinyML INT8 deployment.
- **Not reported by the paper**: False Rejection Rate / False Acceptance Rate (FRR/FAR), DET curves, per-class confusion matrix, model parameter count, SRAM/Flash usage, measured power consumption—these dimensions' absence is elaborated in the "Limitations and Future Work" section.

### Detailed Comparison with Baseline Methods and SOTA

Table II (full dataset, arranged from largest to smallest feature size; italics indicate accuracy under 8-bit quantized features):

| Metric | FE in [8] | iRDTv-B1 | iRDTv-B2 | iRDTv-B3 | MFCC-13 | iRDTv-A2 |
|---|---|---|---|---|---|---|
| Feature Size M×m | 800 | 747 | 558 | 450 | 416 | 372 |
| DS-CNN Accuracy (%) | 90.36 [8] | 92.11 | 91.08 | 91.47 | 91.7 | 91.05 |
| DS-CNN 8-bit (%) | - | *92.11* | *91.89* | *91.77* | *91.06* ([8] reports 91.04) | *91.47* |
| VRES Accuracy (%) | - | 94.73 | 93.53 | 93.27 | 94.4 | 92.25 |
| VRES 8-bit (%) | - | *94.47* | *94.07* | *93.83* | *94.13* | *93.23* |
| DS-CNN K-MAC | - | 4651 | 3433 | 2769 | 2482 | 2061 |
| VRES K-MAC | - | 7807 | 5812 | 4695 | 4212 | 3694 |

Four key readings:

1. **Against Baseline [8]'s Autoencoder FE**: With the same DS-CNN classifier, iRDTv-B1 uses a smaller feature (747 vs 800) to raise accuracy from 90.36% to 92.11% (+1.75 points)—the multiplication-free feature is superior in accuracy, and saves the retraining of the autoencoder.
2. **Against its own LIBROBA MFCC-13**: Under float conditions, MFCC is slightly better (91.7 vs 91.05, A2 configuration); but the paper author points out that the 91.04% reported in [8] uses 8-bit quantized features, so the fair comparison for 8-bit becomes: iRDTv-A2 surpasses with a smaller feature (372 vs 416) (91.47% vs 91.06%), while DS-CNN complexity drops from 2482 K-MAC to 2061 K-MAC (-17%). This is the direct return of the log2 compression innovation—iRDT lags before quantization, surpasses after quantization.
3. **Upper limit of switching classifiers**: VRES-CNN pushes the system to 94.73% (float) / 94.47% (8-bit), i.e., the "94.7%" mentioned in the abstract. Note that MFCC-13 + VRES also has 94.4/94.13—the value proposition of iRDT is not absolute accuracy superiority, but **30 times faster feature extraction and no multiplication under the premise of parity in accuracy (94.73 vs 94.4)**.
4. **Latency**: On Kaggle CPU (Fig.2) iRDT 0.2 ms, MFCC-13 6.7 ms, Baseline [8]'s CNN-FE 60 ms—one order of magnitude for MFCC, two orders of magnitude for CNN-FE; the abstract's statement of "at least one order of magnitude" is conservative; Fig.3 CPU measurement is 0.2-0.3 ms vs 6-7 ms; Fig.1 comparison chart marks MFCC 4.8 ms vs iRDT 0.65 ms (about 7 times, with another label of 32x acceleration). VRES classifier itself is 1 ms, the entire pipeline frontend+classifier is about 1.2 ms.

### Findings from Ablation Studies

Table I (reduced set, 200 epochs, batch 48; Class A = w=64 default chan[1,2,4,8,16,32], Class B = densified chan[1,2,4,6,8,12,16,24,32], Class C = w=128 default list):

| Version | win_per_segm | M | m | Feature Size | DS-CNN (%) | VRES (%) |
|---|---|---|---|---|---|---|
| A0 | 2 | 125 | 6 | 750 | 89.79 | 91.87 |
| A1 | 3 | 83 | 6 | 498 | 89.37 | 91.45 |
| A2 | 4 | 62 | 6 | 372 | 86.87 | 91.04 |
| A3 | 5 | 50 | 6 | 300 | 87.29 | 88.54 |
| A4 | 6 | 41 | 6 | 246 | 87.29 | 90 |
| B1 | 3 | 83 | 9 | 747 | 91.87 | 92.29 |
| B2 | 4 | 62 | 9 | 558 | 91.45 | 90.62 |
| B3 | 5 | 50 | 9 | 450 | 88.75 | 91.04 |
| B4 | 6 | 41 | 9 | 369 | 88.12 | 90.2 |
| B5 | 7 | 35 | 9 | 315 | 87.5 | 89.58 |
| C1 | 2 | 62 | 7 | 434 | 86.45 | 89.16 |
| C2 | 3 | 41 | 7 | 287 | 85.83 | 89.58 |

Five findings:

1. **Frame count M is generally positively correlated with accuracy** (A0 M=125 gets 89.79 better than A2 M=62's 86.87), but **non-monotonic** (A3 M=50 is actually 0.42 points higher than A2 M=62)—time resolution is not better the more the better, there is a sweet spot matching the classifier's receptive field.
2. **Class B (dense delay list) systematically dominates under the same feature size** (B1 747 features 91.87 vs A0 750 features 89.79, smaller size yet 2 points higher): When the budget is the same, spending channels on densifying the frequency axis (inserting intermediate delays like 6, 12, 24) is more cost-effective than spending on the time axis. This is consistent with the note-taker's sin² filter bank interpretation—frequency axis sampling density directly determines spectral resolution.
3. **w=128 comprehensive bias** (Class C 85.83-86.45 at the bottom): Doubling the window length damages both the shortest time granularity and the maximum number of frames, and longer delays (up to 64, corresponding to 125 Hz channels) contribute little to speech command discrimination.
4. **VRES is significantly more robust to small features**: On A2 configuration DS-CNN 86.87 vs VRES 91.04 (4.17 point difference), on A4 87.29 vs 90—the match between classifier and frontend is a system-level variable, discussing "which feature is good" without the classifier is meaningless.
5. **log2 compression ablation** (Section III.B): 8-bit drop is compressed from 2-2.5% (raw output) to within 0.1%; shift approximation (integer 0-9 spectrogram) drop <3%, providing three levels of accuracy-complexity choices.

## Main Contributions

1. **First multiplication-free KWS feature extractor reaching MFCC-level accuracy in this comparative framework**: Under the same data, same classifier, and same training budget, the accuracy difference between iRDT and MFCC/CNN-FE is within ±1 point, with multiple configurations surpassing (92.11 vs 90.36; after quantization 91.47 vs 91.06).
2. **Analytical complexity model matches practice**: Equation (1) Ops≈5mN compresses implementation complexity into the product of two integers, the comparison of 480 k int32 ops vs MFCC 3 M flops provides hardware engineers with a ledger that can be directly brought into cycle estimation.
3. **Three-hyperparameter knob system + 12 ablation groups**: w/chan/win_per_segm allows continuous adjustment of feature size in the range 246-750, and provides clear value rules (powers of 2 delays, d_m≤w/2, prioritize densifying frequency axis).
4. **Synergistic design of log2(1+x) compression and 8-bit quantization**: Identifies and fixes the quantization vulnerability of multiplication-free linear energy features, and the fix method itself maintains zero multiplication (LUT/shift approximation), which is a key link in the end-to-end deployment closed loop.
5. **Fully reproducible**: Code, Colab demo, and pre-trained .tflite models are all open source [18].

## Limitations and Future Work

### Technical Limitations of the Method

- **Coarse frequency resolution**: The upper limit of channel count m is practically 9 (Class B), and each channel's response $-4\sin^2(\omega d/2)$ is comb-like rather than narrowband (note-taker analysis: large delay channels fluctuate multiple times within the band, energy attribution has aliasing), questioning its expressive power for tasks requiring fine formant trajectories (large vocabulary, speaker-dependent wake words).
- **Phase and polarity information completely discarded**: Absolute value accumulation only retains the energy envelope, which is the direct cost of "using |·| instead of squaring to avoid multiplication".
- **Fixed-length input assumption**: The frame count of iRDTv is determined by signal length, facing fixed-length 1-second KWS; streaming/variable-length scenarios need to switch to iRDTf or design separately, the paper does not do streaming verification.
- **Scaler relies on training set statistics**: xmi/xma comes from training set extrema, sensitive to domain shift (changing microphone, changing noisy environment), and the extrema statistics itself is fragile to outlier samples (note-taker analysis).
- **Hyperparameters purely manual**: chan list, w, win_per_segm are all manual grid searches, without an automated (e.g., learnable delay) path.

### Deficiencies in Experimental Design

- **Only one functional metric: validation set accuracy**: No test set, no FRR/FAR, no DET curve, no confusion matrix and per-class recall—and KWS as a detection task, engineering-wise it is the trade-off between false wake rate (FAR) and missed wake rate (FRR) that determines usability; the Silent/Unknown classes are synthetically balanced constructed, far from the unbalanced distribution of the real open world.
- **No target hardware measurements**: All latencies come from CPU (Kaggle CPU / local CPU), cycles on MCU, LUT/FF usage on FPGA, energy (µJ/inference), SRAM/Flash footprint are all "not reported by the paper"—and these are exactly the core claims of TinyML papers, currently only qualitative inferences based on operator types.
- **Single run, no variance reported**: No random seed repetition, no confidence intervals, the statistical significance of differences in the range of 0.3-0.5 points (e.g., 92.11 vs 91.89) is questionable.
- **Incomplete baseline coverage**: The K-MAC and VRES results in the FE in [8] column of Table II are empty ("-"); the 60 ms latency of CNN-FE is indirectly cited; no comparison with other lightweight frontends (Sinc filter banks, GD neural networks, filter bank learning methods) or stronger SOTA (BC-ResNet etc., although the latter is not at the same complexity level).
- **Classifier parameter count not reported**: Only K-MAC, unable to fully evaluate storage costs.

### Possible Directions for Future Improvement

- **Paper self-statement**: Promote to other signal classification problems (RDT family has precedents in environmental sound, emotional speech, EEG [15][16], the optimized version of iRDT awaits re-verification).
- **Note-taker suggestions**: (1) Complete the cycle, energy, LUT/FF measurement closed loop on real MCUs (e.g., Cortex-M4/M55) and FPGAs, turning "very few logic gates" from a claim into numbers; (2) Cascade with binary/integer classifiers (e.g., BiFSMN, fully integer VRES) to form a complete multiplication-free chain from frontend to backend; (3) Frame-level streaming implementation and integration with VAD, evaluating FRR/FAR under continuous listening; (4) Use learnable gating or greedy search to automatically select the delay list, replacing the manual powers of 2 rule; (5) Increase noise/long-range/multi-speaker robustness evaluation, testing the degradation curve of absolute value energy features under low SNR; (6) Joint end-to-end fine-tuning with the backend (currently FE and classifier are completely decoupled in training).
