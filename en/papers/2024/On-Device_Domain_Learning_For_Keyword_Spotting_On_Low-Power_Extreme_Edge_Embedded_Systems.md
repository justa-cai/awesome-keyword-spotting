# On-Device Domain Learning for Keyword Spotting on Low-Power Extreme Edge Embedded Systems

- **Authors/Affiliations**: Cristian Cioflan, Lukas Cavigelli, Manuele Rusci, Miguel de Prado, Luca Benini (Integrated Systems Laboratory, ETH Zurich; Huawei Zurich Research Center; ESAT, KU Leuven; VERSES AI; Department of Information Engineering, University of Bologna)
- **Date**: March 2024 (arXiv v1 submitted on March 12, 2024; IEEE copyright paper, specific conference proceedings not reported)
- **Link**: https://arxiv.org/abs/2403.10549
- **Keywords**: on-device learning, domain adaptation, keyword spotting, TinyML, noise robustness, ultra-low-power microcontroller, GAP9

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS), a lightweight classification task that picks out preset wake-up words from continuous audio streams, serves as the voice entry point for always-listening devices such as smart speakers, wearable devices, and wireless earbuds. The common profile of these devices is: battery-powered, always-on (constantly listening), with clock speeds and memory constrained by cost limits. While KWS models can achieve high classification accuracy in clean, noise-free environments, the paper points out a key pain point at the outset: when the data distribution used for offline training on remote servers is inconsistent with the data distribution in the actual deployment environment (typically due to different on-site noise), performance degrades severely. The authors cite empirical results from their previous work [1], showing a drop of up to 27 percentage points.

Why is this pain point difficult to resolve using traditional methods: The types of noise are theoretically inexhaustible. During offline training, data augmentation can use dozens of noise types, but when users take devices into a new meeting room or a new restaurant, the acoustic environment encountered is essentially a new sample in an infinite space. Moreover, highly non-stationary noise such as human speech in meetings (where speakers switch constantly and spectral structures change drastically over time) overlaps significantly with wake-up words in the time-frequency domain, as both are speech. The robustness learned by general augmentation cannot be specialized to this specific environment.

The second layer of pain points is the coupling of privacy and energy consumption: The ideal adaptation process should be completed entirely on the device itself (referred to as privacy-by-design in the paper, with data never leaving the device), while eliminating the communication energy cost of round-trips to the cloud. This transforms the problem from "how to perform domain adaptation on a server" to "how to perform training on a microcontroller with kilobyte-level read/write memory and milliwatt-level power consumption." The paradigm of migrating machine learning to such low-cost, low-power edge devices is TinyML. Its first phase (deployment frameworks such as TVM [5], TensorFlow Lite Micro [6], DORY [7]) only solved inference; On-Device Learning (ODL) frameworks (TinyOL [8], Tiny Training Engine [9], PULP-TrainLib [10]), which refer to updating model parameters with new data after deployment, have only appeared in recent years. ODL under the four tight constraints of memory, storage, latency, and device lifespan remains an open problem (Chapter 1).

(Terminology Anchor: The "On-Device Domain Learning" in the title and the ODDA—On-Device Domain Adaptation—repeatedly used in the text refer to the same thing: specializing a deployed model to the data distribution of a new environment, with the entire process staying within the device.)

### Specific Deficiencies of Existing Methods

Following the classification by Lopez-Espejo et al. [11] in Chapter 2, noise processing is divided into front-end and back-end routes, and the authors point out the deficiencies of each:

- **Front-end methods** (ANC, Active Noise Cancellation [12][13]; SE, Speech Enhancement [2][14]): The idea is to estimate and subtract noise components from the input signal, restoring "clean speech" before feeding it into the acoustic model. There are two deficiencies: First, front-end modules increase the latency and dimensionality of the KWS system (an additional resident sub-model must run); Second, and most critically, these methods do not support post-deployment model fine-tuning—the denoiser is fixed during training, so it remains the same in a new environment.
- **Back-end methods** ([3][15][16]): During offline training, clean utterances are artificially superimposed with various distortions and noises to obtain noise-aware acoustic models. This eliminates front-end overhead and yields better classification performance under harsh conditions, but it "lacks environmental awareness": the model does not know what specific noise is present on-site, so it cannot specialize for the target noise. In other words, the ceiling of back-end methods is "average robustness to all noises," rather than "specialized proficiency for this noise."
- **ODL framework layer**: Ren et al.'s TinyOL [8] extends a layer customizable via stochastic gradient descent behind a frozen DNN deployed (for incremental new classes); Lin et al.'s Tiny Training Engine [9] generates a backward computation graph, supporting sparse tensor updates and operator rearrangement, targeting single-core ARM MCUs; Nadalini et al.'s PULP-TrainLib [10] accelerates matrix multiplication in ODL by 36.6× compared to previous single-core solutions. These three provide training infrastructure but do not design tasks and workflows specifically for on-site noise conditions in KWS.
- **Closest prior work**: Cioflan et al. [1] (2022 AICAS) from the same team proposed a KWS domain adaptation methodology, which is the direct predecessor of this paper's methodology, but has two hard flaws: First, the energy cost is as high as 5.81 J—for an always-on battery-powered device that needs to adapt every time it encounters new noise, this would significantly shorten battery life; Second, the paper did not demonstrate deployment on actual devices, remaining at the methodology level.

### Key Challenges to be Solved by This Paper

The question the paper aims to answer can be compressed into one sentence: On an ultra-low-power MCU with only kilobyte-level read/write memory and a millijoule-to-joule energy budget per adaptation, how to perform domain adaptation on a strong robustness baseline (NA-KWS) that has "already seen 17 types of noise," and still extract considerable accuracy recovery under the more stringent 0 dB Signal-to-Noise Ratio (SNR, ratio of signal to noise power; 0 dB means signal and noise are equally loud) conditions compared to prior work. This is specifically broken down into four mutually constraining sub-challenges:

1. **Training Memory Equation**: Gradient descent learning requires that the input, parameters, activations, and gradients of a certain layer reside in read/write memory simultaneously (Chapter 3). Fig. 2 (right) shows that pushing adaptation depth to the deepest conv1 requires approximately 2.51 MB of backpropagation-related memory, far exceeding GAP9's 1.5 MB L2 SRAM;
2. **Storage Equation**: Labeled utterances for adaptation and frozen model parameters are placed in read-only storage. A complete training set requires 1.1 GB (main text), which is unrealistic for "extreme edge" devices;
3. **Latency/Energy Equation**: Eq. (1) indicates that adaptation time scales linearly with dataset size and number of epochs, while energy consumption and latency are proportional for always-on devices (Chapter 4);
4. **Engineering Implementation**: No previous work has actually run KWS ODDA on an ultra-low-power multi-core platform (the paper states this is the first).

## Methodology

### Overall Architecture Design and Design Motivation

The system lifecycle is divided into three stages (Fig. 1):

**Stage 1, Offline Noise-Aware Training**. Train the NA-KWS (Noise-Aware KWS) model on a server using the complete keyword dataset—clean utterances are superimposed with equal probability with 17 of the 18 real noise types, plus a completely noise-free clean condition; the one intentionally left out (denoted as the target noise) only appears after deployment, used to simulate "on-site new noise" (described in Chapter 6). The ingenuity of this setting lies in fair comparison: the baseline is already strong (has seen 17 types of noise plus clean conditions), and all gains of ODDA are measured "on top of a strong baseline," avoiding the issue of bullying a weak baseline.

**Stage 2, Deployment**. The acoustic model selects DS-CNN (Depthwise Separable CNN, a lightweight architecture that replaces standard convolutions with depthwise convolutions followed by pointwise convolutions to reduce computational power), with the rationale that [22]-[24] have proven it can form competitive KWS systems on low-power and ultra-low-power platforms (Chapter 5). The model is split into two segments, adopting mixed-precision division: **Frozen backbone** quantized to int8 and deployed using the DORY [7] code generator—because gradients and intermediate activations do not need to be stored, int8 inference energy efficiency can be maximized; **Learnable layers** (such as the classification head) run using fp32 code generated by PULP-TrainLib [10]—because training is sensitive to numerical precision, quantized gradients would introduce uncontrollable noise. The backbone output is dequantized and fed into the fp32 classifier (middle of Fig. 1). This "int8 inference + fp32 training" boundary drawn at the frozen/learnable boundary is the foundation for the memory and energy accounts of the entire scheme: the network to be trained is reduced by two orders of magnitude (as will be seen, only 780 parameters when training only fc1), while the inference segment retains all deployment optimizations.

**Stage 3, On-Site ODDA**. When new noise potentially damaging to performance appears in the environment (how to detect the trigger is not reported in the paper), the process is: Microphone records environmental noise → Noise is superimposed with pre-recorded labeled clean utterances to construct noisy training samples → Compute fp16 MFCC (Mel-frequency cepstral coefficients, the mainstream time-frequency feature for speech) → Pass through frozen backbone → Dequantized output and utterance labels enter the fp32 classifier to calculate cross-entropy loss → Backpropagation to compute gradients → Gradient descent to update learnable parameters → Repeat for each pre-recorded utterance on the board, running multiple adaptation epochs (Chapter 3 + Fig. 1).

There is a design detail that is easily overlooked but very critical: **Why only record noise on-site, not utterances**. Pre-recorded utterances belong to the NA-KWS training set, come with labels, and are stored in non-volatile memory. Users do not need to manually record data or label it—Chapter 3 states this "frees users from the tedious process of manual recording and labeling." Humans only need to let the device "listen to the environment for a while," with zero labeling cost; moreover, the same on-site noise segment can be superimposed onto different clean utterances, effectively providing free data augmentation combinations. Labels come from pre-stored data, and distribution differences come from on-site recordings; the two matters are decoupled.

**Target Hardware GAP9** (Fig. 1 right + Chapter 5): Ten general-purpose RISC-V cores, organized into a fabric controller (managing peripherals and main control logic) plus a nine-core compute cluster; the cluster supports int8 multiply-accumulate (MAC) instructions and shares four floating-point units (FPU, supporting fp16 and fp32); there is also a hardware convolution accelerator, with DNN execution energy as low as 5 µJ [21]. The storage hierarchy consists of 128 kB L1 TCDM (Tightly Coupled Data Memory), 1.5 MB L2 SRAM, and on-chip/off-chip L3 Flash and RAM. This hierarchy directly determines the memory upper limit of the "adaptation depth" strategy. In terms of division of labor (Fig. 1), cores on the fabric controller side are responsible for inference, and the nine-core cluster is responsible for training.

### Mathematical Principles of Core Algorithms

The algorithm itself is naive supervised learning: Cross-entropy loss $L_{CE}$ measures the difference between the classifier's predicted distribution and the utterance labels, backpropagation computes $\partial L_{CE}/\partial w$, and gradient descent $w \leftarrow w - \eta \nabla L$ updates learnable parameters (learning rate $\eta$ is not reported in the paper; batch size is two, see Fig. 2 right description). The paper's mathematical contribution lies not in the algorithm but in the **resource model**—Eq. (1) gives the scaling law for adaptation latency:

$$t_{ODDA} \propto (t_{infer} + t_{backprop}) \times \frac{S_{dataset}}{S_{batch}} \times N_{epochs}$$

where $S_{dataset}$ is the on-board dataset size, $S_{batch}$ is the batch size, $t_{infer}$ is the batch inference latency, $t_{backprop}$ is the parameter update latency, and $N_{epochs}$ is the number of adaptation rounds. This equation puts three optimization axes on the table: switch to a lightweight model (reduce per-sample latency), store less data (reduce $S_{dataset}$), and run fewer rounds (reduce $N_{epochs}$). It is also coupled with memory constraints: although latency decreases as batch size increases (parallelism dilutes the cost), the number of parallel samples is limited by the memory size required to store backpropagation-related variables—wanting to increase batch size requires increasing memory first. This is the dual relationship of "memory for latency" (Chapter 4).

Energy perspective: Device energy consumption is proportional to adaptation latency (under the same operating mode), so Eq. (1) is also an energy model. Based on this, the paper performed empirical accounting (Table II), which the author cross-checked and found completely self-consistent: 100 samples × 21 epochs = 2100 sample-level updates. DS-CNN S under HPM has 6.74 ms / 384 µJ per sample (Table II). 2100 × 6.74 ms ≈ 14.2 s, 2100 × 384 µJ ≈ 806 mJ—matching the "14 s, 806 mJ" reported in the main text precisely. This indicates that the "per-sample cost" in Table II covers the complete pipeline including inference, backpropagation, and parameter updates.

(Author's calculation, for understanding magnitude only: The forward and backward computation volume of one training step is approximately 3 times that of pure inference. In Table II, DS-CNN S inference is 2.95 MFLOPs, corresponding to approximately 8.85 MFLOPs per sample for training; 10% data full-network adaptation totals 1.43 TFLOPs (main text), which translates to approximately 40 sample rounds, which is reasonable in magnitude.)

### Key Technical Innovation 1: Flexible Adaptation Depth—Update Only the Last k Layers

**Method**: Partially freeze the target model, updating only the last k layers, starting from the final linear layer fc1 and unlocking layers downward (Chapter 4 + Fig. 2 right).

**Why it works (Memory Account)**: Gradient descent only requires that the input, activations, and gradients of the "layer being updated" reside in memory simultaneously (Chapter 3). Memory requirements arranged by update depth (batch size is two, Fig. 2 right): Updating to fc1 requires 0.01 MB, to conv9 requires 0.56 MB, to conv8 requires 0.82 MB, to conv6 requires 1.37 MB, to conv4 requires 1.93 MB, to conv2 requires 2.48 MB, and to conv1 (full network) requires 2.51 MB. Two hard facts: First, the 2.51 MB for full-network fine-tuning exceeds GAP9's 1.5 MB L2 SRAM (Chapter 5), hitting the memory wall for deep adaptation on this platform; Second, when training only fc1, only 780 trainable parameters need to be saved, along with the layer's input/output activations and weight gradients, totaling no more than 10 kB of read/write memory (main text), which fits comfortably into the 128 kB L1. The 250:1 memory span (2.51 MB to 0.01 MB, Fig. 2 right) comes almost entirely from the structural fact that "convolution layer activation maps are large, while linear layer activation vectors are small"—this is the physical basis for cutting adaptation depth into a tunable hyperparameter.

**Accuracy Account**: Training only fc1 can exceed the NA-KWS DS-CNN S baseline by 5.5 percentage points, and is comparable in accuracy to the NA-KWS DS-CNN L which is 17 times larger (main text); with 10% data, full-network fine-tuning is only 1.2 percentage points higher than training only the classification head, and the gap widens to 6 percentage points when data is increased to 100% (main text). Translated into engineering language: When data is scarce, the benefit of deep fine-tuning is marginal; "training only the classification head" exchanges less than one percent of memory for the vast majority of the benefit; deep adaptation is only worth the memory cost when data is sufficient. This is a clean ternary trade-off (data volume × adaptation depth × memory).

### Key Technical Innovation 2: Data Sub-sampling—Store Only a Small Bunch of Pre-recorded Utterances

**Method**: ODDA does not use the complete training set, but only stores a randomly sampled subset of pre-recorded labeled utterances on the board, with each GSC12 sample occupying 32 kB of storage (main text).

**Storage-Accuracy Account (Fig. 2 left + main text)**: Storage tiers range from 0.25% (0.003 GB, approx. 3 MB) to 100% (1.18 GB, 37,000 samples). When storing only 3 MB, the adapted DS-CNN S (model 26 kB, labeled in Fig. 2 left) already exceeds its NA-KWS counterpart by 6 points, and catches up to the largest NA-KWS DS-CNN L; when storing 10% (120 MB, approx. 310 samples per class), ODDA's DS-CNN S surpasses the largest noise-robust network by 10 points; when storing the full amount (1.1 GB in main text), the gain is up to 14 points. Corresponding total adaptation computation: DS-CNN S with 10% data is 1.43 TFLOPs, full amount with S is 10.46 TFLOPs, and full amount with DS-CNN L is as high as 98.10 TFLOPs (main text)—both data volume and model size linearly amplify adaptation costs, and edge devices can only afford the corner in the bottom left. Author's check: 37,000 × 0.25% ≈ 93 samples, meaning the "100 samples" flagship setting corresponds exactly to the leftmost tier in Fig. 2 left.

**Deep Judgment (Author's Analysis)**: This result indicates that the "domain adaptation" task itself is small-sample friendly—only the noise distribution needs to be adapted, the word distribution remains unchanged, so 100 samples (approx. 10 per class) are enough to pull the classification head back; this is completely different from the data requirements of few-shot tasks learning new words from scratch, and is the fundamental reason why this technical route can be compressed into 3 MB of storage.

### Key Technical Innovation 3: LPM/HPM Dual-Mode Scheduling and Engineering Implementation

GAP9 supports two operating modes, and the paper pins inference and training to their respective optimal modes (Table II + Chapter 6 C):

- **Low-Power Mode LPM**: 240 MHz, 650 mV, measured average power consumption 36 mW—used for always-on inference, triggered every 50 ms (main text);
- **High-Performance Mode HPM**: 370 MHz, 800 mV, power consumption kept under 55 mW—used for training bursts, saving time while actually having lower energy costs (see table below).

Per-sample complete adaptation cost (excerpt from Table II):

| DS-CNN | Computation [MFLOPs] | Parameters [kB] | Read/Write Memory [kB] | Efficiency [FLOP/cycle] | LPM Time/Energy | HPM Time/Energy |
|---|---|---|---|---|---|---|
| S | 2.95 | 23.7 | 9.5 | 4.94 | 10.89 ms / 424 µJ | 6.74 ms / 384 µJ |
| M | 17.2 | 138.1 | 25.5 | 9.18 | 24.16 ms / 988 µJ | 16.34 ms / 974 µJ |
| L | 51.1 | 416.7 | 40.9 | 11 | 55.04 ms / 2313 µJ | 32.95 ms / 2028 µJ |

(The statistical basis for the Efficiency column FLOP/cycle is not reported in the paper—direct conversion from the time column and main frequency cannot reproduce the value 4.94, it is speculated that equivalent conversions for specific precisions or accelerators are included.)

Flagship numbers: Fine-tuning the classification head of DS-CNN S with 100 samples (main text notes "10 per class") for 21 epochs, completed in 14 s, requiring only 806 mJ under HPM (main text), i.e., 384 µJ per sample (abstract). After switching to a larger model, the read/write memory for classifier updates increases from 10 kB to 40 kB (40.9 kB in Table II, consistent with main text), and average energy consumption increases by 5.3× (Author's check: Ratio of HPM energy for L and S in Table II is 2028/384 ≈ 5.28, matching). The largest DS-CNN requires up to 1.9 minutes to update once under HPM (main text; calculated as 2100 updates × 32.95 ms ≈ 69 s, which does not fully match 1.9 min, speculated that the large model used more samples or epochs, specific settings not reported in the paper).

### Technical Differences from Existing Methods

- **To front-end methods**: No additional resident modules, no increase in inference path latency; adaptation is an event-driven one-time action, with zero changes to the inference system.
- **To back-end NA-KWS**: Not a replacement but a superposition—NA-KWS itself is the baseline and starting point of this paper, ODDA specializes further on top of the strong baseline, with all gains measured relative to "noise-robust models" (abstract).
- **To prior work [1]**: Three substantial advancements—difficulty raised to 0 dB SNR (prior work did not use this difficulty); energy reduced from 5.81 J to 806 mJ (Author's check approx. 7.2× improvement); changed from methodology research to an empirical system on GAP9.
- **To general ODL frameworks (TinyOL/TTE/PULP-TrainLib)**: This paper is a joint design at the task level (noise domain adaptation) and system level (mixed precision, dual-mode, adaptation depth, data sub-sampling), with PULP-TrainLib being just one of the training backends called by it.
- **To few-shot KWS customization works [17]-[20]**: Those solve "learning new words," this paper solves "accuracy recovery of old words under new noise," the problems are orthogonal.

## Experimental Results

### Datasets Used and Their Scales

- **Keyword Data**: Google Speech Commands (GSC) [25], using the official split recommended by the dataset authors (main text). Three task scales: 12 classes (main setting, i.e., GSC12); 35 classes (full vocabulary); Custom 6-word task—following [27], taking the six words yes, down, left, right, off, stop, deliberately excluding words like no, go which are "short and phonetically similar" to avoid confusion dominating the experiments (main text).
- **Noise Data**: DEMAND multi-channel real-environment noise library [26], superimposed additively onto clean utterances. One of the 18 noise types is left out as the target noise (only appears after deployment), while the remaining 17 types plus clean conditions are used for NA-KWS offline training (main text).
- **Signal-to-Noise Ratio**: Training and evaluation experiments are all conducted at SNR = 0 dB (main text), which is more stringent than the scenario in prior work [1].
- **Five Target Noises for Evaluation** (Table I headers): cafeteria, meeting, restaurant, washing, metro.
- **On-Board Storage Scale** (main text + Fig. 2 left): Each GSC12 sample is 32 kB; complete training set of 37,000 samples is approx. 1.1 GB; 10% is 120 MB, approx. 310 samples per class; 0.25% is approx. 3 MB; Fig. 2 left also gives tiers for 30%/50%/70%/90% corresponding to 0.35/0.59/0.83/1.06 GB.

### Definition and Rationale for Evaluation Metrics

- **Accuracy Metrics**: Top-1 classification accuracy (%). The reason is direct: This paper focuses on how much accuracy degradation caused by noise can be recovered. The difference in the same metric before and after adaptation yields the gain, which is how Table I is presented.
- **Resource Metrics**: Total adaptation FLOPs, parameter size (kB), read/write memory requirements (kB, Table II specifically notes that memory refers to the read/write part), computational efficiency (FLOP/cycle), per-sample adaptation time (ms) and energy (µJ), average power consumption (mW). Pulling out such a complete resource coordinate system is because the paper's argument itself is the joint feasibility of "accuracy gain × resource cost"—ODL papers that only report accuracy cannot answer "whether the battery can hold up."
- **Metrics Not Reported in the Paper**: Trade-off between false alarm rate and miss rate, variance and confidence intervals of multiple runs, regression performance of the adapted model on non-target noises.

### Detailed Comparison with Baseline Methods and SOTA

Main comparison table Table I fully excerpted (NA-KWS is baseline, ODDA is after adaptation; three model tiers × three task scales):

| Noise | Model | Mode | 6 Classes | 12 Classes | 35 Classes |
|---|---|---|---|---|---|
| Cafeteria | S | NA-KWS / ODDA | 88.07 / 90.07 | 78.04 / 80.59 | 55.60 / 57.31 |
| Cafeteria | M | NA-KWS / ODDA | 90.29 / 91.78 | 82.60 / 84.25 | 59.60 / 60.69 |
| Cafeteria | L | NA-KWS / ODDA | 91.31 / 92.80 | 83.91 / 85.54 | 62.16 / 63.53 |
| Meeting | S | NA-KWS / ODDA | 79.35 / 88.41 | 65.69 / 78.53 | 49.19 / 57.19 |
| Meeting | M | NA-KWS / ODDA | 83.18 / 91.41 | 69.96 / 84.23 | 51.06 / 57.87 |
| Meeting | L | NA-KWS / ODDA | 85.13 / 92.89 | 70.76 / 84.74 | 55.27 / 63.49 |
| Restaurant | S | NA-KWS / ODDA | 83.30 / 85.59 | 73.52 / 76.22 | 53.68 / 55.97 |
| Restaurant | M | NA-KWS / ODDA | 85.61 / 88.76 | 77.83 / 80.12 | 57.03 / 59.01 |
| Restaurant | L | NA-KWS / ODDA | 87.39 / 88.88 | 79.53 / 81.55 | 59.31 / 61.19 |
| Washing | S | NA-KWS / ODDA | 94.98 / 96.24 | 89.30 / 90.45 | 64.94 / 65.47 |
| Washing | M | NA-KWS / ODDA | 96.63 / 97.10 | 91.64 / 92.86 | 67.58 / 68.15 |
| Washing | L | NA-KWS / ODDA | 96.82 / 97.22 | 92.92 / 93.39 | 71.46 / 71.97 |
| Metro | S | NA-KWS / ODDA | 92.84 / 93.10 | 83.82 / 85.50 | 63.89 / 64.95 |
| Metro | M | NA-KWS / ODDA | 94.30 / 94.15 | 87.10 / 89.51 | 65.82 / 66.66 |
| Metro | L | NA-KWS / ODDA | 94.75 / 95.54 | 87.73 / 90.06 | 68.40 / 69.10 |

(Table I, unit %)

Four core conclusions from reading the table:

**First, noise type determines the magnitude of gain—stationary noise does not need much adaptation, speech noise must be adapted.** On washing noise (stationary), ODDA brings only about 1.15 points (12-class S: 89.30 to 90.45, Table I), attributed in the main text to "general robust NA-KWS can already extract speech features under these noises"; while on meeting noise (non-stationary human speech), the same S model for 12 classes surges by 12.84 points (65.69 to 78.53), M model increases by 14.27 (69.96 to 84.23), L model increases by 13.98 (70.76 to 84.74)—expressed in the main text as "12%" and "up to 15%", which basically matches the table (main text rounds; larger models push top-1 to the 93% level under washing conditions, main text). The mechanism is easy to understand (Author's analysis): Meeting speech shares the time-frequency structure of speech with wake-up words. General augmentation learns the boundary of "speech vs. stationary noise," but cannot learn "the specific reverberation and speaker distribution of this meeting room." Only by having seen the target noise itself can this boundary be drawn correctly.

**Second, small models with ODDA can beat larger models.** In terms of parameters, DS-CNN L (416.7 kB) is 17.6 times that of S (23.7 kB) (Table II; main text takes "17×"). However, in the meeting noise 12-class task, the adapted S (78.53) surpasses the unadapted L (70.76) by 7.77 points (Table I)—expressed in the abstract and conclusion in the same vein as "small DS-CNN refined by ODDA exceeds the 17× larger network by up to 8%." This is the sharpest selling point of the entire paper: rather than stacking parameters for general robustness, it is better to leave a small learnable classification head for specialization on-site.

**Third, in terms of task scale, 12 classes is the gain sweet spot, 35 classes hits the ceiling.** Gains on the 35-class task narrow to 1 (washing) to 8 (meeting) points (main text; Table I shows 35-class meeting S +8.00, L +8.22), and absolute accuracy remains relatively low (35-class meeting ODDA best is 63.49, 35-class ODDA overall falls in the approx. 56 to 72 range, Table I), indicating that the bottleneck for large vocabularies lies in inter-class feature separability rather than the classification head. On the 6-word task, the main text claims peak gain of 11%, dropping to 7% as model capacity increases—checking Table I, 6-class meeting gains are S +9.06, M +8.23, L +7.76. The trend of "decreasing to 7-8 points with capacity" holds, but "peak 11%" does not precisely match the table (the verifiable max gain for 6 classes is +9.06). This is suspected to be due to rounding or differences in statistical basis (Author's note). Additionally, the adapted largest model falls in the 88% to 97% range for the 6-class task across five noises (main text; Table I L ODDA 6-class ranges from 88.88 to 97.22), and 84% to 94% for 12 classes.

**Fourth, gains are not positive everywhere.** For metro noise 6-class M model, there is a negative gain of 0.15 (94.30 to 94.15, Table I), which the paper does not discuss—reminding readers that adaptation is a risky action, not guaranteed to yield monotonic benefits.

**Discrepancy between Abstract and Main Text (Author's Note)**: The abstract states that 100 samples "recover 5%" accuracy, while Chapter 6 and the conclusion both state "improve by 6% within 14 s"; Chapter 1 of the main text also has expressions like "improve by 5% under meeting noise, read/write memory as low as 10 kB." The direction is consistent, but when citing, it is recommended to rely on Table I and Chapter 6.

### Findings from Ablation Experiments

The ablation in this paper covers four axes, each pointing to a translatable engineering judgment:

1. **Task Complexity Ablation** (6/12/35 classes, Table I): As complexity rises, relative gains narrow, and absolute accuracy is dominated by inter-class confusion; 12 classes is the peak gain area (highest approx. 14 points). For 6 classes, because easily confused words are excluded, features are easier to separate, so the peak gain is lower. For 35 classes, it is constrained by the separability ceiling.
2. **Model Capacity Ablation** (S/M/L, Table I + Table II): Larger models have higher baselines and smaller gains (6-class meeting S +9.06 to L +7.76, Table I). The main text attributes this to larger models being able to encode richer class representations, so domain shift hurts them less. On the cost side (Table II): Adaptation energy for M is approx. 2.5× that of S, and for L approx. 5.3×. Read/write memory increases from 9.5 kB to 40.9 kB. Computational efficiency increases from 4.94 to 11 FLOP/cycle (larger models have higher arithmetic intensity and more sufficient parallelism).
3. **Data Volume Ablation** (0.25% to 100%, Fig. 2 left): Accuracy rises monotonically with storage. 3 MB exchanges for +6 points and catches up to the largest NA-KWS model; 120 MB surpasses by 10 points; 1.1 GB exchanges for a max of 14 points (main text); Total adaptation computation increases from 1.43 TFLOPs (S, 10% data) to 98.10 TFLOPs (L, full amount) (main text). Storage is a linear cost, accuracy is a logarithmic return. 3 MB to 120 MB is the cost-performance interval.
4. **Adaptation Depth Ablation** (fc1 layer by layer to conv1, Fig. 2 right): Accuracy rises with depth and data volume. With 10% data, full-network is only 1.2 points higher than the classification head; with 100% data, it is 6 points higher (main text); Memory cost jumps from 0.01 MB (updating to fc1) to 2.51 MB (updating to conv1) (Fig. 2 right), the latter exceeding GAP9's 1.5 MB L2 (Chapter 5).
5. **Operating Mode Ablation** (LPM/HPM, Table II): HPM training is faster and more energy-efficient (S: 6.74 ms/384 µJ vs LPM's 10.89 ms/424 µJ); M's energy is almost identical in both modes (988 vs 974 µJ), L saves approx. 12% in HPM (2313 to 2028 µJ)—the larger the model, the thinner the energy dividend of trading frequency for time (Author's summary based on Table II).

Deployment recipe derived from comprehensive ablation (Chapter 6 C section of main text): DS-CNN S plus training only fc1 plus 100 pre-recorded samples plus 21 epochs plus HPM—10 kB read/write memory, 3 MB read-only storage, 14 s, 806 mJ, exchanging for approx. +6 points on meeting noise 12 classes.

## Main Contributions

1. **First (self-stated by paper) empirical KWS on-device domain adaptation system run on an ultra-low-power multi-core MCU**: GAP9 platform, 14 s, 806 mJ, 384 µJ per sample (abstract + Chapter 6 C).
2. **Two resource reduction strategies and a complete trade-off map**: Flexible adaptation depth (training only the last k layers, compressing read/write memory to 10 kB) and data sub-sampling (storage-accuracy curve starting from 3 MB), combined with the latency model of Eq. (1) to form a reusable design methodology (Chapter 4 + Fig. 2).
3. **Considerable gains on a strong baseline under the high-difficulty 0 dB setting**: Up to +14 points on the noise-aware NA-KWS model (12-word task, abstract and conclusion). The adapted small model surpasses the 17× larger model by up to 8 points (Table I).
4. **A reproducible mixed-precision engineering paradigm**: int8 frozen backbone (DORY deployment) plus fp16 MFCC plus fp32 learnable layers (generated by PULP-TrainLib), with dual-mode scheduling of LPM inference and HPM training (Chapter 5).

## Limitations and Future Work

### Technical Limitations of the Method

- **Side effects of adaptation not evaluated—Risk of Catastrophic Forgetting**: Table I only reports accuracy on the target noise. The performance of the adapted model on the remaining 17 source noises and clean conditions is not reported in the paper. Adaptation essentially pulls the classification head toward the target distribution, and regression toward the old distribution is a mechanistic risk. This blank space is a key unknown for productization.
- **Additive, single-channel noise assumption**: Noise is superimposed additively from DEMAND recordings onto clean utterances. It does not handle convolutive reverberation, multi-microphone arrays, echo, or real overlapping scenarios where "the interrupting speaker is speaking while the wake-up word appears simultaneously." Although meeting noise is real human speech, the target word itself is clean-recorded and then superimposed, so the real mixing of the two types of speech is not modeled.
- **Absence of Trigger Mechanism**: How to detect the startup condition of the entire process "appearance of new noise" (VAD, noise change detector, user manual?) is not reported in the paper. Actual systems must add this link, otherwise the timing of adaptation cannot be discussed.
- **Absolute Accuracy Ceiling for Large Vocabulary Tasks**: Even after adaptation, the 35-class task only falls in the approx. 56 to 72 range (Table I; the adaptation depth and data volume configurations for ODDA in Table I are also not separately labeled), indicating that this paradigm has limited recovery capability for large-vocabulary KWS.
- **Platform and Architecture Binding**: fp32 training relies on cluster FPU, performance relies on nine-core parallelism and hardware convolution accelerators. The feasibility of migrating to lower-end MCUs without FPU is not reported in the paper; the acoustic model only validates one type, DS-CNN.
- **Storage still linear with number of samples**: Each is 32 kB (main text), 100 samples require 3 MB, which is on the large side for some MCUs; further reduction methods such as feature-level storage compression (e.g., storing features directly instead of waveforms) are not explored—this is not reported in the paper.

### Deficiencies in Experimental Design

- **No Report on Statistical Robustness**: All accuracy figures are single-run results, with no multiple seed repetitions, no variance or confidence intervals. It is questionable whether gains of this magnitude, such as +0.47 (M, 6 classes, Table I) on washing noise, are significant.
- **Discrepancies between Main Text and Table Numbers**: Such as deviations of "peak 11%" and "up to 15%" from verifiable values in Table I, and inconsistency between abstract's "recover 5%" and main text's "+6%" (see Author's Note in Experimental Comparison section). Although this does not harm the direction of the conclusion, Table I should be used as the standard when citing.
- **Incomplete Comparison with Prior Work [1]**: Comparison with prior work [1] is only at the text level regarding energy (5.81 J vs 806 mJ), and the two have different SNR settings (this paper's 0 dB is harder). There are no same-condition controlled experiments to isolate the respective contributions of "difficulty increase" and "method improvement."
- **Hyperparameters and Basis Opacity**: Learning rate, optimizer variants, whether 10% data sampling is randomly repeated, the conversion method for the FLOP/cycle efficiency column, and the samples and epochs corresponding to the largest model's 1.9 minutes are all not reported in the paper.
- **Single Evaluation Dimension**: Only top-1 accuracy is reported. The trade-off between false alarm and miss rates, which KWS deployment is truly sensitive to, is not reported. Nor is the impact of the 14 s adaptation process on always-on inference, such as how it interrupts inference and its effects.

### Possible Directions for Future Improvement

- **Automatic Trigger and Continuous Adaptation**: Integrate distribution shift detection (classifier confidence monitoring, feature statistical drift detection) to automatically start ODDA, and evolve towards online continuous adaptation for non-stationary noise streams. End-to-end calculation of trigger frequency and battery lifespan is a natural next step.
- **Adaptation Objectives Anti-Forgetting**: Introduce replay (retaining a small number of source noise samples to participate in adaptation) or regularization constraints, turning "specializing for target noise" and "maintaining performance on old noises" into a multi-objective problem, filling in the blank not evaluated in the paper.
- **Breaking Dependence on Pre-recording and Labeling**: This paper avoids labeling costs by relying on pre-stored labeled utterances. Further, pseudo-labels or self-supervised objectives can be used to directly utilize on-site unlabeled audio, while compressing the 3 MB read-only storage.
- **Lower-Bit Training Paths**: The learnable layers are currently fp32. Exploration of int16/int8 gradients, sparse or low-rank updates is possible to further compress the 10 kB read/write memory, migrating to lower-end MCUs.
- **Joint with Front-End Methods**: SE/ANC is responsible for eliminating modelable stationary components, while ODDA is responsible for specializing in modelable speech noise. The combination of these two complementary routes has not been explored.
- **Expanding Task Boundaries**: Absolute accuracy for larger vocabularies, multi-channel and array inputs, multi-speaker overlapping scenarios, and generalizing the paradigm of "frozen backbone plus on-site specialized classification head" to other TinyML audio tasks such as acoustic event detection—this direction is pointed to by the TinyTrainer project (Swiss National Science Foundation 207913) acknowledged in the paper.
