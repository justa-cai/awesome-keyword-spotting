# Implementing Keyword Spotting on the MCUX947 Microcontroller with Integrated NPU

- **Authors/Affiliations**: Petar Jakuš, Hrvoje Džapo (Faculty of Electrical Engineering and Computing, University of Zagreb, Croatia)
- **Date**: June 10, 2025 (arXiv:2506.08911v1)
- **Link**: https://arxiv.org/abs/2506.08911
- **Keywords**: keyword spotting, microcontroller, MCU, NPU, eIQ Neutron, quantization aware training, edge AI
- **Paper Nature**: Embedded deployment-focused short paper (4 pages). There are no new model architectures or new training algorithms. The core value lies in successfully running the complete pipeline of "MFCC frontend + small CNN + QAT quantization + NPU toolchain porting" on the NXP MCXN947, and providing comparative empirical data for the same model on both CPU and NPU.
- **Naming Correction**: The paper title writes "MCUX947", which is a typo; the text and NXP official naming both use MCXN947 (MCX N series). This note uniformly uses MCXN947.

## Problem Statement

### Problem Background and Domain Pain Points

Always-on voice interfaces are standard features in smart speakers, wearable devices, and home appliance controllers. However, the computing carriers for these devices are often resource-constrained microcontrollers: clock speeds in the hundreds of megahertz, memory in the hundreds of kilobytes, and power budgets in the milliwatt range. Keyword Spotting (KWS) systems must run continuously under these constraints. Traditional approaches press neural network inference onto the CPU, using compression techniques such as INT8 quantization, pruning, and binarization to save resources. The system profile given in the paper's introduction is typical: signal acquisition (mostly MEMS microphones) → feature extraction (MFCC, RASTA-PLP, or simplified FFT) → neural network classifier. All three stages run on the same MCU.

Since 2023, a new variable has emerged: mainstream MCU manufacturers have begun integrating Neural Processing Units (NPUs) into their chips. NXP's MCX N series features its proprietary eIQ Neutron NPU. While hardware accelerators are now available, engineers immediately face three new problems. First, the NPU does not "activate automatically": it only recognizes its own model format and operator set. Models from the training side must be converted via the vendor's toolchain before running on the NPU, a process that rewrites the computational graph. Second, the NPU's native numerical type is fixed-point (8-bit in this paper). Floating-point models must be quantized, which leads to accuracy loss. Third, and most practically: "How much faster, smaller, and less accurate does it become when switching to the NPU?" Public literature lacks strict comparative data for the same model on the same chip between CPU and NPU. Vendor datasheets provide peak compute numbers, but no complete KWS case study places latency, size, and accuracy on the same table.

### Specific Shortcomings of Existing Methods

The paper's related work section outlines four existing routes and points out their differences from this work:

- **Architecture side**: CNNs perform strongly in KWS, especially when combined with pruning and quantization; LSTM, GRU, and their low-power variants (e.g., eGRU) handle temporal modeling; hybrid CNN-RNN architectures strike a balance between performance and efficiency. These works validate "which network suits KWS," but are basically evaluated on CPUs or GPUs, without touching the NPU deployment path.
- **Compression side**: Network binarization further compresses memory usage for edge deployment (the paper cites binarization works like BiFSMN); coprocessors and other hardware accelerations have been proven to significantly reduce power consumption. However, combined evaluations of compression methods and dedicated NPU hardware remain scarce.
- **Frontend side**: To reduce preprocessing complexity, some works directly use FFT to omit the mel-domain mapping, or even use analog frontends and "featureless" techniques to move the frontend out of the digital domain. These routes pursue extreme power efficiency but sacrifice the auditory alignment characteristics of MFCC. Moreover, analog frontends are not programmable and make model iteration difficult.
- **Toolchain side**: Existing literature hardly discloses the structural differences across the three stages of "training graph → quantized graph → NPU graph." After conversion by the vendor's toolchain, operators are replaced and fused. The final graph running is no longer the same graph as during training. This opacity is the main source of risk in porting.

In summary, the gap is: a lack of reference literature that fully connects "QAT quantization + NPU toolchain porting + dual execution path empirical testing" and publicly releases all structural and performance tables. This paper fills this position.

### Key Challenges Addressed by This Paper

To port a single-keyword detector based on MFCC + CNN, compress it to 8-bit fixed-point using Quantization Aware Training (QAT), port it to the Neutron NPU of the MCXN947 via the NXP eIQ Toolkit, and quantitatively answer three questions: how much accuracy is lost, how much the model shrinks, and how many times faster the NPU is relative to the CPU. The implicit fourth challenge (not explicitly stated but running through the entire paper) is: the loss at every step of the entire toolchain must be controllable—simulation errors in QAT during training, operator rewriting during conversion, and fixed-point numerical behavior after deployment. If any link goes out of control, the final accuracy becomes unusable.

## Methodology (Organized by NPU Toolchain and Model Porting Process)

### Process Overview

The methodology of the entire paper can be split into a six-step pipeline, with the output and motivation for each step as follows:

1. Data and Task Definition (Speech Commands, binary classification)
2. MFCC Feature Frontend (25 ms window / 10 ms hop, 20-dimensional coefficients per frame, input tensor 98×20×1)
3. Base CNN (float32, two convolutional layers plus two fully connected layers, Table I)
4. Quantization Aware Training QAT (simulating INT8 during training, Table II)
5. eIQ Toolkit Conversion (TFLite INT8 → Neutron NPU graph, operator rewriting, Table III, Figure 6)
6. MCU-side Deployment (model converted to static array stored in flash, two execution paths: CPU and NPU)

This order itself is a reusable porting methodology: first, narrow the task to the minimum closed loop (single keyword), choose the most conservative mature schemes for the frontend and model, and concentrate all uncertainties on the two links that truly need verification: "quantization + NPU porting."

### Step 1: Data and Task Definition

The data uses the Google Speech Commands dataset (Warden 2018, cited as [16]): 105,829 one-second recordings, 35 words, 16 kHz sampling, recorded on mobile devices. It is split according to the official original division into 84,843 training samples, 11,005 test samples, and 9,981 validation samples (the sum of these three numbers is exactly 105,829, self-consistent).

The task does not perform 35-class classification but narrows it down to a binary classification of "Marvin vs. Others," with only one sigmoid unit in the output layer. Why narrow it down: this is the minimum prototype for a wake-word scenario—the device only needs to answer "whether the wake word appeared," without distinguishing among 35 commands. The cost is that results cannot be directly compared horizontally with public benchmarks for 35-class classification. The paper also does not supplement 35-class experiments (not reported).

Class imbalance is severe: in the test set, there are only 195 Marvin samples and 10,810 non-Marvin samples (inferred from Tables IV/V), a positive-to-negative ratio of approximately 1:55.4. Class weights are used to counterbalance during training: the Marvin class weight is 24.81, and the non-Marvin class weight is 0.51. The ratio between the two is approximately 48.6, which is of the same order of magnitude as the imbalance ratio. This weight selection directly pushes the model toward a high-recall training objective—this drift is related to the confusion matrix later.

### Step 2: MFCC Feature Frontend

The frontend pipeline has four steps, all executed in real-time on the Cortex-M33:

- **Framing and Windowing**: 25 ms frames, 10 ms hop distance (400 sampling points at 16 kHz). Each frame is multiplied by a Hamming window $w(n) = 0.54 - 0.46 \cos(2\pi n/N)$. The 25/10 ratio is a classic trade-off in speech frontends: if the window is longer, the stationarity assumption within the frame fails; if shorter, frequency resolution is insufficient. One second of audio is cut into 98 frames according to this (inferred from $(1000-25)/10+1=98.5$ rounded), which is exactly the source of the first dimension 98 of the model input.
- **Spectral Analysis**: FFT transforms the windowed frames into the frequency domain, then takes the power spectrum $P[k] = \frac{1}{N}|X[k]|^2$.
- **Mel Filtering**: 40 Mel filters cover 40 Hz to 7.6 kHz, aligning with human auditory frequency perception—the Mel scale has high resolution at low frequencies and low resolution at high frequencies, consistent with human hearing. The upper limit is 7.6 kHz instead of the Nyquist 8 kHz, leaving a transition band.
- **Cepstral Calculation**: DCT is performed on the log-Mel spectrum. After decorrelation, the first 20 coefficients are taken to obtain 20-dimensional MFCCs per frame, following the HTK convention. The essence of DCT plus coefficient truncation is to compress the filter bank energy into a low-order representation of the "vocal tract envelope," discarding rapidly changing excitation details—for keyword discrimination, the envelope is sufficient.

The final one-second audio becomes a 98×20 feature map, entering the network as a (98, 20, 1) tensor.

**Why use a manual MFCC frontend instead of an end-to-end learned frontend**: The paper's related work mentions that feature extraction can be incorporated into the network (citations [4], [15]) or directly use FFT (citation [2]) or even analog frontends (citations [3], [14], [17]). There are three engineering reasons for choosing classic MFCC: First, the CPU cost is deterministic and very low—the empirical test shows an average of 431 µs per frame (Section III), which is only about 4.31% duty cycle relative to the 10 ms hop distance (inferred), fully affordable. Second, the auditory alignment characteristics of MFCC have been verified by thirty years of speech literature; the model side does not need to relearn the frontend, requiring less training data. Third, the NPU's operator set is mainly convolution and matrix multiplication. Stuffing operators like FFT and Mel filtering into the NPU risks falling back to the CPU. It is better to run them honestly on the CPU.

### Step 3: Base CNN Architecture (Table I)

The structure is very restrained: two layers of "Convolution + Batch Normalization + Max Pooling," followed by Global Average Pooling, dropout, and two fully connected layers. Listed layer by layer according to Table I:

| Layer | Output Shape | Parameters |
|:---|:---:|:---:|
| input | (98, 20, 1) | 0 |
| conv1 (3×3, 32 channels) | (96, 18, 32) | 320 |
| bn1 | (96, 18, 32) | 128 |
| pool1 (2×2) | (48, 9, 32) | 0 |
| conv2 (3×3, 64 channels) | (46, 7, 64) | 18,496 |
| bn2 | (46, 7, 64) | 256 |
| pool2 (2×2) | (23, 3, 64) | 0 |
| gap (Global Average Pooling) | (64,) | 0 |
| dropout | (64,) | 0 |
| fc1 | (128,) | 8,320 |
| output | (1,) | 129 |

Total parameters: 27,649 (108.00 KB, i.e., 4 bytes per parameter for float32, numbers are self-consistent). Among them, 27,457 are trainable and 192 are non-trainable (statistics for two BN layers). Convolution kernel sizes can be inferred from the shapes: 98→96, 20→18 indicates conv1 is 3×3 with no padding; 46→7 indicates conv2 is also 3×3.

The "why" for each layer's design:

- **Two 3×3 convolutions**: Local patterns on the MFCC feature map (energy envelope of a phoneme segment, resonance trajectory fragments) can be captured by combining 3×3 receptive fields step by step. 3×3 is also the most mature core shape supported by various NPUs. The stepwise doubling of channels from 32 to 64 is a standard parameter-saving approach for small CNNs.
- **GAP instead of Flatten**: This is the most valuable decision in this architecture. The pool2 output is 23×3×64=4,416 dimensions. If flattened and connected to a 128-dimensional fully connected layer, it would require about 565,000 parameters (inferred), which is 20 times more than the rest of the model combined. GAP averages out the spatial dimensions directly, leaving only 64 channels. fc1 only needs 8,320 parameters, a difference of about 68 times. An附带 benefit is translation robustness—the starting position drift of the wake word within the 1-second window does not affect the output.
- **1 output unit**: Binary classification, sigmoid activation (corresponds to the Logistic operator in the NPU graph).
- **dropout**: A small model with 27.6K parameters trained on 85k samples; regularization is a fuse.

Training configuration: Adam optimizer, learning rate 0.001, for 10 epochs, class weights 24.81/0.51. Why only 10 epochs and whether there is a learning rate decay strategy is not explained by the paper; judging from the results, the model has converged (float accuracy 99.14%). However, for QAT, the number of training rounds is on the short side, and the sufficiency of quantized parameter convergence is questionable (training curves not reported).

### Step 4: Quantization Aware Training (Table II)

The paper's quantization route is QAT (Quantization Aware Training), rather than Post-Training Quantization (PTQ). The method involves inserting pseudo-quantization nodes into all layers in the training graph: forward propagation simulates 8-bit fixed-point operations (weights and activations are quantized and then dequantized), while backward propagation maintains full-precision gradients. The paper explicitly states the cost is an increase in training time of about 20%.

**Why choose QAT over PTQ**: There are two layers of reasons. The surface reason is the paper's self-stated "minimization of post-quantization accuracy loss"—QAT allows weights to adapt to quantization errors during training, growing into a distribution friendly to fixed-point, whereas PTQ is a hard truncation after training, which is prone to collapse in small models. The deeper reason is hard constraints: the native numerical type of the Neutron NPU is 8-bit fixed-point. Int8-ization is not an "optional optimization" but the "ticket to enter" for running on the NPU. Since quantization is mandatory, the most stable quantization method must be used.

The paper also describes an alternative route: train with full precision first, then fine-tune with a small number of QAT epochs, resulting in lower training time and comparable accuracy. However, comparative data for the two routes is not reported in the paper. Judging from the text, the main results use the complete QAT route.

Table II gives the structure of the QAT-prepared version: quantize_layer (3 parameters), quant_conv1 (387), quant_bn1 (129), quant_pool1 (1), quant_conv2 (18,627), quant_bn2 (257), quant_dropout (1), quant_fc1 (8,325), quant_output (134) are inserted into the original structure. Total parameters: 27,864 (108.84 KB). Compared to the float version, there are 215 more parameters, all non-trainable (192→407), corresponding to the quantization scale and zero-point inserted per layer. The number of trainable parameters remains exactly the same (27,457), indicating that QAT does not change model capacity, only changes training dynamics.

### Step 5: eIQ Toolkit Conversion and NPU Graph Reconstruction (Table III, Figure 6)

This is the most reusable step for the community in the entire paper. The quantized TFLite model cannot be fed directly to the NPU; it must be converted to an NPU-compatible format via the NXP eIQ Toolkit. The conversion process "reconstructs certain layers to leverage dedicated hardware acceleration capabilities." Table III lists the reconstructed graph completely:

| Layer (NPU Graph) | Output Shape | Parameter Composition |
|:---|:---:|:---|
| InputLayer | [-1, 98, 20, 1] | 0 |
| Conv2D | [-1, 96, 18, 32] | 320 weights + 32 biases + 32 scales + 32 offsets |
| MUL, ADD (BatchNorm) | [-1, 96, 18, 32] | — |
| MaxPool2D | [-1, 48, 9, 32] | 0 |
| Conv2D | [-1, 46, 7, 64] | 18,432 weights + 64 biases + 64 scales + 64 offsets |
| MUL, ADD (BatchNorm) | [-1, 46, 7, 64] | — |
| MaxPool2D | [-1, 23, 3, 64] | 0 |
| Mean (i.e., GAP) | [-1, 64] | 0 |
| FullyConnected | [-1, 128] | 8,192 weights + 128 biases |
| FullyConnected | [-1, 1] | 128 weights + 1 bias |
| Logistic | [-1, 1] | 0 |

Total parameters: 27,521 (28.20 KB, approximately 1 byte per parameter, consistent with int8 dense storage; scales/offsets and other non-single-byte parameters explain the slight discrepancy with 27,521 B of pure weights). Trainable: 27,297, non-trainable: 224. Figure 6 visualizes this converted graph using Netron (citation [13]).

The reconstruction points are broken down one by one, each with a clear "why":

- **BN is split into MUL/ADD two operators**, with per-channel scales and offsets directly attached to the convolution block. Reason: The NPU's operator whitelist does not have an independent BatchNorm operator. Inference-time BN is essentially a per-channel affine transformation (multiply scale, add offset). Folding it into multiply-add incurs zero extra overhead and can seamlessly connect with the preceding convolution output.
- **GAP becomes Mean**, and the final Dense becomes FullyConnected plus Logistic. This is a mapping of "same mathematical operation, different operator naming": the operator names in the training framework are aligned with the operator names in the NPU graph specification. The fixed-point implementation corresponding to sigmoid is called Logistic.
- **Dropout and QuantizeLayer disappear from the inference graph**. Dropout is a training-specific operator; QuantizeLayer is responsible for converting float inputs to int8. Since the input to the NPU path is already a fixed-point tensor, this layer has no meaning.
- **Convolution blocks carry scale/offset, fully connected blocks do not**. Because scale/offset are products of BN folding, not weight quantization parameters—the fc layer has no BN in front of it, so naturally no such pair of parameters. This detail helps readers judge the source of per-channel parameters in others' NPU graphs.

The hidden engineering risk of this step is worth emphasizing: after rewriting by the toolchain, the model is "theoretically equivalent" numerically to the graph during training, but saturation handling and rounding modes are determined by the toolchain. If there is an inexplicable deviation between deployment accuracy and offline evaluation, you can hardly reproduce and debug this problem on the training side—you can only rely on chip-in-the-loop evaluation. The paper does not discuss this. The confusion matrix in Table V is actually an evaluation result of the chip-in-the-loop (or at least the NPU graph). This is why it is more credible than works that "only report TFLite offline accuracy."

### Step 6: MCU-side Deployment Form

Deployment details are given briefly in the paper but contain all key information: the quantized model produces two paths, CPU version and NPU version, via the eIQ Toolkit, executed on the ARM Cortex-M33 CPU and eIQ Neutron NPU of the MCXN947 respectively. The models of both versions are converted into static arrays (C arrays) and burned into the MCU's flash—there is no file system on the MCU; the model is code. Projects not reported by the paper: the inference framework used for the CPU path (TFLite Micro or vendor runtime), chip clock speed, NPU clock, specific SRAM/flash usage, and the audio acquisition link (path from microphone to buffer). These omissions mean that the attribution analysis of "59×" can only stay at the surface level.

## Main Contributions

1. **A reproducible end-to-end porting process**: From data, frontend, QAT to toolchain conversion and dual-path deployment, all six steps provide parameters and structures. In particular, the three comparative tables I/II/III completely disclose the structural evolution of "float training graph → QAT graph → NPU graph"—most deployment papers only give final results. These three tables are the most unique contribution of this paper; such a complete "before and after conversion" comparison is hard to find in other papers.
2. **CPU/NPU empirical anchor for the same chip and model**: The same int8 model runs on the MCXN947 with CPU inference taking 228.2 ms and NPU inference taking 3.847 ms (Table VI), an acceleration ratio of 59.35 times (inferred precise value, paper's口径 59×). This is one of the few credible public quantitative data points on "whether NPU on MCU is worth it."
3. **Complete three-dimensional characterization of quantization benefits**: From float to int8, the size drops from 383,674 B to 35,744 B (a decrease of 90.68%, abstract and conclusion口径, consistent with Table VI numbers); accuracy drops from 99.14% to 97.06% (Tables IV/V); latency on x86 desktop CPU drops from 50.67 ms to 0.42 ms (Table VI).
4. **Fourth point discovered in deep reading (not self-stated by the paper)**: The confusion matrix shows that quantization significantly shifts the operating point towards high false alarms—false alarms increase from 64 cases to 309 cases (×4.83), misses decrease from 31 cases to 15 cases (halved), and the int8 model's 97.06% accuracy is lower than the "never wake up" trivial baseline of 98.23% (inferred). This is a textbook case of "accuracy metric failure in imbalanced single-keyword tasks." It is more warning-valuable for wake-word product developers than the 59× figure.

## Experimental Results

### Evaluation Setup

Evaluation is performed on the Speech Commands official test set (11,005 samples) for window-level binary classification, reporting accuracy and confusion matrices (Tables IV, V). There are no streaming detection-level metrics (false wake-ups per hour, wake-up response latency), no threshold scanning, and no ROC/DET curves—the paper does not report these.

### Overall Comparison of Three Configurations (Table VI)

| Configuration | Size (B) | Accuracy | i5-10210U Latency | MCXN947 Latency |
|:---|:---:|:---:|:---:|:---:|
| Regular (float32) | 383,674 | 99.14% | 50.67 ms | Not Reported |
| Quantized (int8) | 35,744 | 97.06% | 0.42 ms | 228,210 µs (M33 CPU) |
| NPU (int8) | 30,576 | 97.06% | Not Reported | 3,847 µs (Neutron NPU) |

Three observations worth expanding:

- **Acceleration on x86 due to quantization is about 121 times** (50.67/0.42, inferred). This number is abnormally large—the theoretical benefit of int8 relative to float is in the order of 4 times. The remaining ten to twenty times come from desktop CPU int8 SIMD instructions and modern inference framework kernel optimizations. The paper does not report which inference frameworks were used for the two precisions on the i5. Strictly speaking, this cross-precision latency comparison suffers from framework confusion and should be discounted when referenced.
- **The gap between MCU CPU and desktop CPU is about 543 times** (228,210/420 µs, same int8 model, inferred). The paper does not report the M33 main frequency or whether CMSIS-NN optimized kernels are used, making it impossible to further attribute whether the gap is due to frequency or software stack.
- **NPU is still about 9.2 times slower than desktop CPU** (3,847/420, inferred). The positioning of MCU NPU has never been to catch up with desktop compute power, but to pull inference from "unusable" to "usable" within a milliwatt-level power budget—unfortunately, power consumption is exactly what this paper did not measure.

Another detail: the float version model was not tested on the MCXN947 (the cell in Table VI is "-"), and the paper does not explain the reason. A reasonable guess is that a 383 KB float model is unrealistic for MCU memory and latency, but this is the author's inference rather than a statement in the paper.

### Deep Dive into Confusion Matrices (Table IV vs. Table V): What Quantization Truly Changes

This is the most data worth reading carefully in this paper, and the part that the abstract and conclusion do not explain thoroughly. The test set has 10,810 negative samples and 195 positive samples (Marvin):

| | float (Table IV) | int8/NPU (Table V) |
|:---|:---:|:---:|
| True Negative (TN) | 10,746 | 10,501 |
| False Positive (FP) | 64 | 309 |
| False Negative (FN) | 31 | 15 |
| True Positive (TP) | 164 | 180 |
| Accuracy | 99.14% | 97.06% |

Diagnostic metrics inferred from these two tables:

- Recall (Marvin detection rate): 84.10% → 92.31%, an increase of 8.2 percentage points;
- Misses: 31 → 15, halved;
- False Alarm Rate per negative window: 0.59% → 2.86%, an increase of 4.83 times;
- Precision: 71.93% → 36.81%, halved;
- F1 Score: 77.54% → 52.64%, significantly deteriorated;
- Trivial Baseline (accuracy of "never wake up"): 10,810/11,005 = 98.23%. Float's 99.14% is only 0.95 percentage points higher than it, while int8's 97.06% is actually 1.13 percentage points lower.

Interpretation at three levels:

**First, quantization is not "uniform point loss," but operating point drift.** On the surface, accuracy only drops by 2.08 percentage points. In reality, the decision boundary shifts as a whole towards "say Marvin more": the cost of halving misses is nearly five times the false alarms. For always-on wake products, false alarms are an experience killer—speakers waking up by themselves in the middle of the night attract more user complaints than occasionally failing to wake up. A 2.86% false alarm rate per window is completely unusable in a streaming sliding window scenario (sliding multiple windows per second). It must be saved by post-processing such as raising the decision threshold and multi-window voting. The paper does not report any threshold or post-processing information.

**Second, accuracy is an invalid metric for this task.** In a binary classification with a positive-to-negative ratio of 1:55, the zero-cost baseline of "never wake up" has an accuracy of 98.23%. Any model's accuracy number is dominated by the negative class, showing neither recall improvement nor false alarm deterioration. The correct reporting posture is FPR@Target Recall, AUC, or DET curves. This paper inadvertently provides a complete case: if one only looks at "97.06% accuracy" in the abstract, readers will think quantization is almost lossless; only by looking at the confusion matrix does one realize the shape of the loss is "false alarms exchanged for recall."

**Third, possible mechanisms of drift (author's inference, not analyzed by the paper).** The class weight of 24.81 during training has already pushed the objective towards high recall. Int8 quantization compresses the dynamic range of logits, shifting the sigmoid output distribution as a whole, causing more negative samples to fall above the decision threshold. If this inference holds, simple threshold recalibration should pull the operating point back—but the paper did not perform threshold scanning. This is the most regrettable part of the experimental design.

### Frontend Latency and the Claim of "Full Pipeline Less Than 5 ms"

The paper empirically tests the MFCC frontend on the Cortex-M33, averaging 431 µs per frame. The conclusion claims "the complete processing pipeline is less than 5 ms." The most reasonable reading is: within every 10 ms frame period, incremental feature calculation takes 431 µs plus one whole-window NPU inference of 3,847 µs, totaling 4.278 ms, which fits within the 10 ms real-time budget, with a duty cycle of about 43% (inferred). Note two口径 issues: First, if inference is not triggered per frame but once per second (inference only when the window is full), the inference cost is diluted to almost zero, and the duty cycle drops to just the feature's 4.31%. Second, the paper does not explain the inference trigger rhythm and sliding window strategy. "Less than 5 ms" should be understood as the "latency upper limit of a single feature extraction plus a single inference" rather than the duty cycle of a streaming system.

## Configuration Comparison (Instead of Ablation Experiments)

The paper does not perform classic ablation (no layer-by-layer breakdown, no mechanism isolation). The comparable data it provides is a comparison across three configuration axes.

### Axis 1: Numerical Precision (float32 → int8 QAT)

Size: 383,674 → 35,744 B, a decrease of 90.68% (paper's口径, can be recalculated from Table VI); Accuracy: 99.14% → 97.06%, a decrease of 2.08 percentage points; i5 Latency: 50.67 → 0.42 ms. QAT's +20% training time buys the surface result of "accuracy only drops by 2 points"—but the confusion matrix reveals the real cost is operating point drift. The inspiration from this comparison: quantization acceptance must look at the confusion matrix; accuracy will deceive both authors and readers.

### Axis 2: Execution Unit (M33 CPU → Neutron NPU, same int8 model)

Latency: 228,210 → 3,847 µs, an acceleration of 59.35 times (inferred precise value); Accuracy remains unchanged (97.06%), indicating the NPU path is functionally equivalent; Size saves another 5,168 B (35,744 → 30,576, a decrease of 14.46%, inferred)—the NPU graph after eIQ conversion is more streamlined, saving parts corresponding to TFLite runtime metadata and operator fusion (inferred). This comparison is the core selling point of the paper: the net benefit of NPU porting is "more than one order of magnitude latency reduction + double-digit size reduction," at the cost of having to go through the vendor toolchain and accept operator rewriting.

### Axis 3: Platform Span (Desktop x86 vs. MCU)

Same int8 model: 0.42 ms on i5, 3.847 ms on MCU NPU (9.2 times slower), 228.2 ms on MCU CPU (543 times slower). These numbers are very useful for predicting "whether it can be ported to the edge": a model that takes 10 ms for inference on a desktop CPU will likely be two to three orders of magnitude slower when ported to the CPU of a hundred-MHz-class MCU. Even with an NPU, one must be prepared for it to be 5 to 10 times slower (all inferred by order of magnitude based on this paper's data, specifically depending on the model and chip).

### Configuration Dimensions Not Covered by the Paper

Checking thread and clock configurations against task requirements: the paper does not report any multi-core or multi-thread experiments (the number of cores of MCXN947 is not mentioned in the paper itself), nor any main frequency/clock scanning experiments, nor power consumption and energy efficiency (explicitly listed as future work in the conclusion). Additionally missing: comparison between PTQ and QAT (unable to quantify how much accuracy protection QAT itself contributed), exploration of int4 or mixed precision, decision threshold scanning, multi-keyword configuration, noise and far-field conditions. These gaps define the paper's positioning as "process verification"—it proves a path is walkable and provides road signs, but does not exhaust the branches on the path.

## Limitations and Future Work

### Limitations of Evaluation Methodology

1. **Accuracy metric failure** (detailed in the confusion matrix deep dive section): Imbalanced binary classification + negative class dominance. Should report false alarm rate, recall, or DET curves instead.
2. **Window-level offline evaluation, no detection-level metrics**: The acceptance metrics for real wake systems are false wake-ups per hour and wake-up response latency. This requires a full-link evaluation of streaming sliding windows plus post-processing. This paper stops at fixed 1-second window classification.
3. **Single keyword, single dataset**: Only verified Marvin; multi-keyword architecture is stated as future work by the paper.
4. **Single acoustic condition**: Speech Commands consists of near-field recordings on mobile devices, with no noise, far-field, or accent variables. Robustness evaluation is stated as future work by the paper.
5. **No architectural horizontal comparison**: No comparison with same-scale KWS architectures like DS-CNN, BC-ResNet, etc. It is impossible to judge whether this two-layer CNN is a good choice under the same parameter budget.
6. **Power consumption not measured**: The core selling point of NPU is energy efficiency. The paper only gives speed. For battery-powered devices, the energy per inference (µJ/inference) is more critical than latency. This may be the biggest gap in the entire paper.
7. **Thin training details**: 10 epochs, fixed learning rate 0.001, no convergence curves; two routes for QAT (train from scratch vs. fine-tune) are described but no comparative data is given.

### Number Consistency Issues in the Paper (Notes for Citation)

Three number mismatches were found during deep reading. The main conclusions are not affected, but citations should rely on the tables:

1. **Two口径 for i5 float latency**: The text in Section III writes 58.67 ms, while Table VI writes 50.67 ms, a difference of 8 ms. The paper does not explain this.
2. **"98.3% size reduction" cannot be recalculated**: The text claims the NPU version achieves a 98.3% model size reduction relative to the CPU version. However, calculated from Table VI, 30,576 B relative to float's 383,674 B is a decrease of 92.03%, and relative to int8's 35,744 B is a decrease of 14.46%. Neither matches 98.3%. The 90.68% (float→int8) used in the abstract and conclusion is the number consistent with Table VI.
3. **Title typo**: MCUX947 should be MCXN947 (the entire text uses MCXN947).

By the way, "30.58 KB" in the abstract is consistent with 30,576 B when converted using decimal KB, with no contradiction.

### Future Directions

The paper states three directions: power characteristic characterization, multi-keyword architecture, and robustness evaluation under varying acoustic conditions. The author supplements four: decision threshold calibration and multi-window voting post-processing (directly addressing the ×4.83 false alarm problem), PTQ comparative experiments to quantify the real contribution of QAT, an evaluation framework like FPR@Recall, and VAD cascading (first low-power voice activity detection, then KWS, to further reduce always-on power consumption).

## Transferable Engineering Judgments

1. **NPU benefits must be measured by "same chip, same model, CPU vs. NPU."** The anchor given by this paper is 59 times. Translated into product language: running this model on the MCU CPU occupies 228.2 ms per 1-second window (about 22.8% duty cycle, inferred), while on the NPU it is 3.847 ms (about 0.38%). The saved CPU budget can be left for the wireless protocol stack and application logic—the value of the NPU is not just "faster," but "giving the CPU back to you."
2. **Porting is not recompiling; it is rewriting the computational graph.** BN is folded into multiply-add, GAP is renamed Mean, sigmoid is mapped to Logistic. Every rewrite is determined by the toolchain, not by you. Operator selection during the architecture design phase must follow the target NPU's operator whitelist. Any custom operator means falling back to the CPU and tearing the pipeline. Table III in this paper is a realistic template for evaluating such rewrites.
3. **Quantization acceptance looks at the confusion matrix, not accuracy.** In this example, accuracy only drops by 2.08 percentage points, but false alarms increase by nearly five times, and the overall accuracy is lower than the "never wake up" baseline. In an imbalanced task with a 1:55 ratio, accuracy can even point in the wrong direction. The acceptance metric for any wake-word project before launch should be "false alarm rate at target recall," and thresholds must be recalibrated online.

## Terminology Quick Reference Table

| Term | Plain Language Explanation |
|:---|:---|
| KWS (Keyword Spotting) | Staring at one or several wake words in a continuous audio stream, letting all other sounds pass |
| MFCC | Mel-frequency cepstral coefficients, compressing 1 second of audio into a 98×20 "auditory feature map" |
| Hamming Window | A cosine-gradient window added to each frame to suppress spectral leakage caused by truncation |
| DCT | Discrete Cosine Transform, decorrelating Mel filter bank energy and compressing it into low-order coefficients |
| HTK Convention | A set of parameter standards for MFCC implementation (filter shapes, log base, etc.) to ensure reproducibility |
| QAT (Quantization Aware Training) | Simulating 8-bit quantization errors in forward propagation during training, allowing weights to adapt to fixed-point |
| PTQ (Post-Training Quantization) | Directly truncating float weights to int8 after training. Saves training cost but small models are prone to accuracy loss |
| NPU (eIQ Neutron) | A coprocessor in NXP MCUs specifically for running int8 neural networks. The operator set is limited |
| BN Folding | Merging Batch Normalization into per-channel multiply-add during inference. No BN operator remains in the computational graph |
| GAP (Global Average Pooling) | Averaging the entire feature map by channel. Spatial dimensions are compressed to 1, causing a sharp drop in parameters |
| TFLite | TensorFlow's edge-side model format and inference runtime |
| eIQ Toolkit | NXP's model conversion toolchain, converting TFLite models into NPU-executable formats |
| Cortex-M33 | ARM low-power 32-bit CPU core. The CPU execution path and frontend computing carrier in this paper |
| Class Weight | Weighing minority classes during training to counterbalance the 1:55 imbalance of positive and negative samples |
| Confusion Matrix / FPR / Recall | Answering "where the errors are," "proportion of negative sample false triggers," and "proportion of positive sample misses" respectively |
| Speech Commands | Google's 35-word command dataset, 105,829 one-second mobile recordings |
| Duty Cycle | The proportion of time occupied by a specific task within a real-time budget, measuring always-on feasibility |

---

**Final Note**: The value of this paper lies not in the algorithm—two layers of CNN plus QAT contain nothing new—but in unfolding the matter of "NPU deployment," which is glossed over in one sentence in vendor promotions, into a six-step pipeline, three structural comparison tables, and a set of recalculable performance data. For engineers who want to move their own models onto microcontrollers with integrated NPUs, it is one of the few "full-process with numbers" roadmaps available. And the false alarm drift hidden in its confusion matrix, masked by accuracy, reminds all wake-word developers: if you choose the wrong acceptance metrics, quantization will harm your product in ways you cannot see.
