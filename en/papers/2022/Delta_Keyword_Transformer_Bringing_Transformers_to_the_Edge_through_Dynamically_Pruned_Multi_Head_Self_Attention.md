# Delta Keyword Transformer: Bringing Transformers to the Edge through Dynamically Pruned Multi-Head Self-Attention

- **Authors/Affiliations**: Zuzana Jelcicova (Technical University of Denmark; Demant A/S), Marian Verhelst (MICAS, KU Leuven, Belgium)
- **Date**: March 2022 (tinyML Research Symposium 2022)
- **Link**: https://arxiv.org/abs/2204.03479
- **Keywords**: Transformers, delta computations, pruning, compression, keyword spotting, edge devices, multi-head self-attention

## Problem Statement

### Problem Background and Domain Pain Points
The Transformer architecture has demonstrated excellent performance in keyword spotting (KWS) tasks through the Multi-Head Self-Attention (MHSA) mechanism (e.g., KWT series models achieving 98.4% accuracy on GSC-12). MHSA can model relationships between any two tokens in the input sequence, thereby capturing global acoustic dependencies—this is crucial for distinguishing keywords that are similar in local features but differ in global temporal structure (such as "yes" and "yet").

However, the computational complexity of MHSA grows quadratically with sequence length: for a sequence of length $N$ and feature dimension $d$, standard MHSA requires $O(N^2 \cdot d)$ multiply-accumulate (MAC) operations and $O(N^2)$ storage space. In KWS, the typical input length is $N=40$ (40 time frames), and $d=64$. A single layer of MHSA requires approximately $40^2 \times 64 = 102,400$ MAC operations. After stacking 12 Transformer layers, MHSA layers account for 60-80% of the total inference computation, becoming the primary bottleneck for deploying Transformers on resource-constrained edge devices.

### Specific Shortcomings of Existing Methods
- **Pruning methods requiring retraining**: Structured pruning (e.g., removing entire attention heads, layer pruning) or unstructured pruning (e.g., weight magnitude pruning) typically requires fine-tuning or retraining after pruning to recover performance. This means pruning is an offline, time-consuming process that cannot dynamically adjust during inference based on the input—whereas, in reality, different input samples have different "computational difficulties" (e.g., speech in quiet environments is more "regular" than in noisy environments and may have more redundant computation).
- **Information loss from coarse-grained pruning**: Existing dynamic pruning methods usually operate at the token level (skipping the computation of entire tokens, such as DynamicViT, SpAtten) or at the attention head level (removing the computation of entire heads, such as Head Pruning). The problem with such coarse-grained operations is the "all-or-nothing" nature—either all information of a token/head is fully retained, or it is completely discarded. In KWS, a time frame may contain both discriminative information and noise; coarse-grained pruning cannot "retain useful parts and discard redundant parts."
- **Sparse computation requiring dedicated hardware**: The irregular sparse patterns produced by unstructured pruning (i.e., randomly distributed zeros in the weight matrix) require specialized sparse matrix operation hardware (such as GPU sparse tensor cores or custom sparse accelerators) to achieve actual acceleration. On general-purpose ARM Cortex-M series processors, implementing sparse convolution introduces a large number of conditional branches (if-else) and indirect indexing operations. The actual acceleration effect is far lower than the theoretical value, and sometimes even slower than dense computation (because conditional branches disrupt the pipeline).
- **Underutilization of temporal redundancy in speech signals**: A fundamental characteristic of speech signals is "short-term stationarity"—within a frame shift interval of about 10ms, the spectral features of adjacent frames are highly similar (correlation coefficient usually >0.95). This temporal redundancy implies a large amount of repetition in the MHSA computation of adjacent frames, but existing pruning methods do not systematically exploit this characteristic.

### Key Challenges Addressed by This Paper
Design a dynamic compression method for MHSA that requires no retraining, operates at a fine granularity (feature dimension level rather than token level), and relies only on general-purpose hardware support. The method aims to leverage the temporal stability of speech signals to enable pre-trained Transformer KWS models to run efficiently on edge devices.

## Methodology

### Overall Architecture Design and Design Motivation
The core observation of the Delta Keyword Transformer (Delta-KWT) is rooted in the classic concept of delta features from the field of signal processing. In traditional speech recognition, delta cepstrum (first-order differential cepstrum) and delta-delta cepstrum (second-order differential cepstrum) are widely used to capture the dynamic changes of speech. The implicit assumption of this concept is that the "change" (delta) of the speech signal carries more discriminative information than the "absolute value," and the "invariant parts" between adjacent frames are redundant.

Delta-KWT applies this idea to the inference process of Transformers: in MHSA computation, if the difference between the feature vectors of adjacent tokens is small (i.e., delta is small), then the MHSA computation results for the two should also be similar. Therefore, the computation with small variations can be skipped, and the result from the previous frame can be reused directly.

### Mathematical Principles of the Core Algorithm: Delta Threshold Pruning

Let the MHSA computation in the $l$-th layer of the Transformer follow the standard procedure:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $Q = XW_Q$, $K = XW_K$, $V = XW_V$, and $X$ is the input feature matrix.

**Delta Judgment Process**:

For the feature vector $x_t \in \mathbb{R}^d$ of the $t$-th token in the sequence, calculate its feature difference from the previous token:

$$\delta_t = |x_t - x_{t-1}| \in \mathbb{R}^d$$

This is an element-wise absolute difference operation. For each feature dimension $j$, a threshold judgment is performed:

$$\text{skip}_t[j] = \begin{cases} 1, & \text{if } \delta_t[j] < \tau \\ 0, & \text{if } \delta_t[j] \geq \tau \end{cases}$$

where $\tau$ is the preset Delta threshold.

**Feature Propagation Strategy**:

Feature dimensions marked as "redundant" ($\text{skip}_t[j] = 1$) directly reuse the computation results of the previous token:

$$x_t^{(l)}[j] = \begin{cases} x_t^{(l)}[j], & \text{if } \text{skip}_t[j] = 0 \\ x_{t-1}^{(l)}[j], & \text{if } \text{skip}_t[j] = 1 \end{cases}$$

This replacement is applied recursively in all sub-steps of MHSA:

1. **Q/K/V Projection**: In $Q_t = X_t W_Q$, the skipped dimensions are directly replaced by the corresponding dimensions of $Q_{t-1}$. This saves the computation of the projection matrix multiplication.
2. **Attention Score Calculation**: In $S_t = Q_t K^T / \sqrt{d_k}$, since some dimensions of $Q_t$ are replaced, the difference between $S_t$ and $S_{t-1}$ is reduced.
3. **Attention Weighted Sum**: In $\text{out}_t = \text{softmax}(S_t) V$, the computation volume is further reduced.

**Cumulative Effect of the Delta Pipeline**: Since the replacement is recursive (the redundant dimensions of the $t$-th frame reuse the $t-1$-th frame, and the $t-1$-th frame may reuse the $t-2$-th frame's results), within a continuously stable time period (such as the steady-state part of a vowel), only the first frame requires full computation, and most dimensions of subsequent frames can be directly "passed through."

### Key Design: No Training Required

The key innovation of Delta-KWT is that it is a pure inference-time optimization that does not modify any weights or architecture of the model:
- **No model modification**: Does not change architectural parameters such as the number of Transformer layers, number of attention heads, or hidden dimensions.
- **No calibration dataset required**: Does not require representative data to statistically analyze the importance distribution of weights (as required by magnitude pruning).
- **No fine-tuning or retraining required**: Does not require any training process to recover performance loss.
- **Plug-and-play**: Can directly load any pre-trained Transformer model (such as open-source KWT models on Hugging Face) and apply Delta pruning.
- **Dynamically adjustable**: The threshold $\tau$ is set during inference and can be dynamically adjusted based on the device's real-time resource status (battery level, CPU load)—using a small threshold for high resources (high accuracy) and a large threshold for low resources (high efficiency).

### Quantitative Analysis of Fine-Grained Advantages

The core advantage of the Delta method lies in its "feature dimension-level" granularity:
- For a typical KWS input (40 time frames, 64-dimensional features per frame), typically 50-80% of the feature dimensions in each token can be skipped (because the temporal continuity of speech signals leads to high similarity between adjacent frame feature vectors).
- Specifically, for the "steady-state segment" of speech (such as the sustained part of a vowel), the difference between adjacent frames mainly comes from random spectral fluctuations, and the skip ratio can reach 80-90%; for the "transition segment" (such as the transition from consonant to vowel), the skip ratio decreases to 30-50%.
- Compared to token-level pruning (skipping or retaining an entire frame), the Delta method achieves "selective computation" within the same frame—stable dimensions are skipped, and changing dimensions are retained.

### Hardware-Friendly Implementation

- Delta judgment requires only simple comparison operations ($|\delta_t[j]| < \tau$), which are among the most efficient instructions on all general-purpose processors (requiring only 1 clock cycle on ARM Cortex-M).
- No sparse matrix operation hardware is needed—the skipped feature dimensions are directly replaced by the values of the previous frame, and the output remains a dense tensor (unlike the irregular sparse matrices produced by sparse pruning).
- Memory access patterns remain regular (frames are processed sequentially), which is beneficial for cache utilization.
- The additional storage overhead is minimal: only the feature vector of the previous frame ($d$ floating-point values) and the delta judgment results ($d$ boolean values) need to be saved.

### Experimental Setup
- Evaluated on the pre-trained KWT-1 model (Google Speech Commands dataset, 12-class classification).
- KWT-1 contains 12 Transformer encoder layers, with an original accuracy of 98.4%.
- Each Transformer layer contains MHSA (8 attention heads, 8 dimensions per head) and FFN (Feed-Forward Network).

## Main Contributions

1. **Training-free fine-grained dynamic pruning of MHSA**: Proposes a dynamic pruning method at the feature dimension level for the first time, leveraging the temporal stability of speech signals to achieve efficient inference. This method cleverly combines the classic signal processing concept of "delta computation" with modern Transformer architectures—the physical intuition of delta cepstrum (that discriminative information in speech lies in "changes" rather than "absolute values") finds a new application scenario in the attention computation of Transformers.

2. **Plug-and-play post-processing of models**: The method can be directly applied to any pre-trained Transformer model without modifying the model architecture or retraining. This has extremely high practical value in industrial deployment scenarios: developers can directly use pre-trained KWT models on Hugging Face, flexibly trade off between accuracy and efficiency by adjusting a single hyperparameter (Delta threshold), reducing deployment time from hours to zero.

3. **Extreme operation reduction**: Maintains 98.4% original accuracy with 80% operation reduction (4.2x speedup), and loses only 1-4% accuracy with 87-94% operation reduction (7.5-16x speedup). These numbers far exceed other dynamic pruning methods of the same period.

4. **Hardware-friendly design based purely on comparison operations**: The entire pruning decision process requires only comparison operations, which can be efficiently implemented on any general-purpose processor. This enables the deployment of Transformer KWS on edge devices without adding any dedicated hardware.

## Experimental Results

### Datasets Used and Their Scale
- **Google Speech Commands V1-12**: 12 command word categories, approximately 65,000 1-second speech clips. Standard training/validation/test split.

### Definition and Rationale for Evaluation Metrics
- **Accuracy (%)**: Classification accuracy, used to evaluate the impact of Delta pruning on model precision.
- **Operation Reduction Ratio (%)**: The percentage of multiply-accumulate operations skipped out of the total operations.
- **Theoretical Speedup Ratio**: Estimated based on the operation reduction ratio (assuming skipped operations do not incur computational overhead).

### Detailed Comparison with Baseline Methods and SOTA

**Core Performance Data (on KWT-1)**:
| Operation Reduction Ratio | Accuracy (%) | Theoretical Speedup | Accuracy Loss (%) |
|:---:|:---:|:---:|:---:|
| 0% (Original) | 98.4 | 1.0x | 0 |
| 80% | ~98.4 | 4.2x | <0.1 |
| 87% | ~97.5 | 7.5x | ~0.9 |
| 90% | ~96.5 | 10x | ~1.9 |
| 94% | ~95.0 | 16x | ~3.4 |

Key Finding: At 80% operation reduction (4.2x speedup), the accuracy is almost lossless (loss <0.1%). This means that 80% of the MHSA computation in speech signals is indeed redundant—this redundancy stems from the temporal continuity of speech signals, not from redundancy in model design.

### Findings from Ablation Studies

**Threshold Sensitivity Analysis**:
Within the range of Delta threshold $\tau \in [0.01, 0.1]$, the accuracy-operation trade-off curve shows a smooth decreasing relationship (no sudden jumps). This indicates that the method is robust to threshold selection—in practical deployment, developers do not need to precisely locate a single "optimal" threshold, but can choose a suitable trade-off point within a relatively large range.

**Pruning Distribution Across Different Layers**:
- **Lower Transformer layers (Layers 1-4, closer to input)**: Higher pruning ratio (approx. 85-90%), because lower-layer features capture low-level acoustic features (such as spectral energy distribution), which change slowly between adjacent frames.
- **Higher Transformer layers (Layers 9-12, closer to output)**: Lower pruning ratio (approx. 70-75%), because higher-level semantic features change more frequently (reflecting high-level semantic information of keywords).
- This finding is consistent with "hierarchical information processing" in linguistics: lower layers process slowly changing acoustic features, while higher layers process rapidly changing semantic features.

**Comparison with Token-Level Pruning**:
Under the same operation reduction target (e.g., 80%), Delta fine-grained pruning is 3-5 percentage points more accurate than token-level pruning. An 80% operation reduction in token-level pruning means completely skipping 32 tokens (out of 40 total tokens), which leads to severe information loss; whereas Delta pruning skips only 80% of feature dimensions in each token, retaining 20% of key changing information.

**Comparison with Static Pruning (Weight Magnitude Pruning)**:
The Delta method achieves higher accuracy at the same compression rate because it retains "key changes" (dimensions with large delta) rather than retaining randomly. Static pruning might prune certain small-weight but delta-sensitive key connections.

## Limitations and Future Work

### Technical Limitations of the Method
- **Dependence on temporal stability**: The effectiveness of Delta pruning essentially depends on the magnitude of feature differences between adjacent tokens in the input data. For highly dynamic speech signals (such as fast speech rate, or keywords with dense plosives like "stop", "go"), the differences between adjacent frames may be large, leading to a lower pruning ratio and reduced acceleration effect. In extreme cases (such as jump signals), the Delta method may not provide significant acceleration.
- **Manual adjustment of threshold**: Although the threshold-accuracy trade-off curve is smooth, selecting the optimal threshold for specific application scenarios (e.g., requiring accuracy not lower than 97%) still requires manual experimentation. There is a lack of automated threshold search mechanisms (such as automatically determining the threshold based on target accuracy constraints).
- **Only validated on KWT-1**: Experiments were conducted only on a single KWT-1 model, which is a smaller Transformer variant (12 layers, 8 heads, 64 dimensions). For larger Transformer models (such as KWT-3 with approx. 3M parameters, or AST-Base with approx. 85M parameters), the effectiveness of Delta pruning may vary due to differences in model capacity and redundancy levels.
- **Accumulation of Delta errors**: In recursive feature propagation, each frame's replacement introduces a small approximation error. As the sequence length increases, errors may accumulate across layers and frames, leading to increased deviation in the final output. The paper does not systematically analyze the long-term effects of this error accumulation.

### Shortcomings in Experimental Design
- **Speedup ratio based on operation count rather than actual measurement**: The reported 4.2x-16x speedup ratios in the paper are theoretical estimates based on the reduction in multiply-accumulate operations. The actual acceleration effect is limited by: (1) the overhead of the Delta judgment operation itself (although small, it is not negligible in high-speed inference); (2) the memory access overhead of feature replacement (requiring reading and writing the previous frame's cache); (3) the impact of conditional branches on the CPU pipeline. End-to-end inference latency was not measured on actual edge hardware (such as ARM Cortex-M4, RISC-V, etc.).
- **Lack of energy efficiency data**: No measurement data on actual power consumption or energy usage was reported. Although operation reduction theoretically lowers power consumption, the increase in comparison operations and memory access may partially offset this benefit.
- **Noise conditions not evaluated**: All experiments were conducted under clean conditions. Noise may increase the difference between adjacent frames (because noise fluctuates randomly), thereby reducing the pruning ratio and weakening the acceleration effect of the Delta method. This is an important unverified scenario.
- **Lack of systematic comparison with other dynamic inference methods**: Such as comparisons with Dynamic Depth (selecting which layers to execute based on input), Dynamic Width (selecting which channels to execute based on input), etc.

### Possible Directions for Future Improvement
- **Adaptive threshold mechanism**: Design an adaptive threshold strategy based on the statistical properties of current input features (such as the mean and variance of inter-frame differences), making pruning decisions more intelligent and robust. For example, use larger thresholds for steady-state speech segments (more pruning) and smaller thresholds for transition segments (retaining more information).
- **Combination with quantization**: Delta pruning and INT8/INT4 quantization are orthogonal optimization dimensions. Quantization reduces the computational cost of each operation, while Delta reduces the number of operations to be executed. The joint application of the two may bring superimpressed compression effects—preliminary estimates suggest a comprehensive speedup of 50-100x.
- **Learning-based Delta judgment**: Using a small neural network (rather than simple threshold comparison) to determine which feature dimensions can be skipped may more accurately identify truly redundant computations.
- **Extension to other sequence tasks**: The idea of Delta computation can be applied to other tasks using Transformers to process speech sequences, such as Automatic Speech Recognition (ASR) and Speech Translation. These tasks usually have longer sequence lengths (hundreds of frames), and the Delta method may offer greater acceleration potential.
- **Inspiration for the KWS field**: The Delta method reveals an important insight—the computational redundancy of KWS models during inference is "structured and predictable," and this structure stems from the physical characteristics of speech signals (temporal continuity). This insight can inspire more dynamic inference optimization methods based on "input characteristics" rather than "model structure."
