# Efficient Dynamic Filter for Robust and Low Computational Feature Extraction

- **Authors/Affiliations**: Donghyeon Kim, Jeong-gi Kwak, Hanseok Ko (Korea University, Korea); Other authors of the early version include Gwantae Kim, Bokyeung Lee, David K. Han (Drexel University)
- **Date**: 2022 (Accepted by SLT 2022)
- **Link**: https://arxiv.org/abs/2205.01304
- **Keywords**: dynamic filter, keyword spotting, speaker verification, noise robustness, chunk separated convolution, attention pooling, instance-level processing

## Problem Statement

### Problem Background and Domain Pain Points
In real-world speech interaction scenarios, Keyword Spotting (KWS) and Speaker Verification (SV) systems must contend with various unpredictable environmental noises. The diversity of these noise sources manifests across multiple dimensions: (1) Spectral distribution—from narrowband noise (e.g., 60Hz power line hum) to broadband noise (e.g., air conditioner airflow); (2) Temporal patterns—from stationary noise (e.g., fans, engines) to non-stationary noise (e.g., door closing, keyboard typing, intermittent speech from other speakers); (3) Intensity range—from slight background noise (SNR > 20dB) to severe noise masking (SNR < -5dB). It is impossible to enumerate all noise conditions during the training phase; therefore, models must possess the ability to generalize to unseen noise.

Traditional fixed filters (such as convolutional kernels in CNNs) lack adaptive capabilities for noise types not encountered during training because their parameters remain fixed after training. This leads to "train-test domain mismatch"—the model performs well under noise conditions covered by the training set but suffers a sharp decline in performance under new noise types.

### Specific Deficiencies of Existing Methods
- **Fundamental Limitations of Static Filters**: The convolutional kernels of standard CNNs have fixed parameters after training. Although data augmentation (e.g., superimposing various noise types) can expand the noise coverage of the training set, the number of augmented noise types is limited by storage and computational costs. More importantly, augmentation strategies are inherently "reactive"—they only simulate known noises during training and cannot "proactively" adapt to new noise types encountered during inference.
- **Feature Aggregation Bottlenecks in Existing Dynamic Filters**: The Instance-level Dynamic Filter (IDF) framework dynamically generates filter weights for each input sample, endowing the model with the potential to handle unseen noises. The generation process of IDF is: extract a conditioning embedding vector $e = g(x)$ from the input $x$ (conditioning embedding), and then use a generation network to map $e$ to filter weights $\theta = h(e)$. The bottleneck of existing IDF methods lies in the design of the conditioning embedding extraction function $g(\cdot)$: most methods use simple Global Average Pooling (GAP), compressing the entire $T \times F$ time-frequency map into an $F$-dimensional vector (the mean of each frequency bin). This operation loses critical information in two dimensions:
  - **Loss of Temporal Structure**: GAP assigns equal weight $1/T$ to all time frames, failing to distinguish between frames containing keyword speech and those containing pure noise. In noise-robust feature extraction, the informational value of these two types of frames is vastly different.
  - **Loss of Local Patterns in the Frequency Dimension**: GAP compresses each frequency channel independently into a mean, losing the energy contrast patterns between adjacent frequency channels (such as the frequency location and width of formants—key features for distinguishing different keywords).
- **Tension between Computational Efficiency and Expressiveness**: More complex feature extraction methods (such as self-attention mechanisms, bidirectional LSTMs) can capture richer time-frequency structural information but incur excessive computational overhead (self-attention has $O(T^2)$ complexity), making them unsuitable for resource-constrained KWS deployment.

### Key Challenges Addressed by This Paper
To replace simple global average pooling without significantly increasing computational costs (additional parameters <10K, additional computations <2M FLOPs), enabling the dynamic filter to extract more expressive conditioning embedding vectors from the input time-frequency features (preserving time-frequency structural information, distinguishing speech frames from noise frames), thereby enhancing robustness to unseen noises. The method's generality is validated on both KWS and SV tasks.

## Methodology

### Overall Architecture Design and Design Motivation
This paper improves upon the authors' previously proposed Instance-level Dynamic Filter (IDF) framework. The core idea of IDF is: for each input sample $x$, generate a conditioning embedding vector $e$ through a "conditioning network" $g(\cdot)$, then map $e$ to dynamic filter weights $\theta$ using a "generation network" $h(\cdot)$, and finally filter the input using the dynamic weights. This paper focuses on improving the embedding extraction part within the conditioning network $g(\cdot)$—the key link determining the quality of the dynamic filter.

### Mathematical Principles of the Core Algorithm

**Core Technology 1: Chunk Separated Convolution (CS-Conv)**

The design of CS-Conv is inspired by the successful application of Dual-Path RNN (Luo & Yi, 2019) in speech separation. Dual-Path RNN demonstrated that "chunk processing + dual-path modeling" is an efficient and effective method for time-frequency structure modeling—segmenting long time-frequency sequences into short chunks and modeling within chunks and between chunks separately.

**Specific Implementation Steps**:

1. **Chunking Operation**: The input time-frequency feature map $X \in \mathbb{R}^{T \times F}$ is split along the time dimension into $K$ non-overlapping chunks (chunks), each containing $P$ time frames: $X = [X_1, X_2, ..., X_K]$, where $X_k \in \mathbb{R}^{P \times F}$.

2. **Intra-chunk Path**: Within each chunk $X_k$, perform 1D depthwise separable convolution along the time dimension. This step captures local temporal dynamic patterns—such as short-term fluctuation features of noise (periodic modulation of fans, short-term pulses of keyboards). The receptive field of the intra-chunk convolution is $P$ frames (typically 4-8 frames, approx. 40-80ms), sufficient to capture the micro-temporal structure of noise.

   $$\hat{X}_k^{\text{intra}} = \text{DWSConv}_1(X_k; W_{\text{intra}})$$

3. **Inter-chunk Path**: Perform 1D depthwise separable convolution between chunks, operating along the "chunk index" dimension. This step captures global temporal trends—such as gradual changes in noise intensity (the process from quiet to noisy) and switching of noise types (from stationary noise to intermittent noise).

   $$\hat{X}^{\text{inter}} = \text{DWSConv}_2([\hat{X}_1^{\text{intra}}, ..., \hat{X}_K^{\text{intra}}]; W_{\text{inter}})$$

4. **Parameter Efficiency of Separable Convolution**: Standard 1D convolution $(C_{in}, C_{out}, k)$ requires $C_{in} \times C_{out} \times k$ parameters, while separable convolution decomposes it into depthwise ($C_{in} \times 1 \times k$) and pointwise ($C_{in} \times C_{out} \times 1$), reducing the total parameter count to $C_{in} \times k + C_{in} \times C_{out}$, a reduction of approximately $k$ times.

**Comparison with GAP**: GAP outputs an $F$-dimensional vector (mean of each frequency bin), whereas CS-Conv outputs a feature map of dimension $K' \times F$ ($K'$ is the compressed number of chunks), preserving structural information in both time and frequency. The difference in information volume is orders of magnitude—GAP compresses $T \times F$ information into $F$, while CS-Conv retains it as $K' \times F$ (where $K' \ll T$ but is still much greater than 1).

**Core Technology 2: Dynamic Attention Pooling (DAP)**

Although CS-Conv preserves more time-frequency structural information, the importance of frames along the time dimension in its output varies—frames containing keyword speech should receive higher weights than pure noise frames. DAP achieves this selective aggregation along the time dimension through an attention mechanism.

**Mathematical Formulas**:

1. Apply a lightweight 1D convolutional layer to the output $H \in \mathbb{R}^{K' \times F}$ of CS-Conv to generate attention scores for each time frame:

$$s_k = \mathbf{w}^T \cdot H_k + b, \quad k = 1, 2, ..., K'$$

where $\mathbf{w} \in \mathbb{R}^F$ and $b \in \mathbb{R}$ are trainable parameters.

2. Normalize to obtain attention weights via softmax:

$$\alpha_k = \frac{\exp(s_k)}{\sum_{j=1}^{K'} \exp(s_j)}$$

3. Perform weighted summation of time frames using attention weights to obtain the final conditioning embedding vector:

$$e = \sum_{k=1}^{K'} \alpha_k \cdot H_k$$

**Adaptive Nature of DAP**: The attention weights $\alpha_k$ are input-dependent (since $H_k$ is determined by the input), resulting in different attention distributions for different inputs. For example:
- In the KWS task, when the input contains keyword speech, DAP automatically assigns higher weights to speech frames (because these frames have richer spectral patterns, providing more guidance for dynamic filter generation).
- In the SV task, DAP assigns higher weights to frames where speaker features are most prominent (e.g., steady-state vowel segments with clear formants).
- In pure noise inputs, DAP distributes weights relatively uniformly (as there are no particularly prominent frames).

**Comparison with GAP**: GAP assigns equal weight $1/K'$ to all time frames, whereas DAP dynamically adjusts weights. This difference is particularly evident in KWS—when noise frames and speech frames coexist, the uniform weights of GAP "dilute" the information of speech frames with a large number of noise frames, while DAP can "focus" on the frames with the most information.

### Loss Function and Training Strategy

For the KWS task: $L_{KWS} = L_{CE}(y, \hat{y})$ (standard cross-entropy loss)

For the SV task: $L_{SV} = L_{CE}(y, \hat{y}) + \lambda \cdot L_{AM}(\theta_m)$, where $L_{AM}$ is the Angular Margin Loss (e.g., ArcFace), enhancing the discriminability of speaker embeddings. Angular margin loss improves verification accuracy by increasing the spacing between different speakers in the angular space.

The dynamic filter weights $\theta = h(e)$ are jointly optimized with the entire network via backpropagation. The key point is: the quality of the conditioning embedding vector $e$ directly determines the quality of the dynamic filter $\theta$—CS-Conv and DAP improve the quality of $e$, thereby indirectly but significantly enhancing the noise robustness of the entire IDF framework.

### Technical Differences from Existing Methods
- **Compared to Original IDF (GAP)**: CS-Conv + DAP replaces GAP, increasing information volume from $O(F)$ to $O(K' \times F)$, significantly enhancing the expressiveness of the embedding vector without adding significant computational load.
- **Compared to Attention Mechanisms (e.g., Self-Attention)**: The computational complexity of CS-Conv + DAP is $O(K' \times F)$ (linear complexity), far lower than the $O(K'^2 \times F)$ of self-attention. This is because CS-Conv uses fixed-size convolutional kernels (local operations), and DAP calculates attention only along the time dimension (1D rather than 2D).
- **Compared to Data Augmentation**: This method improves noise robustness at the model architecture level, which is orthogonal and complementary to data augmentation—further architectural gains can be achieved on top of augmented training.

## Main Contributions

1. **CS-Conv for Dynamic Filter Embedding Extraction**: For the first time, the chunked separable convolution concept from Dual-Path RNN is migrated from speech separation tasks to the conditional embedding extraction of dynamic filters. The innovation of CS-Conv lies in applying the "chunking + dual-path modeling" paradigm to a new problem: not separating speech from mixed signals, but extracting noise-robust embedding vectors from noise-corrupted features. CS-Conv preserves local and global pattern information along the time dimension, providing a more precise input feature description for dynamic filter weight generation.

2. **DAP for Selective Aggregation along the Time Dimension**: Dynamic Attention Pooling achieves "selective aggregation" of "focusing on key frames and ignoring noise frames" through input-dependent attention weights. The design philosophy of DAP is "information selectivity"—not all time frames are equally important for the generation of dynamic filters; the model should learn to identify and focus on the most important frames. This design is functionally superior to uniform pooling while adding only a lightweight convolutional layer computationally.

3. **Cross-Task Noise Robustness Validation**: Improvements are demonstrated on both KWS and SV tasks, proving that the combination of CS-Conv + DAP has task-agnostic noise robustness enhancement effects. This cross-task generalization indicates that the method's improvement is not overfitting to a specific task but is a fundamental improvement to the general step of "dynamic filter embedding extraction."

4. **Extremely Low Additional Computational Overhead**: CS-Conv introduces approximately 5-10K additional parameters, and DAP introduces several hundred parameters, with a total additional computation of about 1-2M FLOPs. Relative to the entire model (typically hundreds of K parameters), the additional overhead is less than 5%, yet it brings a 2-5 percentage point increase in accuracy, resulting in an extremely high efficiency ratio (performance gain / additional overhead).

## Experimental Results

### Datasets Used and Their Scale
- **KWS**: Google Speech Commands dataset, standard 12-class classification. Noise-augmented testing uses the MUSAN noise library (approx. 900 noise types) and the DEMAND noise library (16 environmental noises, including real-environment recordings from offices, restaurants, traffic, etc.), tested under SNR conditions ranging from 20dB to -5dB.
- **SV**: VoxCeleb1 dataset (approx. 1,251 speakers, approx. 153,000 speech segments, real-environment recordings extracted from YouTube videos) and VoxCeleb2 dataset (approx. 6,112 speakers, approx. 1,128,000 speech segments). The testing protocol uses VoxCeleb1-H (containing difficult negative pairs with the same speaker but different videos) and VoxCeleb1-E (containing an extended test set with cross-domain scenarios).

### Definition and Rationale for Evaluation Metrics
- **KWS**: Classification accuracy (%), reported separately for clean and different SNR conditions. Reporting by SNR level is chosen to evaluate the gradual characteristics of noise robustness.
- **SV**: Equal Error Rate (EER, %, the error rate at the threshold where the false positive rate equals the false negative rate) and minimum Detection Cost Function (minDCF, measuring detection cost at specific operating points). EER is the standard metric in the SV field, unaffected by threshold selection.

### Detailed Comparison with Baseline and SOTA Methods

**KWS Task**:
- Accuracy difference between GAP-IDF (baseline) and CS-Conv+DAP-IDF (this paper) under unseen noise conditions:
  - Clean condition: approx. 93.5% vs. approx. 94.0% (+0.5%)
  - 10dB SNR: approx. 88.2% vs. approx. 91.0% (+2.8%)
  - 0dB SNR: approx. 79.5% vs. approx. 84.0% (+4.5%)
  - -5dB SNR: approx. 72.0% vs. approx. 77.0% (+5.0%)
- Key Finding: The magnitude of improvement increases as SNR decreases (from +0.5% in clean conditions to +5.0% at -5dB), indicating that CS-Conv+DAP is more valuable under more severe noise conditions—because under high noise conditions, the information loss of GAP causes greater damage to performance, while CS-Conv+DAP effectively compensates for this defect by preserving more time-frequency structural information.

**SV Task**:
- On VoxCeleb1-E: EER reduced by 0.5-1.0 percentage points.
- On VoxCeleb1-H: EER reduced by approx. 1.0-1.5 percentage points (the time-frequency modeling advantage of CS-Conv is more pronounced in cross-domain scenarios, as cross-domain recordings typically have more complex noise conditions).

**Computational Overhead Analysis**:
- Additional parameters introduced by CS-Conv + DAP: approx. 5-10K.
- Additional computations introduced by CS-Conv + DAP: approx. 1-2M FLOPs.
- Relative to the entire model (typically 200-500K parameters): Additional parameters account for approx. 2-5%.
- Cost-effectiveness: Approximately 0.5-1.0% accuracy gain per 1% additional parameter.

### Findings from Ablation Studies

**Independent Contributions of Each Component**:
- CS-Conv used alone (DAP replaced by uniform pooling): Accuracy increased by approx. 1-2 percentage points.
- DAP used alone (CS-Conv replaced by GAP): Accuracy increased by approx. 1-2 percentage points.
- Combination of CS-Conv + DAP: Accuracy increased by approx. 3-5 percentage points.

**Evidence that the Two are Complementary Rather Than Redundant**: The combined effect (3-5%) is greater than the sum of their individual effects (1-2% + 1-2% = 2-4%), indicating that the two are functionally complementary—CS-Conv provides better time-frequency structural features, and DAP selects the most critical information from them, with the two synergistically producing excess returns.

**Impact of Chunk Size $P$**:
- $P=2$: Chunks are too small; the intra-chunk path covers only 20ms, failing to effectively capture short-term noise patterns; the inter-chunk path needs to process too many chunks, increasing computational overhead.
- $P=4-8$ (Optimal): The intra-chunk path covers 40-80ms, sufficient to capture short-term patterns of most noises; the inter-chunk path processes approx. 5-10 chunks, making it computationally efficient.
- $P=16$: Chunks are too large; the intra-chunk path loses fine-grained temporal resolution, and the inter-chunk path processes only approx. 2-3 chunks, resulting in insufficient global modeling capability.

**DAP vs. Other Pooling Strategies**:
- DAP (Attention Pooling) > Max Pooling > Uniform Pooling (GAP).
- Max pooling selects the maximum value for each frequency channel; while it highlights peaks, it loses distribution information.
- DAP encodes information about "which frames are most important" in the attention weights, providing the richest aggregation results.

## Limitations and Future Work

### Technical Limitations of the Method
- **Limitations of the Scope of Improvement**: CS-Conv and DAP only improve the "aggregation" part of the embedding extraction in the dynamic filter (from the feature map output by $g(x)$ to the conditioning embedding vector $e$), without involving: (1) architectural improvements to the dynamic filter weight generation network $h(\cdot)$; (2) the method of applying dynamic filters to input features (currently simple convolutional filtering). Overall performance is still limited by the upper bound of the IDF framework's expressiveness—if the generation network $h(\cdot)$ is not powerful enough, even if the conditioning embedding vector $e$ is very precise, the generated filter weights may not be effective.
- **Hyperparameter Sensitivity of Chunk Size**: The chunk size $P$ of CS-Conv needs to be manually adjusted according to the time-frequency resolution of the input features. Different time-frequency representations (e.g., MFCC with frame lengths of 25ms vs. 10ms, or mel-filterbanks with different frequency resolutions) may require different chunk sizes. There is a lack of mechanisms to automatically determine $P$.
- **Information Loss at Chunk Boundaries**: Segmenting the time sequence into non-overlapping chunks creates information breaks at chunk boundaries—temporal patterns spanning chunk boundaries (such as gradual transitions from noise to speech) may be split into two different chunks, causing the intra-chunk path to fail to capture them completely. Introducing overlapping chunks can alleviate this but increases computational load.

### Deficiencies in Experimental Design
- **Insufficient Evaluation in Real Far-Field Scenarios**: Although noise augmentation (superimposing MUSAN noise) was used for testing, this still differs from real far-field recordings (which contain composite degradation factors such as reverberation, multi-speaker interference, and device self-noise). In particular, far-field reverberation introduces long-tailed exponential decay patterns along the time dimension, which may exceed the receptive field of the intra-chunk path of CS-Conv.
- **Incomplete Comparison with State-of-the-Art KWS Systems**: Direct comparisons were not made with some strong baselines from the same period in 2022 (such as BC-ResNet, KWT, ConvMixer+) under the same noise conditions. Comparisons with these architecture-level improvements would help evaluate which direction, "dynamic filters vs. architecture improvements," is more effective for noise robustness.
- **Inference Latency Not Reported**: It has not been verified whether the additional computations of CS-Conv and DAP become bottlenecks in actual deployment.

### Possible Directions for Future Improvement
- **End-to-End Dynamic Filter Optimization**: Jointly optimize CS-Conv + DAP with the generation network $h(\cdot)$ of the dynamic filter and the application process in an end-to-end manner. The current mapping between the conditioning embedding vector $e$ and filter weights $\theta = h(e)$ may not be optimal—end-to-end optimization allows $e$ to better adapt to the input requirements of $h$.
- **Adaptive Chunk Size**: Dynamically adjust the chunk size of CS-Conv based on the time-frequency characteristics of the input. For example, use large chunks for stationary noise (because the statistical properties of stationary noise are stable over long time scales) and small chunks for non-stationary noise (to capture rapid changes).
- **Multi-Scale Modeling**: Combine chunks of different sizes for multi-scale time-frequency modeling. Small chunks capture fine-grained time structures, large chunks capture global time trends, and multi-scale feature fusion can provide a more comprehensive time-frequency description.
- **Joint Design with Front-End Signal Processing**: Jointly optimize the dynamic filter of CS-Conv + DAP with traditional beamforming or noise reduction front-ends, achieving end-to-end design of signal processing and deep learning.
- **Implications for the KWS Field**: This paper verifies the key role of "dynamic feature aggregation" in noise robustness—simple pooling operations (such as GAP) are one of the bottlenecks to noise robustness. This insight applies not only to the dynamic filter framework but can also be generalized to other KWS architecture designs that need to extract discriminative features from noisy inputs (such as query/key projections in attention mechanisms, feature aggregation in classification heads, etc.).
