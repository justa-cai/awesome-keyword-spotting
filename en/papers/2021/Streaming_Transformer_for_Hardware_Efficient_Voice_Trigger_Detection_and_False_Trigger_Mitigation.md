# Streaming Transformer for Hardware Efficient Voice Trigger Detection and False Trigger Mitigation

- **Authors/Affiliations**: Vineet Garg, Wonil Chang, Siddharth Sigtia, Saurabh Adya, Pramod Simha, Pranay Dighe, Chandra Dhir - Apple
- **Date**: 2021.05
- **Link**: https://arxiv.org/abs/2105.06598
- **Keywords**: Streaming Transformer, Voice Trigger Detection, False Trigger Mitigation, Hardware Efficient, Causal Attention, Block Processing, Multi-Task Learning, CTC Loss, Edge Devices, Keyword Spotting

## Problem Statement

Intelligent voice assistants (such as Apple Siri, Amazon Alexa, and Google Assistant) have become the core interaction interface for AI devices like smartphones and smart speakers. A voice assistant query begins with a trigger phrase (e.g., "Hey Siri"), followed by the user's voice request. Currently, state-of-the-art on-device Voice Trigger Detection (VTD) systems typically employ a two-stage cascaded design:

**Stage 1**: A low-computation always-on model continuously processes the audio stream to identify candidate audio segments containing the trigger phrase. However, due to computational constraints, this stage is prone to **False Triggers**—misclassifying audio segments that are acoustically similar to the trigger phrase as trigger events.

**Stage 2**: A larger-capacity model serves as a secondary verifier to decide whether to actually activate the voice assistant.

Even so, two-stage VTD systems may still incorrectly activate the voice assistant due to acoustically similar audio segments. **False Trigger Mitigation (FTM)** systems utilize the audio context following the trigger (post-trigger audio) to determine if the user truly intends to interact with the voice assistant. Traditional FTM systems rely on lattices obtained from Automatic Speech Recognition (ASR) decoding to extract linguistic information. However, full ASR systems are computationally expensive and difficult to run efficiently on devices. Relying on server-side ASR raises privacy concerns, as accidental voice queries from users might be sent to the cloud.

The core problem this paper addresses is: **How to design a unified streaming model architecture that simultaneously handles both VTD and FTM tasks, using only acoustic features (without relying on ASR), achieving hardware-efficient inference on devices, while maintaining or improving the detection accuracy of two-stage systems?**

## Methodology

### Overall Architecture Design

The paper proposes a **joint streaming Transformer encoder architecture** containing three core components:

1. **TF Encoder Backbone (Blue Module)**: Extracts acoustic embedding representations, serving as a shared feature extractor for both VTD and FTM tasks.
2. **Phoneme Transcription Branch (Green Module)**: Used for the VTD task, trained by minimizing the CTC loss.
3. **Phrase Discrimination Branch (Yellow Module)**: Used for the FTM task, trained by minimizing the frame-level Cross-Entropy (CE) loss.

Additionally, an **autoregressive TF Decoder (Orange Module)** can be optionally added during training as a regularizer, but it is not used during inference.

### Baseline Architecture: TF Encoder

The baseline employs a non-streaming TF Encoder using a standard Transformer architecture, trained with CTC loss at the encoder side and Cross-Entropy loss at the decoder side. The Self-Attention (SA) layers of the encoder can view the entire input context at once; the paper refers to this as a **vanilla SA layer**. The trained TF Encoder is used to initialize the Multi-Task Learning (MTL) framework.

### Streaming TF Encoder: Core Innovations

The paper makes three key improvements to the baseline TF Encoder to enable streaming processing:

#### 2.1 Streaming Self-Attention (Streaming SA) Layer

The core improvement is replacing the vanilla SA layer with a streaming SA layer, adopting a **block processing protocol**. Specifically:

Let $Q = [q_1, q_2, ...]$, $K = [k_1, k_2, ...]$, $V = [v_1, v_2, ...]$ be the query, key, and value matrices of the vanilla SA layer, respectively. The streaming SA layer processes data in blocks:

- **Block Shift (S)**: Equals 50% of the block size $B$, i.e., $B = 2S$.
- **First Block**: $Q_1 = [q_1, ..., q_{2S}]$, $K_1 = [k_1, ..., k_{2S}]$, $V_1 = [v_1, ..., v_{2S}]$.
- **Subsequent $i$-th Block** ($i \ge 2$): $Q_i = [q_{(i-1)S+1}, ..., q_{iS}]$, $K_i = [k_{(i-2)S+1}, ..., k_{iS}]$, $V_i = [v_{(i-2)S+1}, ..., v_{iS}]$.

Key Design: The key and value matrices for subsequent blocks contain $S$ frames from the current block plus $S$ frames from the previous block (totaling $2S = B$ frames), while the query matrix contains only the $S$ frames from the current block. This means each block can "see" the current frame plus the context from the previous block, achieving **causal attention with limited context**.

During training, streaming behavior is simulated via **attention masks**. The mask allows each frame to attend only to itself and previous frames (causal constraint) while limiting the context window size (efficiency constraint).

#### 2.2 Frame-wise CE Loss for Phrase Discrimination Branch

The baseline method uses CTC loss for the phrase discrimination branch. However, the blank label mechanism of CTC introduces unnecessary ambiguity when processing post-trigger audio. The paper changes the loss function for the phrase discrimination branch to **frame-wise Cross-Entropy (CE) loss**:

- Independent binary classification for each frame: whether it contains the target trigger phrase.
- Removal of CTC's blank labels and alignment constraints.
- Frame-level labels are obtained via forced alignment.

This improvement allows the phrase discrimination branch to more precisely locate the temporal boundaries of the trigger phrase.

#### 2.3 UniLSTM Layer

A **unidirectional LSTM (uniLSTM)** layer is introduced in the phrase discrimination branch. Since the context window of the streaming SA layer is limited (only $B$ frames), the uniLSTM provides modeling capability for a longer range of historical information, compensating for the limitations of limited-context attention. The unidirectional nature of the uniLSTM ensures causal constraints.

### Multi-Task Learning Framework

The entire model adopts a multi-task learning strategy for joint training:

**Task 1: Phoneme Transcription (VTD)** — CTC Loss
- Input: Trigger audio segments (containing the trigger phrase)
- Target: Phoneme-level transcription sequence
- Purpose: To determine if the audio contains the phoneme sequence of the target trigger phrase

**Task 2: Phrase Discrimination (FTM)** — Frame-wise CE Loss
- Input: Trigger audio + Post-trigger audio
- Target: Binary classification labels for each frame (target phrase vs. non-target)
- Purpose: To judge user intent using the post-trigger audio context

**Task 3 (Training Only): TF Decoder** — CE Loss
- Acts as a regularizer for the encoder, encouraging the encoder to learn richer acoustic representations.
- Not used during inference, adding no inference overhead.

### Streaming Inference Mechanism

During inference, the streaming TF Encoder works as follows:

1. Audio arrives as a continuous stream of frames.
2. The encoder processes audio in blocks ($B$ frames), processing $S$ new frames each time and reusing the $S$ frames of context from the previous block.
3. For the VTD task, the encoder's output is directly used for phoneme transcription judgment.
4. For the FTM task, after trigger detection, the encoder continues to stream-process subsequent audio frames, passing the encoder's embeddings to the phrase discrimination branch.
5. Since the encoder is shared, FTM can directly reuse the audio context already processed by VTD, without requiring recalculation.

This design allows **VTD and FTM to share computation**, significantly reducing runtime memory and inference latency.

### Hardware Optimization

The paper implements several hardware optimizations on-device:
- **Key-Value Caching**: The key and value matrices of the streaming SA layer can be cached and reused, avoiding redundant calculations.
- **Linear Scaling**: The runtime memory and inference time of the streaming model grow **linearly** with audio length, as opposed to the **quadratic growth** of non-streaming models.

## Main Contributions

1. **Proposed Streaming TF Encoder Architecture**: This is the first application of streaming self-attention mechanisms to the joint task of voice trigger detection and false trigger mitigation. Through block processing protocols and attention masks, it achieves streaming inference using only causal context while maintaining performance comparable to bidirectional attention models.

2. **Unified VTD and FTM Tasks**: Proposes a single model architecture to handle both tasks simultaneously. The two tasks share the TF Encoder backbone, and FTM directly reuses the encoded audio context results from VTD, without requiring an additional full forward pass. This is the first work in the literature to handle both tasks simultaneously in a streaming manner on-device.

3. **Frame-wise CE Loss Replacing CTC Loss**: In the phrase discrimination branch, replacing the traditional CTC loss with frame-wise Cross-Entropy loss eliminates the ambiguity of blank labels, improving the discrimination accuracy of the FTM task.

4. **Order-of-Magnitude Improvement in Hardware Efficiency**: The streaming model reduces runtime memory by 32% and inference time by 56% (compared to equivalent non-streaming models). More importantly, resource consumption changes from quadratic growth to linear growth, making it possible to process post-trigger audio of arbitrary length.

5. **Elimination of Dependence on Server-Side ASR**: The entire system uses only acoustic features and does not rely on computationally expensive ASR decoding, enabling fully on-device deployment and fundamentally solving privacy issues.

## Experimental Results

### VTD Task Results

Using the baseline TF Encoder (non-streaming, vanilla SA + CTC in both branches) as a reference, the performance of the streaming TF Encoder on the VTD task was evaluated:

**Core Metric**: False Reject Rate (FRR) at a fixed False Alarm Rate.

- The streaming joint model (VTD + FTM) achieved an **average relative reduction of 18% in FRR** on the VTD task compared to the baseline.
- This improvement is attributed to: (1) Frame-wise CE loss being more suitable for phrase discrimination tasks than CTC loss; (2) The uniLSTM layer providing longer-range context modeling; (3) Joint training enabling the encoder to learn better features serving both tasks.

### FTM Task Results

**Core Metric**: The proportion of false triggers correctly suppressed when using post-trigger audio of varying lengths.

| Post-Trigger Audio Length | False Trigger Suppression Rate |
|---------------------------|--------------------------------|
| 0.5 seconds               | ~85%                           |
| 1.0 second                | **95%**                        |

- With only an additional **1 second** of post-trigger audio, the model can correctly suppress **95%** of false triggers.
- This result indicates that short post-trigger audio context contains sufficient discriminative information, allowing for high-precision intent judgment without complete ASR decoding.

### Hardware Efficiency Evaluation

Measurements on real Apple devices (processing 1 second of post-trigger audio):

| Metric          | Non-Streaming Model | Streaming Model | Improvement |
|-----------------|---------------------|-----------------|-------------|
| Runtime Memory  | Baseline            | -32%            | Significant |
| Inference Time  | Baseline            | -56%            | Significant |

**Resource Consumption Growth Characteristics**:
- Non-streaming model: Memory and inference time grow **quadratically** with audio length (because SA layers require global attention).
- Streaming model: Memory and inference time grow **linearly** with audio length (because SA layers use only local context).

For longer post-trigger audio (e.g., 2 seconds, 3 seconds), the advantages of the streaming model will be even more significant.

### Ablation Studies

The paper validates the contribution of each component through ablation studies:

**Comparison of SA Layer Types**:
- Vanilla SA (Bidirectional Attention) vs. Streaming SA (Causal Limited Context)
- On the VTD task, the performance loss of Streaming SA is minimal.
- On the FTM task, the combination of Streaming SA + uniLSTM even outperforms Vanilla SA.

**Comparison of Loss Functions**:
- CTC Loss vs. Frame-wise CE Loss (in the phrase discrimination branch)
- Frame-wise CE loss significantly outperforms CTC loss on the FTM task.

**Contribution of UniLSTM Layer**:
- Removing the uniLSTM layer results in a performance drop of approximately 2-3% on FTM.
- The uniLSTM effectively compensates for information loss due to limited context attention.

## Limitations and Future Work

### Technical Limitations

1. **Information Loss from Causal Attention**: The streaming SA layer uses only left-side (past) context and cannot utilize discriminative information from the right-side (future) context. Although the uniLSTM compensates for this to some extent, performance may still lag behind bidirectional models in complex scenarios requiring bidirectional context (e.g., confirmation in strong noise environments).

2. **Tuning Overhead for Block Size**: The block size $B$ is a key hyperparameter that requires careful trade-offs between computational efficiency and detection accuracy. If $B$ is too small, context information is lost; if $B$ is too large, latency and memory increase. The paper does not provide systematic guidelines for selecting block sizes.

3. **Training Complexity**: The multi-task learning framework requires simultaneously balancing CTC loss, frame-wise CE loss, and the optional decoder CE loss. Training stability and convergence speed may be affected. The selection of loss function weights requires careful tuning.

4. **Sequence Dependency of UniLSTM**: The uniLSTM layer introduces serialized computational dependencies, which may limit the degree of parallelization. On hardware accelerators (such as NPUs), this could become an inference bottleneck.

5. **Limitations of Acoustic-Only Features**: Relying entirely on acoustic features means the model cannot utilize linguistic information (e.g., lexical semantics, syntactic structures). For complex FTM scenarios requiring an understanding of the user's query meaning (e.g., determining if the user is speaking to the assistant or conversing with another person), acoustic features alone may be insufficient.

### Experimental Design Shortcomings

1. **Confidentiality of Evaluation Data**: As this is an Apple production system, the paper does not disclose details of the specific training and evaluation datasets, making it difficult for external researchers to reproduce and verify the results.

2. **Lack of Comparison with Competing Methods**: There is no direct comparison with other streaming Transformer variants (such as Emformer, Blockformer) or streaming Conformer on the same tasks.

3. **Insufficient Granular Analysis of False Trigger Types**: The paper does not analyze the suppression effects for different types of false triggers (e.g., TV sounds, pet sounds, other phrases with similar phonemes). The difficulty of handling different types of false triggers may vary significantly.

4. **Incomplete Latency Analysis**: While percentage improvements in inference time are reported, absolute latency values and end-to-end system latency analyses are not provided, which are crucial for deploying real-time systems.

### Future Improvement Directions

1. **Integration of Acoustic and Linguistic Features**: Explore the fusion of lightweight on-device language models (such as RNN-LMs or small Transformer-LMs) with acoustic models, introducing linguistic discriminative information without significantly increasing computational load.

2. **Adaptive Context Windows**: Design mechanisms that can adaptively adjust block sizes based on audio complexity, using smaller $B$ to reduce latency in simple scenarios and larger $B$ to improve accuracy in complex scenarios.

3. **More Efficient Attention Mechanisms**: Explore Linear Attention or Low-Rank Attention to replace standard dot-product attention, further reducing the computational complexity of streaming inference.

4. **Support for Multiple Languages and Trigger Phrases**: Extend the framework to multi-lingual scenarios, supporting joint detection of multiple languages and multiple trigger phrases.

5. **Unification with Wake Word Detection**: Explore unified architectures where the streaming TF Encoder is used simultaneously for wake word detection and command word recognition, further reducing the system complexity of on-device AI.

### Implications for the KWS Field

This paper provides important insights for the Keyword Spotting (KWS) field, particularly in the design of on-device KWS systems:

**Feasibility of Streaming Transformers**: It proves that Transformer architectures can be adapted to streaming KWS scenarios through block processing and causal attention without significant accuracy loss. This breaks the traditional notion that "KWS must use CNNs/RNNs," paving the way for subsequent Transformer-based KWS research (such as Conformer KWS).

**Joint Modeling Paradigm of VTD + FTM**: The design where two tasks share a backbone encoder achieves optimal utilization of computational resources. Moreover, the regularization effect of the FTM task actually improves the performance of the VTD task. This approach of "using an auxiliary task to enhance the main task" can be generalized to other multi-task scenarios related to KWS.

**Sufficiency of Acoustic-Only Features**: The 95% false trigger suppression rate indicates that for short-duration intent discrimination, acoustic features may be sufficient, eliminating the need for complete ASR decoding. This finding has guiding significance for the architectural design of on-device KWS systems—simple and focused acoustic models may be more practical than complex multi-module systems.

**Hardware Efficiency as a Primary Goal**: The paper treats runtime memory and inference time as optimization goals equally important to detection accuracy. This engineering perspective holds significant reference value for the actual productization of KWS systems.
