# A Streaming End-to-End Framework For Spoken Language Understanding

- **Authors/Affiliations**: Xian Shi, Wei Wang, Yuxuan Wang, Qiang Huo - University of Waterloo; Huawei; Tsinghua University
- **Date**: 2021.05
- **Link**: https://arxiv.org/abs/2105.10042
- **Keywords**: Spoken Language Understanding, Streaming, End-to-End, Intent Detection, Keyword Spotting, Conformer, CTC, Chunk-based Processing

## Problem Statement

Traditional Spoken Language Understanding (SLU) systems adopt a cascaded pipeline architecture: the Automatic Speech Recognition (ASR) module transcribes speech into text, and then the Natural Language Understanding (NLU) module extracts intent and slot information from the text. This cascaded architecture has several fundamental flaws:

1. **Error Propagation**: Recognition errors from the ASR module are directly passed to the NLU module, causing the final intent classification accuracy to depend heavily on ASR quality. Experiments show that when the ASR Word Error Rate (WER) increases from 5% to 15%, the SLU intent classification accuracy may drop by 10-20%.

2. **Latency Accumulation**: The inference latencies of the ASR and NLU modules accumulate, resulting in high end-to-end latency for the entire pipeline. In voice assistant scenarios requiring real-time response, a latency exceeding 200-300ms significantly impacts user experience.

3. **Inconsistent Optimization Objectives**: The ASR module optimizes for character/word-level recognition accuracy, while the SLU module optimizes for intent-level classification accuracy; their optimization objectives are inconsistent. ASR may over-optimize on semantically irrelevant details while misrecognizing key phonemes that are crucial for intent classification.

4. **High Training Data Requirements**: The cascaded architecture requires large amounts of paired data (audio-text-intent annotations), whereas end-to-end models can learn directly from audio-intent pairs.

However, streaming processing in end-to-end SLU faces unique technical challenges: how to achieve accurate intent classification while maintaining low-latency streaming inference (i.e., using only current and past information, without waiting for future context). For Keyword Spotting (KWS), a sub-task of SLU, streaming processing is an absolute necessity—KWS systems must listen to the audio stream in real-time and respond immediately.

The core problem this paper addresses is: How to design an end-to-end SLU framework that supports streaming inference, eliminating the ASR-NLU cascaded latency while achieving low-latency multi-task inference including keyword detection and intent classification.

## Methodology

### Overall Framework Design
The proposed streaming end-to-end SLU framework maps speech directly to intent/semantic representations, without intermediate text output. The core components of the framework include: a streaming encoder, multi-task prediction heads (ASR auxiliary task + SLU main task), and a chunk-based streaming inference mechanism.

### Streaming Conformer/Transformer Encoder
The encoder is the core of the framework, responsible for extracting high-level semantic representations from raw audio or acoustic features:

- **Base Architecture**: Uses a Conformer encoder (a hybrid architecture combining convolution and self-attention) as the backbone. Conformer has been proven superior to pure Transformers in ASR tasks; its convolution module provides local feature modeling capability, while the self-attention module provides global context modeling.

- **Causal Attention**: Modifies standard bidirectional self-attention into causal (left-to-right) attention, ensuring the model only uses current and past information, avoiding "peeking" into the future. This is the core requirement for streaming processing.

- **Limited Context Attention**: Further restricts the size of the attention window (e.g., focusing only on the most recent L frames, L=10-50), reducing computational complexity and latency. Each frame only attends to its own context and the preceding L frames.

- **Causal Convolution Module**: Modifies the convolution module in Conformer to causal convolution (left-padding), ensuring no future information is used.

### Chunk-based Streaming Inference
To achieve true streaming processing, the framework adopts a chunk-based inference mechanism:

1. **Audio Chunking**: The continuous audio stream is divided into overlapping chunks of fixed size (e.g., 320ms or 640ms). Each chunk contains the acoustic features of the current chunk and optional cached context.

2. **Incremental Encoding**: The encoder independently encodes each chunk while maintaining a state cache (carryover state), passing partial information from the previous chunk to the next, simulating limited context lookback.

3. **Chunk-level Prediction**: After processing each chunk, the model outputs intermediate predictions for that time segment. The final keyword/intent decision is made by aggregating predictions from multiple chunks.

4. **Latency-Accuracy Trade-off**: The chunk size determines the lower bound of latency—larger chunks provide more context information (usually implying higher accuracy) but also increase latency. The paper explores the impact of different chunk sizes on accuracy and latency.

### Multi-task Learning: ASR Auxiliary + SLU Main Task
The framework adopts a multi-task learning strategy, optimizing both ASR and SLU objectives simultaneously:

- **ASR Auxiliary Task (CTC Loss)**: Uses Connectionist Temporal Classification (CTC) loss to train the encoder to learn acoustic-text alignment. CTC loss provides frame-level supervision signals, helping the encoder learn accurate phoneme-level feature representations. CTC does not require frame-level alignment annotations and efficiently computes probabilities using the forward-backward algorithm.

  L_CTC = -log P(text|audio)

- **SLU Main Task (Intent Classification Loss)**: Uses the encoder's output for intent classification. The classification head typically consists of: a pooling layer (aggregating frame-level features into segment-level features) -> a fully connected layer -> softmax classification.

  L_intent = CrossEntropy(predicted_intent, true_intent)

- **Joint Loss**: L_total = alpha * L_CTC + (1 - alpha) * L_intent, where alpha controls the weight of the two tasks.

- **Motivation for Multi-task**: The ASR auxiliary task provides rich acoustic supervision signals for the encoder, helping it learn better speech representations. Even if the text output is not directly used in the SLU main task, the CTC loss still improves intent classification performance by forcing the encoder to learn accurate phoneme representations.

### Keyword Spotting as a Special Case of Intent Detection
The framework treats keyword spotting as a special form of intent detection:
- Each target keyword corresponds to an intent class
- "Non-keyword" corresponds to a "no intent" class
- KWS can be implemented by adding keyword classes to the intent classification head

The advantage of this unified perspective is that KWS and more complex intent detection can share the same streaming encoder, reducing the number of models and computational overhead during deployment.

### Training Strategy
- **Input Features**: 80-dimensional log-mel spectrogram, 25ms window, 10ms stride.
- **Optimizer**: AdamW, with learning rate warmup + Noam decay.
- **Data Augmentation**: SpecAugment (time and frequency masking), speed perturbation.
- **Streaming Simulation Training**: Uses the same chunk-based processing logic during training as in inference, ensuring consistency between training and inference.

## Main Contributions

1. **Introduction of a Streaming End-to-End SLU Framework Supporting Keyword Spotting**: For the first time, unifies streaming end-to-end SLU framework with the KWS task, demonstrating that KWS can be a special instance of more general SLU tasks. This unification allows KWS systems to benefit from the semantic understanding capabilities of the SLU framework.

2. **Elimination of Cascaded ASR-NLU Pipeline Latency**: By mapping directly from speech to intent end-to-end, the intermediate text generation step is eliminated, significantly reducing end-to-end latency. In experiments, the framework's end-to-end latency was reduced by approximately 40-60% compared to cascaded systems.

3. **Low-Latency Streaming Inference via Chunk-based Processing**: The chunk-based processing mechanism allows the model to run incrementally on the audio stream, achieving true real-time inference. The chunk size can be flexibly adjusted to accommodate different latency budgets.

4. **Proof that Joint ASR+SLU Training Benefits Both Tasks**: The ASR auxiliary task provides acoustic alignment supervision via CTC loss, helping the encoder learn more accurate speech representations; the SLU main task provides semantic-level supervision, with both mutually reinforcing each other. Ablation studies show that joint training improves accuracy by 2-5% compared to training SLU alone.

5. **Quantification of the Performance Gap between Streaming and Non-Streaming**: Systematically analyzes the accuracy loss of streaming processing (limited context) relative to non-streaming processing (full context), providing practitioners with references for selecting appropriate context window sizes.

## Experimental Results

### Datasets and Setup
- **SLU Evaluation**: Uses standard SLU benchmark datasets (such as Fluent Speech Commands, SNIPS, etc.) to evaluate intent classification accuracy.
- **KWS Evaluation**: Evaluates keyword detection performance on Google Speech Commands and internal KWS datasets.
- **Evaluation Metrics**: Intent classification accuracy, KWS accuracy, end-to-end latency (ms).

### SLU Intent Classification Performance
- Compared to cascaded ASR+NLU systems, the end-to-end framework achieved competitive intent classification accuracy (within a 1-2% gap).
- On some intent classes, the end-to-end framework even outperformed the cascaded system—particularly for words that ASR easily misrecognizes but have little impact on intent classification (the end-to-end model can skip these "noises" and focus directly on semantically critical information).

### Streaming Processing Effects
- **Significant Latency Reduction**: The end-to-end framework's latency is approximately 200-400ms (depending on chunk size), reducing latency by about 40-60% compared to cascaded systems (approx. 500-800ms).
- **Accuracy-Latency Trade-off Curve**: As the chunk size increases from 160ms to 1280ms, intent classification accuracy continues to improve, but latency also increases linearly. A chunk size of 640ms achieved the best accuracy-latency balance.
- **Impact of Context Window Size**: Larger context windows (focusing on more historical frames) continuously improve accuracy, but the gains diminish after L=30 frames.

### KWS Performance
- Keyword detection performance is comparable to dedicated KWS models (such as DS-CNN, CRNN), with an accuracy gap of less than 1%.
- As a byproduct of the unified SLU framework, KWS is achieved without additional model overhead.

### Ablation Studies
- **Contribution of ASR Auxiliary Task**: After removing the CTC loss, intent classification accuracy dropped by approximately 2-5%, and KWS accuracy dropped by approximately 1-3%. This proves the importance of ASR auxiliary training for learning acoustic feature representations.
- **Streaming vs. Non-Streaming**: The accuracy of using full context (non-streaming) is approximately 2-4% higher than the best streaming configuration, which is the inherent cost of streaming processing.
- **Conformer vs. Transformer**: The Conformer encoder is approximately 1-2% more accurate than the pure Transformer encoder, validating the advantage of the convolution+attention hybrid architecture in speech tasks.
- **Impact of Alpha Value**: SLU task performance is best when alpha=0.3 (lower ASR weight); ASR performance is best when alpha=0.7 (higher ASR weight). Alpha=0.4-0.5 achieves the best balance between the two tasks.

## Limitations and Future Work

### Technical Limitations
- **Accuracy Loss in Streaming Processing**: Compared to full-context models, streaming processing introduces an accuracy loss of approximately 2-4%, because causal attention and limited context windows cannot utilize future information. In applications requiring the highest accuracy, non-streaming or two-stage processing (streaming pre-screening + non-streaming re-ranking) may still be necessary.
- **Training Data Requirements**: The framework requires paired audio-text-intent annotations, where obtaining intent annotations is costly. Compared to simple KWS classification tasks, the data preparation cost increases significantly.
- **Framework Complexity**: Contains multiple components such as streaming encoder, CTC loss, intent classification head, and chunk-based processing logic, making it much more complex than dedicated KWS models. For scenarios requiring only KWS functionality, this complexity may be unnecessary.
- **Handling Long-Tail Intents**: For rare intent classes with limited training data, end-to-end models may not perform as well as cascaded systems (because the NLU module can leverage pre-trained knowledge at the text level).

### Experimental Design Shortcomings
- Evaluation under noise and far-field conditions is limited; real-world voice assistants often face these challenges.
- Inference performance (latency, memory usage, power consumption) was not tested on real edge devices.
- Insufficient analysis of applicability to different languages; it remains unclear whether the framework is equally effective for tonal languages (such as Chinese).
- Comparisons with other end-to-end SLU methods (such as SpeechTransformer, audio-text multimodal methods) are not comprehensive.
- Lack of evaluation on multi-intent detection in continuous speech (the framework primarily evaluated single-intent classification).

### Future Improvement Directions
- Explore two-stage streaming processing—a lightweight streaming model for pre-screening and a more precise non-streaming model for re-ranking—to balance latency and accuracy.
- Combine self-supervised pre-training (such as wav2vec 2.0, HuBERT) to improve the quality of the encoder's acoustic representations, reducing dependence on annotated data.
- Study multi-lingual and cross-lingual transfer in end-to-end SLU frameworks, enabling rapid adaptation to new languages.
- Explore incorporating slot filling tasks into the streaming framework to achieve more complete SLU capabilities.
- Combine model distillation and quantization techniques to enable the framework to run in real-time on edge devices.
- Insights for the KWS field: Viewing KWS within a more general SLU framework is a valuable perspective shift. KWS is not just a classification problem, but a problem of extracting specific semantic information from speech. End-to-end frameworks allow KWS systems to leverage higher-level semantic information (such as contextual intent), potentially breaking through the performance limits of pure classification methods.
