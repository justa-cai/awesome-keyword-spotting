# Streaming End-to-End Speech Recognition and Keyword Spotting Using Sequence-to-Sequence Models

- **Authors/Affiliations**: Yanzhang He, Rohit Prabhavalkar, Kanishka Rao, Wei Li, Anton Bakhtin, Ian McGraw (Google)
- **Date**: 2017
- **Link**: https://arxiv.org/abs/1706.02563
- **Keywords**: Keyword Spotting, RNN-T, Sequence-to-Sequence, Streaming Speech Recognition, Keyword Biasing, End-to-End

## Problem Statement

Traditional keyword spotting (KWS) systems typically use separate acoustic and language model components, which are trained independently and combined during decoding. This separated design has several fundamental issues: (1) independently trained components may not be globally optimized, leading to suboptimal overall performance; (2) the system architecture is complex, requiring the maintenance of multiple models and sophisticated decoders; (3) KWS and ASR functions are usually provided by completely independent systems, preventing the sharing of computation and model parameters.

End-to-end models such as the RNN Transducer (RNN-T) provide a unified framework for ASR and KWS, but applying them to streaming scenarios with real-time constraints and optimizing them for specific keywords remains challenging. The core problem addressed by the paper is: How to build a streaming-capable end-to-end sequence-to-sequence model that performs both general speech recognition and targeted keyword spotting, while improving the detection rate of specific keywords through keyword biasing?

## Methodology

### RNN-T Architecture

The paper uses the RNN Transducer as the core architecture for unified ASR+KWS:

1. **RNN-T Components**:
   - **Encoder (Transcription Network)**: Processes audio feature sequences, mapping each audio frame to a high-dimensional encoded representation. The encoder uses LSTM or similar architectures and supports streaming processing (unidirectional).
   - **Prediction Network**: Predicts the next output based on the history of previously output text, functioning similarly to a language model. It uses an LSTM to encode the sequence of previously output characters/subwords.
   - **Joint Network**: Combines the encoder output and the prediction network output to produce logits for each output unit at every time step.

2. **RNN-T Loss Function**:
   - RNN-T extends CTC by introducing conditional dependencies on the prediction network.
   - The forward variable $\alpha(t, u)$ represents the cumulative probability at the $t$-th encoded frame and the $u$-th output unit.
   - The recursive relationship considers two types of transitions: emit (output a token) and blank (do not output).
   - Loss function: $L = -\log P(y|x) = -\log \sum \alpha(T, U)$, summing over all valid paths.

3. **Streaming Capability**:
   - The encoder uses a unidirectional LSTM, relying only on current and past audio frames.
   - The decoder processes audio frame by frame, outputting zero or more tokens per frame.
   - This achieves low-latency inference at the frame level, suitable for always-on KWS scenarios.

### Keyword Biasing

The key technique introduced in the paper is keyword biasing, which enhances the model's sensitivity to specific keywords during decoding:

1. **Principle of Biasing Mechanism**:
   - During standard RNN-T decoding, a probability distribution over all candidate tokens is calculated at each time step.
   - For characters/subwords belonging to the target keyword, a positive bias term is added to their logits.
   - The bias term increases the probability of selecting keyword tokens, thereby improving the detection rate.

2. **Context-Aware Biasing**:
   - The bias strength is dynamically adjusted based on the current decoding state.
   - When the partially output sequence matches the prefix of the keyword, subsequent characters receive stronger bias.
   - If the output sequence does not match the keyword, no bias is applied (or the bias is very weak).
   - This context-aware mechanism avoids inappropriately biasing keyword tokens in non-keyword contexts.

3. **Bias Strength Control**:
   - The magnitude of the bias term is a tunable hyperparameter that controls detection sensitivity.
   - Larger bias improves keyword detection rates but may increase false alarms (incorrectly outputting the keyword in non-keyword contexts).
   - Smaller bias maintains more natural ASR behavior but may miss some keyword instances.
   - In practical deployment, the bias strength needs to be adjusted according to application requirements.

### Joint Optimization

- **Multi-task Training**: The model simultaneously optimizes general ASR quality (Word Error Rate, WER) and keyword detection performance.
- **Training Data**: Uses large-scale labeled speech data, containing both general speech and speech containing keywords.
- **Loss Function**: Standard RNN-T loss, potentially with an auxiliary loss for keyword detection.
- **Evaluation Metrics**: ASR Word Error Rate (WER) + KWS detection rate/false alarm rate.

### System Architecture

- **Encoder**: Multi-layer unidirectional LSTM, with 512-1024 hidden units per layer.
- **Prediction Network**: 2-layer LSTM, with 256-512 hidden units per layer.
- **Joint Network**: Fully connected layer, with output dimension equal to the output vocabulary size + 1 (blank).
- **Output Units**: Character-level (a-z) or subword-level (wordpiece/BPE).
- **Feature Input**: 80-dimensional log-mel filterbank energies, frame length 25ms, frame shift 10ms.

## Main Contributions

1. **Unified ASR and KWS with RNN-T**: The paper is the first to demonstrate that the RNN Transducer can serve as a unified framework to perform both streaming speech recognition and keyword detection simultaneously. A single model provides both general transcription capabilities and targeted keyword detection, eliminating the need to maintain separate KWS and ASR systems. This unified framework simplifies system architecture and reduces deployment resources.

2. **Keyword Biasing Technique**: Innovatively introduces a keyword biasing mechanism that improves detection rates by applying positive bias to the logits of specific keywords during decoding, without modifying model parameters. This is an elegant and efficient solution—there is no need to retrain the model; enabling biasing during inference is sufficient to enhance keyword detection.

3. **End-to-End Streaming KWS**: Achieves fully streaming end-to-end KWS—the model processes audio frame by frame, producing outputs at the frame level, meeting the low-latency requirements of always-on applications. This proves that complex end-to-end models (like RNN-T) can operate under strict real-time constraints.

4. **Balancing ASR Quality and KWS Performance**: The paper demonstrates how to maintain general ASR quality while optimizing keyword detection. The impact of keyword biasing on the overall ASR word error rate is minimal, indicating that ASR and KWS can coexist harmoniously within the same model.

5. **Industrial-Grade System Design**: Coming from Google, the paper provides a system design scheme oriented towards actual product deployment, covering aspects from model architecture to inference optimization, offering direct reference value for industrial KWS system development.

## Experimental Results

### Experimental Setup
- **Dataset**: Large-scale internal Google speech dataset, containing thousands of hours of labeled speech.
- **Evaluation Metrics**: ASR Word Error Rate (WER), KWS detection rate, false alarm rate (FA/hour).
- **Baseline Methods**: Traditional independent KWS systems, RNN-T without biasing, RNN-T with different biasing strengths.

### Key Results
- **KWS Detection Rate**: The detection rate of RNN-T + keyword biasing is comparable to or better than traditional KWS systems.
- **Biasing Effect**: Keyword biasing significantly improves the detection rate of target keywords, with an increase of approximately 5-15% (absolute value).
- **ASR Quality**: The impact of biasing on overall ASR word error rate is minimal (<0.5% WER increase), indicating that biasing does not impair general transcription capabilities.
- **Streaming Latency**: The model produces outputs with frame-level latency (approximately 10ms), meeting the requirements for real-time KWS.
- **False Alarm Rate**: Biasing strength requires careful tuning—excessive bias increases false triggers in non-target contexts.

### Detailed Analysis
- **Impact of Biasing Strength**: Detection rate increases with biasing strength, but the false alarm rate also rises. There exists a "sweet spot" where detection rate is significantly improved while the false alarm rate remains acceptable.
- **Advantages of Context-Aware Biasing**: Context-aware biasing (applying bias only when matching keyword prefixes) performs better than global uniform biasing, resulting in lower false alarm rates.
- **Effect of Encoder Depth**: Deeper encoders improve both ASR and KWS performance but also increase inference latency.
- **Position Invariance of Keywords**: RNN-T + biasing is insensitive to the position of keywords in the audio; it can effectively detect keywords regardless of whether they appear at the beginning, middle, or end.

## Limitations and Future Work

### Limitations

1. **Computational Cost**: RNN-T models have significantly higher computational costs than dedicated lightweight KWS models (such as DS-CNN). Although RNN-T provides unified ASR+KWS capabilities, its computational demands may be too high for extremely low-power devices (such as microcontrollers). RNN-T is better suited for devices with certain computational capabilities (such as mobile application processors).

2. **Bias Parameter Tuning**: The strength of keyword biasing needs to be tuned for each keyword and application scenario. Different keywords may require different biasing strengths (keywords with unique pronunciations may need smaller bias, while keywords with pronunciations similar to common words require more refined biasing strategies).

3. **Insufficient Evaluation in Noise and Far-Field**: The paper does not sufficiently evaluate the model's KWS performance under highly challenging acoustic conditions (far-field recording, strong noise, strong reverberation). These are typical scenarios in smart speakers and in-car systems.

4. **Training Data Requirements**: RNN-T models require a large amount of labeled speech data to achieve good performance. For low-resource languages or specific domains, obtaining sufficient training data may be a challenge.

5. **Limitations of Biasing**: Keyword biasing only adjusts the probability distribution during decoding and does not change the model's acoustic understanding capability. For interfering words that are acoustically very similar to the keyword, biasing may not effectively distinguish them.

### Future Work

1. **Lightweight RNN-T**: Reduce the computational cost of RNN-T through model compression (quantization, pruning, knowledge distillation) and architectural optimization (such as smaller encoders), enabling it to run on lower-power devices.
2. **Adaptive Biasing**: Develop adaptive biasing mechanisms that automatically adjust biasing strength based on the current acoustic environment and speaker characteristics.
3. **Multi-Keyword Biasing**: Extend the biasing mechanism to support the detection of multiple keywords simultaneously, with independent biasing parameters for each keyword.
4. **Conformer Encoder**: Replace the LSTM encoder with a Conformer (an architecture combining the advantages of CNN and Transformer) to improve acoustic modeling capabilities.
5. **End-to-End Bias Learning**: Integrate the biasing mechanism into model training (rather than only as an adjustment during inference), allowing the model to actively learn to increase the output probability of keywords at appropriate times.
