# Audiomer: A Convolutional Transformer for Keyword Spotting

- **Authors/Affiliations**: Surya Kant Sahu, Sai Mitheran, Juhi Kamdar, Meet Gandhi - George Mason University / Independent Research
- **Date**: 2021.09
- **Link**: https://arxiv.org/abs/2109.10252
- **Keywords**: Transformer, Convolution, Hybrid Architecture, Performer Attention, Keyword Spotting, Self-Attention, Linear Complexity

## Problem Statement

Keyword Spotting (KWS) models need to effectively extract discriminative features from short speech segments (typically around 1 second). CNN-based models have dominated the KWS domain by capturing local time-frequency patterns in spectrograms through local convolution kernels. However, the receptive field of CNNs is limited by kernel size and network depth, making it difficult to directly model long-range temporal dependencies in speech. For KWS tasks, although the speech segments are short, global prosodic patterns, pitch change trajectories, and temporal structures spanning multiple phonemes across the entire segment remain crucial for keyword discrimination.

Pure Transformer architectures can capture global context through self-attention mechanisms, theoretically making them well-suited for modeling such long-range dependencies. However, the computational complexity of standard self-attention is $O(n^2)$, which is prohibitively expensive for inputs with large sequence lengths. More importantly, in KWS scenarios, edge devices have strict requirements for latency and power consumption, and the quadratic complexity of pure Transformers becomes a bottleneck for practical deployment. Furthermore, Transformers lack the translation invariance and locality inductive biases of CNNs, potentially requiring more data to achieve optimal performance on small datasets.

The core problem this paper addresses is: How to design a hybrid architecture that can efficiently extract local time-frequency features via CNNs while modeling global context through an efficient attention mechanism, all while maintaining the computational efficiency suitable for KWS deployment. Solving this problem is of significant importance for promoting the application of Transformer architectures in resource-constrained KWS scenarios.

## Methodology

### Overall Architecture Design
Audiomer adopts the classic "CNN backbone + Transformer head" hybrid architecture design, a paradigm widely validated as effective in both vision (hybrid models prior to ViT) and speech processing. The overall pipeline is: Raw Audio -> MFCC/Spectrogram Feature Extraction -> 1D Residual Network -> Performer Attention Layer -> Classification Head. The core idea of this design is that the CNN is responsible for transforming the raw time-frequency representation into high-level feature sequences, while the Performer establishes global dependencies on these high-level features.

### 1D Residual Convolutional Network (1D ResNet)
The convolutional frontend of Audiomer uses a 1D residual network to process the input spectrogram features. Unlike traditional 2D convolutions (which convolve along both time and frequency dimensions), 1D convolutions operate along the time axis on the spectrogram, treating the frequency dimension as the channel dimension. The motivation for this design lies in:

- **Temporal structure of speech is core information for keyword recognition**: The discriminative power of keywords is primarily manifested in the temporal sequence patterns of phonemes (e.g., the temporal sequence of /y/, /E/, /s/ in "yes"). 1D temporal convolutions are naturally suited to capture this temporal structure.
- **Parameter efficiency**: It reduces the number of parameters (compared to 2D convolutions), facilitating deployment with a small footprint. The number of parameters in 1D convolutions is independent of the frequency dimension, depending only on the kernel size and number of channels.
- **Gradient flow via residual connections**: Residual connections in deep networks ensure effective gradient propagation, allowing the network to learn more complex temporal features without suffering from vanishing gradients.

Each residual block contains: 1x1 Convolution (channel dimensionality reduction, reducing computation) -> Batch Normalization -> ReLU Activation -> Temporal Convolution (capturing temporal patterns) -> Batch Normalization -> Residual Skip Connection (maintaining information flow) -> ReLU Activation. This bottleneck structure design draws on the success of ResNet, reducing computational overhead while maintaining expressive power.

### Performer Attention Mechanism
The core innovation of Audiomer lies in using Performer attention to replace standard multi-head self-attention. Performer (Choromanski et al., 2020) reduces the quadratic complexity $O(n^2)$ of standard softmax attention to linear complexity $O(n)$ through Random Feature Mapping and the FAVOR+ algorithm (Fast Attention Via Orthogonal Random features):

**Standard Attention**: $\text{Attention}(Q,K,V) = \text{softmax}(QK^T / \sqrt{d}) * V$. Computing $QK^T$ requires $O(n^2 * d)$ time complexity and $O(n^2)$ space complexity.

**Performer Attention**: Leveraging the mathematical property that the softmax kernel function can be approximated by random features, it uses a random feature function $\phi(.)$ to approximate the softmax kernel as $\phi(Q) * \phi(K)^T$. Specifically:

$\phi(x) = (1/\sqrt{r}) * [\exp(w_1^T * x + b_1), ..., \exp(w_r^T * x + b_r)]$

where $w_i$ are random vectors sampled from a specific distribution, and $r$ is the random feature dimension. Through this approximation, the attention calculation becomes:

$\text{Attention}(Q,K,V) = (\phi(Q) * (\phi(K)^T * V)) / (\phi(Q) * (\phi(K)^T * 1))$

This reduces the computational complexity from $O(n^2 * d)$ to $O(n * d * r)$, where $r$ is the controllable random feature dimension (typically much smaller than the sequence length $n$). Crucially, this approximation can be proven to be unbiased with controllable variance.

This allows Audiomer to efficiently model long-range temporal dependencies in the convolutional feature sequences without being constrained by computational bottlenecks related to sequence length. Even when processing longer speech segments, Performer maintains linear computational complexity.

### Positional Encoding and Classification Head
Before the Performer attention layer, learnable positional encoding is added to the feature sequence output by the convolution to inject positional information. Since the random feature approximation of Performer destroys position-awareness (unlike standard attention, which can implicitly encode position differences via $QK^T$), positional encoding is indispensable.

Classification uses the CLS token method: a special classification token is added at the beginning of the sequence, and its output after passing through the Performer layer is fed into a fully connected classification head. The CLS token acts as an aggregator of global information, collecting relevant information from all patches via the attention mechanism.

### Training Strategy
Standard cross-entropy loss is used for training, employing the Adam optimizer with a learning rate warmup (linearly increasing the learning rate for the first 10% of training steps) and a cosine annealing schedule (learning rate decays according to a cosine function). This learning rate scheduling strategy is widely used in Transformer training; the warmup phase helps stabilize initial training, while cosine annealing aids convergence to better local minima.

## Main Contributions

1. **Introduction of Performer Attention for KWS**: This is the first application of Performer, a linear-complexity attention mechanism, to keyword spotting, effectively solving the computational efficiency issues of standard Transformers in KWS scenarios. This choice allows the model to maintain efficient inference speed without sacrificing global modeling capabilities. Compared to contemporaneous KWS-Transformer methods using standard attention, Audiomer has significant computational advantages in processing long sequences.

2. **Hybrid Architecture of 1D Residual Convolution and Performer**: A concise yet effective hybrid architecture is proposed, where 1D residual convolutions are responsible for local feature extraction and Performer handles global context modeling. The two complement each other: CNNs provide translation invariance and efficient extraction of local features (inductive biases lacking in Performer), while Performer provides global interaction modeling (which CNNs cannot achieve due to their limited local receptive fields).

3. **Demonstration of the Effectiveness of Linear Attention in KWS**: Experiments show that Performer's approximate attention can achieve performance close to that of standard attention on KWS tasks, while offering significant computational advantages. This finding is of great significance to the KWS field—it indicates that in practical short-sequence KWS tasks, exact calculation of attention is not necessary; approximate attention is sufficient.

4. **New Paradigm for Processing Spectrograms with 1D Convolutions**: Treating the frequency dimension as the channel dimension in 1D temporal convolutions provides a new perspective for spectrogram processing in the KWS field. This approach offers advantages in parameter efficiency and temporal modeling capabilities.

## Experimental Results

### Datasets
- **Google Speech Commands (GSC) v2**: A 12-class subset containing common command words such as "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence", and "unknown".
- Evaluation uses standard test set splits to ensure reproducibility of results.
- Input features are 40-dimensional MFCCs, covering speech segments of approximately 1 second.

### Main Performance
- Achieved competitive accuracy (approximately 95-96%) on GSC v2 (12 classes), surpassing pure CNN baseline models (DS-CNN approximately 94-95%).
- The hybrid architecture has advantages over both pure Transformers and pure CNNs:
  - **Compared to pure CNNs**: Global context modeling brought an accuracy improvement of about 1%, particularly for easily confused word pairs (e.g., "go" vs "no").
  - **Compared to pure Transformers**: Local feature extraction is more efficient, the model converges faster, and less training data is required.
- Demonstrates better generalization ability to unseen speakers, attributed to the global prosodic patterns (such as intonation curves and rhythm features) captured by the Performer layer.
- The model exhibits robustness when processing variable-length speech, as the linear complexity of Performer ensures that processing inputs of different lengths does not lead to a sharp increase in computational cost.

### Comparison with Other Methods
- Compared to standard self-attention Transformers, Performer shows a clear speed advantage (approximately 2-3x acceleration) when sequences are long, while the speed difference is smaller for short sequences (such as typical sequence lengths in KWS).
- Compared to classic KWS models like DS-CNN, Audiomer achieves an accuracy improvement of about 1%, but with a slightly larger model parameter count.
- Compared to KWT (Keyword Transformer, using standard attention), Audiomer has better scalability under comparable performance.

### Ablation Studies (Limited)
- Removing the Performer layer (using only the CNN backbone) resulted in an accuracy drop of approximately 1%, confirming the contribution of global attention.
- Using standard attention instead of Performer yielded similar performance on short sequences but increased inference time.

## Limitations and Future Work

### Technical Limitations
- **Model Size and Latency**: Although the computational complexity of the Performer layer is linear, the additional random feature mapping and linear projections still increase the number of parameters and latency. It is more suitable for edge devices with ample computational resources compared to pure CNN models. The constant factor of Performer (determined by the random feature dimension $r$) may make inference speed lower than expected in actual implementations.
- **Paper Length Constraints**: This paper is a 2-page short paper (workshop/extended abstract format), with insufficient experimental analysis, including ablation studies, detailed comparisons under different configurations, error analysis, etc. Many important experimental details and hyperparameter settings were not reported.
- **Streaming Processing Capability**: Although Performer attention reduces computational complexity, it still requires complete sequence input (non-causal random feature mapping) and has not been adapted for streaming/causal inference modes. Practical KWS systems typically require frame-level real-time processing.
- **Instability of Random Features**: The random feature mapping used by Performer may produce different performance under different random seeds. The paper does not analyze the impact of this variance on KWS performance.

### Insufficient Experimental Design
- Evaluation was conducted only on the Google Speech Commands dataset, without validation in noise, far-field, or more diverse datasets. GSC is the most basic KWS benchmark, but the robustness of the method under more difficult conditions remains unclear.
- No comparison was made with other efficient attention mechanisms (such as Linformer, Nystromformer, Linear Transformer), making it impossible to determine if Performer is the optimal choice for KWS tasks.
- Lack of deployment performance analysis after model compression (quantization, pruning), which is an important omission for KWS research aimed at edge deployment.
- Specific numbers for inference latency and model size were not reported, making it difficult to assess actual deployment feasibility.
- Lack of statistical significance analysis (e.g., variance over multiple runs).

### Future Improvement Directions
- Explore causal Performer for streaming KWS inference by modifying the random feature mapping to depend only on current and past features.
- Combine model distillation techniques to transfer the knowledge of Audiomer to smaller student models, achieving true edge deployment.
- Validate the generalization of the method on more languages (especially tonal languages like Chinese) and acoustic conditions.
- Combine with the Conformer architecture to explore the performance of convolution-enhanced Performer in KWS.
- **Insights for the KWS field**: Linear attention mechanisms such as Performer provide a feasible path for deploying Transformer-like models on resource-constrained devices. The CNN-Performer hybrid paradigm of Audiomer provides a valuable reference for subsequent KWS model design.
