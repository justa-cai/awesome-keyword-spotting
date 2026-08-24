# Keyword Transformer: A Self-Attention Model for Keyword Spotting

- **Authors/Affiliations**: Axel Berg, Mark O'Connor, Miguel Ventresca - Arm Machine Learning Research Lab / Lund University
- **Date**: 2021.04 (arXiv), Interspeech 2021
- **Link**: https://arxiv.org/abs/2104.00769
- **Keywords**: Transformer, Self-Attention, Keyword Spotting, Small Footprint, Vision Transformer, Patch Embedding, Google Speech Commands

## Problem Statement

The field of Keyword Spotting (KWS) has long been dominated by Convolutional Neural Networks (CNNs), ranging from early simple CNNs to various variants such as Depthwise Separable CNNs (DS-CNNs) and Attention-Enhanced CNNs. The success of CNNs stems from their local receptive fields and translation invariance; however, these characteristics also limit the model's ability to capture global context and long-range dependencies.

Transformer architectures have achieved revolutionary success in Natural Language Processing (BERT, GPT series), Computer Vision (Vision Transformer), and Automatic Speech Recognition (Conformer, Whisper). However, in the domain of KWS, the exploration of pure Transformer architectures has been relatively scarce. The main concerns include:
1. **Data Efficiency**: Transformers typically require large amounts of training data to leverage their advantages, whereas training data for KWS is relatively limited.
2. **Computational Efficiency**: The quadratic complexity of self-attention may not be suitable for low-power edge devices.
3. **Inductive Bias**: The translation invariance and locality inductive biases of CNNs are beneficial for speech processing; pure Transformers lack these biases.

The core problem this paper addresses is: Can a pure self-attention KWS architecture (without relying on a CNN backbone) be designed to surpass CNN-based State-of-the-Art (SOTA) methods while maintaining a model size suitable for edge deployment?

## Methodology

### Overall Architecture - KWT (Keyword Transformer)
The design of KWT is inspired by the Vision Transformer (ViT). It treats the spectrogram as an "image," implementing end-to-end keyword classification through patch embedding, positional encoding, and a Transformer encoder.

### Patch Embedding
The first key step in KWT is to divide the input spectrogram (typically 40-dimensional MFCCs with T time steps) into fixed-size patches:
- **Patch Size**: The spectrogram is divided into non-overlapping patches of size $(P_t \times P_f)$, where $P_t$ is the size in the time dimension and $P_f$ is the size in the frequency dimension.
- **Linear Projection**: Each patch is flattened and mapped to a D-dimensional embedding vector via a linear projection layer.
- **CLS Token**: A learnable classification token ([CLS]) is prepended to the sequence of patch embeddings, similar to ViT.
- **Result**: The input is converted into a sequence of shape $(N_p + 1) \times D$, where $N_p$ is the number of patches and $D$ is the embedding dimension.

### Positional Encoding
Since the Transformer itself lacks position awareness, KWT adds learnable positional encodings to the patch embeddings:
- The length of the positional encoding equals the number of patches plus one (including the CLS token).
- The learnable positional encodings can adaptively encode the spatial relationships between patches based on the training data.

### Transformer Encoder
KWT uses a standard Post-Norm Transformer encoder (i.e., LayerNorm is applied after the residual connection), consisting of multiple identical Transformer layers. Each layer includes:
1. **Multi-Head Self-Attention (MHSA)**: H attention heads, each with dimension $D/H$, computing global attention between patches.
2. **Residual Connection + LayerNorm**
3. **Feed-Forward Network (FFN)**: A two-layer MLP with an intermediate dimension of $4D$, using GELU activation.
4. **Residual Connection + LayerNorm**

The choice of Post-Norm over Pre-Norm was based on experimental findings that Post-Norm provides more stable training for the KWS task.

### Classification Head
The final classification uses the output representation of the CLS token, which is passed through a LayerNorm and then into a linear classification layer.

### Model Configurations
The paper explores various configurations:
- **KWT-1**: 1 Transformer layer, embedding dimension 64, 2 attention heads
- **KWT-2**: 2 Transformer layers, embedding dimension 128, 4 attention heads
- **KWT-3**: 3 Transformer layers, embedding dimension 192, 6 attention heads
- Patch sizes vary from $(16, 16)$ to $(32, 32)$

### Training Details
- Optimizer: AdamW with weight decay of 0.1
- Learning Rate: Warmup + Cosine Annealing
- Data Augmentation: SpecAugment (time/frequency masking)
- Label Smoothing: 0.1

### Official Code
ARM released the official implementation: github.com/ARM-software/keyword-transformer

## Main Contributions

1. **Introduction of one of the first pure Transformer architectures for KWS**: KWT is a fully self-attention-based KWS model without any CNN components. This work demonstrates that the Transformer paradigm can be successfully transferred to small-footprint audio classification tasks.

2. **Adaptation of Vision Transformer methods to speech spectrograms**: By converting spectrograms into sequences processable by Transformers via patch embedding, it showcases the applicability of ViT's "image as a sequence of patches" concept to the speech domain.

3. **Achieving SOTA accuracy on GSC with competitive model size**: KWT achieved approximately 97.5% accuracy on Google Speech Commands v2 (12 classes), surpassing CNN methods like DS-CNN of similar parameter counts, without requiring pre-training or additional data.

4. **Systematic analysis of patch size**: The paper provides a detailed analysis of how patch size affects KWS performance, finding that larger patches perform better on short speech segments.

## Experimental Results

### Datasets
- **Google Speech Commands v2**: 12-class and 35-class classification tasks
- **Audio Features**: 40-dimensional MFCCs, approximately 49 time steps (1-second audio)

### Accuracy
- **12-Class Task**: KWT achieved approximately 97.5% accuracy, surpassing similarly sized DS-CNNs (approximately 96.5%).
- **35-Class Task**: KWT also achieved competitive accuracy, reaching optimal performance with approximately 200k parameters.
- **No Pre-training**: All results were obtained from scratch training, without large-scale pre-training data.

### Comparison with Other Methods
- **KWT vs DS-CNN**: KWT achieves approximately 1% higher accuracy at the same parameter count.
- **KWT vs Att-RNN**: KWT achieves similar or higher accuracy with fewer parameters.
- **KWT vs Larger Models**: The accuracy of KWT-3 (approximately 500k parameters) is close to that of models with 10 times the number of parameters.

### Impact of Patch Size
- Smaller patches (e.g., 16x16) produce more sequence tokens, providing finer-grained features but increasing computational cost.
- Larger patches (e.g., 32x32) reduce sequence length, lowering computational cost but potentially losing fine-grained information.
- For KWS (1-second audio), medium-sized patches achieve the best balance between accuracy and efficiency.

### Generalization Ability
- KWT demonstrates better generalization performance on unseen speakers compared to CNN baselines, which may be attributed to the global prosodic patterns captured by self-attention.

## Limitations and Future Work

### Technical Limitations
- **Fixed-Size Input**: KWT requires input spectrograms of fixed length, making it unable to directly process streaming audio. This limits its application in continuous listening scenarios.
- **Quadratic Attention Complexity**: The computational complexity of self-attention is $O(n^2)$, and the computational load grows rapidly as the number of patches increases. However, in KWS scenarios, the sequence length is typically short (~50 patches), so the practical impact of this limitation is small.
- **Data Efficiency**: Compared to CNNs with strong inductive biases, KWT may require more training data or more data augmentation to achieve optimal performance.
- **Inference Memory Footprint**: The storage of key-value pairs in Transformers consumes more memory than lightweight CNNs.

### Experimental Design Shortcomings
- Patch size selection requires tuning for different keyword durations.
- Evaluation was not conducted in noisy, far-field, or custom keyword scenarios.
- Performance after model quantization was not explored.

### Future Improvement Directions
- Adapt causal/streaming Transformers for real-time KWS inference.
- Explore efficient linear attention alternatives to standard attention.
- Combine CNN inductive biases (e.g., ConvNet stem) to improve data efficiency in small-data scenarios.
- **Implications for the KWS Field**: The success of KWT proves the feasibility of pure Transformers in KWS, laying the foundation for subsequent research such as Conformer-KWS and Streaming Transformers.
