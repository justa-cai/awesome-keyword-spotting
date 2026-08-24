# Broadcasted Residual Learning for Efficient Keyword Spotting

- **Authors/Affiliations**: Bum Jun Kim, Sang Hyuk Chang - Qualcomm AI Research / KAIST
- **Date**: 2021.06
- **Link**: https://arxiv.org/abs/2106.04140
- **Keywords**: Broadcasted Residual Learning, BC-ResNet, Efficient Model, 1D Temporal Convolution, Depthwise Separable Convolution, Google Speech Commands

## Problem Statement

In the field of Keyword Spotting (KWS), models must satisfy strict resource constraints on edge devices—low memory footprint, low computational cost, and low power consumption—while maintaining high accuracy. In recent years, depthwise separable convolutions (such as those in the MobileNet series) and channel-frequency decomposition strategies have been widely used to build lightweight KWS models. However, the design of residual connections still has room for optimization.

Standard residual connections (identity mapping or projection shortcuts introduced in ResNet) require dimension matching via 1x1 convolutions (projection layers) when input and output channel dimensions are inconsistent. This introduces additional parameters and computational overhead. In applications like KWS, which are extremely sensitive to parameter efficiency, these extra costs are non-negligible. For example, when the channel expansion ratio is 2, the number of parameters in the projection layer can account for 10-15% of the entire residual block.

Furthermore, traditional two-dimensional convolutions perform convolution along both time and frequency dimensions simultaneously. However, the time-frequency structure of speech signals is special: the time dimension carries critical temporal patterns (phoneme sequences, prosodic changes), while the frequency dimension primarily provides feature channel information. Treating both equally may lead to a waste of computational resources, as local convolution in the frequency dimension contributes relatively little to the KWS task.

The core problem this paper addresses is: How to design a parameter-efficient residual connection method while optimizing the convolution strategy for time-frequency dimensions, enabling KWS models to achieve higher accuracy with fewer parameters. This issue is not only academically significant but also has practical engineering value for deploying KWS models on ultra-low-power MCUs (Microcontrollers).

## Methodology

### Overall Architecture Design - BC-ResNet
The paper proposes BC-ResNet (Broadcasted Residual Learning Network), the core idea of which is to decompose standard two-dimensional convolution into one-dimensional convolution along the time dimension and a broadcast operation along the frequency/channel dimension. The overall architecture follows the standard KWS pipeline: MFCC feature input -> stacked BC-ResNet modules -> Global Average Pooling -> Fully Connected classification head.

The design philosophy of BC-ResNet is "to concentrate computational resources on the most critical information dimension"—for the KWS task, the time dimension is more critical than the frequency dimension, so most of the computational load is invested in modeling the time dimension.

### Broadcasted Residual Learning (BRL)
The core innovation of BRL lies in replacing the projection layer in the residual connection with a broadcast (Broadcast) operation:

**Standard Residual Connection**: $output = F(x) + Projection(x)$
Where $Projection(x)$ is usually a 1x1 convolution, requiring additional parameters to match the output dimension of $F(x)$. When the number of channels changes from $C_{in}$ to $C_{out}$, the number of parameters in the projection layer is $C_{in} * C_{out}$.

**Broadcasted Residual Connection**: $output = F(x) + Broadcast(x)$
Where $Broadcast(x)$ matches the dimension of $x$ to the output dimension of $F(x)$ through simple tensor replication/expansion operations (without learnable parameters).

Specifically, when the input tensor $x$ has shape $(B, T, C_{in})$ and $F(x)$ outputs $(B, T, C_{out})$, the broadcast operation replicates $x$ along the channel dimension to become $(B, T, C_{out})$. If $C_{in}$ divides $C_{out}$ evenly, each channel is replicated $C_{out}/C_{in}$ times; otherwise, zero-padding is used to match dimensions.

Mathematical representation: $Broadcast(x)_{b,t,c} = x_{b,t, c \mod C_{in}}$

The core advantage of this operation is that it is completely parameter-free and involves only memory copy operations in hardware, resulting in extremely low computational overhead. From a signal processing perspective, the broadcast operation assumes that different output channels require the same input information—which is reasonable in the context of KWS, as different output channels need to extract information from the same input channels in the initial stages.

### Time-Frequency Decomposed Convolution
Another key design of BC-ResNet is decomposing two-dimensional convolution into:

1. **1D Temporal Convolution**: Depthwise separable convolution along the time axis to capture the temporal dynamic patterns of keywords. The kernel size is typically 3 or 5, capable of capturing transition features between phonemes within a local time window. Each input channel is convolved independently along the time dimension, with the number of parameters being $C * K_t$.

2. **Frequency-wise Transform**: Information mixing in the frequency/channel dimension via pointwise convolution (1x1 convolution). The role of this step is to combine information from different frequencies along the channel dimension to form higher-level feature representations. The number of parameters is $C_{in} * C_{out}$.

The theoretical basis for this decomposition is: the temporal structure of speech signals (phoneme sequences, prosodic patterns, pitch contours) is more important than the frequency structure. By concentrating most of the computational load on modeling the time dimension and using lightweight pointwise convolutions to handle the frequency dimension, parameters can be utilized more efficiently. Compared to standard two-dimensional depthwise separable convolution, this decomposition reduces the number of parameters by approximately $K_f$ times ($K_f$ is the kernel size in the frequency dimension).

### Detailed Structure of BC-ResNet Module
Each BC-ResNet module contains the following computational steps:
1. Depthwise 1D Temporal Convolution, kernel size $K_t$
2. Batch Normalization + ReLU activation
3. Pointwise Convolution (i.e., 1x1 convolution), channel expansion
4. Batch Normalization
5. Broadcasted Residual Skip Connection
6. ReLU activation

The design of this structure ensures that the residual connection spans the entire transformation module, ensuring effective gradient flow during training, while the broadcast operation eliminates the parameter overhead of the projection layer.

### Model Scale Configurations
The paper provides different variants of BC-ResNet to accommodate different parameter budgets:
- **BC-ResNet-6**: Approximately 12K parameters, suitable for extremely resource-constrained devices
- **BC-ResNet-8**: Approximately 25K parameters, suitable for MCU-class devices
- **BC-ResNet-14**: Approximately 48K parameters, suitable for higher-performance edge devices
- **BC-ResNet-18**: Approximately 100K parameters, suitable for scenarios with ample resources

These variants achieve different parameter counts by adjusting the number of modules (6/8/14/18 BC-ResNet modules) and channel expansion factors.

## Main Contributions

1. **Introduction of Broadcasted Residual Learning for Parameter-Efficient KWS**: BRL replaces the learnable projection layer with a parameter-free broadcast operation, eliminating extra parameter overhead while maintaining the gradient flow advantages of residual connections. This is a concise and effective improvement of ResNet residual connections in the KWS domain. From an engineering perspective, BRL demonstrates the important insight that "not all dimension transformations require learning."

2. **Time-Frequency Decomposed Convolution Strategy**: Decomposing two-dimensional convolution into 1D temporal convolution and frequency/channel transformation better aligns with the time-frequency characteristics of speech signals (the dynamics of the time dimension are much higher than those of the frequency dimension), allowing the model to capture more effective features with fewer parameters. This design philosophy is consistent with concurrent work on BC-ResNet, further validating the effectiveness of specialized processing for the time dimension.

3. **SOTA Performance on GSC Dataset**: BC-ResNet achieved 98.0% and 98.7% top-1 accuracy on Google Speech Commands v1 and v2, respectively (12-class task), consistently outperforming previous methods at the same parameter count. These results were among the best on GSC in 2021.

4. **Open-Source Code Promoting Reproducibility**: The Qualcomm AI Research team released the official implementation (github.com/Qualcomm-AI-research/bcresnet), facilitating community reproduction and further research. The code is concise and clear, making it easy to integrate into existing KWS systems.

## Experimental Results

### Dataset and Evaluation Setup
- **Google Speech Commands (GSC) v1 and v2**: Standard 12-class and 35-class classification tasks
- **Evaluation Metrics**: Top-1 Accuracy, Parameters (K), Multiply-Accumulate Operations (MADD)
- **Input Features**: 40-dimensional MFCC, 49 time steps (corresponding to 1-second audio, 16kHz sampling rate, 25ms window, 10ms stride)
- **Training Configuration**: Cosine annealing learning rate scheduler, label smoothing 0.1

### Comparison with SOTA
- **GSC v1 (12 classes)**: BC-ResNet achieved 98.0% accuracy, surpassing all known methods at the time (including Att-RNN's 97.7% and DS-CNN's 97.5%).
- **GSC v2 (12 classes)**: BC-ResNet achieved 98.7% accuracy, consistently outperforming baselines such as DS-CNN, Att-RNN, and KWT under the same parameter budget.
- **Parameter Efficiency**: BC-ResNet-8 (approx. 25K parameters) achieved accuracy comparable to or higher than DS-CNN-S (approx. 24K parameters), validating the parameter efficiency advantages of broadcasted residuals and time-frequency decomposition.
- **Computational Efficiency**: Due to the replacement of 2D convolution with 1D temporal convolution, the number of MADD operations was also significantly reduced. The MADD for BC-ResNet-8 is approximately 5M, far lower than that of DS-CNN with similar accuracy (approx. 8M).

### Ablation Studies
- **Broadcast vs. Projection Residual**: Replacing the 1x1 projection convolution with the broadcast operation reduced parameters by approximately 15% while causing almost no drop in accuracy (difference less than 0.1%), proving the effectiveness of the broadcast operation in KWS. This indicates that dimension matching can be achieved through simple replication rather than learning.
- **1D vs. 2D Convolution**: The time-frequency decomposition strategy outperformed standard two-dimensional convolution in both parameter efficiency and accuracy. Replacing the 1D decomposition with 2D convolution increased parameters by approximately 30% but improved accuracy by no more than 0.2%.
- **BC-ResNet Variants of Different Scales**: From very small models (~10K parameters, approx. 96% accuracy) to medium models (~200K parameters, approx. 98.7% accuracy), the BC-ResNet series demonstrated excellent accuracy-parameter trade-off curves across different parameter budgets.
- **Type of Residual Connection**: Completely removing the residual connection led to a drop in accuracy of approximately 1-2%. The accuracy of standard projection residuals and broadcasted residuals was similar, but the former had more parameters.

### Deployment Characteristics
- The model is suitable for edge deployment, with fast inference speed and low memory footprint.
- The broadcast operation is implemented very efficiently in hardware (involving only memory copying, no computational operations) and does not become a bottleneck for inference.
- All model variants can run in real-time on MCU-level devices (inference time less than 10ms).

## Limitations and Future Work

### Technical Limitations
- **Upper Bound of Expressiveness of Broadcast Operation**: The broadcast operation is fixed (no learnable parameters). In scenarios requiring complex non-linear dimension transformations, its expressiveness may be inferior to learnable projection layers. For larger-scale tasks or longer speech segments, the simplifying assumption of the broadcast operation may no longer hold. However, the relative simplicity of the KWS task means this limitation has a minor impact.
- **Evaluation Only on GSC**: All experiments were limited to the Google Speech Commands dataset and were not validated in noisy, far-field, or custom keyword scenarios. The recording conditions of the GSC dataset are relatively clean, and it remains unclear whether broadcasted residuals remain effective under harsher acoustic conditions.
- **Insufficient Theoretical Analysis**: There is a lack of in-depth theoretical explanation for why the broadcast operation is so effective in KWS. Why can dimension matching in KWS tasks be achieved through simple replication? Is it related to some structural property of KWS features? These questions require further theoretical analysis.
- **Loss of Frequency Information**: 1D temporal convolution treats the frequency dimension as channels, potentially losing local structural information in the frequency dimension. For keywords requiring fine-grained frequency distinction (e.g., different tones), this simplification may lead to information loss.

### Experimental Design Shortcomings
- Not evaluated in streaming/continuous KWS scenarios; only fixed 1-second audio segment classification was evaluated.
- Lack of deployment performance data after quantization (impact of INT8, INT4 quantization on broadcast operations).
- No detailed comparison with concurrent efficient Transformer methods (such as KWT); only CNN and RNN baselines were compared.
- Not validated on KWS tasks in different languages (especially tonal languages like Chinese).

### Future Improvement Directions
- Extend the BRL concept to more complex speech tasks (such as ASR, speech enhancement) to verify the effectiveness of broadcasted residuals in larger models and more complex tasks.
- Further optimize edge deployment by combining model compression techniques (quantization, pruning, distillation) and study the performance retention of BC-ResNet after compression.
- Explore adaptive broadcast strategies—dynamically adjusting broadcast weights based on channel importance, rather than simple uniform replication. For example, a lightweight gating mechanism could be introduced to control the intensity of the broadcast.
- Combine the time-frequency decomposition idea of BC-ResNet with attention mechanisms to enhance global modeling capabilities while maintaining parameter efficiency.
- **Heuristics for the KWS Domain**: The design philosophy of time-frequency decomposition + parameter-free residual connections provides an effective paradigm for building ultra-lightweight KWS models. The simplicity of BC-ResNet (no complex modules, no special training tricks) makes it a strong candidate for a KWS baseline model.
