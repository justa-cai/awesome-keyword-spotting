# Depthwise Separable Convolutional ResNet with Squeeze-and-Excitation Blocks for Small-footprint Keyword Spotting

**Authors/Affiliations**: Yiming Zou, Xiaodong Wei, Xiaowei Qin (Northwestern Polytechnical University)

**Date**: April 2020 (arXiv:2004.12200)

**Link**: https://arxiv.org/abs/2004.12200

**Keywords**: Depthwise Separable Convolution, ResNet, Squeeze-and-Excitation, Keyword Spotting, Small-footprint Model

## Problem Statement

Small-footprint KWS models need to achieve the highest possible detection accuracy within extremely limited parameter counts and computational budgets. The computational complexity of traditional convolution operations is $O(C_{in} * C_{out} * K * K)$, where $C_{in}$ and $C_{out}$ are the number of input and output channels, and $K$ is the kernel size. The high computational cost of standard convolutions limits the expansion of model depth and width.

Depthwise Separable Convolution reduces the computational cost by approximately 8-9 times by decomposing standard convolution into depthwise convolution and pointwise convolution. However, simply replacing convolution operations may not be sufficient to fully leverage the model's expressive power.

Core Question: How to enhance the feature representation capability of small models while maintaining low computational complexity? Channel attention mechanisms (Squeeze-and-Excitation) represent a promising direction—by learning channel-level attention weights, the model can automatically focus on the most informative feature channels.

## Methodology

### Overall Architecture Design

This paper proposes DS-ResNet-SE, which integrates three key components:

**1. Depthwise Separable Convolution**:
- **Depthwise Conv**: Performs spatial convolution independently for each input channel, with computational complexity $O(C_{in} * K * K)$.
- **Pointwise Conv**: Uses 1x1 convolution to achieve cross-channel information fusion, with computational complexity $O(C_{in} * C_{out})$.
- The total computational cost is reduced by approximately $1/C_{out} + 1/K^2$ compared to standard convolution.

**2. Residual Connection**:
- **Skip Connection**: $y = F(x) + x$, where $F(x)$ is the residual mapping.
- Mitigates the vanishing gradient problem in deep networks.
- Enables the network to learn incremental features rather than complete feature mappings.
- Supports the construction of deeper networks while maintaining training stability.

**3. Squeeze-and-Excitation (SE) Attention Module**:
- **Squeeze**: Global average pooling compresses the features of each channel into a single scalar value, producing a channel descriptor.
- **Excitation**: A two-layer fully connected network learns non-linear relationships between channels:
  - FC1: Dimensionality reduction ($C \rightarrow C/r$, where $r$ is the reduction ratio), using ReLU activation.
  - FC2: Dimensionality expansion ($C/r \rightarrow C$), using Sigmoid activation.
  - The output is the attention weight for each channel (between 0 and 1).
- **Reweighting**: Multiplies the attention weights with the original features channel-wise.

### Network Structure

A typical DS-ResNet-SE Block:
1. 1x1 convolution to expand channels (optional)
2. Depthwise separable convolution to extract spatial features
3. SE module for channel attention weighting
4. 1x1 convolution to project back to the original dimension
5. Residual connection

### Hyperparameter Configuration

- **Network Depth**: Stacking multiple DS-ResNet-SE blocks.
- **Number of Channels**: Increasing layer by layer (e.g., 64 -> 128 -> 256).
- **SE Reduction Ratio**: $r=4$ or $r=8$.
- **Total Parameters**: Controlled to be within approximately 200K.

## Main Contributions

1. **Integration of SE Attention with DS-ResNet**: This is the first work to integrate the Squeeze-and-Excitation attention mechanism into a Depthwise Separable Convolution ResNet for the KWS task. The SE module enables small models to automatically learn channel-level feature importance, enhancing the weights of discriminative features.

2. **Channel Attention Enhances Small Models**: It is demonstrated that even models with very few parameters can effectively enhance feature representation through attention mechanisms. The SE module introduces very few additional parameters (approximately 1-2%), but brings significant accuracy improvements.

3. **Residual Connections Support Deep Networks**: Combined with residual connections, small-footprint networks can stack more layers without experiencing training degradation, increasing the model's non-linear expressive power.

4. **Systematic Ablation Study**: A detailed analysis of the independent contributions of each component (DS convolution, SE module, residual connection) validates the necessity of combining all three.

## Experimental Results

### Experimental Setup
- **Dataset**: Google Speech Commands dataset (12 classes).
- **Baselines**: Standard CNN, DS-CNN (without SE), DS-ResNet (without SE).
- **Evaluation Metric**: Classification accuracy.

### Main Results
- **DS-ResNet-SE vs. DS-ResNet**: The SE module brings an accuracy improvement of approximately 0.5-1%.
- **DS-ResNet-SE vs. DS-CNN**: The residual connection and SE module together improve accuracy by approximately 1-2%.
- **Parameter Count**: Total parameters are controlled to approximately 200K, suitable for embedded deployment.
- **SE Module Overhead**: The SE module adds only about 1-2% extra parameters, offering extremely high cost-effectiveness.

### Ablation Experiments
- **DS Convolution Only**: High computational efficiency but limited expressive power.
- **+ Residual Connection**: Deeper network training is more stable, with accuracy improvements.
- **+ SE Module**: Further significant improvements on top of residuals, especially in distinguishing difficult keyword pairs.
- **SE Reduction Ratio**: $r=4$ provides the best performance-complexity trade-off.

### SE Module Analysis
- Visualization of SE weights shows that SE modules in different layers focus on different feature channels.
- Shallow layer SE modules focus on spectral shape features.
- Deep layer SE modules focus on more abstract discriminative features.

## Limitations and Future Work

### Method Limitations
- **SE Module Computational Overhead**: Although the parameter increase is minimal, the SE module introduces additional fully connected layer computations.
- **Manual Architecture Design**: Hyperparameters such as network depth and number of channels require manual tuning.
- **Limited Evaluation Conditions**: Evaluation is primarily conducted on clean, single-channel near-field audio; effects in far-field and noisy conditions are not fully verified.
- **Generalization of SE Module**: It remains to be verified whether the SE module can still effectively identify informative channels under noisy conditions.

### Future Directions
- Investigate the effects of spatial-channel joint attention (e.g., CBAM) in KWS.
- Combine Neural Architecture Search (NAS) to automatically discover optimal DS-ResNet-SE configurations.
- Expand evaluation to far-field and noisy scenarios.
- Research lightweight attention alternatives (such as those used in EfficientNet).
- Explore the combination of temporal attention mechanisms with SE channel attention.
