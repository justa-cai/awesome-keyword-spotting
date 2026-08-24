# Depth Pruning with Auxiliary Networks for TinyML

- **Authors/Affiliations**: Josen Daniel De Leon (University of the Philippines; Samsung Research Philippines), Rowel Atienza (University of the Philippines)
- **Date**: April 2022 (ICASSP 2022)
- **Link**: https://arxiv.org/abs/2204.10546
- **Keywords**: pruning, depth pruning, tinyML, keyword spotting, visual wakewords, model compression, microcontroller deployment

## Problem Statement

### Problem Background and Domain Pain Points
TinyML (Tiny Machine Learning) aims to deploy neural networks onto microcontrollers (MCUs) with extremely limited resources, typically ARM Cortex-M0/M4 series processors. The hardware constraints of these MCUs are severe: Cortex-M0 has only 32-64KB SRAM (for runtime activation values and weight caching) and 128-256KB Flash (for storing model parameters and code), with a clock speed of only 48MHz; Cortex-M4 is slightly more powerful, with 128-256KB SRAM, 512KB-1MB Flash, and a clock speed of 120-180MHz. Running neural networks on these platforms requires the model to simultaneously satisfy three hard constraints: (1) Flash storage must accommodate all model parameters (weights + biases + BatchNorm parameters); (2) SRAM must accommodate all runtime activation values (intermediate feature maps of each layer); (3) Inference latency must meet real-time requirements (KWS typically requires <100ms latency).

When pre-trained models exceed these constraints (e.g., MobileNetV1-0.25x with approximately 226K parameters requires about 902KB Flash for FP32 storage), model compression is the only way out. The goal of compression is not merely to reduce the number of parameters—the SRAM constraint (activation size) and latency constraint (computational volume) are equally important, and there are complex interdependencies among the three.

### Specific Shortcomings of Existing Methods
- **Hardware Mismatch of Magnitude Pruning**: Magnitude pruning creates sparse models by setting small weight values to zero, theoretically significantly reducing the number of non-zero parameters. However, the resulting unstructured sparse patterns require specialized sparse computing support to achieve actual acceleration. On general-purpose ARM Cortex-M series processors, implementing sparse convolution introduces a large number of conditional branches (if weight != 0 then multiply) and index operations (reading column indices of non-zero values from CSR/CSC formats), resulting in actual acceleration far below theoretical values. On Cortex-M0, the overhead of conditional branches is particularly severe (due to the lack of a branch predictor, every branch miss flushes the 3-stage pipeline), and sometimes sparse computation is even slower than dense computation.
- **Accuracy Collapse in Standard Depth Pruning**: Depth pruning (layer dropping) reduces model size by removing deeper layers of the network. The pruned model remains a standard dense network and can run directly on any MCU. However, simply removing deeper layers and attaching a new linear classifier head to shallow feature maps usually leads to a severe drop in accuracy (5-15%), because shallow feature maps lack sufficient semantic abstraction capability to support accurate classification decisions. Specifically, the shallow layers of a network tend to capture low-level features (such as edges, textures), which are insufficiently discriminative for classification.
- **Research Gap in Pruning Compact Models**: Most pruning research focuses on large models (e.g., ResNet-50 with 25M parameters, VGG-16 with 138M parameters), which have a large amount of redundant parameters (studies show that 50-90% of parameters in large CNNs can be safely pruned). However, for compact models already optimized for tinyML (e.g., MobileNetV1-0.25x with only about 22K parameters, DS-CNN with only about 42K parameters), the redundancy is inherently small, making pruning significantly more difficult and risky—every parameter has a greater impact on accuracy.

### Key Challenges Addressed by This Paper
How to effectively perform depth pruning on already highly compact tinyML models without requiring specialized hardware support (i.e., the pruned model remains a dense network), such that the pruned model can run in real-time on extremely low-power MCUs like ARM Cortex-M0, while maintaining acceptable accuracy (accuracy loss controlled within 1%).

## Methodology

### Overall Architecture Design and Design Motivation
The core insight of the "Auxiliary Network-Enhanced Depth Pruning" method proposed in this paper is: **The fundamental reason for accuracy drop due to depth pruning is not "loss of deep parameters," but "insufficient semantic expression capability of shallow feature maps to support classification."** Therefore, the solution is not simply to reduce parameter loss, but to enhance the classification capability of shallow features—this is the role of the auxiliary network.

The method consists of two stages:
1. **Depth Pruning**: Remove several layers from the end of the pre-trained model, selecting the intermediate layer feature maps as the new "end features."
2. **Auxiliary Network Attachment**: Attach a carefully designed small auxiliary network (rather than a simple linear classifier head) at the pruning point to extract more effective classification features from the intermediate layer feature maps.

### Mathematical Principles of Core Algorithms

**Depth Pruning Strategy**:
In sequential network architectures (such as MobileNetV1, DS-CNN), network layers gradually extract features from low-level to high-level from shallow to deep. Let the original network have $L$ layers, and the pruning depth $k$ represents retaining the first $k$ layers ($1, 2, ..., k$), removing all layers from the $(k+1)$-th layer onwards.

After pruning, the output feature map of the $k$-th layer is $F_k \in \mathbb{R}^{H_k \times W_k \times C_k}$ ($H_k, W_k$ are spatial dimensions, $C_k$ is the number of channels). For spectrogram inputs in KWS, $H_k$ corresponds to the time dimension, and $W_k$ corresponds to the frequency dimension.

The selection of pruning depth $k$ needs to balance model size and accuracy. The paper determines the optimal $k$ through layer-by-layer evaluation: for each candidate $k$, train the auxiliary network and evaluate validation set accuracy, selecting the $k$ with the best accuracy-model size trade-off.

**Auxiliary Network Architecture Design**:

The auxiliary network adopts a small network based on depthwise separable convolutions, containing 2-3 depthwise separable convolution blocks:

The structure of each block is:
$$\text{DWSConv}(C_{in}, C_{out}, k=3) \to \text{BatchNorm} \to \text{ReLU6} \to \text{PWConv}(C_{out}, C_{out}) \to \text{BatchNorm} \to \text{ReLU6}$$

Where DWSConv is depthwise separable convolution (depthwise + pointwise), $k=3$ is a 3x3 convolution kernel.

Finally, classification results are output through Global Average Pooling (GAP) and a fully connected layer.

**Parameter Control of Auxiliary Networks**: A key design constraint is that the auxiliary network itself must have an extremely small number of parameters (typically <5K parameters), so that the auxiliary network does not offset the parameter reduction brought by depth pruning. This is achieved by using small channel numbers (e.g., 32 or 64 output channels) and depthwise separable convolutions (parameter count is approximately $1/C_{out}$ of standard convolutions).

**Training Strategy - Two-Stage Training**:

Stage 1 (Auxiliary Network Training):
- Freeze the retained original network layers (parameters of the first $k$ layers remain unchanged)
- Train only the parameters of the auxiliary network
- Loss function: Standard cross-entropy $L_{CE}$
- Training epochs: Usually 20-30 epochs are sufficient for convergence
- Learning rate: 0.001 (Adam optimizer)
- The reason for rapid convergence in this stage is: the retained layers already provide good feature representations, and the auxiliary network only needs to learn how to extract classification information from these features.

Stage 2 (Optional Joint Fine-tuning):
- Unfreeze all layers (including retained layers and auxiliary network)
- Perform end-to-end fine-tuning with a smaller learning rate (0.0001)
- The purpose is to allow the retained layers and auxiliary network to optimize synergistically
- The paper finds that the improvement in the second stage is usually small (about 0.3-0.5%), but it helps to some extent for the KWS task.

### Technical Differences from Existing Methods
- Compared to magnitude pruning: The auxiliary network method produces a dense standard network that can run directly on any MCU without requiring sparse computing support. This is a fundamental advantage—the efficiency loss of sparse computing on MCUs is far greater than on GPUs/TPUs.
- Compared to standard depth pruning + linear head: The auxiliary network significantly improves the classification capability of intermediate layer features through additional non-linear transformations (2-3 layers of depthwise separable convolutions), improving accuracy by 3-8%. Intuitively, a linear classifier assumes that features are linearly separable in space, but intermediate layer feature maps usually retain complex spatial structures (such as time-frequency patterns in spectrograms), requiring non-linear transformations for effective classification.
- Compared to knowledge distillation: Does not require a large number of samples from the original training data, only a small amount of calibration data (e.g., 10% of the original training set) to train the auxiliary network. Knowledge distillation requires a complete training set and multiple rounds of training.
- Compared to Neural Architecture Search (NAS): Does not require an expensive search process; the architecture of the auxiliary network is fixed (only hyperparameters such as channel numbers are tuned).

### Experimental Setup
- **MLPerfTiny Benchmark**: A standardized benchmark in the tinyML field, maintained by the MLCommons organization, containing two tasks.
- **Target Hardware**: ARM Cortex-M0 (one of the lowest-end MCUs, 48MHz clock speed, no floating-point unit, no branch predictor).
- **Inference Framework**: TensorFlow Lite for Microcontrollers (TFLite Micro).

## Main Contributions

1. **Auxiliary Network-Enhanced Depth Pruning Paradigm**: First proposes replacing the simple linear classifier head with a carefully designed small auxiliary network, fundamentally solving the problem of accuracy drop caused by depth pruning. The core value of this design idea lies in redefining the problem of depth pruning—not "how to reduce parameter loss," but "how to enhance the classification capability of shallow features." The auxiliary network acts as a "shallow feature semantic enhancer."

2. **Extreme Parameter Reduction with Minimal Accuracy Cost**: Achieved 93% parameter reduction (from about 226K to about 16K) on the VWW task, with only a 0.65% accuracy cost. Achieved 28% parameter reduction on the KWS task, with a 1.06% accuracy cost. A 93% parameter reduction means the model's Flash usage drops from about 902KB to about 64KB, making deployment on Cortex-M0 (typically 128KB Flash) feasible instead of impossible.

3. **Real MCU Deployment Verification**: Performed real deployment and performance measurement on ARM Cortex-M0, rather than just reporting theoretical operations. Real-world results on Cortex-M0: VWW model shrunk by 4.7x, inference accelerated by 1.6x. Deployment at the Cortex-M0 level means this method can be scaled to almost all IoT devices (the cost of this level of MCU is less than $1).

4. **No Need for Specialized Hardware and Framework Modifications**: The pruned model is a standard dense network and can run on any MCU using standard CMSIS-NN or TensorFlow Lite for Microcontrollers libraries, without modifying the underlying inference engine.

## Experimental Results

### Datasets Used and Their Scales
- **Visual Wakewords (VWW)**: The standard visual wakeword task in MLPerfTiny, built on the COCO dataset, binary classification (person/no person). The training set consists of about 115,000 images (64x64 grayscale), and the test set consists of about 8,000 images.
- **Keyword Spotting (KWS)**: The standard KWS task in MLPerfTiny, based on Google Speech Commands V1, 12-class classification. The training set consists of about 51,000 1-second audio clips (MFCC features, 49x10 time-frequency maps), and the test set consists of about 6,800 audio clips.

### Definition and Rationale for Evaluation Metrics
- **Accuracy (%)**: Standard classification accuracy.
- **Parameters (K)**: Total number of model parameters, directly determining Flash storage usage.
- **Parameter Reduction Ratio (%)**: Percentage reduction in parameters relative to the original model.
- **Inference Latency (ms)**: Measured inference time on ARM Cortex-M0.
- **SRAM Usage (KB)**: SRAM size required for runtime activation values.

### Detailed Comparison with Baseline Methods and SOTA

**VWW Task (Visual Wakewords)**:

| Method | Parameters | Parameter Reduction | Accuracy | Accuracy Change |
|:---|:---:|:---:|:---:|:---:|
| Original MobileNetV1-0.25x | ~226K | - | 88.5% | baseline |
| Magnitude Pruning (20%) | ~181K | 20% | 87.5% | -1.0% |
| Standard Depth Pruning (Linear Head) | ~70K | 68.9% | 82.5% | -6.0% |
| **Auxiliary Network Depth Pruning (This Paper)** | **~16K** | **93%** | **87.85%** | **-0.65%** |

Key Comparison: Standard depth pruning achieves only 82.5% accuracy at a similar parameter count (about 70K), while auxiliary network depth pruning achieves 87.85% accuracy with fewer parameters (about 16K). The introduction of the auxiliary network allows for higher accuracy under more aggressive pruning—this subverts the traditional belief that "more pruning leads to lower accuracy."

**Measured Performance on Cortex-M0 (VWW)**:
- Flash Usage: Reduced from about 902KB to about 192KB (4.7x shrinkage)
- SRAM Usage: Reduced from about 80KB to about 40KB
- Inference Latency: Reduced from about 470ms to about 290ms (1.6x acceleration)
- Accuracy: 88.5% -> 89.5% (actually increased by 1%, possibly because the auxiliary network acted as a regularizer—the smaller model avoided overfitting of the original model)

**KWS Task (Keyword Spotting)**:

| Method | Parameters | Accuracy | Accuracy Change |
|:---|:---:|:---:|:---:|
| Original DS-CNN | ~42K | ~93.8% | baseline |
| **Auxiliary Network Depth Pruning** | **~30K** | **~92.7%** | **-1.06%** |

The improvement in the KWS task is smaller than in VWW (28% vs 93% parameter reduction). Analysis of reasons:
- DS-CNN itself is already a very compact model (only about 42K parameters), leaving limited pruning space.
- KWS task classification decisions rely more heavily on high-level temporal features, and the loss of these features due to depth pruning is harder to compensate for through the auxiliary network.
- When deployed on Cortex-M0 (using INT8 quantization), quantization errors further caused a 2.21% drop in accuracy.

### Findings from Ablation Studies

**Auxiliary Network vs. Linear Classifier Head**:
At the same pruning depth $k$, the auxiliary network is 3-8 percentage points more accurate than the linear head. This gap becomes more significant as the pruning depth increases—because shallower feature maps require non-linear transformations to extract classification information.

**Auxiliary Network Architecture Selection**:
Depthwise separable convolution auxiliary networks outperform pure fully connected auxiliary networks in both parameter efficiency and accuracy. Specifically, at the same parameter count (about 5K parameters), the DWS auxiliary network is about 2% more accurate than the FC auxiliary network.

**Necessity of Two-Stage Training**:
The first-stage strategy of freezing pre-trained layers + training only the auxiliary network converges faster (20 epochs vs 80 epochs) and achieves comparable final accuracy than end-to-end training. This is because freezing preserves the feature extraction capability of pre-training, avoiding "catastrophic interference" in joint training.

**Selection of Pruning Depth $k$**:
- VWW: $k$=7 (retaining the first 7 layers) works best, reducing the model from 13 original layers to 7 layers + auxiliary network.
- KWS: $k$=5 (retaining the first 5 layers), reducing the model from 6 original layers of DS-CNN to 5 layers + auxiliary network.
- After exceeding the optimal $k$, accuracy drops sharply (because feature maps that are too shallow lack sufficient semantic information).

## Limitations and Future Work

### Technical Limitations of the Method
- **Semantic Bottleneck of Shallow Features**: The assumption of depth pruning is that intermediate layer features already contain sufficient classification information. However, for some KWS tasks requiring fine-grained temporal modeling (such as distinguishing "yes" and "yet"—which differ only in the last approximately 50ms of phonemes), the high-level temporal patterns captured by deeper networks may be indispensable. Excessive depth pruning leads to irreversible loss of this fine-grained temporal discrimination capability—the auxiliary network can compensate to some extent, but cannot create information that does not exist in the original network.
- **Limitations of Sequential Architectures**: The method only applies to sequential network architectures (such as VGG, MobileNet, DS-CNN) that can be removed layer by layer from the end. For networks with dense residual connections (such as ResNet, EfficientNet), depth pruning would disrupt skip connections—because the output of the $k$-th layer may be referenced by multiple subsequent layers, removing the referencing layers would disrupt the residual addition operation. More complex handling strategies are required (such as pruning both ends of the skip connection simultaneously).
- **Limitations of Fixed Auxiliary Network Architecture**: The depthwise separable convolution architecture of the auxiliary network is manually designed and may not be the optimal choice for all pruning depths. Different pruning depths may require auxiliary networks of different complexities.

### Shortcomings in Experimental Design
- **Limited Improvement in KWS Task**: On the KWS task, parameters were reduced by only 28% (far lower than the 93% in VWW), and the accuracy drop was relatively more significant. The paper does not deeply analyze why there is such a large difference between KWS and VWW. Speculated reasons may be: (1) DS-CNN itself is already very compact (42K parameters), with much lower parameter redundancy than MobileNetV1-0.25x (226K parameters); (2) The feature hierarchy of the KWS task (from low-level spectral textures to high-level phoneme sequences) may be deeper than that of VWW (from low-level edges to high-level object contours), requiring more layers to extract.
- **Insufficient Dataset Coverage**: Experiments were only conducted on two tasks in MLPerfTiny, and were not validated on larger-scale or more diverse KWS datasets (such as datasets containing noise, multiple speakers, and far-field conditions).
- **Joint Effects of Quantization and Pruning**: The paper reports performance after INT8 quantization but does not systematically analyze the interaction effects between quantization and depth pruning—quantization errors may be amplified on the shallow features after pruning.

### Possible Directions for Future Improvement
- **Task-Aware Adaptive Pruning Depth**: Automatically determine the optimal pruning depth based on the characteristics of each task (such as the number of classification categories, acoustic similarity between categories). NAS or reinforcement learning can be used to search for the optimal $(k, \text{auxiliary network configuration})$ combination.
- **Combination with Width Pruning**: Perform depth pruning and width pruning (channel pruning) simultaneously to achieve more flexible model compression. For example, reduce the number of channels in the retained layers as well, further reducing computational volume.
- **Knowledge Distillation Enhancement for Auxiliary Networks**: Use the original complete model as a teacher and the auxiliary network as a student, providing richer supervision signals through knowledge distillation (not only hard labels, but also feature representations of intermediate layers).
- **Inspiration for the KWS Field**: The idea of the auxiliary network indicates that for already compact KWS models, instead of simply "reducing parameters" (pruning), it is better to "reconstruct the classifier"—using a more sophisticated classifier head to compensate for the lack of feature extraction capability. This "reconstruction rather than deletion" approach provides a new perspective for KWS model compression.
