# End-to-end Keyword Spotting using Neural Architecture Search and Quantization

- **Authors/Affiliations**: Martin Gjoreski, Ana Rebersek, Simon Ostermann, Matjaz Gams - Graz University of Technology
- **Date**: 2021.04
- **Link**: https://arxiv.org/abs/2104.06666
- **Keywords**: Neural Architecture Search, Quantization-Aware Training, DARTS, Keyword Spotting, End-to-end, Model Compression, INT8

## Problem Statement

Deploying Keyword Spotting (KWS) models on edge devices requires architectures that simultaneously satisfy two constraints: high classification accuracy and low computational/storage overhead. The traditional "design architecture first, then compress model" workflow suffers from two fundamental issues:

1.  **Mismatch between architecture and quantization**: Architectures optimized for floating-point precision may experience significant accuracy degradation when quantized to low precision (e.g., INT8, INT4). Certain architectural components (such as specific activation functions or attention mechanisms) are particularly sensitive to quantization errors.
2.  **Suboptimal design workflow**: Failing to consider quantization constraints during the architecture design phase results in compressed models that are far from the Pareto optimal frontier in terms of the accuracy-efficiency trade-off.

The core problem this paper addresses is: Can we automatically discover KWS architectures that are inherently quantization-friendly by jointly optimizing Neural Architecture Search (NAS) and Quantization-Aware Training (QAT), thereby achieving end-to-end optimization from architecture search to quantized deployment?

## Methodology

### Overall Framework
The end-to-end workflow proposed in the paper is as follows:
1.  Define a search space containing diverse components.
2.  Use Differentiable Architecture Search (DARTS) to search for the optimal architecture within the search space.
3.  Apply Quantization-Aware Training simultaneously during the search process, ensuring that the discovered architectures are naturally suitable for low-precision inference.
4.  Deploy the quantized model on the target hardware.

### Differentiable NAS (DARTS)
The paper employs DARTS (Differentiable Architecture Search) for architecture search:
-   **Continuous Relaxation**: Discrete architecture choices (e.g., selecting 3x3 or 5x5 convolutions) are transformed into continuous probability distributions (softmax weights), allowing architecture search to be optimized via standard gradient descent.
-   **Search Space**: A cell-based search space, where each cell is a Directed Acyclic Graph (DAG). Nodes represent feature maps, and edges represent candidate operations.
-   **Candidate Operation Set**:
    -   3x3 Depthwise Separable Convolution
    -   5x5 Depthwise Separable Convolution
    -   3x3 Dilated Convolution
    -   3x3 Max Pooling
    -   3x3 Average Pooling
    -   Identity Mapping (Skip Connection)
    -   Zero Operation (No Connection)
-   **Bi-level Optimization**: Alternating updates of architecture parameters ($\alpha$) and network weights ($w$), enabling the search process to accurately evaluate the contribution of each operation.

### Quantization-Aware Training (QAT)
Quantization is applied synchronously during the NAS search process:
-   **Weight Quantization**: Floating-point weights are quantized to INT8 (or lower precision). The Straight-Through Estimator (STE) is used to approximate the gradients of the quantization operation.
-   **Activation Quantization**: Intermediate feature values are also quantized to low precision.
-   **Quantization-Aware Inference Simulation**: The effects of quantization (quantization/dequantization of weights and activations) are simulated during the forward pass of training, allowing network weights to adapt to quantization errors during the learning process.
-   **Batch Normalization Folding**: Parameters of BN layers are folded into the weights of the preceding layer, reducing computation during inference.

### Joint NAS + Quantization Optimization
The key innovation lies in integrating Quantization-Aware Training into the search loop of NAS:
-   During each forward pass of DARTS, quantization simulation is applied to candidate operations simultaneously.
-   The update of architecture parameters $\alpha$ considers performance after quantization, rather than floating-point performance.
-   This ensures that the architectures discovered by the search maintain high performance even after quantization.

### Analysis of Discovered Architectures
Architectures discovered by the search typically exhibit the following characteristics:
-   A tendency to use Depthwise Separable Convolutions (small parameter count, quantization-friendly).
-   Less reliance on pooling operations (pooling is less sensitive to quantization but causes greater information loss in KWS tasks).
-   Moderate skip connections (maintaining gradient flow while avoiding chains of identity mappings that may be unstable under quantization).

## Main Contributions

1.  **Joint Optimization of Architecture and Quantization**: For the first time in the KWS domain, NAS and QAT are integrated into a unified optimization framework to discover architectures that are naturally quantization-friendly. This joint optimization avoids the suboptimality inherent in the traditional pipeline between the "architecture design" and "model compression" stages.
2.  **Discovery of Quantization-Friendly Architectures**: The discovered architectures suffer almost no accuracy loss after INT8 quantization, demonstrating better quantization robustness compared to baseline methods that design first and quantize later.
3.  **End-to-End Pipeline from Search to Deployment**: Provides a complete toolchain from NAS search -> QAT fine-tuning -> quantized model export, offering practical engineering value.
4.  **Surpassing Hand-Designed Models**: Architectures discovered by joint NAS+QAT outperform classic hand-designed models (such as DS-CNN) in both accuracy and efficiency.

## Experimental Results

### Datasets
-   **Google Speech Commands v2**: 12-class classification task.
-   **Evaluation Metrics**: Classification accuracy, model size (KB), number of multiply-accumulate operations (MADD).

### Floating-Point vs. Quantized Performance
-   **INT8 Quantization**: Architectures discovered by joint NAS+QAT suffer less than 0.5% accuracy loss after INT8 quantization.
-   **Comparison with Traditional Methods**: Standard DS-CNN experiences a 1-2% accuracy drop after INT8 quantization, indicating that jointly optimized architectures are more quantization-friendly.
-   **Lower Precision (INT4)**: INT4 quantization leads to significant accuracy degradation (5-10%), suggesting that INT8 is currently the safe lower bound for KWS quantization.

### Comparison with Baseline Models
-   Architectures discovered by NAS+QAT achieve competitive accuracy (approx. 95-96%) on Google Speech Commands.
-   At the same accuracy level, model size and MADD are reduced by 20-30% compared to standard DS-CNN.
-   Joint NAS+QAT outperforms the sequential method of "NAS search for floating-point architecture, followed by quantization."

### Ablation Studies
-   **Joint vs. Sequential Optimization**: Architectures from joint optimization show approximately 0.5-1% higher accuracy after quantization compared to sequential optimization.
-   **Impact of Search Space**: A larger search space (more candidate operations) can discover better architectures, but at the cost of longer search time.
-   **Impact of Quantization Precision**: Lossless transition from FP32 to INT8; noticeable degradation begins at INT4.

## Limitations and Future Work

### Technical Limitations
-   **Search Computational Cost**: Although DARTS is more efficient than evolutionary search methods, the joint NAS+QAT search process still requires several days to weeks of GPU time.
-   **Extremely Low Precision Quantization**: Accuracy drops remain significant for INT4 and binary quantization; joint optimization has not yet fully resolved this issue.
-   **Search Space Constraints**: Emerging architectural elements such as attention mechanisms and Transformer components are not included.

### Experimental Design Shortcomings
-   Lack of real-world inference time measurements on actual edge hardware (e.g., MCUs, DSPs).
-   Streaming/continuous KWS processing requirements are not considered.
-   Comparison with other contemporary NAS methods (such as Once-for-All NAS) is not comprehensive enough.

### Directions for Future Improvement
-   Explore hardware-aware NAS, using measured latency from target hardware directly during the search process.
-   Expand the search space to include mixed-precision quantization (different layers using different precisions).
-   Investigate zero-cost NAS proxies to further accelerate the search.
-   **Implications for the KWS Domain**: Architecture design and model compression should be jointly optimized rather than executed sequentially; this philosophy applies to all speech models requiring deployment on edge devices.
