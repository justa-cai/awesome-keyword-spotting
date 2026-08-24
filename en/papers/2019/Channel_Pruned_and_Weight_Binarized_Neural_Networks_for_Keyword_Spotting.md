# Channel Pruned and Weight Binarized Neural Networks for Keyword Spotting

- **Authors/Affiliations**: Jiancheng Lyu (UC Irvine), Spencer Sheen (UC San Diego)
- **Date**: September 2019 (arXiv)
- **Link**: https://arxiv.org/abs/1909.05623
- **Keywords**: Keyword Spotting, Channel Pruning, Weight Binarization, Model Compression, Group Lasso, CNN, Structured Sparsity

## Problem Statement

Deploying neural network-based Keyword Spotting (KWS) systems on resource-constrained mobile and embedded devices requires significant model compression to meet strict memory, computational, and energy constraints. As a "always-on" feature that needs to run continuously, keyword detection imposes extremely high demands on model lightweighting.

Existing model compression methods have their respective limitations:

1. **Limitations of Pruning Alone**: Unstructured pruning (setting individual weights to zero) produces sparse models, but achieving actual acceleration requires specialized sparse matrix operation hardware; its effectiveness on general-purpose hardware is limited. Structured pruning (such as channel pruning) directly reduces computational load, but determining the optimal pruning structure remains challenging.
2. **Limitations of Quantization Alone**: Traditional quantization methods (such as INT8 quantization) reduce model size and accelerate inference, but the compression ratio is still insufficient in extremely resource-constrained scenarios. Extreme quantization, such as binarization (1-bit weights), can significantly compress the model, but using it alone leads to a substantial drop in accuracy.
3. **Underutilized Synergy of Combined Compression**: Structured pruning and weight binarization are complementary compression strategies—pruning reduces the number of channels (reducing computation), while binarization reduces the bit-width of each weight (reducing storage and accelerating multiplication). However, there is a lack of in-depth research on how to systematically combine the two to achieve the best compression-performance trade-off.

Therefore, the core challenge is: to design a systematic training pipeline that effectively combines structured channel pruning and weight binarization, achieving substantial model compression while keeping KWS accuracy almost unchanged.

## Methodology

This paper proposes a **Three-Stage Training Pipeline** that systematically combines channel pruning based on the Relaxed Group-wise Splitting Method (RGSM) with weight binarization.

### 1. Baseline CNN Architecture

The baseline model is a simple CNN architecture containing:
- Input: Single-channel acoustic features (e.g., MFCC or spectrogram)
- Example convolutional layer configuration: 8x20_c1_f64_s1 denotes kernel size 8x20, input channels 1, number of filters 64, stride 1
- Pooling layer: MaxPool2d
- Fully Connected (FC) layer + ReLU activation
- Output layer: Keyword classification

### 2. Stage I: Group Lasso Channel Pruning Based on RGSM

#### 2.1 Group Lasso Penalty

Weights in the convolutional layer are grouped by channel, and the Group Lasso (GL) penalty is defined as:

$$\|w\|_{GL} = \sum_{g=1}^{G} \|w_g\|_2$$

where $G$ is the number of groups, and $w_g$ is the weight vector of the $g$-th group. The purpose of Group Lasso is to drive the weights of certain groups toward zero as a whole, thereby achieving **structured channel-level sparsity**.

#### 2.2 Relaxed Variable Splitting Method (RGSM)

Directly adding the GL penalty to the training objective ($\ell(w) + \mu P(w)$) yields poor results. This paper adopts the Relaxed Group-wise Splitting Method, constructing an augmented Lagrangian function by introducing an auxiliary variable $u$:

$$\mathcal{L}_\beta(u, w) = \ell(w) + \mu P(u) + \frac{\beta}{2}\|w - u\|_2^2$$

where $\ell(w)$ is the cross-entropy loss, $P(u)$ is the Group Lasso penalty, and $\beta > 0$ is the augmented Lagrangian parameter. RGSM alternately optimizes $u$ and $w$:

- **Update $u$**: Solved via the proximal operator of Group Lasso; groups with small norms are directly set to zero:

$$u_g^{t+1} = \text{Prox}_{GL,\lambda}(w_g^t) = \max\left(0, 1 - \frac{\lambda}{\|w_g^t\|_2}\right) \cdot w_g^t$$

- **Update $w$**: Updated using standard Stochastic Gradient Descent (SGD):

$$w^{t+1} = w^t - \eta \nabla_w \mathcal{L}_\beta(u^{t+1}, w^t)$$

The convergence of RGSM is guaranteed by Theorem 1: under the conditions that the loss function is bounded from below and satisfies Lipschitz gradient conditions, the iterative sequence converges to a critical point of the augmented Lagrangian function, which satisfies the Group Lasso proximal condition and the gradient balance condition.

#### 2.3 Pruning Operation

After Stage I training is completed, channels with norms below a threshold are pruned—removed entirely—to generate a compact network with reduced channel count.

### 3. Stage II: Accuracy Recovery Retraining

Full-precision (float32) weight retraining is performed on the pruned compact network. The key aspects of this stage are:
- Keep the post-pruning channel structure unchanged (do not restore pruned channels)
- Fine-tune only the full-precision weights of the remaining channels
- The goal is to **recover accuracy loss caused by pruning**, restoring the compact network's performance to be close to that of the original network

### 4. Stage III: Weight Binarization

Based on the full-precision compact network from Stage II, weights are binarized to 1-bit precision. Binarized weights take the form of **floating-point scalar x sign vector**:

$$w \approx \alpha \cdot \text{sign}(w)$$

where $\alpha$ is the floating-point scalar coefficient. Binarized weights allow convolution operations to be converted into efficient bitwise operations (XNOR + popcount), achieving significant acceleration on hardware supporting bitwise operations. Training for weight binarization involves a projection operator:

$$\text{proj}_{Q,w}(w) := \text{argmin}_{z \in Q} \|z - w\|$$

where $Q = \mathbb{R}_+ \times \{\pm 1\}^D$. Stage III uses the full-precision weights from Stage II as a **warm start** initialization, fine-tuning weights through binarization-aware training.

### 5. Overall Effect of the Three-Stage Process

| Stage | Operation | Purpose |
|------|------|------|
| Stage I | RGSM + Group Lasso training → Pruning | Identify and remove unimportant channels |
| Stage II | Full-precision retraining (post-pruning) | Recover accuracy loss |
| Stage III | Weight binarization training | Further compress storage and accelerate inference |

## Main Contributions

1. **Systematic Three-Stage Compression Pipeline**: Proposes for the first time a three-stage training pipeline that systematically combines RGSM channel pruning and weight binarization. Each stage has clear objectives and optimization strategies, rather than simply chaining the two techniques together.
2. **Efficient Structured Sparsity Method**: Adopts the Relaxed Group-wise Splitting Method (RGSM) instead of direct Group Lasso training, achieving more efficient channel sparsification on the KWS task. Experiments prove that the direct GL method performs poorly on KWS CNNs, making RGSM the better choice.
3. **Complementary Compression Strategies**: Demonstrates that structured channel pruning and weight binarization are complementary—the former reduces the number of computational channels (reducing multiply-accumulate operations), while the latter reduces the bit-width of each weight (turning multiplication into bitwise operations). Their combination achieves "double compression."
4. **Extremely High Compression Efficiency**: Achieves over 50% channel sparsity with an accuracy loss of less than 0.25%, proving that substantial compression can be achieved with almost no impact on KWS performance.
5. **Hardware-Friendly Inference**: Binarized weights enable doubling of inference speed on mobile devices such as the Samsung Galaxy J7 using standard TensorFlow functions (conv2d, matmul).

## Experimental Results

### Dataset
- Google Speech Commands Dataset (keyword classification task)
- Classification accuracy across multiple keyword categories serves as the evaluation metric

### Main Results

| Method | Channel Sparsity | Accuracy Change |
|------|-----------|-----------|
| Original Full-Precision CNN | 0% | Baseline |
| Stage I (RGSM Pruning) | >50% | Moderate Drop |
| Stage II (After Retraining) | >50% | Close to Original Accuracy |
| Stage III (+ Binarization) | >50% + 1-bit Weights | <0.25% Loss |

- The combined method controls final accuracy loss within 0.25% while maintaining channel sparsity exceeding 50%
- Direct Group Lasso regularization (non-RGSM) alone fails to effectively achieve channel sparsification
- Full-precision retraining in Stage II is crucial for accuracy recovery—skipping this stage and binarizing directly leads to greater accuracy degradation

### Hardware Acceleration Validation
- On the Samsung Galaxy J7 phone, weight binarization alone improved inference speed by approximately 2x
- Channel pruning further reduced computational load by 50%
- The combination of both achieved an overall inference acceleration of approximately 4x

## Limitations and Future Work

### Technical Limitations
- **Time and Complexity of Three-Stage Training**: The three-stage process significantly increases total training time—each stage requires an independent training process, and dependencies between stages increase the difficulty of hyperparameter tuning (e.g., $\mu$ for Group Lasso, $\beta$ for RGSM, learning rate for binarization, etc.).
- **Limitations of Binarization on Complex Tasks**: Although binarization causes only a 0.25% accuracy loss on simple KWS classification tasks, it may lead to greater performance degradation in more complex KWS tasks (more keyword categories, noisy environments, multi-speaker scenarios).
- **Insufficient Hardware-Specific Optimization**: Although acceleration on the Samsung Galaxy J7 was validated, hardware-specific optimizations for binarized weights (such as dedicated XNOR-Net accelerators, FPGA implementations) have not been fully explored.

### Experimental Design Limitations
- Evaluated only on the Google Speech Commands Dataset, lacking validation on more challenging wake-word detection tasks (which require handling continuous audio streams and false alarm control).
- Lacks direct comparison with other contemporary compression methods (such as knowledge distillation, neural architecture search).
- No systematic parametric analysis of the accuracy-efficiency trade-off under different compression ratios.

### Future Directions
- Explore extending the three-stage process to more complex KWS architectures (such as CRNN, attention models) to verify the generalizability of the method.
- Investigate the possibility of joint optimization (performing pruning and binarization simultaneously within a unified training framework) rather than sequential execution to reduce training time.
- Combine Quantization-Aware Training (QAT) techniques to explore the performance of finer quantization granularities (such as 2-bit, 4-bit) after channel pruning.
- Evaluate the actual acceleration effects of combined compression on dedicated hardware (such as NPU, DSP), establishing a complete workflow from model compression to hardware deployment.
- Explore adaptive pruning strategies—dynamically adjusting pruning ratios based on the importance of different layers and computational bottlenecks.
