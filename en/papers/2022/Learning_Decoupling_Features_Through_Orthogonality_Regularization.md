# Learning Decoupling Features Through Orthogonality Regularization

- **Authors/Affiliations**: Li Wang, Rongzhi Gu (Shenzhen Graduate School, Peking University, ADSPLAB); Weiji Zhuang, Peng Gao, Yujun Wang (Xiaomi Corporation, Beijing); Yuexian Zou (Shenzhen Graduate School, Peking University)
- **Date**: March 2022
- **Link**: https://arxiv.org/abs/2203.16772
- **Keywords**: keyword spotting, orthogonality regularization, speaker verification, feature decoupling, multi-task learning, joint optimization

## Problem Statement

### Problem Background and Domain Pain Points
Keyword Spotting (KWS) and Speaker Verification (SV) are two fundamental and closely related tasks in speech interaction systems. In devices such as smart speakers and mobile assistants, KWS determines "what word was said" (e.g., wake-word detection for "Xiao Ai Tongxue"), while SV determines "who said it" (e.g., voiceprint authentication ensuring only the owner can wake the device). These two functions typically need to run simultaneously and continuously—each time the device is woken, the system must confirm "the correct wake word was spoken" (KWS) and verify "it was spoken by an authorized user" (SV).

From a signal processing perspective, a single speech signal simultaneously encodes two types of information: linguistic content information (what words are said, determined by phoneme sequences and prosodic patterns) and speaker identity information (who is speaking, determined by physiological features such as vocal tract length, vocal cord characteristics, and pronunciation habits). The human auditory system can naturally separate these two types of information, but whether artificial neural networks can also learn this separation is an important research question.

### Specific Shortcomings of Existing Methods
- **Resource Waste of Independent Training**: KWS and SV are trained independently using different datasets and architectures, failing to leverage complementary information between the two tasks. A deeper issue is: (1) KWS models may implicitly encode speaker information—making them more sensitive to specific speakers (e.g., if the training set is predominantly male, the KWS recognition rate for male users is higher), leading to unfair bias; (2) SV models may implicitly encode linguistic content information—making verification more accurate for specific words (e.g., performing better on words frequently appearing in training), leading to content-dependent bias.
- **Limitations of Unidirectional Information Flow**: Some joint methods use concatenation (splicing SV features after KWS features) or attention mechanisms (using SV features to weight KWS features) to pass information between tasks, but the information flow is unidirectional (e.g., SV features assist KWS), lacking bidirectional feature interaction and reciprocal learning. More importantly, unidirectional information flow cannot guarantee the "purification" of information—auxiliary information may contain noise (e.g., content information mixed into SV features may mislead KWS).
- **Coupling Issues in Dual-Attention Methods**: While learning shared representations for both KWS and SV, these methods do not explicitly decouple the feature spaces of the two tasks. As a result, the learned representations may be "mixed"—containing both linguistic content and speaker information. This coupling can lead to severe bias in practical applications (e.g., KWS systems being more sensitive to specific speakers—in cases with large amounts of male voice training data, the wake-up rate for female users may decrease significantly). More critically, coupled feature representations are more vulnerable to adversarial attacks—attackers can bypass the dual verification of SV+KWS by mimicking the target speaker's voice.
- **Limitations of Adversarial Decoupling**: Some methods use adversarial training (e.g., Gradient Reversal Layer) to remove specific information from features (e.g., removing speaker information from KWS features). However, adversarial training faces three fundamental difficulties: (1) Instability of min-max optimization—training the discriminator and generator requires precise balance, otherwise training diverges; (2) It only guarantees unidirectional decoupling (e.g., ensuring KWS features do not contain speaker information, but not ensuring SV features do not contain content information); (3) Adversarial loss is difficult to tune (the choice of $\lambda$ has a huge impact on training stability).

### Key Challenges Addressed by This Paper
How to extract shared underlying acoustic features for both tasks in a dual-branch network jointly trained for KWS and SV, while ensuring that the KWS branch features are "speaker-independent" (unaffected by speaker identity) and the SV branch features are "content-independent" (unaffected by what specific words are said), i.e., achieving true bidirectional feature decoupling. At the same time, the decoupling method needs to be stable in training, efficient in computation, and capable of guaranteeing bidirectional (rather than unidirectional) decoupling effects.

## Methodology

### Overall Architecture Design and Design Motivation
This paper proposes a dual-branch deep network with orthogonality regularization. The reason for choosing a symmetric dual-branch architecture (both branches using the same GRU structure) is: (1) Symmetric design allows the orthogonality constraint to operate on feature spaces of the same dimension, making the geometric meaning clearer; (2) Shared underlying convolutional layers can learn general acoustic features (such as spectral texture and short-term dynamics), which are underlying representations commonly needed for both KWS and SV tasks.

The overall architecture consists of three core parts:
1. **Shared Temporal Conv Layer**: 2-3 layers of 1D convolution to extract low-level acoustic features. The input is a 40-dimensional Mel-filterbank energy sequence. The design motivation for the shared layer is: low-level acoustic features (such as the shape of the spectral envelope and patterns of short-term energy change) are fundamental and general for both KWS and SV.
2. **KWS Branch**: 2-3 layers of GRU (hidden dimension 128) to extract keyword-discriminative features from shared features. Outputs a 128-dimensional KWS feature vector $f_{KWS}$.
3. **SV Branch**: A GRU network with the same structure as the KWS branch to extract speaker-discriminative features from shared features. Outputs a 128-dimensional SV feature vector $f_{SV}$.

The key constraint for the two branches is that their output feature vectors must be orthogonal.

### Mathematical Principles of the Core Algorithm: Orthogonality Regularization

**Definition of Orthogonality**:
Let the output feature vector of the KWS branch for the $i$-th sample in the training batch be $f_{KWS}^i \in \mathbb{R}^d$, and the output feature vector of the SV branch be $f_{SV}^i \in \mathbb{R}^d$. The orthogonality constraint requires that these two sets of feature vectors be statistically orthogonal to each other:

$$L_{orth} = \left\| \frac{1}{N} \sum_{i=1}^{N} f_{KWS}^i \cdot (f_{SV}^i)^T \right\|_F^2$$

where $\|\cdot\|_F$ is the Frobenius norm, and $N$ is the batch size.

**In-depth Analysis of Intuitive Explanation**:

In linear algebra, two vectors $\mathbf{u}$ and $\mathbf{v}$ are orthogonal if $\mathbf{u}^T \mathbf{v} = 0$, meaning their inner product is zero. The inner product $\mathbf{u}^T \mathbf{v} = \sum_{j=1}^d u_j v_j$ measures the "degree of linear correlation" between the two vectors—zero inner product means the two vectors have no "covarying" relationship in any dimension.

Extending this concept to a set of feature vectors: $L_{orth}$ measures the statistical correlation between KWS features and SV features within the batch. When $L_{orth} = 0$:
- KWS features $f_{KWS}$ do not contain any information linearly correlated with speaker identity (otherwise they would have a positive inner product with $f_{SV}$)
- SV features $f_{SV}$ do not contain any information linearly correlated with linguistic content (otherwise they would also have a positive inner product with $f_{KWS}$)
- The two feature spaces are "statistically independent"—knowing the value of one feature vector allows no inference about any information in the other feature vector

**Why Orthogonality is an Appropriate Constraint—Analysis from an Information Theory Perspective**:
- In information theory, the mutual information $I(X; Y) = 0$ between two random variables $X$ and $Y$ implies they are completely independent. Linear orthogonality guarantees that the linear correlation between $X$ and $Y$ is zero, which is a necessary condition for mutual information to be zero (though not a sufficient condition—higher-order statistical dependencies may still exist).
- The orthogonality constraint is "well-behaved" in optimization: $L_{orth}$ is a quadratic function of the inner product of two feature vectors, and its gradient $\nabla L_{orth}$ is well-defined and continuous everywhere (unlike the discontinuous gradients in min-max optimization of adversarial training).
- Orthogonality guarantees "information compression efficiency" of features—in a $d$-dimensional space, a set of orthogonal vectors has the best information compression efficiency (each dimension carries unique information with no redundancy).

**Comparison with Adversarial Decoupling**:
Adversarial decoupling (e.g., training a "speaker discriminator" using a Gradient Reversal Layer to remove speaker information from KWS features) requires training an additional discriminator network and optimizing within a min-max game. The advantages of orthogonality regularization are:
1. No additional discriminator network is needed—only the inner product of two feature vectors needs to be calculated.
2. Training is more stable—there is no oscillation problem associated with min-max optimization.
3. It simultaneously guarantees bidirectional decoupling—$f_{KWS} \perp f_{SV}$ is equivalent to $f_{SV} \perp f_{KWS}$.

### Complete Form of the Loss Function

The total loss function consists of three parts:

$$L_{total} = L_{KWS} + L_{SV} + \lambda \cdot L_{orth}$$

where:

$L_{KWS} = L_{CE}^{KWS}(y_{kw}, \hat{y}_{kw}) + \alpha \cdot L_{triplet}^{KWS}$

$L_{SV} = L_{CE}^{SV}(y_{spk}, \hat{y}_{spk}) + \alpha \cdot L_{triplet}^{SV}$

**Role of Triplet Loss**:
For KWS: $L_{triplet}^{KWS} = \max(0, d(f_a, f_p) - d(f_a, f_n) + m)$, where $f_a$ is the anchor (embedding of a certain keyword), $f_p$ is the positive (embedding of the same keyword), $f_n$ is the negative (embedding of a different keyword), $d(\cdot,\cdot)$ is the Euclidean distance, and $m$ is the margin.

The role of triplet loss is to pull embeddings of the same class closer and push embeddings of different classes further apart, forming compact and separated class clusters in the embedding space. This complements cross-entropy loss (which optimizes classification boundaries in logit space)—cross-entropy optimizes the "classification surface," while triplet loss optimizes the "embedding space structure."

### Technical Differences from Existing Methods
- Compared to independent training: Joint training + orthogonality allows KWS and SV to mutually enhance each other, and KWS features do not contain speaker bias, while SV features do not contain content bias.
- Compared to concatenation/attention methods: The orthogonality constraint provides an explicit mathematical guarantee (feature independence), rather than implicit feature mixing.
- Compared to adversarial decoupling: Orthogonality regularization is simpler (no discriminator needed), training is more stable (no min-max optimization), and it simultaneously guarantees bidirectional decoupling.

## Main Contributions

1. **Orthogonality Regularization for KWS-SV Feature Decoupling**: This is the first application of orthogonality regularization to explicit bidirectional decoupling of KWS and SV features in a shared network. This method provides an elegant mathematical tool—achieving feature independence by minimizing the inner product of the two feature spaces. The elegance of orthogonality regularization lies in its ability to solve feature decoupling in both directions using a simple scalar loss (the norm of the inner product of two feature vectors), avoiding the complexity and instability of adversarial training.

2. **Empirical Verification of Reciprocal Learning**: Proven on both English (GSCD) and Chinese (real-world) datasets that there is a reciprocal relationship between KWS and SV tasks—joint training + orthogonality not only does not harm the performance of either task, but simultaneously improves the SOTA performance of both. This breaks the inherent belief that "joint training inevitably leads to performance trade-offs"—when the information of the two tasks is correctly decoupled, the shared underlying features actually become better (because they are jointly optimized by two complementary supervisory signals).

3. **Efficiency Improvement**: The parameter count and computational cost of the joint model are lower than the sum of running two independent models (sharing the underlying feature extraction layer saves approximately 30-40% of parameters), offering greater deployment advantages on devices with limited storage and computation.

4. **Mitigation of Bias**: The orthogonality constraint naturally mitigates speaker bias in KWS and content bias in SV—this is crucial for fairness and security. In practical deployment, a speaker-independent KWS system provides consistent service quality for all user groups (different genders, ages, accents).

## Experimental Results

### Datasets Used and Their Scale
- **Google Speech Commands Dataset (GSCD) V2**: 35 classes, approximately 105,000 1-second English command words. Contains recordings from thousands of different speakers. Standard train/validation/test splits.
- **Real-world Chinese Keyword Dataset**: Chinese wake-word data collected by Xiaomi Corporation, containing multi-speaker (approximately 1,000 speakers) and multi-environment (quiet, in-car, outdoor, office) recordings. Data scale is approximately 100,000 samples.

### Definition and Rationale for Evaluation Metrics
- **KWS**: Accuracy (%) and Equal Error Rate (EER, %). The reason for choosing EER rather than just reporting accuracy is: EER is unbiased in threshold selection (error rate at the threshold where FPR=FNR), making it more suitable for evaluating KWS performance in actual deployment.
- **SV**: Equal Error Rate (EER, %) and Minimum Detection Cost Function (minDCF, defined by NIST SRE standards). minDCF considers prior probabilities and cost weights for different application scenarios, and is the standard metric in the SV field.

### Core Performance Data

**KWS Performance on GSCD**:
- Independent KWS Training -> EER approx. 1.8-2.0%
- Joint Training + Orthogonality -> EER 1.31%
- Improvement: EER reduced by approx. 0.5-0.7%, equivalent to a relative error reduction of approx. 30-35%

**SV Performance on GSCD**:
- Independent SV Training -> EER approx. 2.2-2.5%
- Joint Training + Orthogonality -> EER 1.87%
- Improvement: EER reduced by approx. 0.3-0.6%, equivalent to a relative error reduction of approx. 15-25%

**Performance on Chinese Dataset**: Also showed significant performance improvements (specific numbers are not disclosed as the dataset is not public; the paper only reports relative improvement trends).

### Ablation Study on Orthogonality Regularization

**With vs. Without Orthogonality (Core Ablation)**:
- KWS Accuracy: With orthogonality is approx. 1-2% higher than without orthogonality.
- SV EER: With orthogonality is approx. 0.5-1% lower than without orthogonality.
- Key Finding: Orthogonality not only did not harm performance (as one might worry that "decoupling constraints limit expressive power"), but actually improved the performance of both tasks. This indicates that feature decoupling actually helped each task learn more "pure" discriminative features—the KWS branch is no longer "disturbed" by speaker information, and the SV branch is no longer "disturbed" by content information, allowing each to focus more on its own discriminative task.

**Impact of $\lambda$ Value (Orthogonality Weight)**:
- $\lambda \in [0.1, 1.0]$: Stable effect, accuracy change <0.3%.
- $\lambda > 10$: Optimization is dominated by the orthogonality constraint, leading to a decline in classification performance (because the gradient of the classification loss is suppressed by the gradient of the orthogonality loss).
- $\lambda = 0$ (No orthogonality): Degenerates to standard joint training, with performance declining for both tasks.
- The optimal value for $\lambda$ is approximately 0.5, where the orthogonality constraint and classification loss achieve a good balance.

**Impact of Triplet Loss $\alpha$**:
- $\alpha = 0$ (No triplet loss): Accuracy drops by approx. 0.5-1%, indicating that triplet loss makes a significant contribution to the discriminability of the embedding space.
- $\alpha \in [0.1, 0.5]$: Best performance.
- Synergistic effect of triplet loss and orthogonality regularization: Triplet loss forms compact class clusters in the embedding space, while orthogonality ensures that KWS clusters and SV clusters are in different subspaces—the two together create a well-structured dual-task embedding space.

### Comparison with Independent Training

| Method | KWS EER(%) | SV EER(%) | Total Parameters |
|:---|:---:|:---:|:---:|
| Independent KWS Model | 1.8-2.0 | - | ~200K |
| Independent SV Model | - | 2.2-2.5 | ~200K |
| Joint Training (No Orthogonality) | ~1.5 | ~2.0 | ~280K |
| **Joint Training + Orthogonality (This Paper)** | **1.31** | **1.87** | **~280K** |

The total parameter count of the joint model is approximately 60-70% of the sum of the two independent models (because sharing the underlying convolutional layer saves approximately 120K parameters). With fewer parameters, the performance of both tasks is better—this is an ideal "win-win" result in multi-task learning.

## Limitations and Future Work

### Technical Limitations of the Method
- **Requirement for Symmetric Architecture**: The two branches need the same network structure (same feature dimension $d$) for the orthogonality constraint $L_{orth} = \|\frac{1}{N}\sum f_{KWS}^i (f_{SV}^i)^T\|_F^2$ to make geometric sense (the inner product of two $d$-dimensional vectors is well-defined). If the optimal architectures for KWS and SV differ significantly (e.g., KWS requires CNN while SV requires Transformer), direct application is difficult—additional linear projection to the same dimension would be required for the features.
- **Limitations of Linear Orthogonality Constraint**: The orthogonality constraint guarantees linear independence of feature vectors ($\mathbb{E}[f_{KWS} \cdot f_{SV}^T] \approx 0$), but does not guarantee higher-order statistical independence. The two features may be higher-order statistically correlated (e.g., $f_{KWS}^2$ correlated with $f_{SV}^2$) but linearly orthogonal. In such cases, the linear orthogonality constraint may be insufficient—KWS features may still contain higher-order speaker information.
- **Hyperparameter Sensitivity**: $\lambda$ (orthogonality weight) and $\alpha$ (triplet loss weight) require careful tuning. Although the effect is stable within the range $\lambda \in [0.1, 1.0]$, re-tuning is still required for new datasets or tasks.
- **Approximation Error of Batch Statistics**: $L_{orth}$ uses the sample mean within the batch to approximate the population expectation $\mathbb{E}[f_{KWS} \cdot f_{SV}^T]$. When the batch size is small, the approximation error is large, which may lead to unstable effects of the orthogonality constraint.

### Shortcomings in Experimental Design
- **Text-Dependent vs. Text-Independent SV**: The experiments mainly focus on text-independent speaker verification (not restricting what words are said). In text-dependent SV scenarios (e.g., specific voice passwords), the information coupling between KWS and SV is stronger (because SV must verify both "what was said" and "who said it"), and the orthogonality constraint may be too strict—it may constrain away information that is useful for both tasks.
- **Robustness to Noise Not Evaluated**: All experiments were conducted under relatively clean conditions, without testing performance under noise, far-field, or reverberation conditions. Under noisy conditions, the quality of shared underlying features may degrade, and whether the orthogonality constraint remains effective needs verification.
- **Lack of Systematic Comparison with Other Decoupling Methods**: There is a lack of comparison with other statistical independence measures based on HSIC (Hilbert-Schmidt Independence Criterion) or CCA (Canonical Correlation Analysis).

### Possible Directions for Future Improvement
- **Non-linear Independence Constraints**: Use kernel-based non-linear independence measures such as HSIC (Hilbert-Schmidt Independence Criterion) to replace linear orthogonality constraints. HSIC measures the independence of two random variables in RKHS (Reproducing Kernel Hilbert Space) and can capture higher-order statistical dependencies. The cost is that computational complexity increases from $O(d)$ to $O(N^2)$ ($N$ is the batch size).
- **Multi-task Extension**: Extend orthogonality regularization to joint learning of more tasks (e.g., KWS + SV + Emotion Recognition + Language Identification), verifying the effectiveness of multi-way orthogonality constraints. In a scenario with $M$ tasks, orthogonality needs to be constrained for $\binom{M}{2}$ pairs of features.
- **Adaptive Weights**: Design a strategy to dynamically adjust $\lambda$ based on the training progress of the two tasks—for example, a smaller $\lambda$ in the early stages of training (allowing both branches to freely learn their respective features) and a larger $\lambda$ in the later stages (forcing decoupling of already learned features).
- **Inspiration for the KWS Field**: This paper proves that there is a "positive reciprocal" relationship between KWS and SV—joint training is not only more efficient but also yields better performance. This finding has important implications for the design of speech interaction systems: rather than deploying independent KWS and SV modules, it is better to build a unified acoustic feature extraction system and ensure feature decoupling through orthogonality constraints. This "joint + decoupling" design principle can be generalized to other multi-task speech processing scenarios (e.g., KWS + Voice Activity Detection, SV + Anti-spoofing Detection, etc.).
