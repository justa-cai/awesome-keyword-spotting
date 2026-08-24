# Improving Feature Generalizability with Multitask Learning in Class Incremental Learning

- **Authors/Affiliations**: Dong Ma, Chi Ian Tang, Cecilia Mascolo (University of Cambridge, UK; Dong Ma is also affiliated with Singapore Management University)
- **Date**: April 2022
- **Link**: https://arxiv.org/abs/2204.12915
- **Keywords**: class incremental learning, continual learning, multitask learning, keyword spotting, catastrophic forgetting, feature generalizability

## Problem Statement

### Problem Background and Domain Pain Points
In real-world keyword spotting (KWS) products, user requirements evolve dynamically—today you may need to recognize "Ok Google," while tomorrow you might need to add "Hey Alexa," custom wake words, or commands in specific languages. The goal of Class Incremental Learning (CIL) is to enable models to gradually learn new keyword classes without retraining the entire system, while maintaining recognition capabilities for all previously learned classes. The core challenge of CIL is "catastrophic forgetting": when a model learns new classes, neuron weights are overwritten by gradients from the new task, leading to a sharp degradation in recognition performance for old classes.

In edge AI scenarios (such as KWS on smart speakers or wearable devices), the constraints for incremental learning are stricter: (1) original training data for old classes may no longer be available (due to privacy regulations limiting long-term data storage); (2) device computational resources are limited, making it impossible to store and replay large amounts of historical data; (3) the incremental learning process must be completed on-device (due to privacy requirements) and cannot rely on cloud training.

### Specific Shortcomings of Existing Methods
- **"Treating symptoms rather than the root cause" of incremental step methods**: Existing CIL methods (such as Knowledge Distillation like LwF, Sample Replay, and Parameter Regularization like EWC) primarily focus on the incremental learning phase—i.e., how to preserve old knowledge during the learning of new classes via distillation loss, replaying old samples, or constraining important weight updates. However, these methods generally ignore a more fundamental problem: if the feature extractor of the base model is overfitting to the base classes (i.e., the learned features are discriminative only for base classes and have poor generalization to new classes), then regardless of the strategy used in the incremental phase, subsequent learning performance will be limited—because the new classes learned incrementally must reuse the feature extractor of the base model.
- **Fragility of base model features**: Models trained jointly on all base classes in a single classification task tend to learn the "shortest path" that "just happens to distinguish these classes"—i.e., class-specific features. For example, if the base classes include "yes," "no," "up," and "down," the model might learn "spectral patterns to distinguish these specific words" rather than "general phoneme recognition capabilities." When the acoustic characteristics of new classes (such as "stop," "go") differ significantly from those of the base classes, the shared feature extractor cannot provide effective feature representations for the new classes.
- **Insufficient handling of data imbalance**: In practical incremental learning scenarios, data for old classes may no longer be available (due to privacy and storage constraints), while the number of samples for new classes may be limited. This data imbalance exacerbates the forgetting problem. More critically, there is a lack of systematic training termination strategies (early stopping mechanisms) to prevent overfitting or underfitting in data-imbalanced scenarios—different incremental learning scenarios (number of new classes, sample size, similarity to old classes) require different numbers of training epochs.

### Key Challenges Addressed by This Paper
How to optimize the generalization capability of the feature extractor from the base model training phase, so that it is not only effective for base classes but also provides good feature initialization for subsequent incremental learning of new classes. The core insight is: **Good incremental learning starts with a good base model**—the upper limit of incremental learning performance is largely determined by the quality of the base model's features.

## Methodology

### Overall Architecture Design and Design Motivation
The core method of this paper consists of three interacting components: Multi-Task Base Training (MT-BT), the incremental learning phase (compatible with existing CIL methods), and Adaptive Early Stopping (AS-ES). The design motivations for these three components are:
- MT-BT addresses the problem of "poor feature generalization in the base model" (root-cause improvement)
- Existing CIL methods address the problem of "forgetting in the incremental phase" (operational-level improvement)
- AS-ES addresses the problem of "uncertain training time under data imbalance" (engineering-level improvement)

### Mathematical Principles of Core Algorithms

**Core Technology 1: Multi-Task Base Training (MT-BT)**

**Deep Analysis of Design Motivation**:
When all $N$ base classes are trained jointly in a single classification task, the loss function is:

$$L_{\text{single}} = -\sum_{i=1}^{N_{\text{train}}} \sum_{c=1}^{N} y_{ic} \log p(y_c | x_i; \theta)$$

where $\theta$ are the parameters of the shared feature extractor. In this single-task setting, the gradient update direction is to "minimize the classification error of base classes"—the model will look for feature dimensions that are easiest to distinguish these classes. If certain acoustic features (such as formant frequencies of specific keywords) happen to perfectly distinguish the base classes, the model will over-rely on these features while ignoring other feature dimensions that are equally useful but currently unnecessary.

**Multi-task Decomposition**:
MT-BT randomly divides the $N$ base classes into $K$ subsets $S_1, S_2, ..., S_K$, with each subset containing approximately $N/K$ classes. Each subset is equipped with an independent classification head (fully connected layer + softmax), but they share the same feature extractor (backbone).

During training, samples from different subsets are trained alternately (or mixed in a single minibatch). The parameters of the shared feature extractor are updated via gradient aggregation from all sub-tasks:

$$\theta^{(t+1)} = \theta^{(t)} - \eta \sum_{k=1}^{K} \nabla_\theta L_k(\theta^{(t)})$$

where $L_k$ is the cross-entropy loss of the $k$-th sub-task.

**Why This Improves Generalization—Analysis from the Perspective of Gradient Conflict**:
The classification objectives of different sub-tasks differ, and their gradient directions may conflict—i.e., a certain feature dimension may be useful for sub-task 1 but useless (or even harmful) for sub-task 2. During gradient aggregation, these conflicts cancel each other out, forcing the feature extractor to find "compromise features" that are useful for all sub-tasks—these features happen to be more general and transferable acoustic features (such as phoneme-level features, rather than specific word-level features).

This mechanism is similar in spirit to meta-learning (such as MAML) but simpler: meta-learning improves transferability by "optimizing initialization parameters on multiple tasks," while MT-BT improves generalization by "optimizing the feature extractor on multiple sub-tasks." The difference is that MT-BT does not require the inner-outer loop optimization of MAML, making training simpler.

**Analysis of Negative Transfer**:
A potential risk of multi-task training is "negative transfer"—conflicts between different sub-tasks may lead to a decrease in performance for each sub-task. The paper's experiments show that in CIL scenarios, the base accuracy of MT-BT may be 0.5-1% lower than single-task training (due to gradient conflicts between sub-tasks), but this slight loss in base performance is exchanged for significant gains in the incremental learning phase (+4-5%), with the total benefit far exceeding the base loss.

**Core Technology 2: Adaptive Early Stopping (AS-ES)**

**Design Motivation**:
In the incremental learning phase, data is typically imbalanced: (1) samples for new classes are limited (users only provide a few registration samples); (2) if a replay strategy is used, the replayed samples for old classes are also limited. On such imbalanced data, both overfitting (too many training epochs) and underfitting (too few training epochs) will harm performance. The optimal number of training epochs for different incremental learning scenarios may vary drastically (from 10 epochs to 200 epochs).

**Specific Implementation**:
1. Divide the limited incremental training data into a validation set (validation set, approximately 10-20% of the training data)
2. Evaluate performance metrics (accuracy or loss value) on the validation set after each training epoch
3. When validation set performance does not improve for $P$ consecutive epochs (patience strategy, usually $P=5-10$), terminate training
4. Save the model parameters with the best validation set performance (best checkpoint)

The simplicity of AS-ES is its advantage—it introduces no additional computational overhead or hyperparameters (except for the patience value $P$), but significantly improves the stability and reliability of incremental learning in practical deployment.

### Compatibility with Existing CIL Methods
MT-BT and AS-ES are orthogonal and compatible with any existing incremental learning strategy:
- **Knowledge Distillation (LwF)**: MT-BT provides better base features -> LwF preserves old knowledge during the incremental phase via distillation
- **Parameter Regularization (EWC)**: MT-BT provides better base features -> EWC constrains important weight updates during the incremental phase
- **Sample Replay (Replay, GEM)**: MT-BT provides better base features -> Replay replays old samples during the incremental phase
- The key argument of the paper is: MT-BT does not replace these existing strategies, but acts as a "base training enhancement module" that stacks with them, providing consistent additional performance gains on top of any existing method.

### Experimental Setup
- **Datasets**: Google Speech Commands (GSC, variants with 12 and 35 classes) and UrbanSound8K (10 classes of urban environmental sounds)
- **Base Classes**: 10 base classes used on GSC, 5 base classes used on UrbanSound8K
- **Incremental Steps**: Adding 1-5 new classes at each step, for a total of 5-10 incremental steps
- **Backbone Network**: Small CNN (3-4 convolutional layers + fully connected, approx. 50K parameters)
- **Number of Subsets $K$**: Mainly using $K=2-4$

## Main Contributions

1. **Re-examining the Base Training Phase of CIL**: For the first time, this paper systematically points out and proves the key insight that "the quality of base model training has a decisive impact on incremental learning performance." Previous research has almost entirely focused on forgetting mitigation strategies in the incremental phase, ignoring that the quality of the base model is the determinant of the upper limit of incremental learning performance. The paper proves through systematic ablation experiments that, under the same incremental strategy, base models trained with MT-BT have an average accuracy 4-5% higher after incremental learning compared to base models trained with standard training.

2. **Improving Feature Generalizability with Multi-Task Learning**: By decomposing base classes into multiple subsets for multi-task training, gradient conflicts between sub-tasks force the feature extractor to learn more general acoustic features. This method is simple (requiring only $K$ independent classification heads), general (compatible with any CIL strategy), and does not increase computational cost at inference time (only one classification head is used during inference).

3. **Practicality of Adaptive Early Stopping**: Introduces a validation-set-based adaptive early stopping strategy for CIL, replacing empirical fixed training epoch selection. This seemingly simple improvement solves the problem of "not knowing how many epochs to train for" in practical deployment, improving accuracy by 1-2 percentage points in data-imbalanced scenarios.

4. **Orthogonal Stackability with Existing Methods**: The proposed methods do not replace existing CIL strategies (LwF, EWC, GEM, Replay) but act as a "base training enhancement module" that stacks orthogonally with them. The paper verifies consistent gains (3-5%) in combination experiments with 4 CIL strategies, proving the generality of the method.

## Experimental Results

### Datasets Used and Their Scales
- **Google Speech Commands (GSC-12)**: 12 command word classes, approx. 65,000 1-second speech samples. Uses standard train/validation/test splits. 10 base classes (excluding "silence" and "unknown"), with 2 classes added incrementally.
- **Google Speech Commands (GSC-35)**: 35-class variant, approx. 105,000 speech samples.
- **UrbanSound8K**: 10-class urban environmental sound classification (engine sounds, children playing, dog barking, street music, etc.), approx. 8,700 audio clips. Used to verify the generalization of the method on non-speech audio tasks.

### Definition and Rationale for Evaluation Metrics
- **Average Incremental Accuracy (%)**: After all incremental steps are completed, the average classification accuracy of the model on all seen classes (base + new). This is the standard metric in the CIL field, comprehensively measuring the model's "learning ability" and "memory retention ability."
- **Accuracy Curve per Step**: Shows the trend of accuracy changes as incremental steps increase. Used to analyze the dynamic process of forgetting—ideal CIL methods should have a平缓 (flat) accuracy decline curve.

### Detailed Comparison with Baseline and SOTA Methods

**Typical Configuration on GSC-12 (10 base classes + 5 incremental steps, adding 2 new classes per step)**:

| CIL Strategy | Standard Training | +MT-BT | +MT-BT+AS-ES | Total Gain |
|:---|:---:|:---:|:---:|:---:|
| Fine-tuning (Lower Bound) | ~75% | ~78% | ~79% | +4% |
| LwF | ~85% | ~89% | ~90.5% | +5.5% |
| EWC | ~83% | ~86% | ~87.5% | +4.5% |
| GEM | ~88% | ~91% | ~92% | +4% |
| Replay | ~87% | ~90% | ~91.5% | +4.5% |

Key Findings: MT-BT provides consistent gains (3-5%) across all CIL strategies, and AS-ES provides an additional gain of approx. 1% on top of that. The strongest combination is MT-BT + AS-ES + LwF, achieving an average incremental accuracy of 90.5%.

**Validation on UrbanSound8K**:
MT-BT + AS-ES also shows consistent improvements (3-5 percentage points) on UrbanSound8K, verifying the task-agnostic nature of the method—it applies not only to speech KWS but also to general sound classification CIL.

### Findings from Ablation Experiments

**Multi-Task vs. Single-Task Base Training**:
- MT-BT ($K=3$) achieves higher accuracy than single-task training across all incremental steps
- The advantage increases with the number of incremental steps: only approx. 1% higher at the 1st incremental step, but approx. 3-5% higher at the 5th incremental step
- Reason Analysis: Features with good generalization benefit from more new classes—each time a new class is added, good base features provide a "bonus" for it

**Impact of Number of Subsets $K$**:
- $K=1$: Degenerates to standard single-task training, no extra gain
- $K=2$: The regularization effect of multi-task training begins to appear, gain approx. +2%
- $K=3-4$ (Optimal): Maximum gain, approx. +3-4%
- $K$ too large ($K \geq 6$): Too few classes per subset (e.g., only 1 class per subset when $K=10$), making the classification task too simple and losing the multi-task training effect. Sub-tasks that are too simple mean too few gradient conflicts, so the feature extractor is not forced to learn more general features.

**Quantitative Effect of Early Stopping Mechanism**:
- Compared to fixed training for 50 epochs: AS-ES improves accuracy by 1-2 percentage points in data-imbalanced scenarios
- The number of training epochs automatically selected by AS-ES varies greatly across different scenarios: approx. 20 epochs when new class samples are abundant, approx. 5-8 epochs when samples are scarce (rapid termination to prevent overfitting)
- Without AS-ES, fixed training for 50 epochs leads to severe overfitting when new class samples are scarce—accuracy for new classes is artificially high, but accuracy for old classes drops sharply

**Impact of Subset Partitioning Strategy**:
- Random partitioning vs. Partitioning by acoustic similarity: The performance difference between the two strategies is not significant (<0.5%), indicating that MT-BT is insensitive to the partitioning strategy. This is because randomization ensures class diversity in each subset, which is sufficient to generate effective gradient conflicts.

## Limitations and Future Work

### Technical Limitations of the Method
- **Design Choice of Subset Partitioning**: Dividing base classes into $K$ subsets is a hyperparameter that requires empirical decision-making. Although experiments show that $K=3-4$ works best in most scenarios, the optimal $K$ may differ for different numbers of base classes and data distributions. The paper does not provide a mechanism to automatically determine $K$.
- **Computational Overhead of Base Training**: Although multi-task training does not increase inference cost, it requires training $K$ classification heads simultaneously during the training phase, with computational load approx. 1.2-1.5 times that of single-task training. This is because each minibatch requires calculating loss and gradients for $K$ sub-tasks separately.
- **Randomness of Subset Partitioning**: Although random partitioning is insensitive to performance, in some extreme cases (e.g., two acoustically very similar classes are assigned to different subsets), it may generate excessive gradient conflicts, reducing training efficiency.

### Shortcomings in Experimental Design
- **Limitation on Number of Base Classes**: The number of base classes in experiments is small (10), and the number of incremental steps is small (5-10). When the number of base classes is large (e.g., 100+), the effect of multi-task decomposition may differ—because even with $K=4$, each subset still has 25+ classes, and sub-tasks may still be too simple.
- **Single Backbone Network**: All experiments use a small CNN (3-4 layers) as the backbone network, and the effect on Transformers or deeper CNNs has not been verified. Deeper networks may have more redundant parameters, and their sensitivity to base training strategies may differ.
- **Not Evaluated in Combination with Data Augmentation**: The combined effect of MT-BT and data augmentation (such as SpecAugment, noise augmentation) has not been evaluated. Data augmentation itself is a regularization method, which may have synergistic or redundant effects with the implicit regularization of MT-BT.

### Possible Directions for Future Improvement
- **Automatic Subset Partitioning**: Use acoustic similarity between classes (e.g., based on DTW distance or embedding space distance) or semantic relationships (e.g., based on phoneme composition similarity) to guide subset partitioning. The goal is to maximize differences between subsets (producing more effective gradient conflicts) while ensuring sufficient discriminability among classes within each subset.
- **Deep Integration with Meta-Learning**: Combine MT-BT with meta-learning algorithms such as MAML. MT-BT provides better base features, and MAML further optimizes the rapid adaptation capability of the feature extractor on this basis—adapting to the distribution of new classes within a few gradient steps.
- **Online Multi-Task Base Training**: Explore the feasibility of continuously performing multi-task training during the incremental learning process—when new classes arrive, recombine new classes with some old classes into sub-tasks for incremental multi-task training. This can continuously improve the generalization capability of the feature extractor.
- **Larger Base Class Sets**: Verify the scalability of MT-BT on large speech datasets containing hundreds of classes (such as Speech Commands Full, LibriSpeech).
- **Implications for the KWS Field**: This paper reveals the important rule that "the quality of the base model determines the upper limit of incremental learning." This has direct guiding significance for the product design of KWS systems—when deploying KWS products, more effort should be invested in optimizing feature representation during the base training phase (e.g., using multi-task training, data augmentation, pre-training, etc.), rather than just patching forgetting problems in the incremental phase. This principle of "prevention is better than cure" applies to all AI systems that require continuous learning.
