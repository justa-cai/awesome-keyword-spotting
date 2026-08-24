# Progressive Continual Learning for Spoken Keyword Spotting

- **Authors/Affiliations**: Yizheng Huang, Nana Hou, Nancy F. Chen (Institute for Infocomm Research, A*STAR, Singapore; Nana Hou is also affiliated with Nanyang Technological University, Singapore)
- **Date**: January 2022
- **Link**: https://arxiv.org/abs/2201.12546
- **Keywords**: continual learning, incremental learning, keyword spotting, catastrophic forgetting, network instantiation, dynamic architecture

## Problem Statement

### Problem Background and Domain Pain Points
In practical keyword spotting (KWS) products, user requirements evolve dynamically. A smart home system might initially recognize only "turn on light" and "turn off light," later needing to add "turn on air conditioner" and "turn off air conditioner," and subsequently requiring custom commands like "good night mode." The goal of Class Incremental Learning (CIL) is to enable the model to sequentially learn new keyword classes without retraining the entire system, while maintaining recognition capabilities for all previously learned classes.

CIL faces the core challenge of catastrophic forgetting: when the model learns new classes, gradient updates for the new task overwrite the neuron weights encoding knowledge of old classes. Forgetting is particularly severe in KWS because the acoustic features of different keywords may share low-level representations (e.g., identical phonemes) but have unique combination patterns at higher levels. Learning new keywords may alter shared low-level feature representations, indirectly affecting the feature quality of old keywords.

### Specific Shortcomings of Existing Methods
- **Regularization-based methods (EWC, SI)**: EWC (Elastic Weight Consolidation) estimates the importance of each weight for the old task via the Fisher Information Matrix and restricts updates to important weights. SI (Synaptic Intelligence) estimates importance by online tracking of weight contributions. However, second-order importance estimation in high-dimensional parameter spaces is imprecise (the diagonal approximation of the Fisher matrix ignores correlations between parameters), and as the number of tasks increases, the degrees of freedom for learning continuously decrease (more and more weights are "frozen"), ultimately making it difficult to learn new tasks. The paper's experiments show that EWC and SI improve performance by only about 5-10% over direct fine-tuning on CIL tasks in KWS, which is far from satisfactory.
- **Replay-based methods (GEM, NR)**: GEM (Gradient Episodic Memory) constrains the gradient direction of the new task by replaying a small number of samples from old tasks (ensuring it does not increase the loss of old tasks). However, replay methods face storage constraints on edge devices. Assuming a total storage budget of 100KB and each spectrogram sample being approximately 4KB, only 25 old samples can be stored. As the number of tasks grows (e.g., 10 tasks), only about 2-3 samples can be allocated per task, causing replay effectiveness to drop sharply.
- **Methods requiring task IDs**: Some CIL methods (e.g., Expert Gate, Progressive Neural Networks) require knowing the task ID of the current input during inference, which is infeasible in practical KWS systems—the device does not know which keyword the user will say next.
- **Methods relying on large pre-trained models**: Some methods use embeddings from large pre-trained speech models (e.g., wav2vec 2.0 with approximately 300M parameters) as feature extractors. Although the features from pre-trained models have strong generalization capabilities, the models themselves are too large (hundreds of MBs in storage, tens of MFLOPs in computation) to be deployed on resource-constrained edge devices.
- **Problems with existing dynamic architecture methods**: Some methods (e.g., Progressive Nets, Dynamically Expandable Networks) add new network branches or layers for each new task, but parameter growth is linear (the model size increases by a fixed proportion for each added task). After multiple incremental tasks, the model quickly exceeds the storage limits of edge devices.

### Key Challenges Addressed by This Paper
How to design a continual learning method that does not require task IDs, does not require a storage buffer (no sample replay), and has strictly controllable model size growth (sub-linear growth), enabling it to sequentially learn multiple new keyword tasks on a lightweight backbone network (e.g., TC-ResNet-8, with approximately 80K parameters) while maintaining high recognition accuracy for all learned classes.

## Methodology

### Overall Architecture Design and Design Motivation
The core idea of PCL-KWS (Progressive Continual Learning for KWS) is "progressive network instantiation"—dynamically generating a lightweight sub-network for each new task while transferring knowledge between all sub-networks through shared memory blocks. The architecture is based on TC-ResNet-8 (a time-convolutional residual network, a lightweight backbone network specifically designed for KWS, with approximately 80K parameters).

The design philosophy of PCL-KWS is "protect old knowledge, compress new knowledge":
- **Protect old knowledge**: By freezing all parameters of old sub-networks, forgetting is prevented fundamentally (more thorough than regularization methods).
- **Compress new knowledge**: By using keyword-aware network scaling, each new sub-network uses only the necessary amount of parameters.

### Mathematical Principles of Core Algorithms

**Core Technology 1: Network Instantiator**

**Initial Task (Task 0)**: Train a complete baseline network $f_0$, consisting of convolutional layers $C_0$, a shared memory block $M$, and a classification head $H_0$. $M$ stores general acoustic feature representations learned from the initial task.

**When the $t$-th new task arrives ($t \geq 1$)**:
1. Keep the parameters of all existing sub-networks $f_0, f_1, ..., f_{t-1}$ unchanged (to prevent forgetting).
2. Instantiate a new classification head $H_t$ (the output layer for new classes, output dimension = $K_t$, number of new classes).
3. Copy the convolutional layer parameters from the baseline network as the initial parameters for the new sub-network $C_t$: $C_t^{(0)} = C_0$ (using baseline parameters for initialization rather than random initialization accelerates convergence).
4. The new sub-network is $f_t = (C_t, M, H_t)$, where $M$ is the "public memory block" shared by all sub-networks.

**Dual Role of the Shared Memory Block $M$**:
- **Forward Transfer**: The new sub-network $C_t$ acquires general acoustic feature representations learned from all previous tasks by reading $M$. This provides a "pre-trained" feature foundation for the new task.
- **Continuous Update**: The parameters of $M$ are updated during the training of new tasks (gradients propagate through $M$), allowing $M$ to continuously accumulate knowledge from all tasks. However, old sub-networks use snapshots of $M$ from the time of their training (they do not change with subsequent updates to $M$), avoiding interference from new tasks on old sub-networks.

**Core Technology 2: Keyword-Aware Network Scaling**

**Problem**: If the sub-network $C_t$ for each new task copies the full baseline convolutional layer parameters, the model size will grow linearly: $|f_{total}| = |f_0| + \sum_{t=1}^T |C_t| + \sum_{t=0}^T |H_t| + |M|$. After 5 new tasks, the model size could be 6 times that of the baseline, far exceeding the storage limits of edge devices.

**Solution**: Adaptively reduce the width of the sub-network based on the scale of the new task (number of keywords):

$$w_t = \min\left(1.0, \frac{K_t}{K_0}\right)$$

where $K_t$ is the number of keywords in the new task, and $K_0$ is the number of keywords in the initial task.

The number of channels in each layer of the new sub-network $C_t$ = Baseline channels $\times w_t$.

**Intuitive Explanation**:
- When the new task has only 1-2 keywords (e.g., $K_t=3$, $K_0=15$), $w_t = 3/15 = 0.2$, meaning the new sub-network uses only 20% of the baseline channel width.
- The fewer keywords to recognize, the less feature extraction capacity is required—this aligns with the principle of "using a small network to solve simple tasks."
- This scaling ensures that the growth of the total model size is sub-linear—the size of the new sub-network is proportional to the complexity of the new task.

**Mathematical Analysis of Parameter Growth**:
Assume the baseline model has $P_0$ parameters, and $T$ new tasks each add $K_t$ keywords:
- Total parameters = $P_0 + \sum_{t=1}^T P_0 \times w_t + \sum_{t=0}^T |H_t| + |M|$
- When $K_t \ll K_0$ (fewer keywords in new tasks), $\sum_{t=1}^T P_0 \times w_t \ll T \times P_0$, so the growth is much slower than linear.
- In the experimental setup of the paper ($K_0=15$, $K_t=3$, $T=5$): The total parameters of the 5 new sub-networks are approximately 100% of the baseline, meaning the total model is about 2 times the baseline.

### Comprehensive Comparison with Baseline Methods
The paper systematically evaluates four representative CIL strategies:
- **EWC (Elastic Weight Consolidation)**: Diagonal approximation based on the Fisher Information Matrix, $\lambda_{EWC} \in \{100, 1000, 5000\}$.
- **SI (Synaptic Intelligence)**: Based on online parameter importance estimation, $c_{SI} = 0.5$.
- **NR (Network Roulette)**: Based on random sub-network selection, randomly selecting one sub-network for inference each time.
- **GEM (Gradient Episodic Memory)**: A replay-based method with gradient constraints, buffer size $b \in \{50, 100, 200\}$.

### Experimental Setup
- **Backbone Network**: TC-ResNet-8 (approximately 80K parameters).
- **Initial Task**: 15 basic keywords.
- **Incremental Tasks**: 5 tasks, each adding 3 new keywords.
- **Total**: 15 + 15 = 30 keywords, 6 tasks.
- **Dataset**: Google Speech Commands.

## Main Contributions

1. **Progressive Network Instantiation Framework**: PCL-KWS is the first dynamic architecture continual learning method specifically designed for small KWS models. By instantiating dedicated sub-networks for each new task and sharing feature memory, it achieves an elegant solution of "old knowledge remains unchanged (frozen old sub-networks), new knowledge grows incrementally (instantiated new sub-networks), and knowledge is shared (shared memory block)." Compared to regularization methods, PCL-KWS addresses forgetting at the structural level (rather than the parameter constraint level)—frozen old sub-networks are not affected by any gradient updates.

2. **Keyword-Aware Network Scaling**: The innovative scaling mechanism $w_t = \min(1.0, K_t/K_0)$ dynamically adjusts the sub-network size based on the scale of the new task. The deep value of this design lies in explicitly linking "model capacity" with "task complexity"—allocating few parameters to simple tasks to avoid parameter waste. In practical edge deployment scenarios, this predictable parameter growth allows devices to estimate "how much additional storage is needed to add N new keywords."

3. **Buffer-Free Continual Learning**: Unlike replay methods such as GEM, PCL-KWS does not require storing any historical data samples. Knowledge retention is achieved entirely through parameter snapshots (freezing old sub-networks), eliminating the complexity of buffer management (such as buffer size decisions, sample selection strategies, and buffer update strategies).

4. **Forward Transfer via Shared Memory Block**: The shared memory block $M$ provides a "knowledge hub" for all sub-networks—new sub-networks can acquire knowledge from old tasks via $M$ without directly accessing the training data of old tasks. This solves the data availability problem of replay methods.

## Experimental Results

### Datasets Used and Their Scale
- **Google Speech Commands V2**: Standard KWS benchmark. 30 classes were selected from 35 (excluding "silence" and several low-frequency classes). The training set has approximately 84,000 samples, and the test set has approximately 11,000 samples.
- **Incremental Learning Configuration**: 15 basic keywords (Task 0) + 5 incremental tasks (Task 1-5, 3 new keywords per task), totaling 30 keywords and 6 tasks.

### Definition and Rationale for Evaluation Metrics
- **Average Accuracy (%)**: The average classification accuracy of the model across all 30 seen classes after all incremental steps are completed. This is a standard metric in the CIL field.
- **Parameter Growth Ratio**: The multiple of the total parameters relative to the baseline model. Used to evaluate the storage efficiency of the method.
- **Per-Task Accuracy**: The independent accuracy for each learned task, used to analyze the distribution characteristics of forgetting.

### Core Performance Data

**PCL-KWS vs. Baseline Methods (Average Accuracy after 5 Incremental Tasks)**:

| Method | Avg Accuracy (%) | vs Fine-tuning | Parameter Growth |
|:---|:---:|:---:|:---:|
| Fine-tuning (Lower Bound) | ~75-80 | baseline | 1.0x |
| EWC | ~85-87 | +7 | 1.0x |
| SI | ~85-88 | +8 | 1.0x |
| NR | ~86-89 | +9 | ~1.5x |
| GEM (b=100) | ~88-90 | +12 | 1.0x |
| **PCL-KWS** | **92.8** | **+15** | **~2.0x** |

PCL-KWS significantly outperforms all baseline methods with an average accuracy of 92.8%. Compared to the strongest baseline, GEM (approximately 89-90%), PCL-KWS is higher by about 3%.

### Parameter Growth Analysis

Details of PCL-KWS parameter growth:
- Baseline model $f_0$: Approximately 80K parameters.
- Each new sub-network ($w_t = 0.2$): Approximately 16K parameters (only 20% of the baseline).
- 5 new sub-networks: 5 x 16K = 80K parameters.
- New classification heads: Approximately 5 x 3 x d = approximately 15K parameters.
- Shared memory block: Approximately 10K parameters.
- **Total Parameters**: Approximately 80K + 80K + 15K + 10K = approximately 185K, which is about 2.3 times the baseline.

Comparison: If each sub-network uses full width ($w_t = 1.0$), 5 new sub-networks would require 5 x 80K = 400K parameters, and the total model would be approximately 500K (6.25 times the baseline). The scaling mechanism compresses parameter growth from 6.25 times to 2.3 times.

### Findings from Ablation Studies

**Shared Memory Block vs. No Sharing**:
- Removing the shared memory block $M$ (each sub-network uses an independent feature extractor) -> Accuracy drops by approximately 3-5%.
- Reason analysis: Without $M$, new sub-networks cannot utilize knowledge from old tasks and must start learning from the initial parameters of the baseline—equivalent to optimizing each new task in an "isolated" parameter space, lacking cross-task knowledge transfer.

**Impact of Network Scaling $w_t$**:
- $w_t = 1.0$ (No scaling): Total parameters approximately 500K (6.25 times baseline), accuracy approximately 93.3% (only 0.5% higher than $w_t=0.2$).
- $w_t = 0.5$: Total parameters approximately 280K (3.5 times baseline), accuracy approximately 93.0%.
- $w_t = 0.2$ (Selected in paper): Total parameters approximately 185K (2.3 times baseline), accuracy approximately 92.8%.
- **Key Finding**: From $w_t=1.0$ to $w_t=0.2$, the parameter count decreases by approximately 63%, but the accuracy drops by only 0.5%—the scaling mechanism saves significant storage with almost no sacrifice in performance.

**Sub-network Initialization Strategy**:
- Copying from baseline $C_0$ (Selected in paper) -> Accuracy 92.8%.
- Random initialization -> Accuracy approximately 89-90% (drop of approximately 3%).
- Copying from the most recent sub-network $C_{t-1}$ -> Accuracy approximately 92.5% (comparable to copying from the baseline).
- The advantage of baseline initialization is that it provides "original, unshifted by subsequent tasks" feature representations, offering a fair starting point for all new tasks.

## Limitations and Future Work

### Technical Limitations of the Method
- **Requirement for Task ID (or Task Router) during Inference**: Since the classification heads $H_t$ for different tasks are different, the system needs to know which task the current input belongs to in order to select the correct classification head. In practical KWS systems, users may say any learned keyword, and the system cannot predict this in advance. The paper does not fully discuss how to solve this problem—possible solutions include: (1) Parallel inference on all classification heads and selecting the result with the highest confidence; (2) Training an additional "task router" to predict which task the input belongs to; (3) Using a unified output space (the union of all learned classes as the output layer).
- **Continuous Growth of Model Size**: Although the scaling mechanism constrains the growth rate (sub-linear), the model size still increases monotonically with the number of tasks. In scenarios with a very large number of tasks (e.g., hundreds of incremental tasks, such as users continuously adding custom command words), storage constraints will eventually be breached.
- **Knowledge Rigidity Due to Fixed Sub-networks**: Once a sub-network is trained and frozen, its parameters are no longer updated. If the acoustic characteristics of subsequent tasks differ significantly from the training distribution of a certain old sub-network, the performance of the old sub-network may degrade (because the shared memory block $M$ is updated, but the old sub-network uses an old snapshot of $M$, potentially causing a mismatch).
- **Capacity Bottleneck of Shared Memory Block $M$**: The size of $M$ is fixed. As the number of tasks increases, $M$ needs to encode knowledge from more and more tasks, potentially leading to capacity saturation—the "memory interference" problem where new knowledge overwrites old knowledge.

### Shortcomings in Experimental Design
- **Single Backbone Network**: All experiments were conducted only on TC-ResNet-8. TC-ResNet is a time-convolutional network, and its sequential architecture is naturally suitable for deep pruning and sub-network instantiation. Its applicability on CNNs (e.g., DS-CNN), RNNs (e.g., LSTM), or Transformer architectures has not been verified.
- **Sensitivity to Task Order Not Tested**: CIL methods are typically sensitive to task order (which keywords are learned first vs. later may affect final performance). The paper did not test the impact of different task orders on PCL-KWS.
- **Scenario with Overlapping Keywords Between Tasks Not Tested**: If the keywords in a new task have acoustic similarities with those in old tasks (e.g., "stop" and "top"), can PCL-KWS effectively distinguish them?
- **Comparison with Other Dynamic Architecture CIL Methods Not Conducted**: Such as Der++ (Deriving from Dark Experience), AANet (Adaptive Aggregation Network), etc.

### Possible Directions for Future Improvement
- **Unified Classification Head Design**: Design a unified classification scheme that does not require task IDs. For example, using a dynamically expanding output layer (adding new output nodes for each new task while keeping old output nodes unchanged), combined with a unified feature space mapping.
- **Periodic Updating of Sub-network Parameters**: Explore mechanisms to periodically update the parameters of old sub-networks while ensuring no forgetting. For example, using knowledge distillation to "feed back" knowledge from the current global model (containing knowledge from all sub-networks) to the old sub-networks, allowing them to adapt to updates in $M$.
- **Automatic Scaling Strategy**: Use NAS or reinforcement learning to automatically determine the optimal sub-network width $w_t$ for each new task, rather than using a simple proportional formula. Keywords with different acoustic characteristics may require different feature extraction capacities.
- **Inspiration for the KWS Field**: The "instantiation + sharing + scaling" paradigm of PCL-KWS finds an elegant balance between model compression and incremental learning. This approach can be generalized to other edge AI scenarios requiring continuous adaptation to new tasks (such as voice command recognition, acoustic event detection, gesture recognition). The core principle is: "Freeze old knowledge, share general knowledge, compress new knowledge."
