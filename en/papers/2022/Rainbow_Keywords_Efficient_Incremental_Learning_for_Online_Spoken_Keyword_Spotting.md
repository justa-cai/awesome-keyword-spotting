# Rainbow Keywords: Efficient Incremental Learning for Online Spoken Keyword Spotting

- **Authors/Affiliations**: Yang Xiao, Nana Hou, Eng Siong Chng (School of Computer Science and Engineering, Nanyang Technological University, Singapore)
- **Date**: March 2022 (Interspeech 2022)
- **Link**: https://arxiv.org/abs/2203.16361
- **Keywords**: incremental learning, knowledge distillation, online keyword spotting, catastrophic forgetting, diversity-aware sampling, memory replay

## Problem Statement

### Problem Background and Domain Pain Points
Keyword recognition systems need to continuously adapt to new keyword requirements after deployment—users may want to add custom wake words or command words at any time. In this online incremental learning scenario, training data for new keyword categories is learned immediately upon arrival, while training data for old categories is typically no longer available (due to storage limits or data expiration). This makes catastrophic forgetting an inevitable challenge.

In practical product scenarios, the constraints for incremental learning are even stricter: the storage budget is extremely low (e.g., tens of KB on a smartwatch for storing old samples), the number of registration samples for each new keyword is very small (users are usually only willing to speak 3-5 times), and the incremental learning process must be completed in real-time on the device (cannot rely on cloud training).

### Specific Shortcomings of Existing Methods
- **Methods requiring task ID auxiliary information**: Some CIL methods assume that task IDs (i.e., knowing which incremental step the current input belongs to) can be obtained during inference, which is unrealistic in actual KWS systems—users may say any learned keyword, and the system cannot know this in advance. For example, after 5 incremental steps where 20 keywords have been learned, the system needs to monitor all 20 keywords simultaneously, and the task ID is unknown.
- **Methods where storage grows linearly with tasks**: Some methods (e.g., iCaRL, GEM) store a fixed number of old samples for each incremental task (e.g., 50 samples per task), causing total storage to grow linearly as the number of tasks increases. For edge devices with extremely low storage budgets (e.g., 50KB on a smartwatch for the incremental learning buffer, with each spectrogram sample being about 2-4KB), linear growth quickly becomes unbearable—after 10 incremental tasks, 500 samples (about 1-2MB) would be needed, far exceeding the budget.
- **Low information efficiency of random sampling**: Under limited memory budgets, if old samples to be retained are selected randomly, it may retain a large number of redundant or low-information samples (e.g., very typical, easy-to-classify samples—the model can already classify them correctly, so retaining them offers almost no help in preventing forgetting), while discarding difficult samples near the decision boundary (samples on which the model is most uncertain and most prone to forgetting).
- **Insufficient training data for replay methods**: Even with carefully selected old samples, only a small number of samples can be retained per category (e.g., 5-10). These few samples are insufficient to represent the complete distribution of the original data, leading the model to overfit on the limited replay data.

### Key Challenges Addressed by This Paper
How to preserve the most critical information of old categories with minimal storage space (through intelligent sampling rather than random sampling) without requiring task IDs, enabling the KWS model to maintain recognition capability for all learned categories while sequentially learning multiple new keywords. The specific goal is to achieve an absolute average accuracy improvement of more than 4% over the best baseline on the Google Speech Command dataset.

## Methodology

### Overall Architecture Design
Rainbow Keywords (RK) is a sample-replay-based incremental learning method containing three closely cooperating components: Diversity-Aware Sampler (DAS), Mixed-Label Data Augmentation (MLDA), and Knowledge Distillation (KD). The design philosophy of RK is "less but better"—instead of storing a large number of old samples, it carefully selects a small number of most representative samples and virtually expands their diversity through data augmentation.

### Mathematical Principles of Core Algorithms

**Core Technology 1: Diversity-Aware Sampler (DAS)**

**Deep Analysis of Design Motivation**:
Under a limited memory budget $M$ (assuming a total of $m$ old samples are stored, $m = M / \text{sample\_size}$), which old samples to retain is crucial for the effect of preventing forgetting. Formally, let the complete dataset for old category $c$ be $\mathcal{D}_c = \{x_1, x_2, ..., x_{N_c}\}$, and we need to select a subset $\mathcal{S}_c \subset \mathcal{D}_c$, $|\mathcal{S}_c| = m_c$ ($m_c = m / C$, where $C$ is the total number of old categories), such that the accuracy of the model after replay training on $\mathcal{S}_c$ on the complete test set is maximized.

Random sampling uniformly and randomly selects $m_c$ samples from $\mathcal{D}_c$. The problem with this approach is: (1) Most samples are located in the central region of the category distribution (typical samples), for which the model can already classify correctly, so retaining them offers almost no help in preventing forgetting; (2) A small number of samples are located near the category decision boundary (difficult samples), for which the model is most uncertain and most prone to forgetting, but random sampling may miss these critical samples.

**Specific Implementation of DAS**:
1. **Classification Uncertainty Calculation**: For each sample $x_i$ in old category $c$, calculate its uncertainty using the predicted probability distribution of the current model $f_\theta$:

$$H(x_i) = -\sum_{j=1}^{C} p_\theta(y_j | x_i) \log p_\theta(y_j | x_i)$$

where $H(x_i)$ is the prediction entropy. High entropy means the model is uncertain about the prediction for this sample (the distribution of $p_\theta(y|x_i)$ is close to uniform), while low entropy means the model is very certain (the distribution of $p_\theta(y|x_i)$ is sharp).

2. **Keyword Grouping**: Group samples by keyword category into $\{\mathcal{D}_1, \mathcal{D}_2, ..., \mathcal{D}_C\}$, ensuring that each category has representation.

3. **Intra-group Diversity Selection**: Within each category group $\mathcal{D}_c$, sort samples in descending order of entropy $H(x_i)$, prioritizing the selection of high-entropy samples (i.e., samples that the model is most likely to "confuse"). These samples represent the most vulnerable areas of the model's knowledge within that category—they are located near the decision boundary and are the "boundary knowledge" most prone to forgetting.

4. **Memory Budget Allocation**: Divide the total memory budget equally among all old categories, retaining $m_c = m / C$ samples for each category.

**Connection between DAS and Active Learning**:
The core idea of DAS—"selecting the samples the model is least certain about"—is consistent with Uncertainty Sampling in Active Learning. However, their goals differ: Active Learning selects uncertain samples to "gain more information to train a better model," while DAS selects uncertain samples to "protect the model's most vulnerable knowledge." In the context of continual learning, DAS is a "knowledge-protection-oriented uncertainty sampling" method.

**Core Technology 2: Mixed-Label Data Augmentation (MLDA)**

**Design Motivation**:
Even if $m$ old samples are carefully selected, with $m_c$ samples per category (usually 5-10), this is still insufficient to represent the complete diversity of the original data distribution. The model may overfit on these few replay samples—memorizing the features of these specific samples but failing to generalize to other samples of that category.

**Specific Implementation of MLDA**:
Perform diverse data augmentation on each stored old sample $x_i$ to generate multiple variants:

$$\tilde{x}_i^{(k)} = T_k(x_i), \quad k = 1, 2, ..., K$$

where $T_k$ is the $k$-th augmentation operation, including:
- **Time stretching/compression**: Change the speaking speed of the speech (stretching factor $\alpha \in [0.8, 1.2]$), simulating different speaking rates
- **Frequency masking** (SpecAugment): Randomly mask $\Delta f$ consecutive frequency channels ($\Delta f \in [1, 5]$), simulating loss of spectral information
- **Time masking** (SpecAugment): Randomly mask $\Delta t$ consecutive time frames ($\Delta t \in [1, 10]$), simulating brief interruptions
- **Volume variation**: Adjust signal amplitude (gain factor $g \in [0.5, 2.0]$), simulating different distances and volumes
- **Noise superposition**: Add random background noise (SNR $\in [5, 20]$dB), simulating different noise environments

The augmented samples share the label of the original sample (mixed-label strategy): $y(\tilde{x}_i^{(k)}) = y(x_i)$. This ensures the correctness of the supervision signal—the augmentation does not change the semantic category of the sample.

**Quantitative Effect**: Each stored sample generates 5-10 variants through $K=5-10$ augmentation operations. The original $m$ samples are virtually expanded into $m \times K$ training samples, effectively increasing storage efficiency by a factor of $K$. When $m=100$ and $K=10$, the equivalent training set size expands from 100 to 1000.

**Core Technology 3: Knowledge Distillation Loss (KD)**

**Design Motivation**:
Sample replay (DAS + MLDA) only protects the feature space regions corresponding to the retained samples. For regions not covered by retained samples (the central region of the category distribution—typical samples), the model may still forget. Knowledge distillation provides "implicit protection" for the entire feature space.

**Mathematical Formulas**:
1. Before learning the new task, copy the current model as the "teacher": $f_{old} = f_\theta$
2. During incremental training, for each training sample $x$ (including new data and replay data), calculate the distillation loss:

$$L_{KD} = D_{KL}(p_{old}(y|x) \| p_{new}(y|x))$$

where $p_{old}$ is the output probability distribution of the teacher model (calculated under softmax temperature $\tau$), and $p_{new}$ is the output distribution of the student model (current model).

3. Total loss:

$$L = L_{CE}^{new} + L_{CE}^{replay} + \alpha \cdot L_{KD}^{all}$$

**Protection Mechanism of KD**: The KL divergence constrains the output distribution of the student model to remain consistent with that of the teacher model. Even for category regions not covered by replay samples, the "soft labels" of the teacher model still provide a supervision signal—telling the student model "for this input, what probability distribution you should output." This is equivalent to performing "fuzzy protection" on the entire category region in the output space.

### Analysis of Synergistic Effects of the Three Components
- DAS ensures that replay samples cover the most critical knowledge regions (decision boundaries)
- MLDA expands the diversity of replay samples, making distillation effective over a wider area
- KD provides additional protection for regions not covered by replay samples (soft constraints in the output space)
- The three form a "complementary protection network": DAS selects key points, MLDA expands the neighborhood of key points, and KD fills the gaps between neighborhoods

### Design Without Task ID
RK does not require task ID information because:
- All learned categories share a unified output layer (dynamically expanded softmax layer)
- During inference, the model outputs a probability distribution $p(y|x)$ for all learned categories, directly selecting the category with the highest probability $\hat{y} = \arg\max_c p(y_c|x)$
- When new categories are added, only the dimensions of the output layer are expanded (adding new output nodes), while old output nodes remain unchanged

## Main Contributions

1. **Diversity-Aware Sampler**: First introduces an uncertainty-based sample selection strategy based on prediction entropy in CIL tasks for KWS, significantly improving information efficiency under limited storage budgets. The core insight of DAS—"retaining samples the model is least certain about rather than random samples"—is consistent with the ideas of active learning, but plays a completely new "knowledge protection" role in continual learning scenarios.

2. **Mixed-Label Data Augmentation**: Virtually expands sample diversity from limited storage through augmentation, increasing the equivalent training set size by 5-10 times without increasing physical storage. MLDA is a simple but extremely effective "storage efficiency multiplier."

3. **Lightweight Framework with Three-Component Synergy**: The three components of RK (DAS + MLDA + KD) each have their strengths and complement each other, forming a complete solution for preventing forgetting. The overall framework does not require task IDs, has lower memory overhead than competing methods, and the computational overhead of each component is low.

4. **Significant Performance Improvement**: Achieves an absolute average accuracy improvement of 4.2% over the best baseline on the Google Speech Command dataset, achieving better forgetting mitigation effects under the same memory budget.

## Experimental Results

### Datasets Used and Their Scales
- **Google Speech Command V2-35**: 35 categories, approximately 105,000 1-second speech samples. Standard train/validation/test split.
- **Incremental Learning Configuration**: Uses 20 categories from the 35 as base categories and 15 categories as incremental categories. Incremental tasks are divided into several steps, adding 3-5 new categories per step.

### Definition and Rationale for Evaluation Metrics
- **Average Accuracy (%)**: The average classification accuracy of the model for all seen categories after all incremental steps are completed. This is the standard metric for CIL.
- **Memory Requirement (KB)**: The additional storage space required to store old samples. Directly evaluates the storage efficiency of the method.

### Comparison with Baseline Methods

| Method | Average Accuracy (%) | Memory Requirement | vs Best Baseline |
|:---|:---:|:---:|:---:|
| Fine-tuning (Lower Bound) | ~65-70 | 0 | - |
| Random Replay | ~80-82 | Same | - |
| iCaRL | ~83-85 | Same | - |
| LwF (No Replay) | ~78-80 | 0 | - |
| **RK (This Paper)** | **~88-89** | **Same or Less** | **+4.2%** |

Key Finding: RK achieves an average accuracy 4.2% higher than the best baseline (iCaRL or Random Replay + Distillation) under the same or lower memory budget.

### Findings from Ablation Studies

**Independent Contributions of Each Component**:
- Complete RK -> ~88-89%
- Remove DAS (replace with random sampling) -> Accuracy drops by ~3-4% (to ~84-85%)
- Remove MLDA -> Accuracy drops by ~1.5-2%
- Remove KD -> Accuracy drops by ~1-2%
- Use only DAS (no MLDA, no KD) -> ~84-85%
- Use only MLDA (no DAS, no KD) -> ~82-83%

**Synergistic Effects of the Three Components**:
The effect of combining the three components (88-89%) is greater than the combination of any two components (~85-87%), and greater than the sum of the independent effects of each component (DAS ~+3-4% + MLDA ~+1.5-2% + KD ~+1-2% = Total +5-8%, but the actual improvement from baseline to complete is ~+6-8%). Evidence of synergy: Variants near the decision boundary generated by MLDA augmentation of boundary samples selected by DAS further enhance the protective effect of KD.

**DAS vs Other Sampling Strategies**:
- DAS (Entropy Sampling) > Forgetting-Aware Sampling (selecting samples most likely to be forgotten in previous training) > Random Sampling > Center-based Sampling (Herding, selecting samples closest to the class mean)
- Center-based sampling performs the worst (~2-3% drop), because it selects the most "typical" samples—samples that are easiest for the model to classify, offering the lowest marginal benefit for protection.

### Performance Curves for Sequential Learning
- The accuracy drop curve of RK is significantly flatter than all baseline methods
- After 5 incremental steps, RK accuracy is ~88%, Random Replay ~82%, iCaRL ~85%, Fine-tuning ~65%
- As the number of incremental steps increases, the advantage of RK continues to expand—because DAS selects the most protective samples at each step, and MLDA expands the coverage of these samples

## Limitations and Future Work

### Technical Limitations of the Method
- **Computational Overhead of Uncertainty Estimation**: DAS requires calculating the classification entropy $H(x_i)$ for all candidate samples during sampling, which requires one forward pass for all old samples. If the number of old samples is large (e.g., tens of thousands), the computational cost of this step cannot be ignored. However, this is executed only once during the "sampling phase" of incremental learning (not during each inference), so it has no impact on inference performance.
- **Quality of Uncertainty Estimation Depends on Model**: The effectiveness of DAS depends on the accuracy of the model's uncertainty estimation. In early training stages (when the model is not fully trained, and $H(x_i)$ does not accurately reflect the true difficulty of samples) or when the model is overconfident (when $H(x_i)$ is generally low, making it difficult to distinguish between difficult and simple samples), the quality of entropy estimation declines.
- **Predefined Nature of Augmentation Strategies**: The augmentation strategies of MLDA (time stretching, frequency masking, etc.) are manually designed and may not cover all possible acoustic variations. Especially in real applications, new keywords may have unique acoustic characteristics (e.g., fundamental frequency pattern changes in tonal languages), and predefined augmentation strategies may be insufficient.
- **Privacy Issues with Stored Samples**: RK requires storing spectrogram data of a small number of old samples on the device. Although the storage amount is small, in strict privacy scenarios (e.g., medical, financial), even the storage of a small number of samples may not be allowed.

### Shortcomings in Experimental Design
- **Evaluation Limited to a Single Dataset**: All experiments are evaluated only on Google Speech Commands. Limitations of this dataset include: (1) It only contains short English command words, and its applicability to multi-word wake words (e.g., "Hey Google") and tonal languages has not been verified; (2) The recording conditions are relatively clean, and the performance of incremental learning under real far-field noise conditions is unknown.
- **Incomplete Comparison with Latest CIL Methods**: Direct comparisons with methods from the same period in 2022, such as PCL-KWS (dynamic architecture methods) or MT-BT+CIL (multi-task foundation training methods), were not conducted.
- **Performance Under Different Storage Budgets Not Evaluated**: The paper evaluates performance under a fixed storage budget and does not show performance curves of DAS under different $m$ values (e.g., $m=50, 100, 200, 500$), making it impossible to quantify the advantages of DAS under different resource constraints.

### Possible Directions for Future Improvement
- **Adaptive Memory Allocation**: Dynamically allocate memory budgets based on the "forgetting risk" of each category, rather than dividing equally. Forgetting risk can be estimated through loss changes or gradient norms during training. High-risk categories are allocated more storage, while low-risk categories are allocated less.
- **Learning-Based Sampling Strategies**: Train a "sampling strategy network" using reinforcement learning or meta-learning—inputting sample features and model state to output a "retain or not" decision. This may be superior to heuristic strategies based on entropy.
- **Integration with Dynamic Architectures**: Combine the DAS+MLDA sampling strategy of RK with the dynamic architecture of PCL-KWS—simultaneously gaining the advantages of sample-level (replaying key samples) and architecture-level (dynamic subnetworks) incremental learning.
- **Learnable Augmentation Strategies**: Use the ideas of AutoAugment or RandAugment to automatically search for the combination of augmentation strategies most suitable for KWS incremental learning, replacing the manually designed MLDA.
- **Implications for the KWS Field**: RK proves that "smart data selection" is far more important than "large amounts of data storage"—under the same storage budget, carefully selecting 5 difficult samples is more effective than randomly selecting 20 samples. This principle has guiding significance for all scenarios requiring continual learning on edge devices (e.g., personalized speech recognition, online acoustic event detection).
