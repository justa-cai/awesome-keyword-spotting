# Target Aware Network Architecture Search and Compression for Efficient Knowledge Transfer (TASCNet)

- **Authors/Affiliations**: S.H. Shabbeer Basha (RV University, India), Debapriya Tula (Google Research, India), Sravan Kumar Vinakota (New Jersey Institute of Technology, USA), Shiv Ram Dubey (IIIT Allahabad, India)
- **Date**: May 2022 (arXiv), accepted by Multimedia Systems Journal
- **Link**: https://arxiv.org/abs/2205.05967
- **Keywords**: network architecture search, neural network compression, filter pruning, transfer learning, Bayesian optimization, knowledge distillation

## Problem Statement

### Problem Background and Domain Pain Points
Transfer Learning is one of the most successful paradigms in deep learning—Convolutional Neural Networks (CNNs) pre-trained on large source datasets (e.g., 1,000 classes, ~1.2M images in ImageNet) can be adapted to various downstream tasks via fine-tuning. The shallow layers (layers close to the input) of pre-trained models learn general visual features (e.g., edge detection, texture patterns, color distributions) that are useful for many tasks; whereas the deep layers (layers close to the output) learn task-specific features (e.g., fine visual discrimination required for 1,000-class object classification in ImageNet), which may be over-parameterized when transferred to target tasks with significant differences.

In the Keyword Spotting (KWS) scenario, a common transfer learning approach treats audio spectrograms as "images," using CNNs pre-trained on ImageNet (e.g., VGG-16, ResNet-50) as feature extractors. However, spectrograms and natural images differ fundamentally in statistical properties: (1) The two axes of a spectrogram have different physical meanings (time axis and frequency axis), whereas both axes of a natural image are spatial axes; (2) Spectrogram pixel values represent energy (non-negative), while natural image pixel values represent color (which can be negative, e.g., in normalized images); (3) The local patterns in spectrograms (e.g., frequency trajectories of formants) are statistically distinct from the local patterns in natural images (e.g., object edges). These differences mean that the deep layers pre-trained on ImageNet may be over-parameterized for KWS tasks—containing fine-grained object recognition capabilities needed for ImageNet (e.g., distinguishing among 120 dog breeds) that are entirely useless for KWS.

### Specific Shortcomings of Existing Methods
- **Efficiency Issues with Direct Fine-Tuning**: Standard transfer learning fine-tunes all layers of the pre-trained model on target data. This retains the over-parameterization of the source task (e.g., ~138M parameters for VGG-16, ~25M for ResNet-50). When KWS datasets are small (e.g., Google Speech Commands has only ~65K samples), this leads to overfitting (the model has sufficient capacity to memorize the training set) and inefficient inference.
- **Task Mismatch in General Pruning**: Standard model pruning methods (e.g., Magnitude Pruning, FPGM, HRank) use generic redundancy metrics to decide which parameters to prune, ignoring the specific needs of the target task. For instance, magnitude pruning removes filters with "small weights," but for the KWS task, these "small weight" filters might exactly be responsible for detecting key spectral features of keywords (e.g., broadband noise patterns of high-frequency consonants)—they just happen to have small numerical values under the current weight initialization.
- **Computational Cost of NAS**: Neural Architecture Search (NAS) can design optimal architectures from scratch for a target task, but the search cost is extremely high (e.g., DARTS requires ~1.5 GPU days, ENAS requires ~0.5 GPU days). Furthermore, it cannot leverage the knowledge of pre-trained models—the search starts from random initialization, wasting the general features already learned in the pre-trained model.

### Key Challenges Addressed by This Paper
How to automatically find the optimal configuration of "which layers in the pre-trained CNN are over-parameterized" and "how many parameters should be retained" in the pre-trained CNN, such that the fine-tuned model is both efficient (low parameter count, low FLOPs) and effective (high accuracy) on the target task, while utilizing Bayesian optimization to efficiently search the configuration space and avoid the high cost of exhaustive search.

## Methodology

### Overall Architecture Design
TASCNet (Target Aware Architecture Search and Compression Network) is a two-stage automated framework:

**Stage 1: Target-Aware Architecture Search** – Automatically adjusts the configuration of the deep layers of the pre-trained model using Bayesian optimization.
**Stage 2: Cosine Similarity Filter Pruning** – Further removes redundant filters after training.

The two stages are executed sequentially: first determining the optimal architecture, then performing redundancy elimination on that optimal architecture.

### Mathematical Principles of Core Algorithms

**Stage 1: Architecture Search Based on Bayesian Optimization**

**Definition of Search Space**:
For the deep layers (typically the last $D$ convolutional blocks and fully connected layers) of the pre-trained CNN, the hyperparameter space is defined as:

$$\mathcal{H} = \{h_1, h_2, ..., h_D\}$$

where $h_i$ represents the proportion of parameters or the number of filters retained in the $i$-th block. For example, for the last convolutional block of VGG-16 (originally 512 filters), $h_i \in \{64, 128, 256, 512\}$.

The design of the search space is based on the observation that the shallow layers (first $D_{early}$ layers) of pre-trained CNNs learn general low-level features (e.g., edges, textures) that are useful for most tasks and can remain unchanged; whereas the features of the deep layers (last $D_{deep}$ layers) are more task-specific and require adjustment of parameter counts according to the target task.

**Bayesian Optimization Process**:
1. **Surrogate Model**: A Gaussian Process (GP) is used as the surrogate model to model the "configuration-performance" mapping in the search space:

$$f(\mathbf{h}) \sim \mathcal{GP}(m(\mathbf{h}), k(\mathbf{h}, \mathbf{h}'))$$

where $m(\cdot)$ is the mean function (usually set to 0), and $k(\cdot, \cdot)$ is the kernel function (usually the Matérn 5/2 kernel, which strikes a good balance between smoothness and flexibility).

2. **Acquisition Function**: Expected Improvement (EI) is used as the acquisition function to balance "exploration" (trying new configurations) and "exploitation" (improving known good configurations):

$$\alpha_{EI}(\mathbf{h}) = \mathbb{E}[\max(f(\mathbf{h}) - f(\mathbf{h}^+), 0)]$$

where $\mathbf{h}^+$ is the current optimal configuration. The advantage of EI is that it considers both the magnitude of the predicted value (exploitation) and the uncertainty of the prediction (exploration).

3. **Evaluation Process**:
   - For each candidate configuration $\mathbf{h}$, modify the corresponding layers of the pre-trained CNN (e.g., reduce the number of filters in a layer from 512 to $h_i$).
   - Fine-tune the modified model on the target dataset (a small number of epochs, e.g., 5-10 epochs for rapid evaluation).
   - Record the validation set accuracy as the feedback signal for Bayesian optimization.
   - Update the posterior distribution of the GP.

4. **Iterative Search**: Bayesian optimization updates the surrogate model based on historical evaluation results, gradually focusing on more promising regions of the configuration space. Near-optimal configurations are typically found within 20-50 evaluations.

**Bayesian Optimization vs. Other Search Strategies**:
- vs. Grid Search: In a search space with $D$ dimensions and 4 candidate values per dimension, grid search requires $4^D$ evaluations. For $D=5$, this requires 1,024 evaluations—each requiring training a model, totaling thousands of GPU hours.
- vs. Random Search: While random search is more efficient than grid search (Bergstra & Bengio, 2012), it cannot utilize historical evaluation information to guide the search direction.
- vs. Bayesian Optimization: Utilizes GP uncertainty estimates to efficiently explore the search space, finding near-optimal configurations within 20-50 evaluations.

**Stage 2: Filter Pruning Based on Cosine Similarity**

**Design Motivation**:
After the architecture search determines the optimal parameter count for each layer, the model may still contain redundant filters during fine-tuning—i.e., filters that are functionally similar and have overlapping contributions to the output. Although these filters are "allowed" in number, they are functionally redundant.

**Mathematical Definition of Cosine Similarity**:
For a pair of filters $(W_i^l, W_j^l)$ in layer $l$ (where $W_i^l \in \mathbb{R}^{k \times k \times C_{in}}$ is the weight tensor of the $i$-th convolutional kernel), their cosine similarity is calculated over the entire training process:

$$\cos(W_i^l, W_j^l) = \frac{W_i^l \cdot W_j^l}{\|W_i^l\|_2 \cdot \|W_j^l\|_2}$$

**Why Use Cosine Similarity Instead of Magnitude**:
- Magnitude Pruning removes filters with small $\|W_i^l\|_2$. However, small magnitude does not imply unimportance—a filter might have small weights but extract critical sparse features (e.g., high-frequency transient feature detectors in KWS).
- Cosine similarity measures the similarity of the "direction" of filters. Filters with similar directions extract almost identical features (regardless of magnitude), so removing one is indeed redundancy elimination. Geometrically, cosine similarity measures the angle between two vectors on the unit sphere—the smaller the angle, the more similar the direction.

**Pruning Strategy**:
1. During fine-tuning, calculate the cosine similarity of all filter pairs every $T$ epochs.
2. Mark filter pairs with cosine similarity consistently above a threshold $\theta$ (e.g., $\theta > 0.9$) as "redundant pairs."
3. In redundant pairs, remove the filter with the smaller $\|W\|_2$ (keeping the "stronger" one).
4. Gradually remove redundant filters while monitoring validation set accuracy.
5. Stop pruning when accuracy drops by more than $\delta$ (e.g., 1%).
6. Perform a few epochs of fine-tuning (recovery fine-tuning) after pruning to recover performance loss caused by pruning.

**Consistency Check "Over the Entire Training Process"**:
Two filters are only marked as redundant if they maintain high cosine similarity across multiple training checkpoints. This avoids misjudging "accidental similarities"—two filters might happen to be similar at one checkpoint but differ significantly at others; such cases should not be considered redundant.

### Application to KWS
Although TASCNet is primarily validated on image classification tasks, the paper also demonstrates its application in KWS:
- Uses pre-trained VGG-16/ResNet-50 as feature extractors.
- Input: Mel spectrograms of audio (40 frequency channels x ~100 time frames), transforming KWS into an "image classification" task.
- TASCNet automatically adjusts the configuration of the deep layers of the pre-trained CNN (reducing the number of filters) to adapt to the spectrogram classification needs of KWS.

## Main Contributions

1. **Two-Stage Automated Framework**: TASCNet is the first to unify NAS and pruning into a single automated workflow for efficient knowledge transfer. The NAS stage solves the "architecture adaptation" problem (how many parameters are appropriate for each layer), and the pruning stage solves the "redundancy elimination" problem (which parameters are redundant). The two stages are complementary—NAS provides macro-level architecture optimization, while pruning provides micro-level weight optimization.

2. **Target-Aware Architecture Search**: Unlike general NAS (which searches for a complete architecture from scratch), TASCNet's search is an "incremental search" based on a pre-trained model—it only adjusts the deep parameters that need adaptation, retaining the pre-trained features of the shallow layers. This significantly reduces the search space and search cost (from thousands of candidate configurations to dozens) while preserving the general feature extraction capability of the pre-trained model in the shallow layers.

3. **Cosine Similarity Pruning Criterion**: Redundancy detection based on the cosine similarity of filter pairs during training is more robust than pruning methods based on single snapshots (e.g., magnitude pruning using only the weight values of the final training state). Multi-checkpoint consistency checks avoid mis-pruning due to accidental similarities.

4. **Cross-Domain Validation**: The method's generality is validated on image classification (CalTech-101/256, Stanford Dogs) and KWS. The successful application across domains indicates that TASCNet does not rely on the statistical properties of specific data domains.

## Experimental Results

### Datasets Used and Their Scales
- **Image Classification**:
  - CalTech-101: 101 object classes, ~9,000 images
  - CalTech-256: 256 object classes, ~30,000 images
  - Stanford Dogs: 120 dog breeds, ~20,000 images
- **KWS**: Transfer learning using VGG-16, ResNet-50 on spectrograms (the specific KWS dataset is not explicitly stated in the paper).
- **Pre-trained Models**: VGG-16 (~138M parameters), ResNet-50 (~25M parameters), pre-trained on ImageNet.

### Definition and Rationale for Evaluation Metrics
- **Accuracy (%)**: Classification accuracy.
- **Trainable Parameters (M)**: Total number of parameters for fine-tuning.
- **FLOPs (G)**: Floating-point operations required for inference.
- **Compression Ratio**: Percentage reduction in parameters.

### Detailed Comparison with Baseline Methods and SOTA

**Results on CalTech-101 (Using VGG-16 as the pre-trained model)**:
| Method | Accuracy (%) | Parameter Reduction | FLOPs Reduction |
|:---|:---:|:---:|:---:|
| Direct Fine-Tuning of VGG-16 | ~95.0 | 0% | 0% |
| Magnitude Pruning (50%) | ~94.2 | 50% | 40% |
| TASCNet (This Paper) | ~95.5 | **60%** | **45%** |

TASCNet achieves slightly higher accuracy than direct fine-tuning (+0.5%) while reducing parameters by 60%. This may be because reducing parameters acts as regularization, preventing overfitting on small datasets.

**Results on Stanford Dogs**:
- TASCNet achieves approximately 0.3-0.5% higher accuracy than direct fine-tuning, while reducing parameters by ~40%.

**Specific Results on KWS**:
- Using VGG-16 as the pre-trained model on spectrogram-based KWS:
  - Parameter count reduced by ~40%
  - FLOPs reduced by ~30%
  - Accuracy remains largely unchanged (drop <1%)

### Findings from Ablation Studies

**Contribution of Stage 1 (NAS)**:
- Using only the NAS stage can reduce parameters by ~30-40%.
- The configurations found by NAS typically reduce the number of filters in deep layers by 50-75%, while keeping shallow layers unchanged.
- This validates the hypothesis that "the deep layers of pre-trained models are over-parameterized."

**Contribution of Stage 2 (Pruning)**:
- Further reduces parameters by 10-20% on top of NAS.
- Pruning mainly removes redundant filters in the "middle layers" that NAS did not touch.
- Cosine similarity pruning yields ~0.5-1% higher accuracy than magnitude pruning at the same compression rate.

**Necessity of Two Stages**:
The effect of using only pruning (skipping NAS) is significantly worse than the two-stage approach. For example, directly applying cosine similarity pruning to the original VGG-16 results in a ~2% accuracy drop when parameters are reduced by 30%; whereas the NAS+Pruning scheme achieves nearly lossless accuracy with a 40% parameter reduction. This indicates that architecture adaptation (determining the optimal parameter count for each layer) is a more important optimization dimension than weight pruning (removing redundant weights).

## Limitations and Future Work

### Technical Limitations of the Method
- **Serial Complexity of Two Stages**: The two stages need to be executed sequentially (NAS -> Fine-tuning -> Pruning -> Re-fine-tuning), making the total workflow more time-consuming than single-stage methods. The Bayesian optimization phase requires rapid fine-tuning evaluations for 20-50 candidate configurations (5-10 epochs each), totaling 100-500 fine-tuning epochs.
- **Limitations of Search Space**: Only the number of filters/neurons in deep layers is searched, ignoring other architectural dimensions (e.g., kernel size, network depth, skip connection patterns). These dimensions may significantly impact KWS performance—for example, KWS might benefit from larger time-dimension convolutional kernels (to capture longer temporal dependencies).
- **Domain Gap of Pre-trained Models**: Using ImageNet pre-trained models for audio spectrograms involves a fundamental domain mismatch—the statistical priors of ImageNet (e.g., 1/f spectral characteristics of natural images) are completely different from the statistical properties of spectrograms. The meaning of "general features" (e.g., edge detection) learned in the shallow layers differs between spectrograms and natural images.
- **Dimensionality Limitations of Bayesian Optimization**: As the search dimension increases (e.g., searching configurations for all layers), the complexity of GP posterior updates is $O(N^3)$ ($N$ is the number of evaluations), requiring more evaluations to converge in high-dimensional spaces.

### Shortcomings in Experimental Design
- **Insufficiently Deep KWS-Specific Evaluation**: The KWS experiments only use the simple scheme of spectrograms + pre-trained image CNNs, without comparison to specialized KWS architectures (e.g., DS-CNN, TC-ResNet, KWT). These specially designed architectures are typically more effective than the "spectrogram + image CNN" scheme.
- **Lack of Detailed Evaluation on Standard KWS Datasets**: Detailed results (e.g., performance under different noise conditions, accuracy comparison with SOTA KWS models) are not reported on standard KWS benchmarks like Google Speech Commands.
- **Insufficient Discussion of Domain Gap from Spectrogram to Image**: Treating audio spectrograms as images for transfer learning involves a fundamental domain mismatch—the two axes of a spectrogram have different physical meanings (time axis and frequency axis), whereas the two axes of a natural image are homogeneous (both are spatial axes). The spatial invariance learned by pre-trained models may not apply to spectrograms.

### Possible Directions for Future Improvement
- **End-to-End Integrated Optimization**: Merge NAS and pruning into a single optimization process. For example, use Differentiable NAS to simultaneously search for architecture configurations and pruning masks, reducing serial steps.
- **Audio-Specific Pre-trained Models**: Replace ImageNet pre-trained models with models pre-trained on large-scale audio data (e.g., PANNs—Pre-trained Audio Neural Networks on AudioSet, AST—Audio Spectrogram Transformer). Features learned in the shallow layers of audio-specific pre-trained models (e.g., spectral textures, time-frequency patterns) are more directly useful for KWS.
- **Task-Aware Search Space**: Automatically define the search space based on the characteristics of the target task. For example, the KWS task may require larger time-dimension convolutional kernels and fewer frequency-dimension filters—these priors can narrow the search space and improve search efficiency.
- **Single-Stage Joint Optimization**: Explore schemes that perform architecture search and pruning simultaneously (e.g., using learnable pruning gates + differentiable architecture parameters) to determine the optimal architecture and optimal pruning mask in a single training run.
- **Insights for the KWS Domain**: TASCNet reveals the problem of "over-parameterization in transfer learning"—the architecture of a pre-trained model is not necessarily suitable for the target task. This insight encourages KWS researchers to perform target-aware architecture adaptation when using pre-trained models, rather than blindly fine-tuning end-to-end. More broadly, for any scenario involving cross-domain transfer learning (e.g., from images to audio, from text to speech), target-aware architecture adaptation is worth considering.
