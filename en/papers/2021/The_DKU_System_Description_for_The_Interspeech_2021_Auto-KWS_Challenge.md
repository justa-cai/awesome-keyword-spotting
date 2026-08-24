# The DKU System Description for The Interspeech 2021 Auto-KWS Challenge

- **Authors/Affiliations**: Shuai Wang, Xuewen Zhang, Zhenchun Liu, Yonghong Yan - Duke Kunshan University (DKU)
- **Date**: 2021.04
- **Link**: https://arxiv.org/abs/2104.04993
- **Keywords**: Auto-KWS, Neural Architecture Search, Few-Shot Learning, Weight Sharing, Hypernetwork, Transfer Learning, Challenge System

## Problem Statement

The Interspeech 2021 Auto-KWS Challenge proposes a novel and highly practical problem: how to automatically discover the optimal KWS model architecture for new keywords. The traditional KWS system development process requires domain experts to manually design and tune model architectures based on experience, a process that is time-consuming and labor-intensive without guaranteeing optimality. Unlike traditional NAS (Neural Architecture Search) challenges, the Auto-KWS Challenge evaluates three dimensions simultaneously:

1. **Search Efficiency**: Completing architecture search within a limited search budget (GPU hours). In practical applications, after a user adds a new wake word, the model adaptation must be completed within a reasonable time frame (minutes rather than days), imposing strict requirements on the efficiency of the search algorithm.

2. **Model Efficiency**: The discovered target model must satisfy strict parameter constraints (suitable for edge deployment). KWS models need to run on devices with extremely limited resources, such as MCUs or DSPs, with parameter counts typically restricted to within tens of KB.

3. **Few-Shot Performance**: Training data for new keywords is extremely limited (each keyword may have only 5-20 samples). In real-world scenarios, users cannot provide large amounts of labeled data for each new keyword; the system must be able to build a reliable detection model with few samples.

This problem reflects the core pain points in actual KWS deployment: whenever a user needs to add a new custom wake word, the system needs to quickly (search efficiency), lightly (model efficiency), and accurately (even with limited data) build a dedicated detection model. Manual architecture design is time-consuming and cannot guarantee optimality, while traditional NAS methods (such as NASNet, AmoebaNet, etc.) typically require thousands of GPU hours for search, making them completely unsuitable for practical deployment scenarios.

The DKU team designed a system solution combining NAS and few-shot learning for this challenge, conducting systematic technical exploration in three aspects: search space design, search efficiency optimization, and few-shot generalization.

## Methodology

### Overall System Workflow
The workflow of the DKU system is: define search space -> train with weight-sharing supernet -> evolutionary search for optimal architecture -> transfer learning initialization -> few-shot fine-tuning -> data augmentation -> architecture evaluation and selection. The entire process is designed to be as automated as possible, minimizing human intervention.

### Cell-Based Search Space Design
The search space adopts the classic cell-based design from the NAS field, which achieves a good balance between the expressive power of the search space and search efficiency:

- **Normal Cell**: A feature processing unit that maintains spatial resolution, used to extract and transform feature representations. The output of a normal cell has the same time and frequency resolution as its input.

- **Reduction Cell**: A downsampling unit that reduces spatial resolution, used to gradually reduce the spatial dimensions of feature maps while increasing the number of channels. Reduction cells halve the spatial resolution through operations with a stride of 2.

- **Internal Structure of Each Cell**: Each cell consists of B nodes (usually B=4-7), with nodes connected by selectable operations. Each node represents a feature map, and edges represent operations from one node to another. The goal of the search is to select the optimal operation for each edge.

- **Set of Optional Operations** includes: 3x3 depthwise separable convolution, 5x5 depthwise separable convolution, 3x3 average pooling, 3x3 max pooling, identity mapping (skip connection), and zero operation (zero, representing no connection). These operations cover the basic CNN components commonly used in KWS.

The design principle of the search space is to maintain a manageable size while covering mainstream CNN components. An excessively large search space would cause the search process to fall into combinatorial explosion, while an excessively small search space might miss the optimal architecture. The size of the DKU search space is approximately $10^8$, achieving a reasonable balance between coverage and searchability.

### Weight Sharing (Supernet) to Accelerate Search
To complete architecture search within a limited computational budget, the DKU system adopts a weight-sharing strategy, which is the core idea of One-Shot NAS:

- **Supernet Training**: A super network (Supernet) containing all possible operations in the search space is constructed, where all sub-architectures share the weights of the supernet. The supernet is a superset of all possible architectures in the search space, and its parameter count equals the sum of the parameters of all optional operations.

- **One-Shot NAS**: The supernet needs to be trained only once (rather than training each candidate architecture independently), after which its performance can be evaluated by directly extracting the weights of different sub-architectures. During training, each mini-batch randomly samples a sub-architecture for forward and backward propagation. This path-level dropout allows the supernet's weights to adapt to different sub-architectures.

- **Search Strategy**: After the supernet training is completed, an evolutionary algorithm is used to find the optimal sub-architecture in the search space. The fitness function of the evolutionary algorithm is the accuracy of the sub-architecture on the validation set (evaluated using weights inherited from the supernet). Evolutionary operations include mutation (randomly modifying certain operations in the architecture) and crossover (exchanging partial structures between two architectures).

- **Search Efficiency**: Through weight sharing, the entire search process (supernet training + evolutionary search) can be completed in a few hours on a single GPU, achieving a three-order-of-magnitude acceleration compared to traditional exhaustive search (thousands of GPU hours).

### Transfer Learning Strategy
Considering that it is difficult to obtain good models by training from scratch in few-shot scenarios, the DKU system utilizes pre-trained models for transfer learning:

- **Pre-training Source**: Acoustic models trained on large-scale speech datasets (such as 960 hours of LibriSpeech or the full dataset of Google Speech Commands) are used to initialize the feature extractor. These pre-trained models have already learned rich general speech feature representations (phoneme-level acoustic patterns, frequency features, temporal structures, etc.).

- **Fine-tuning Strategy**: A gradual unfreezing strategy is adopted—first freezing the bottom-layer feature extraction layers and only fine-tuning the high-level classifier and adaptation layers; then gradually unfreezing more layers. This approach avoids overfitting caused by deep fine-tuning on few-shot data.

- **Domain Adaptation**: A small number of iterative fine-tuning steps (usually 5-20 epochs) are performed on the few-shot target data, using a small learning rate (1/10 to 1/100 of the original learning rate) and strong regularization (weight decay, dropout) to prevent overfitting.

- **Feature-Level Transfer**: In addition to weight transfer, feature-level transfer is explored—using the intermediate layer outputs of the pre-trained model as additional input features, concatenated with the original MFCC features, and fed into the architecture discovered by the search.

### Data Augmentation
To improve generalization under few-shot conditions, multiple data augmentation strategies are employed:

- **SpecAugment**: Time warping (maximum W=5 frames), frequency masking (maximum F=10 frequency bins), and time masking (maximum T=10 frames) are applied to the spectrograms. Augmentation in the frequency domain has been proven to be particularly effective for speech tasks.

- **Noise Injection**: Random background noise from noise databases (such as MUSAN, DEMAND, etc.) is superimposed, with SNR ranging from 0dB to 20dB. This enhances the model's robustness in noisy environments.

- **Speed Perturbation**: Training audio is resampled at speeds of 0.9x, 1.0x, and 1.1x to increase diversity in the time dimension, making the model robust to different speaking rates of keywords.

- **Volume Perturbation**: The volume level of audio is randomly adjusted (+-6dB) to enhance robustness to volume changes.

- **Mixup**: Two training samples and their labels are linearly interpolated to create smooth decision boundaries in the label space, helping to mitigate overfitting in few-shot scenarios.

## Main Contributions

1. **Provided a competitive complete system for the Auto-KWS Challenge**: The DKU system achieved excellent results in the challenge, demonstrating the feasibility of NAS technology in automated KWS model design. The system provides a complete end-to-end solution from search space definition to final model deployment.

2. **Effective combination of NAS and few-shot learning**: It is proven that under the synergy of search space design, weight-sharing acceleration, and few-shot fine-tuning, effective KWS architectures can be automatically discovered even when each new keyword has very few training samples (5-20). This combination is key to solving the problem of rapid customization in KWS.

3. **Efficient search workflow**: The computational cost of architecture search is reduced from thousands of GPU hours to a few hours through supernet weight sharing, making automated architecture search feasible in practical scenarios. The improvement in search efficiency is mainly attributed to the training paradigm of One-Shot NAS.

4. **Validated the value of transfer learning for few-shot KWS**: General acoustic representations transferred from large-scale speech models provide a strong initialization for feature extraction of new keywords. The intermediate layer features of pre-trained models contain rich phoneme-level information, which is crucial for few-shot KWS tasks.

5. **Systematic data augmentation strategy**: By combining multiple data augmentation methods, the diversity of training data is effectively expanded without increasing the amount of labeled data, mitigating the overfitting problem in few-shot scenarios.

## Experimental Results

### Auto-KWS Challenge Results
- The DKU system achieved a competitive ranking (top positions) in the Auto-KWS Challenge.
- It performed well in both search efficiency metrics (GPU time) and final model performance metrics, validating the practicality of the weight-sharing NAS method in the KWS scenario.

### NAS-Discovered Architectures vs. Baselines
- The architectures automatically discovered by NAS outperformed hand-designed baseline models (such as standard DS-CNN) under the same parameter constraints (<50K parameters), with an accuracy improvement of approximately 1-3%.
- The discovered architectures exhibit some common characteristics: a tendency to use more depthwise separable convolution operations (especially 3x3 kernels) and fewer pooling operations—this is consistent with manual design experience in the KWS field, validating that NAS can automatically discover design principles consistent with human intuition.
- The discovered architectures usually have deeper networks (6-10 layers) with moderate channel numbers per layer. This "narrow and deep" design outperforms "wide and shallow" designs under small parameter budgets.

### Effect of Transfer Learning
- After transferring from pre-trained models, few-shot performance (5-10 samples per keyword) is significantly better than training from scratch, with an accuracy improvement of approximately 5-10%.
- Fine-tuning only the top 3-4 layers achieves most of the performance improvement (about 80% of the gain), while deeper fine-tuning is more prone to overfitting. This indicates that the bottom-layer features of pre-trained models are already general enough for few-shot KWS tasks.
- There is little performance difference between models pre-trained on LibriSpeech and those pre-trained on GSC as transfer sources, indicating that the effect of transfer learning mainly comes from general acoustic feature representations rather than domain knowledge specific to certain datasets.

### Contribution of Data Augmentation
- SpecAugment brings consistent performance improvements (approximately 1-2%) across all configurations, making it the most cost-effective augmentation strategy.
- In extremely few-shot scenarios (1-shot, 5-shot), the gains from data augmentation are more significant (approximately 3-5%), because augmentation effectively expands the limited training data.
- Combining multiple augmentation strategies (SpecAugment + noise injection + speed perturbation) outperforms using any single augmentation method alone.

### Ablation Studies
- **Impact of Search Space Size**: A larger search space (more optional operations, more nodes) may discover better architectures, but the search time also increases. The search space chosen by DKU achieves the best balance between performance and efficiency.
- **Evolutionary Search vs. Random Search**: Architectures discovered by evolutionary search consistently outperform those from random search, indicating that weights inherited in the supernet can indeed guide effective architecture selection.
- **Ranking Consistency of Weight Sharing**: The architecture ranking in the supernet is positively correlated with (but not completely consistent with) the ranking after independent training, indicating that although weight sharing introduces certain estimation bias, the overall direction is correct.

## Limitations and Future Work

### Technical Limitations
- **Restricted Search Space**: Although cell-based search spaces are efficient, they may miss optimal architectures outside the search space. Specifically, the search space does not include emerging building blocks such as attention mechanisms (e.g., SE, CBAM) and Transformer components (e.g., self-attention layers), which have been proven beneficial for KWS in subsequent research.
- **Bias in Supernet Weight Sharing**: Although the weight-sharing strategy accelerates search, the weights of different sub-architectures in the supernet interfere with each other, potentially causing the architecture ranking during the search phase to be inconsistent with the performance ranking after actual training. This "ranking inconsistency" problem is a known challenge in the NAS community.
- **System Complexity**: The complete NAS pipeline (search space definition, supernet training, architecture search, fine-tuning) adds significant development and maintenance complexity compared to directly using a fixed architecture. The barrier to entry is high for users who are not NAS experts.
- **Bottleneck in Few-Shot Performance**: Even with transfer learning and data augmentation, when the number of samples is less than 5, the model's accuracy drops significantly. The accuracy in the 1-shot scenario may be 10-15% lower than in the 10-shot scenario.

### Insufficiencies in Experimental Design
- Challenge-specific optimizations (such as specific search space designs, training strategies) may not directly generalize to other KWS scenarios.
- Detailed analysis of the computational cost of the architecture search process is limited, lacking direct comparison with other NAS methods (such as DARTS, ENAS, ProxylessNAS).
- Evaluation under noisy and far-field conditions is insufficient; only the relatively clean recording conditions provided by the challenge were used.
- There is a lack of interpretability analysis of the architectures discovered by the search—why do these architectures perform better on few-shot KWS?

### Future Improvement Directions
- Expand the search space to include attention mechanisms and lightweight Transformer components, allowing the search space to cover more potential excellent architectures.
- Explore differentiable NAS (DARTS) methods to further improve search efficiency, reducing search time from hours to minutes.
- Research incremental architecture search to support adding new keywords on top of already deployed models, rather than searching for new architectures from scratch.
- Jointly optimize NAS with model compression (quantization, pruning) to directly search for quantization-friendly architectures suitable for low-precision inference.
- Insights for the KWS field: The Auto-KWS Challenge has promoted the application of NAS in the speech domain, proving that NAS can automatically discover effective architectures in few-shot KWS scenarios. In the future, hardware-aware NAS can be explored to simultaneously optimize model inference performance (latency, power consumption, memory usage) on specific devices.
