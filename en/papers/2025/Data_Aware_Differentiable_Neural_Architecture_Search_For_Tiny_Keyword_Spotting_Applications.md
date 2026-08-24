# Data Aware Differentiable Neural Architecture Search for Tiny Keyword Spotting Applications

- **Authors/Affiliations**: Yujia Shi, Emil Njor, Xenofon Fafoutis (Technical University of Denmark, DTU); Pablo Martínez-Nuevo, Sven Ewan Shepstone (Bang & Olufsen, Danish audio equipment manufacturer)
- **Date**: July 21, 2025 (arXiv v1); officially published at IEEE International Workshop on Machine Learning for Signal Processing (MLSP) 2025 (August 31 – September 3, 2025, Istanbul)
- **Link**: https://arxiv.org/abs/2507.15545 ; Code implementation https://github.com/ss1k/Data-Aware-NAS
- **Keywords**: differentiable NAS, DARTS, data-centric AI, TinyML, keyword spotting, MFCC hyperparameter search, neural architecture search
- **Funding**: Danish Innovation Fund DIREC project (9142-00001B)

## Problem Statement

### Problem Background and Domain Pain Points

TinyML deploys machine learning models directly onto low-power devices, promising three benefits: low energy consumption, no dependency on network infrastructure, and data staying on-device (privacy). Keyword Spotting (KWS) is one of the most typical deployment scenarios for TinyML—smart earbuds and speakers must continuously listen for wake words under a milliwatt-level power budget. However, the industry pain point identified at the beginning of the paper is not that models are not small enough, but that **there is a severe scarcity of people who can build such systems**: successfully designing TinyML systems requires engineers to master three things simultaneously—designing high-performance machine learning models, optimizing models to minimize resource consumption, and implementing them under embedded system constraints. Engineers with this composite background are extremely rare, causing the development pipeline for new TinyML applications to often bottleneck on human resources (Section 1).

For KWS, this pain point has a frequently overlooked specific manifestation: **feature extraction configuration itself is a set of heavy design decisions**. To extract MFCC features, one must first determine the window length, frame shift, and number of Mel filter banks. The window length determines the time-frequency resolution trade-off (long windows offer high frequency resolution but poor time resolution, and vice versa), the frame shift determines the number of frames (i.e., sequence length; larger frame shifts result in fewer frames, reducing subsequent computation but potentially losing phoneme transients), and the number of Mel filter banks determines the number of channels on the frequency axis (directly affecting the input width and parameter count of the first convolutional layer). In traditional workflows, these parameters are manually determined by engineers based on experience **before** architecture design. Once fixed, the model architecture can only be optimized on a predetermined input. Input fidelity (high resolution, complex features) and model complexity (depth, width) are actually competing for the same memory and compute budget, but traditional workflows split them into two sequential stages where they cannot see each other: the compute saved at the input end is unknown to the model end, and the input precision required by the model end is not perceived by the input end.

### Specific Shortcomings of Existing Methods

Section 2 of the paper systematically reviews four generations of technologies for automating this pipeline and points out their respective gaps:

- **Early Black-Box NAS** (Reinforcement Learning [7], Evolutionary Algorithms [8]): Samples architectures one by one in a discrete search space, trains from scratch to evaluate, and uses evaluation results to guide the next round of sampling. The fatal problem is computational cost—evaluating each candidate architecture often requires training from scratch. Although the **supernet paradigm** [9] was later developed—using a pre-trained large network to encapsulate the entire search space, where extracting sub-networks only requires fine-tuning—the exploration of the search space itself still relies on discrete black-box optimization. The paper accurately states that black-box methods **cannot understand the landscape** of the search space, leading to "unguided search," where sampling merely revolves around the current optimal solution.
- **Differentiable Architecture Search (DARTS)** [2]: Relaxes the discrete search space into a continuous one, navigating with gradient descent, significantly improving efficiency. However, DARTS only searches for the internal structure of the model; **the preprocessing configuration of input data is given**, and the design freedom on the data side is completely outside the search scope.
- **Hardware-Aware NAS** [13]: Simultaneously compresses accuracy and resource consumption (e.g., parameter count) in multi-objective optimization, forming the basis for well-known TinyML architectures like MCUNet [10] and MicroNets [11]. However, its "resources" are only counted on the model side; **data preprocessing configurations remain fixed**, and the resource allocation problem between input fidelity and model complexity does not enter the optimization view.
- **Data-Aware NAS** [14][15]: This is a recently emerging direction that incorporates data preprocessing configurations into the search space, allowing the optimization process to explicitly manage the resource allocation of "input data fidelity vs. model complexity." However, existing work follows the black-box route of supernet-accelerated evaluation [15], and the paper explicitly states that its validation on standard benchmark datasets "is still emerging."

In summary: Differentiable search (DARTS family) is efficient but does not touch data; Data-Aware NAS touches data but does not use gradients. Each line lacks half of what the other has.

### Key Challenges to Be Solved by This Paper

Merging these two lines into "Data Aware Differentiable NAS" requires solving three specific technical challenges:

1. **Differentiability of Discrete Data Configurations**. Choosing a window length of 400 or 640 is a discrete choice; gradient descent cannot directly differentiate discrete options. It is necessary to design continuous surrogate variables and mechanisms that allow all candidate configurations to participate in the forward pass simultaneously, which directly modifies the input end of DARTS.
2. **Input Dimension Conflict**. Feature maps produced by different MFCC configurations have different dimensions (number of frames × Mel channels vary with configuration), while DARTS' continuous relaxation requires all inputs to be merged into a single dimension sample entering the supernet—the multi-configuration mixing on the data side directly contradicts the single-dimension requirement on the architecture side, necessitating a dimension alignment scheme.
3. **Convergence and Efficiency of Joint Search**. The data configuration space (only 8 MFCC combinations in this paper) is much smaller than the architecture space. If data parameters participate in alternating updates throughout the process, it is wasteful and may disturb the convergence of architecture search. A mechanism for exiting data search needs to be designed.

On the application level, the paper also needs to answer a proof-of-concept question: Can this merged scheme achieve **fewer parameters and higher accuracy** simultaneously on standard KWS benchmarks, and can it migrate successfully with **almost no framework changes** when switching to a custom task (name wake-up)?

## Methodology

### Overall Architecture Design and Design Motivation

The overall skeleton is taken from **PC-DARTS** [16] (Partial Channel Connections for Memory-Efficient DARTS), inheriting its cell-based search space and continuous relaxation parameters $\alpha$, $\beta$, and then grafting the data search space on this basis.

**Why PC-DARTS was chosen over original DARTS**: The core improvement of PC-DARTS is that during search, each edge only performs operation mixing on partial channels, significantly compressing GPU memory usage during the search phase. This choice is particularly critical for this paper—after data awareness, **each input sample must simultaneously carry the weighted mixture of all candidate data configurations**, making the forward computation and activation memory heavier than standard DARTS. Memory efficiency during the search phase becomes a hard constraint. Experimental evidence: Searching GSC for 50 epochs, the peak memory on an 80 GB Nvidia A100 was 14.59 GB, taking about 16 hours (Section 4.1)—without the memory-saving design of partial channel connections, this number would be larger.

**Cell Structure and Network Assembly** (Section 3): The search target is the internal wiring of two basic cells—the normal cell maintains feature map resolution, and the reduction cell halves the resolution and doubles the feature channels. The final network consists of multiple stacked normal cells, with reduction cells inserted at one-third and two-thirds of the network depth. Inside each cell, $\alpha$ parameters are attached to each candidate operation (various convolutions, pooling) on each edge, and $\beta$ parameters express the importance of the edge itself. The magnitude of the parameters indicates the importance of the corresponding operation/edge, optimized jointly with network weights.

**The definition of the data search space is modality-dependent** (Section 3): Image inputs can search for resolution, color encoding; audio inputs can search for sampling rate or feature extraction techniques (e.g., whether to use MFCC). This layer of abstraction means the framework is not bound to KWS but is a general interface. For the KWS experiments in this paper, the data search space consists of three groups of hyperparameters for MFCC (Table 1):

- Window length: {400, 640} samples (i.e., 25 ms / 40 ms at 16 kHz)
- Frame shift: Window length 400 paired with {100, 200}, window length 640 paired with {160, 320} (i.e., 1/4 and 1/2 of the window length, converting to 6.25/12.5/10/20 ms)
- Number of Mel filter banks: {40, 80}

Table 1 lists the 8 specific combinations. The paper emphasizes that these options "reflect typical engineering choices"—meaning the search space encloses the few档位 (levels) that engineers normally struggle with, rather than arbitrary continuous values.

### Mathematical Principles of the Core Algorithm

The basic mechanism of DARTS relaxes the output of each edge from "select one operation" to "convex combination of all candidate operations," with combination weights normalized by softmax of $\alpha$; PC-DARTS then uses $\beta$ to weight-select edges. This paper introduces the data-side $\gamma$ parameter on this basis and writes the alternating update rules as two explicit formulas (Section 3.2, equation numbers follow the original text):

**Before** the early stopping mechanism is triggered, the three search parameters are updated jointly:

$$[\alpha^{t+1}, \beta^{t+1}, \gamma^{t+1}] = [\alpha^{t}, \beta^{t}, \gamma^{t}] - \eta_{arch} \nabla_{\alpha,\beta,\gamma} L_{arch} \tag{1}$$

**After** early stopping is triggered, $\gamma$ exits the update, leaving only architecture parameters:

$$[\alpha^{t+1}, \beta^{t+1}] = [\alpha^{t}, \beta^{t}] - \eta_{arch} \nabla_{\alpha,\beta} L_{arch} \tag{2}$$

Where $\eta_{arch}$ is the architecture parameter learning rate, and $L_{arch}$ is the architecture loss, calculated on the validation set.

**Training Process** (Section 3.2): First, perform warm-up identical to DARTS, updating only model weights $w$; entering the main search phase, alternate execution—update $w$ on the training set, update $\alpha$, $\beta$, $\gamma$ on the validation set. Why use this alternating two-layer structure: This is standard practice in the DARTS lineage. Weights are fitted on the training set, while architecture/data parameters are selected on the validation set to pick the configuration with the best generalization, preventing the search process from overfitting the training set, which would lead to poor performance upon deployment.

**Discretization Exit of Search**: At the end of the search, the data configuration with the highest $\gamma$ value is taken as the final system's data configuration—completely isomorphic to how DARTS discretizes the architecture at the end of the search based on $\alpha/\beta$ strength (Section 3.1). Afterward, the entire selected system (data preprocessing + model architecture) is trained from scratch for 100 epochs to obtain final performance (Section 4).

### Key Technical Innovation 1: Data Gamma Parameter – Turning Data Configurations into Differentiable Search Variables

This is the most core step in the entire paper (Section 3.1). Breaking down the mechanism:

1. Each candidate data configuration $d$ is assigned a continuous relaxation parameter $\gamma_d$;
2. During training, **all data configurations participate simultaneously**: all $\gamma$ parameters pass through softmax, transforming raw values into percentages, and each configuration contributes to the "combined input sample" according to its percentage—i.e., the single input seen by the supernet is the weighted sum of features from all configurations;
3. Thus, $\gamma_d$ dynamically tracks the contribution of configuration $d$ to the final loss: as search moves toward lower loss, a configuration that contributes significantly to low loss and high accuracy will see its $\gamma_d$ rise, thereby increasing its proportion in subsequent combined inputs, forming a positive feedback convergence;
4. At the end of the search, take argmax.

**Why use softmax weighted mixing instead of evaluating configurations one by one**: This is the essential divergence from black-box Data Aware NAS [15]. The black-box route requires sampling and evaluating among discrete configurations, where the performance signal for each configuration must wait for a complete evaluation cycle; whereas weighted mixing allows **the gradient of every training batch to flow to all $\gamma$s simultaneously**—one forward-backward pass updates the importance estimates for all 8 configurations. This is precisely the replication of the DARTS philosophy of "continuousizing discrete space to exchange for gradient navigation" in the data dimension. The cost is that during the search phase, the supernet sees mixed inputs rather than the pure input of any single configuration. Whether this surrogate maintains ranking consistency with the performance of the final discrete configuration is not verified by the paper (see Limitations).

### Key Technical Innovation 2: Two Strategies for Dimension Alignment (Fig. 1 Zero Padding / Fig. 2 Preprocessing)

Feature maps produced by different data configurations have different dimensions (number of frames varies with frame shift, frequency channels vary with the number of Mel filter banks), while DARTS' continuous relaxation requires inputs to be aligned to a single dimension to be combined into a single input sample (Section 3.3). The paper provides two alignment strategies, explicitly stating their respective trade-offs:

- **Zero Padding Strategy (Fig. 1)**: Pads all low-dimensional data configurations with zeros to the highest dimension among all configurations. The benefit is that it **preserves all information from all data samples**; the cost is that it may introduce significant processing overhead. The paper adds a pragmatic engineering observation: if the search ultimately wins with a certain low-dimensional configuration, the deployment system can **chip away** both the input dimension and the padding zeros in the internal tensors—i.e., the padding during the search phase is merely an alignment tool and does not enter deployment.
- **Preprocessing Strategy (Fig. 2)**: Performs early processing (the paper example uses a convolutional layer) on high-dimensional configurations to reduce them to the lowest dimension among all configurations. The benefit is **low processing demand**; the risk is that important information stored in high-dimensional configurations is processed away prematurely, leading to decreased prediction performance.

**Experimental Verdict**: The paper reports that the preprocessing strategy achieved better prediction performance on GSC v0.02 (beginning of Section 4), so all main results adopt the preprocessing strategy. The quantitative difference between the two strategies is not reported by the paper.

An implicit logic worth noting: Choosing to reduce dimensions via preprocessing to the lowest dimension rather than a median dimension indicates that the search framework defaults to "letting the preceding convolutional layers decide how to compress the extra dimensional information," handing the decision of information compression over to learnable parameters rather than hard truncation.

### Key Technical Innovation 3: Early Stopping Mechanism for Data Configuration Search

When the largest $\gamma$ among all configurations reaches **twice** the second-largest $\gamma$, the data configuration search stops, and subsequently, only architecture parameters are updated according to formula (2) (Sections 3.1, 3.2). Why is this mechanism needed: The data search space (8 MFCC combinations) is several orders of magnitude smaller than the architecture space. Once the $\gamma$ of a certain configuration significantly wins, the marginal benefit of continuing to update $\gamma$ approaches zero, while it may instead slow down architecture search or introduce perturbations. Early stopping concentrates search resources on the truly large architecture space. The threshold of 2x is a heuristic setting; the paper does not perform threshold sensitivity analysis.

### Technical Differences with Existing Methods

| Dimension | DARTS/PC-DARTS | Hardware Aware NAS | Data Aware NAS [14][15] | This Paper |
|---|---|---|---|---|
| Search Model Architecture | Yes (Gradient Navigation) | Yes (Mainly Black-Box) | Yes (Supernet Black-Box) | Yes (Gradient Navigation) |
| Search Data Configuration | No | No | Yes | Yes |
| Source of Search Efficiency | Continuous Relaxation | — | Supernet Weight Reuse | Continuous Relaxation |
| Resource Allocation: Input Fidelity vs. Model Complexity | Not Handled | Not Handled | Explicitly Managed | Explicitly Managed |

One-sentence summary of differences: This paper elevates "data configuration selection" from a manual hyperparameter decision before the search begins to a first-class citizen that is **on the same level and shares gradient signals** with the model architecture during the search process; meanwhile, it retains the search efficiency of the DARTS family (compared to the black-box evaluation route of [15]). Additionally, it reuses the complete training protocol of DARTS (warm-up, alternating updates, validation set selection), resulting in low integration cost—this is also the confidence behind the "minimal change migration" of the name detection experiment.

## Experimental Results

### Datasets Used and Their Scales

**Main Benchmark: Google Speech Commands (GSC) v0.02** (Section 4.1, [17][18]): 105,829 audio clips, 2,618 speakers, 35-word vocabulary, each audio clip at most 1 second, sampling rate 16 kHz. Samples shorter than 1 second are zero-padded to align duration. This is the standard audio benchmark for the TinyML community (MLPerf Tiny also uses it [17]).

**Custom Task: Name Detection** (Section 4.2): Derived from GSC v0.02—retaining two name classes, "Marvin" and "Sheila," as detection targets, and grouping all other classes into a single unknown class. The motivation comes directly from the partner Bang & Olufsen's product scenario: **personalized keyword detection where smart headsets react to a user's name**. To mitigate natural class imbalance (two name classes vs. 33 unknown classes), four data augmentations were performed: pitch shifting, time stretching, reverberation, and background noise injection. The specific parameters for augmentation are not reported in the paper, pointing to the code repository.

### Definition and Rationale for Evaluation Metrics

There are only two metrics: **Parameter Count** (Parameters) and **Accuracy** (Accuracy), see Table 2, Table 3. The reason for selecting parameter count is that it serves as a proxy metric for model size/storage in TinyML—this continues the tradition of Hardware-Aware NAS. The weakness of this rationale is equally obvious: the title hangs "Tiny," yet no real embedded hardware metrics are reported (latency, energy consumption, SRAM activation memory are all unreported); parameter count is only one-third of the resource story. Search cost is reported, however: GSC search for 50 epochs, peak VRAM 14.59 GB, about 16 hours on A100 (Section 4.1); name detection search for 100 epochs (Section 4.2), its duration and VRAM are not reported.

### Detailed Comparison with Baseline Methods and SOTA

**GSC v0.02 (Table 2)**, all baselines evaluated using the discovered data configuration, "Ours Not Data Aware" uses a fixed configuration of window length 640, frame shift 160, 40 Mel filters for search (Section 4.1):

| Model | Parameters | Accuracy |
|---|---|---|
| **Ours (Data-Aware Search)** | **298 K** | **97.61%** |
| Ours Not Data Aware | 391 K | 96.48% |
| DS-CNN [19] | 1.40 M | 94.68% |
| MobileNetV3 [20] | 946 K | 82.76% |
| MobileNetV2 [21] | 2.27 M | 90.84% |
| GhostNet [22] | 5.6 M | 92.94% |
| EfficientNet [23] | 5.93 M | 88.32% |

Interpretation: The data-aware version wins on **both** axes simultaneously compared to the non-data-aware version—accuracy is 1.13 percentage points higher (97.61% vs. 96.48%), and parameters are 93 K fewer (298 K vs. 391 K, relatively ~24% fewer). Compared to the strongest baseline DS-CNN: accuracy is 2.93 percentage points higher, and parameters are only 21% of its size. Compared to EfficientNet: parameters are ~95% fewer (298 K vs. 5.93 M), and accuracy is actually 9.29 percentage points higher.

**Name Detection (Table 3)**, baselines also evaluated using the discovered data configuration (Section 4.2):

| Model | Parameters | Accuracy |
|---|---|---|
| **Ours (Data-Aware Search)** | **1.64 M** | **95.43%** |
| Ours Not Data Aware | 2.76 M | 91.62% |
| DS-CNN | 1.39 M | 93.20% |
| MobileNetV3 | 0.94 M | 84.94% |
| MobileNetV2 | 2.26 M | 93.03% |
| GhostNet | 5.59 M | 91.21% |
| EfficientNet | 5.92 M | 90.86% |

Interpretation: Data awareness brings a 3.81 percentage point improvement (95.43% vs. 91.62%) and saves 1.12 M parameters (2.76 M → 1.64 M, relatively ~41% fewer). MobileNetV3 has fewer parameters (0.94 M vs. 1.64 M) but accuracy is 10.49 percentage points lower; compared to DS-CNN (1.39 M) and MobileNetV2 (2.26 M) with similar parameter counts, accuracy is higher. The paper's expression is very restrained: admitting that it is not necessarily the unique parameter-accuracy Pareto optimal solution among all baselines (Section 4.2), but it indeed falls into the top-tier accuracy bracket.

**Relationship with True SOTA**: The paper explicitly states that direct comparison with highly optimized models like BC-ResNet [24] is not possible—they use different training pipelines, extensive data augmentation (not adopted in this paper), and dedicated network modules that may not exist in standard deployment frameworks. Making a fair comparison based solely on metrics reported in literature is difficult (Section 4.1). This is an honest statement, implying that 97.61% should not be read as "GSC New SOTA."

**What the Searched Results Look Like**:

- GSC optimal data configuration: **Window length 400, Frame shift 200, 40 Mel filters** (Section 4.1). Calculated based on 16 kHz, 1-second audio, approximately 79 frames × 40 channels (this is an estimation by the author based on the paper's sampling rate, not a number from the paper text).
- Name detection optimal data configuration: **Window length 640, Frame shift 320, 40 Mel filters** (Section 4.2), calculated as approximately 49 frames × 40 channels. The configurations selected for the two tasks are different, and both chose the cheaper 40 Mel tier rather than 80—this is precisely the value of data-aware search: **the optimal feature configuration is task-dependent and should not be cut by a global convention**. Name detection can tolerate much coarser time resolution (frame shift 20 ms); a reasonable speculation is that the discriminative information for disyllabic names (Marvin, Sheila) is concentrated in the low-frequency envelope rather than phoneme-level transients, but this is the author's analysis, not explained in the paper.
- Searched cell structures (Fig. 3–6): The GSC normal cell consists of dil_conv_3x3, dil_conv_5x5, avg_pool_3x3, skip_connect, and the reduction cell features sep_conv_3x3; in the name detection's two cell types, max_pool_3x3 appears significantly more frequently. Intuitive reading: On coarser-grained inputs (640/320), name detection search tends to use cheaper pooling aggregation instead of some convolutions. This interpretation is the author's analysis.

### Findings from Ablation Experiments

The paper's only systematic ablation is **removing the data search space** ("Not Data Aware"), with results highly consistent across both tasks:

- GSC: Accuracy 97.61% → 96.48% (**−1.13 pp**), Parameters 298 K → 391 K (**+31%**);
- Name Detection: Accuracy 95.43% → 91.62% (**−3.81 pp**), Parameters 1.64 M → 2.76 M (**+68%**).

Conclusion: Joint optimization of data and architecture **simultaneously improves accuracy and reduces parameters** on these two tasks, rather than trading one for the other. Moreover, the fixed configuration used in the ablation experiment for name detection (window length 512, frame shift 160, 40 Mel) actually has higher time resolution than the one used for GSC ablation (window length 640, frame shift 160, 40 Mel), yet the model is larger and less accurate—indirectly confirming that fixed configuration engineering intuition is unreliable.

The blanks in the ablation must also be pointed out: The zero-padding vs. preprocessing alignment strategies only have a qualitative conclusion that "preprocessing is better" (Section 4), with no quantitative numbers; the 2x early stopping threshold has no sensitivity analysis; the initialization scheme for $\gamma$ and the evolution process of $\gamma$ for data search across epochs are not reported.

## Main Contributions

1. **First combination of Differentiable NAS and Data-Aware NAS for KWS** (paper positions this as initial results): Expanded the data search space on the PC-DARTS skeleton, with model architecture and input data features jointly optimized by gradients (Abstract, Sections 3, 5).
2. **$\gamma$ Continuous Surrogate Variable Mechanism**: Introduced continuous relaxation parameters for each discrete data configuration, mixed all configurations into a single input via softmax weighting, allowing discrete configuration selection to obtain gradient signals, and taking argmax $\gamma$ at the end of search (Section 3.1).
3. **Dimension Alignment Methodology**: Two strategies of zero-padding and preprocessing and their trade-off analysis, solving the structural contradiction between multi-data configurations and DARTS' single-dimension input requirements (Section 3.3, Fig. 1/2).
4. **Dual-Task Validation**: Achieved 97.61% accuracy with 298 K parameters on the GSC v0.02 standard benchmark (Table 2), and achieved 95.43% accuracy with 1.64 M parameters on the custom name detection task for smart headset scenarios (Table 3), with the latter migrating with almost zero framework changes—proving the method is not benchmark-specific.
5. **Ablation Evidence**: After removing the data search space, accuracy dropped by 1.13 and 3.81 percentage points on the two tasks, and parameters increased by 31% and 68% respectively (Table 2/3), providing direct evidence that "data configurations are worth entering the search space."
6. **Open Source**: Complete code is on GitHub (ss1k/Data-Aware-NAS).

## Limitations and Future Work

### Technical Limitations of the Method

- **Physical Meaning of Mixed Inputs is Dubious**. During the search phase, the supernet sees the softmax weighted sum of feature maps from different MFCC configurations (different time-frequency grids)—frames produced by different window lengths/frame shifts are inherently misaligned on the time axis, and element-wise weighted averaging lacks a clear signal processing explanation. This brings the known risk of the DARTS family: **there may be a ranking gap between the surrogate ranking during the search phase and the true performance of each configuration after discretization**; the paper does not verify that the configuration with argmax $\gamma$ indeed outperforms other candidates when evaluated separately.
- **Data Search Space is Still a Small Discrete Space**. The 8 MFCC combinations (Table 1) merely bring a few habitual levels of engineers into the search; greater degrees of freedom such as sampling rate, feature type (MFCC vs. Mel Filter Bank Energy vs. Raw Waveform) are not included; the paper itself admits this in the conclusion, listing "direct gradient descent on continuous data options" as future work (Section 5).
- **Parameter Count is the Only Resource Metric**. For MCU deployment, SRAM activation memory (determining if it can run) and inference latency (determining real-time performance) often become binding constraints earlier than parameter count (Flash storage); the paper has no real hardware measurements, and the "Tiny" nature is indirectly represented by parameter count.
- **Early Stopping Threshold of 2x is a Guess**, with no sensitivity analysis; the applicability of this threshold under different data search space sizes is unknown.
- **Name Detection Ablation Configuration Selection is Opaque**: Window length 512 is not within the search space of Table 1, and the paper does not explain why (512, 160, 40) was chosen as the fixed configuration; the choice of the ablation baseline may affect the magnitude of the difference.

### Shortcomings in Experimental Design

- **No Statistical Robustness**: Search and final training were run only once, with no multi-seed repetition or variance reporting (not reported by the paper). NAS methods are notoriously sensitive to random seeds; a 1–4 percentage point improvement should ideally be backed by confidence intervals.
- **Baselines May Be Under-Tuned**: The paper admits to not using data augmentation (Section 4.1). MobileNetV3 achieves only 82.76% on GSC (Table 2), significantly lower than the common levels for this family in literature, suggesting that the training pipeline for baselines was not optimized for their respective architectures—the relative improvement brought by data awareness may partly come from weak baselines.
- **No True SOTA Comparison**: Strong models like BC-ResNet are explicitly excluded from comparison (Section 4.1); the method's competitiveness under strong training recipes is unknown.
- **Self-Positioning of Length and Completeness**: This is a 4-page workshop paper; the abstract and introduction both call it initial/early results. Many components (augmentation parameters, $\gamma$ evolution, quantitative comparison of the two strategies, search overhead for name detection) are left in the code repository and not in the paper.
- **Single Metric**: 35-word classification accuracy is not an operational metric for KWS deployment—actual systems care about the DET trade-off between false alarm rate and miss rate, which is especially needed for binary classification tasks like name detection; the paper does not report this.

### Possible Directions for Future Improvement

The paper lists three items itself (Section 5): direct gradient descent on continuous data options (breaking free from discrete level limits), designing new architecture search spaces for TinyML (the current cell space is inherited from visual NAS and may not fit audio and MCUs well), and incorporating more hyperparameter optimization. Combining with the limitation analysis in this note, the author believes the following are also worth doing: explicitly writing real resource constraints such as latency/SRAM into the search objective (truly unifying Hardware-Aware and Data-Aware in the same differentiable framework); multi-seed statistics and ranking gap verification; completing deployment measurements on real Cortex-M class hardware to fulfill the TinyML narrative; incorporating data augmentation strategies themselves into the data search space (complementing the four fixed augmentations in this paper); and using false acceptance/false rejection rate evaluation for class-imbalanced tasks like name detection, directly aligning with headset wake-up product metrics. Overall, the value of this paper lies not in the number 97.61%, but in clearing a clean path for "data configuration and model architecture to be co-optimized at the same differentiable level," and open-sourcing the code—for KWS operators, it is more suitable to be read as a method skeleton rather than a results leaderboard.
