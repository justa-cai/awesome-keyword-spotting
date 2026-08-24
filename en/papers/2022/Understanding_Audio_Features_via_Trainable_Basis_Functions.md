# Understanding Audio Features via Trainable Basis Functions

- **Authors/Affiliations**: Kwan Yee Heung, Kin Wai Cheuk, Dorien Herremans (Singapore University of Technology and Design; Kin Wai Cheuk is also affiliated with the Agency for Science, Technology and Research)
- **Date**: April 2022
- **Link**: https://arxiv.org/abs/2204.11437
- **Keywords**: spectrogram, trainable front-end, feature extraction, keyword spotting, speech recognition, basis functions, nnAudio, FastAudio

## Problem Statement

### Problem Background and Domain Pain Points
The performance of audio deep learning systems largely depends on how input features are represented. The spectrogram is the most common audio input representation, and its computation involves a multi-stage signal processing pipeline, where each step relies on specific basis functions:

1. **STFT (Short-Time Fourier Transform)**: Uses sinusoidal window functions (such as Hanning or Hamming windows) as basis functions to decompose the time-domain signal into a time-frequency representation. The parameters of the window function (e.g., window lengths of 256/512/1024 samples) determine the trade-off between time resolution and frequency resolution—shorter windows provide better time resolution (suitable for capturing transient features of consonants), while longer windows provide better frequency resolution (suitable for resolving the fine positions of formants).

2. **Mel Filterbank**: Uses triangular or trapezoidal filters as basis functions to map the linear frequency axis to the Mel scale (simulating the human ear's non-linear perception of frequency—higher resolution in low-frequency regions and lower resolution in high-frequency regions). The parameters of the Mel filterbank (e.g., center frequencies and bandwidths of 40 filters) determine the distribution of frequency resolution along the frequency axis.

3. **MFCC (Mel-Frequency Cepstral Coefficients)**: Uses DCT (Discrete Cosine Transform) basis functions to decorrelate the output of the Mel filterbank, extracting the most discriminative low-dimensional features. The first 13 DCT coefficients typically contain the most critical spectral envelope information.

The parameters of these basis functions (window size, center frequencies and bandwidths of Mel filters, number of DCT coefficients retained) are usually manually set based on prior knowledge of human auditory perception and decades of experience in speech processing research. However, whether these manually set parameters are optimal for specific deep learning tasks is a fundamental question that has not been fully explored.

### Specific Shortcomings of Existing Methods
- **"Black Box" Input of Fixed Feature Extraction**: In standard audio deep learning pipelines, spectrogram computation is treated as a fixed preprocessing step, and its parameters do not participate in end-to-end backpropagation optimization. This means: (1) the model cannot adaptively adjust frequency resolution according to task requirements—certain frequency bands crucial for the KWS task (e.g., high-frequency energy of the consonant /s/ concentrated in 3000-7000Hz) may be overly compressed in the Mel filterbank (40 filters have very low resolution in the high-frequency region); (2) the optimal frequency resolution may differ across tasks (KWS requires distinguishing fine spectral textures, ASR requires capturing phoneme sequences, music classification requires identifying chords and rhythms), yet the same manually set parameters are used.
- **Lack of Theoretical Understanding of "Optimal Basis Functions"**: Why does the Mel scale perform well on KWS? Is it because it simulates human auditory perceptual characteristics, or because its non-linear frequency resolution happens to match the spectral energy distribution of speech signals? Are 40 Mel filters better than 20 or 80? If so, why? These important questions lack systematic answers based on experimental evidence.
- **Shape Constraints of FastAudio**: Previous work, FastAudio, proposed trainable Mel basis functions but introduced shape constraints (such as monotonicity—filter amplitude monotonically decreases from the center to both sides; symmetry—filters are symmetric about the center frequency) to ensure that the learned filter shapes "look like Mel." While the rationale for this constraint is reasonable (preventing filters from degenerating into meaningless shapes), it may limit the model's ability to discover completely different optimal filter shapes—perhaps the filters required for the KWS task do not resemble Mel filters at all.

### Key Challenges Addressed by This Paper
If the basis functions in spectrogram computation (particularly the Mel filterbank) are allowed to be freely optimized via backpropagation (without imposing any shape constraints), can deep learning models learn basis functions better suited to specific tasks than those manually designed? What do these learned basis functions look like, and what information do they reveal about the task? How does the effectiveness of trainable basis functions change under different model complexities?

## Methodology

### Overall Experimental Framework
This paper uses two trainable front-end tools—nnAudio and FastAudio—to embed the spectrogram computation layer into an end-to-end deep learning pipeline, allowing the parameters of the Mel filterbank (gMel) and the STFT window function (gSTFT) to be optimized via backpropagation.

### Technical Details of Trainable Front-End Tools

**nnAudio (Unconstrained Trainable Front-End)**:
- A PyTorch-based library for trainable spectrogram computation
- Allows parameters of gMel (Mel filterbank weight matrix) and gSTFT (STFT window function coefficients) to participate in gradient calculation
- Imposes no shape constraints—learned filters can be of any shape (multi-peaked, asymmetric, or even oscillatory)
- gMel parameter matrix: $W_{mel} \in \mathbb{R}^{n_{mel} \times (N_{FFT}/2 + 1)}$, where $n_{mel}$ is the number of Mel filters and $N_{FFT}$ is the number of FFT points
- gSTFT parameter vector: $\mathbf{w}_{STFT} \in \mathbb{R}^{N_{FFT}}$

**FastAudio (Constrained Trainable Front-End)**:
- A variant of nnAudio that introduces shape constraints
- Constraints include: (1) non-negative filter amplitudes ($W_{mel} \geq 0$); (2) each filter monotonically increases then monotonically decreases along the frequency axis (maintaining a Mel-like triangular shape); (3) appropriate overlap between adjacent filters
- Constraints are implemented via parameterization (e.g., using sigmoid functions to ensure non-negativity) rather than hard constraints

### Controlled Experimental Design with Four Training Settings

| Setting | gMel | gSTFT | Description |
|:---:|:---:|:---:|:---|
| A (Baseline) | Fixed | Fixed | Standard fixed spectrogram front-end; parameters do not participate in gradient updates |
| B | Trainable | Fixed | Only optimizes Mel filter shapes; STFT window function remains standard (Hanning window) |
| C | Fixed | Trainable | Only optimizes STFT window function; Mel filterbank remains standard Mel scale |
| D | Trainable | Trainable | Simultaneously optimizes Mel filters and STFT window; maximum freedom |

The scientific value of this controlled design lies in: comparing A vs B quantifies the contribution of "Mel filter optimization"; comparing A vs C quantifies the contribution of "STFT window optimization"; comparing A vs D quantifies the overall effect of "joint optimization"; comparing D vs B+C verifies whether the optimizations of the two dimensions are complementary.

### Experimental Tasks

- **KWS (Keyword Spotting)**:
  - Dataset: Google Speech Commands V2-12, 12 command word classes, approximately 105,000 1-second audio clips
  - Models: BC-ResNet (approximately 500K parameters) and a simple linear model (only one fully connected layer, approximately 5K parameters)
  - Number of Mel basis functions: 10, 20, 30, 40 (systematically varied to study the relationship between the number of basis functions and trainable effectiveness)
  - Input features: 40-dimensional Mel filterbank energy (standard configuration)

- **ASR (Automatic Speech Recognition)**:
  - Evaluated the impact of trainable basis functions on more complex speech tasks using the TIMIT dataset
  - Evaluation metric: Phone Error Rate (PER)

### Loss Functions and Optimization
- KWS: Cross-entropy loss $L_{CE}(y, \hat{y})$
- ASR: CTC loss
- Optimizer: Adam, learning rate 0.001
- Trainable basis functions use the same learning rate as model parameters or a separately tuned learning rate (sensitivity analysis shows little difference between the two)
- During training, the parameters of trainable basis functions change with gradient updates—initial values use standard Mel filterbanks (good initialization is crucial for convergence)

### Analysis Methods
- **Visualization of Learned Filter Shapes**: Plot the trained Mel filterbanks (weight distribution of each filter along the frequency axis) to analyze their frequency response characteristics
- **Frequency Importance Analysis**: Infer which frequency bands are most important for specific tasks by analyzing the energy distribution of learned filters
- **Model Complexity Interaction**: Test the effectiveness of trainable basis functions under different model complexities (from simple linear models to BC-ResNet)

## Main Contributions

1. **Up to 14.2 percentage point improvement in KWS accuracy**: Using trainable basis functions (Setting D, nnAudio, 40 Mel basis functions) achieved an accuracy improvement of up to 14.2 percentage points over fixed basis functions (Setting A) on the KWS task. This is a very significant improvement—in deep learning research, it is extremely rare for a single technique to contribute an absolute accuracy improvement of 14%. It proves that the choice of basis functions has a far greater impact on KWS performance than previously recognized.

2. **Shape constraints (FastAudio) are counterproductive on KWS**: Surprisingly, the shape constraints of FastAudio (forcing filters to maintain a Mel shape) not only failed to help but actually harmed KWS performance. This indicates that "Mel-like shapes" are not the optimal choice for the KWS task—deep learning models can find completely different filter shapes better suited to specific tasks. This finding has important theoretical implications: it challenges the long-held assumption that "the Mel scale is optimal"—the Mel scale was designed based on human auditory perceptual characteristics, but deep learning models may not need to adhere to the constraints of the human auditory system.

3. **Trainable basis functions help simple models the most (14.2%) and complex models the least (2%)**: When model complexity is low (simple linear model, approx. 5K parameters), the benefits of trainable basis functions are most significant (14.2% improvement). As model complexity increases (BC-ResNet, approx. 500K parameters), the benefits gradually decrease (2% improvement). The deeper implication of this finding is: simple models have limited expressive power and cannot learn complex frequency transformations internally, so they rely more heavily on the quality of input features—trainable basis functions take on the role of "feature engineering"; whereas complex models (such as deep CNNs) have sufficient capacity to learn similar feature transformations internally (implicitly achieving frequency weighting and recombination through convolution layer weights), so they are less sensitive to the quality of input features.

4. **Visual analysis of frequency importance**: By analyzing the shapes of learned filters, the paper provides direct evidence regarding "which frequency bands the KWS task relies on most." The learned filters are no longer smooth triangles but exhibit more complex shapes (multi-peaked, asymmetric), indicating that the KWS task requires more complex frequency resolution than assumed by human auditory perceptual models.

## Experimental Results

### Datasets Used and Their Scales
- **KWS**: Google Speech Commands V2-12, 12 classes, approx. 105,000 1-second audio clips. Standard train/validation/test split. Performance evaluated under different numbers of Mel basis functions (10, 20, 30, 40).
- **ASR**: TIMIT dataset, approx. 6,300 English audio clips, standard train/test split.

### Detailed Comparison of KWS Accuracy

**Simple Linear Model + 40 Mel Basis Functions** (Model approx. 5K parameters):

| Setting | nnAudio Accuracy (%) | FastAudio Accuracy (%) | vs Baseline A |
|:---|:---:|:---:|:---:|
| A (Fixed) | ~75.0 | ~75.0 | baseline |
| B (gMel Trainable) | ~82.0 | ~78.0 | +7.0 / +3.0 |
| C (gSTFT Trainable) | ~78.0 | ~76.5 | +3.0 / +1.5 |
| D (All Trainable) | **~89.2** | ~80.0 | **+14.2 / +5.0** |

Key Findings:
- nnAudio (unconstrained) outperforms FastAudio (constrained) in all settings, proving that shape constraints limit the optimization space of filters
- Setting D (all trainable) > Setting B (only gMel) > Setting C (only gSTFT), indicating that Mel filter optimization is more important than STFT window function optimization
- The difference between nnAudio D and FastAudio D is approximately 9 percentage points, the largest single difference—shape constraints are the main bottleneck for performance

**BC-ResNet + 40 Mel Basis Functions** (Model approx. 500K parameters):

| Setting | nnAudio Accuracy (%) | vs Baseline A |
|:---|:---:|:---:|
| A (Fixed) | ~93.0 | baseline |
| D (All Trainable) | ~95.0 | +2.0 |

The trainable gain for complex models (+2.0%) is much smaller than for simple models (+14.2%), validating the hypothesis that "complex models compensate for insufficient input features internally."

### ASR Performance
- Using the TIMIT dataset, the Phone Error Rate (PER) decreased by up to 9.5 percentage points (using trainable basis functions)
- The magnitude of improvement in the ASR task is similar to that in KWS, validating the generalizability of trainable basis functions across different speech tasks

### Impact of the Number of Mel Basis Functions

| Number of Mel Basis Functions | Fixed Accuracy (%) | Trainable Accuracy (%) | Trainable Improvement |
|:---:|:---:|:---:|:---:|
| 10 | ~70 | ~82 | +12 |
| 20 | ~73 | ~85 | +12 |
| 30 | ~74 | ~87 | +13 |
| 40 | ~75 | ~89 | +14 |

Key Findings: The improvement from training is larger when the number of basis functions is smaller (12% improvement with 10 basis functions, 14% with 40—although the absolute improvement increases with the number of basis functions). This is because: when the number of basis functions is very small, the optimal positioning (frequency range and weight distribution) of each basis function is crucial to performance—trainable optimization precisely adjusts these positions; whereas a large number of basis functions already provide sufficient frequency coverage, and the marginal benefit of optimization is relatively small.

### Analysis of Learned Filters

**Characteristics of Filters Learned for the KWS Task**:
- Learned filters are no longer smooth triangles (standard Mel filters) but exhibit more complex shapes
- **Multi-peaked Structure**: Some learned filters have multiple peaks on the frequency axis (standard Mel filters have only one peak). This suggests that the KWS task may need to attend to multiple discontinuous frequency regions simultaneously—for example, the same keyword may have discriminative information in both low frequencies (vowel formants) and high frequencies (consonant noise)
- **Asymmetry**: Learned filters are asymmetric on both sides of the center frequency—the decay slope on the low-frequency side may differ from that on the high-frequency side. This contrasts with the symmetric triangular shape of Mel filters
- **High-Frequency Enhancement**: Some learned filters allocate more energy to the high-frequency region (3000-8000Hz), which is consistent with the high-frequency discriminative information of consonants (such as /s/, /t/, /k/) in the KWS task—standard Mel filters have very low resolution in the high-frequency region (because the Mel scale compresses frequency resolution in high-frequency regions), and trainable optimization corrects this "mismatch"

**Impact of Model Complexity on Filter Shapes**:
- Filters learned by simple models show significant changes (large differences from standard Mel)—because simple models rely entirely on the quality of input features
- Filters learned by complex models show smaller changes (close to standard Mel)—because complex models can compensate internally

## Limitations and Future Work

### Technical Limitations of the Method
- **Diminishing Returns with Model Complexity**: When models are sufficiently complex (e.g., large Transformers, SOTA CNNs), the effect of trainable basis functions weakens (only approx. 2% improvement). This indicates that for SOTA large KWS models, optimizing input features may no longer be the main bottleneck—the convolutional layers within the model have already implicitly learned similar frequency transformations.
- **Sensitivity to Initialization**: The final performance of trainable basis functions is affected by initial values (initialized using standard Mel filters). If the initialization is too far from the optimal value, it may fall into local optima. The paper does not explore different initialization strategies (such as random initialization, PCA-based initialization, etc.).
- **Complexity of Learning Rate Adjustment**: Parameters of trainable basis functions require different learning rate strategies than model parameters (usually smaller learning rates to prevent drastic changes in basis functions during early training, which could cause feature space collapse). This increases the complexity of hyperparameter tuning during training.
- **Additional Computational Overhead**: Spectrogram computation in trainable front-ends requires gradient computation during forward propagation (standard spectrogram computation does not), increasing VRAM usage and computational load during training. However, there is zero additional overhead during inference (using trained fixed filters).

### Shortcomings in Experimental Design
- **Limitations in Depth of Analysis**: Although visualizations of filter shapes are provided, the acoustic mechanism analysis of "why certain frequency bands are enhanced" is not deep enough. There is a lack of systematic comparative analysis with known phonetic knowledge (such as frequency ranges of specific consonants, typical positions of formants)—such comparisons could verify whether the learned filters have physical interpretability.
- **Evaluation Limited to Two Tasks**: It remains unclear whether the conclusions for KWS and ASR apply to other audio tasks (such as acoustic scene classification, music information retrieval, speech emotion recognition, sound event detection). Different tasks may have vastly different requirements for frequency resolution.
- **Lack of Robustness Evaluation in Noise**: All experiments were conducted under relatively clean conditions. The robustness of trainable basis functions in noisy environments has not been verified—are the learned filters sensitive to noise? Is it necessary to retrain basis functions on noise-enhanced data?
- **No Comparison with Data Augmentation Methods**: Data augmentation methods such as SpecAugment also improve performance by modifying spectrograms. The relationship between trainable basis functions and data augmentation (complementary or redundant) has not been verified.

### Possible Directions for Future Improvement
- **Task-Adaptive Front-End Design**: Based on experimental findings with trainable basis functions, design optimal front-ends for specific tasks (e.g., filterbank configurations dedicated to KWS—possibly using non-Mel scale non-linear frequency resolution, asymmetric filter shapes, multi-peaked filters) to replace generic Mel filterbanks. This "data-driven feature engineering" may be more effective than manual design.
- **Multi-Task Joint Optimization**: Optimize basis functions within a multi-task learning framework, allowing learned filters to serve multiple speech processing tasks simultaneously (e.g., joint training of KWS + ASR + SV). Multi-task constraints may help learn more robust filters.
- **Physically Constrained Trainable Basis Functions**: Introduce soft constraints based on acoustic physical principles (such as the physical lower limit of frequency resolution—constrained by the uncertainty principle, energy conservation constraints) to replace the pure geometric constraints of FastAudio. Physical constraints ensure the physical rationality of filters without overly limiting the optimization space like geometric constraints do.
- **Layer-Adaptive Basis Functions**: Use different basis function configurations for different layers of the model—shallow layers use basis functions with high time resolution (capturing transient features), while deep layers use basis functions with high frequency resolution (capturing fine spectral patterns). This hierarchical front-end design may better match the feature extraction hierarchy of the model.
- **Implications for the KWS Field**: The most important insight from this paper is "do not take feature extraction for granted as a mere preprocessing step"—the design space of feature representation is as important as the design space of model architecture and deserves equal optimization effort. For KWS, this means that researchers should first confirm whether the input feature representation is sufficiently good before designing new model architectures. A simple model + good features may outperform a complex model + poor features.
