# Keyword Spotting Based on Dual-Branch Broadcast Residual and Time-Frequency Coordinate Attention

- **Authors/Affiliations**: Zeyu Wang, Jian-Hong Wang (School of Computer Science and Technology, Shandong University of Technology, Zibo); Kuo-Chun Hsu (Department of Information Management, National Taipei University of Business, Taipei)
- **Date**: Submitted September 5, 2025; Accepted November 27, 2025; Published February 10, 2026 (Comput Mater Contin. 2025;87(1):9, Tech Science Press)
- **Link**: https://doi.org/10.32604/cmc.2025.072881
- **Keywords**: keyword spotting, broadcast residual learning, dual-branch parallel structure, coordinate attention, curriculum learning, OHEM, noisy far-field

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) is the technology of detecting predefined wake-up words from continuous speech streams, serving as the voice entry point for contemporary smart devices: Xiaomi’s "Xiao Ai Tong Xue" and Apple’s "Hey Siri" both rely on always-on wake-word detectors (Introduction). In Internet of Things (IoT) scenarios, KWS is further deployed in smart homes, in-vehicle systems, and other environments where users control speakers, TVs, or car infotainment systems via voice commands. These tasks almost always run on edge devices with limited computing power and memory. Therefore, KWS systems must simultaneously satisfy two mutually constraining metrics: first, noise robustness—real-world acoustic environments are dynamic with fluctuating noise levels; insufficient robustness leads to false alarms or missed detections, directly degrading user experience; second, model size and memory footprint must be small to fit within the power and performance budgets of mobile platforms (Introduction).

This paper focuses on the specific pain point of "noisy far-field" scenarios: under ideal near-field, low-noise conditions, existing small-footprint models already perform well; however, in complex real-world environments with significant far-field noise interference, these models degrade sharply. Models with few parameters and simple structures have limited generalization capabilities, becoming fragile when encountering acoustic conditions unseen during training, manifesting as decreased accuracy, increased response latency, and frequent false triggers (Introduction). The target of this paper is very clear: **lightweight design and noise robustness must coexist**, whereas existing works often sacrifice one for the other.

### Specific Deficiencies of Existing Methods

The paper systematically reviews the defects of existing technical routes in the related work section, which can be summarized into six points:

- **Receptive Field Bottleneck of Pure CNNs**: After Sainath et al. introduced convolutional networks to KWS in 2015, CNNs became mainstream due to their low parameter count, computational efficiency, and ability to model time-frequency correlations in speech. However, the fixed receptive field of CNNs limits long-range context modeling, often causing them to ignore the global temporal structure of speech.
- **Latency and Complexity Costs of RNNs**: LSTM and GRU units mitigate gradient issues via gating mechanisms, and bidirectional LSTMs can utilize both past and future contexts in non-real-time scenarios. However, wake-word detection involves short speech segments that do not require long dependencies. Moreover, RNN-based methods are generally more complex and have higher latency than CNNs, making them unsuitable for real-time edge deployment.
- **Noise Robustness Shortcomings of Pure 1D Depthwise Separable Convolutions**: Stacking 1D time-domain depthwise separable convolutions, as represented by MatchboxNet, indeed results in low parameters and computational cost. However, the experimental analysis in this paper (Table 2 discussion) points out that relying solely on 1D structures limits feature modeling capabilities under complex noise interference, showing a significant gap compared to the proposed model at low Signal-to-Noise Ratios (SNR).
- **Time-Frequency Coupling Issue in BC-ResNet’s Serial Residuals (The Most Critical Criticism of This Paper)**: BC-ResNet introduces broadcast residual learning to reconcile the contradiction between "the translational invariance of 1D convolutions is not preserved in frequency-domain processing" and "2D convolutions are computationally expensive"—it first performs convolution along the frequency dimension, then averages along the frequency axis to obtain a 1D time representation, performs time-domain operations, and finally broadcasts the 1D residual back to the 2D feature map for residual mapping. However, the paper points out that BC-ResNet employs **serial residual stacking**: time convolution and frequency convolution are executed sequentially on the same path. The latter convolution overwrites or interferes with features extracted by the former, causing time-frequency coupling, longer gradient propagation paths, and mutual interference between features.
- **Information Loss in ConvMixer’s Serial Processing**: ConvMixer (ICASSP 2022) uses both 2D frequency-domain sub-blocks and 1D time-domain sub-blocks, but the feature extraction phase remains serial, leading to information loss during propagation.
- **Deployment Costs of Attention and Transformers**: Attention mechanisms can improve noise robustness to some extent. Self-attention structures (e.g., AST, Keyword Transformer) surpass convolutional and hybrid models in speech recognition, but their high computational and memory overhead makes them impractical for smart speakers, wearables, and in-vehicle systems.
- **Failure of Multi-Condition Training with Large SNR Gaps**: Multi-condition training is a common technique to improve the robustness of small-footprint speech models. However, when the gap between different SNR levels is too large, the model struggles to learn discriminative features in such a complex noise space, resulting in low training efficiency and limited robustness.

### Key Challenges to Be Solved

The core insight of this paper is: **noise interference with speech manifests differently in the time and frequency domains**—in the time domain, noise is random and irregular; in the frequency domain, noise introduces additional spectral components. This implies that time-domain dynamic features (phoneme duration, energy changes) and frequency-domain distribution features (formants, timbre) should be modeled separately, rather than stepping on each other in a single serial path. Consequently, the key challenges this paper aims to solve can be broken down into three: (1) How to decouple time-frequency modeling from serial to parallel while maintaining BC-ResNet-level computational efficiency, allowing features in both domains to be extracted independently with independent gradient backpropagation; (2) How to cover the full difficulty spectrum from clean far-field to extremely low SNR of −10 dB with very few parameters (approx. 103K, Table 2), and generalize to unseen SNR conditions during training (20 dB test, Section 4.1.2); (3) How to design training strategies to avoid the decreased learning efficiency caused by multi-condition training throwing all difficulty levels at the model at once.

## Methodology

### Overall Architecture Design and Design Motivation

The full name of the model is Time-Frequency Dual-Branch Parallel Residual Network (TF-DBPResNet). The overall framework is shown in Figure 1 and consists of three parts:

1.  **SE-Res Pre-Convolution Block**: For initial significant feature extraction. The structure consists of seven consecutive layers of 3×3 2D convolutions, each followed by Batch Normalization (BN) and ReLU (Figure 2). At the end of the convolution stack, a Squeeze-and-Excitation (SE) attention module is attached. The design motivations are twofold: first, stacking multiple small receptive field kernels instead of a single large kernel achieves a better balance between parameter efficiency and modeling capability, while continuously expanding the overall receptive field to improve the modeling of local speech patterns and short-term context; second, the SE module explicitly models inter-channel dependencies and adaptively learns weights for each channel, enhancing the network's sensitivity to key channel features, highlighting key acoustic regions, suppressing redundant background information, and improving discriminability and robustness (Section 3.2).
2.  **Four Consecutive TF-Blocks**: Each TF-Block contains two core components—the Dual-Branch Broadcast Residual (DBBR) module and the Time-Frequency Coordinate Attention (TFCA) module. DBBR is responsible for the parallel decoupled extraction and cross-dimensional fusion of time-frequency features, while TFCA models attention weights along the time and frequency axes respectively to dynamically adjust the importance of each position in the feature map. The two collaborate to jointly enhance the representation capability for complex speech patterns and robustness against noise interference (Section 3.1).
3.  **Post-Convolution Block (Post-Block)**: Consists of three layers of 1D depthwise separable convolutions to continue refining high-level discriminative features under low parameter complexity, followed by max-pooling along the time dimension for compression, and finally outputting class probabilities via a fully connected layer and Softmax (Section 3.1).

The input feature is FBank: Short-Time Fourier Transform uses a 25 ms frame length and 10 ms frame shift to extract 64-dimensional log-Mel filter banks, with the Mel spectrogram normalized to 98×64 (Section 4.1.2). The convolution configurations for the four layers of TF-Blocks are shown in Table 1: The frequency branch 2D depthwise separable convolution kernels are 5×1, 5×1, 7×1, 7×1 from front to back (expanding only along the frequency dimension), with channel numbers decreasing stepwise from 64 to 32, 16, and 8; The time branch 1D depthwise separable convolution kernels are 1×9, 1×11, 1×13, 1×15 (time windows widen stepwise), with channel numbers constant at 64. The intention behind widening the time kernels with depth can be understood as expanding the time receptive field layer by layer to cover longer phoneme dynamics; the specific motivation for the stepwise decrease in frequency channels is not reported in the paper, but from a parameter control perspective, it compresses the overhead of 2D convolutions.

It should be noted: The paper does not report whether downsampling operations exist between TF-Blocks, nor does it provide a parameter breakdown for each module (only the TFCA parameter count of approx. 12.3K can be inferred from Table 4).

### Mathematical Principles of Core Algorithms

**DBBR Module (Section 3.3, Equations 1–8)**. Input feature tensor $x \in R^{B \times H \times W}$, where B is the batch size, and H, W are the frequency and time dimensions respectively. Two branches process in parallel:

Time Branch: The input passes through a time convolution sub-block (1D depthwise separable convolution) to extract time dependencies while maintaining feature dimensions, followed by BN and Swish activation. Frequency Branch: The input passes through a frequency convolution sub-block (2D depthwise separable convolution along the frequency dimension), using sub-band normalization and Swish activation here.

After independent modeling in both branches, average pooling is performed to compress them into 1D representations. The time branch pools along the time dimension W:

$$F^{(T)}(b,h) = \frac{1}{W}\sum_{w=1}^{W} F^{(T)}(b,h,w) \tag{1}$$

Obtaining a 1D representation indexed by frequency; The frequency branch pools along the frequency dimension H:

$$F^{(F)}(b,w) = \frac{1}{H}\sum_{h=1}^{H} F^{(F)}(b,h,w) \tag{2}$$

Obtaining a 1D representation indexed by time. (Note: The paper text states "pooling along the non-modeled dimension," but based on the subscripts in Equations 1 and 2, the time branch actually pools along W, and the frequency branch pools along H. That is, each branch convolves along its responsible axis and then compresses along that axis into a 1D vector indexed by the other axis, with the gated convolution running on the other axis—there is a slight discrepancy between the text and formulas; the formulas take precedence.)

Subsequently, a **dual-path gating mechanism** is introduced on the 1D features of both branches, i.e., a GLU-style gate formed by the element-wise multiplication of two independent 1D convolutions with Sigmoid and Tanh activations:

$$f_T^{gate}(F^{(T)}) = \sigma(\text{Conv1D}_1(F^{(T)})) \odot \tanh(\text{Conv1D}_2(F^{(T)})) \tag{3}$$

$$f_F^{gate}(F^{(F)}) = \sigma(\text{Conv1D}_3(F^{(F)})) \odot \tanh(\text{Conv1D}_4(F^{(F)})) \tag{4}$$

Where the four Conv1Ds are independent learnable convolution kernels, each followed by BN to stabilize training. The gated outputs are expanded back to the original input dimensions via a broadcast mechanism and added element-wise to the input feature x to form dual residuals:

$$R^{(T)} = x + \text{Broadcast}(f_T^{gate}) \tag{5}$$

$$R^{(F)} = x + \text{Broadcast}(f_F^{gate}) \tag{6}$$

The two residual outputs are concatenated along the channel dimension:

$$R_{concat} = \text{Concat}[R^{(T)}, R^{(F)}] \tag{7}$$

Finally, a 1×1 2D convolution performs channel mapping, followed by BN and Swish activation, to obtain the final output of the module:

$$\text{Output} = \delta(\text{BN}(\text{Conv}^{1 \times 1}(R_{concat}))) \tag{8}$$

**TFCA Module (Section 3.4, Equations 9–14)**. Input 2D feature map $x \in R^{B \times C \times H \times W}$, where C is the number of channels. Average pooling is performed along the frequency and time directions respectively to obtain direction-aware features:

$$G_T(i) = \frac{1}{H}\sum_{j=1}^{H} x(i,j), \quad G_F(j) = \frac{1}{W}\sum_{i=1}^{W} x(i,j) \tag{9,10}$$

Obtaining $G_F \in R^{B \times C \times 1 \times W}$ and $G_T \in R^{B \times C \times H \times 1}$. After reshaping and concatenating into $G_{cat} \in R^{B \times C \times (H+W) \times 1}$, a compact representation is obtained via a convolution layer plus BN plus h-swish activation:

$$G_{emb} = h\_swish(\text{BN}(\text{Conv}(G_{cat}))) \tag{11}$$

$G_{emb}$ is split into two sub-tensors $V_F, V_T = \text{Split}(G_{emb})$, each passing through a convolution layer with Sigmoid to generate attention weights for the two directions:

$$a_F = \sigma(\text{Conv}^{1 \times 1}(V_F)) \in R^{B \times C \times 1 \times w} \tag{12}$$

$$a_T = \sigma(\text{Conv}^{1 \times 1}(V_T)) \in R^{B \times C \times H \times 1} \tag{13}$$

The attention maps are applied back to the original feature map, performing joint re-weighting on the frequency and time dimensions:

$$Y = x(E_p(a_F)E_p(a_T)) \tag{14}$$

Where $E_p(\cdot)$ is the expansion broadcast operator (Equation 14 does not explicitly write the element-wise multiplication symbol; based on context, it is understood as outer-product-style joint re-weighting). The specific value of the channel reduction ratio r is not reported in the paper.

**Curriculum Learning combined with Online Hard Example Mining (OHEM) (Section 3.5, Equations 15–17)**. Training is divided into five stages of increasing difficulty: The first stage uses clean speech; the subsequent three stages sequentially introduce noise samples at 0, −5, and −10 dB; the final stage adds far-field reverberation (RIR) to half of the training samples to simulate real far-field conditions. The criterion for advancing each stage, crit, is defined as the difference between normalized validation accuracy and validation loss, normalized as follows:

$$\text{Norm}(a_m) = \frac{a_m - \min(A)}{\max(A) - \min(A)}, \quad A = \{a_1, a_2, \ldots, a_m\} \tag{15}$$

If crit does not exceed the historical best for five consecutive epochs, the current stage is considered sufficiently learned, the best parameters are reloaded, and the next stage begins. OHEM calculates the loss for all samples in each mini-batch and backpropagates using the subset of hardest examples with the highest loss:

$$L(t, p) = -\sum_{k=1}^{C} t_k \cdot \log(p_k) \tag{16}$$

$$L_{OHEM} = \text{mean}(L(t_i, p_i), i \in K_{num}) \tag{17}$$

Where $K_{num}$ is the index set of the K hardest samples in the batch (in actual implementation, the top 70% of samples with the highest loss in the batch are selected, Section 4.1.2).

### Key Technical Innovation 1: DBBR—Parallelization of BC-ResNet Broadcast Residuals

This is the core of the inheritance relationship with the BC-ResNet lineage throughout the paper and deserves separate discussion.

**Inherited Parts**. DBBR directly adopts the broadcast residual learning skeleton proposed by BC-ResNet (Interspeech 2021, Kim et al.). The paper explicitly states in Section 3.3 that "this module extends the concept of broadcast residual learning" (Figure 4 describes this path): first, convolution is performed along the frequency dimension for modeling, then features are compressed to the time dimension via average pooling from frequency to time, followed by further convolution operations along the time axis, and finally the 1D residual information is broadcast back to the original dimensions and added element-wise to the input. The frequency branch (2D depthwise separable convolution + frequency pooling + time-axis gated convolution + broadcast residual, Equations 2/4/6) is a complete replica of the original BC block of BC-ResNet. Additionally, three details clearly inherit the design language of BC-ResNet: the Sigmoid multiplied by Tanh structure of the dual-path gate (Equations 3, 4) corresponds to the time gate in the BC block; the stepwise decrease in frequency channels (64→32→16→8, Table 1) continues the practice of BC-ResNet scaling frequency channels per stage; and Swish activation is also used in BC-ResNet.

**Divergent Parts**. The first modification of DBBR is **serial to parallel**: In BC-ResNet, frequency convolution is performed first, followed by time operations on the same path, meaning the input to the time convolution has already been processed by frequency operations; DBBR allows the time branch and frequency branch to start from the same input x and run independently—the time convolution branch focuses on speech dynamics (phoneme duration, energy changes), while the frequency convolution branch captures spectral distribution features (formants, timbre), with the two branches not polluting each other. The reasons given by the paper fall into two levels: forward pass level, in serial structures, later convolutions overwrite or interfere with features extracted by earlier convolutions; parallelism avoids this inter-feature interference, and during fusion, it retains details from each domain while capturing cross-domain correlations; backward pass level, gradients for the two branches are calculated and backpropagated independently, avoiding the gradient coupling issues common in serial architectures like BC-ResNet, stabilizing model optimization, and promoting balanced learning of time-frequency representations. The second modification is **adding a dual-mirror time branch**: The time branch (1D depthwise separable convolution along time + time pooling + frequency-axis gate + broadcast, Equations 1/3/5) is an axial symmetric mirror of the frequency branch, allowing the "compress-gate-broadcast" mechanism to hold in both directions, achieving time-frequency decoupled modeling. The third modification is the **fusion method**: The dual residuals are concatenated along the channel dimension (Equation 7) and then undergo channel mapping via 1×1 convolution + BN + Swish (Equation 8), replacing the single element-wise residual of BC-ResNet, explicitly fusing "information learned by each branch" into a unified representation.

From the data, the benefits of this modification are evident (Table 2): TF-DBPResNet uses 103K parameters and 38.65M MACs, compared to BC-ResNet-6's 206K parameters and 58.25M MACs—parameters halved, multiply-accumulate operations reduced by about one-third—while simultaneously achieving accuracy improvements of 2.39, 4.10, 5.66, 9.02, and 7.01 percentage points higher than BC-ResNet-6 under five conditions: clean far-field, 20, 0, −5, and −10 dB; Compared to the larger BC-ResNet-8 (353K parameters, 90.57M MACs), the parameters are only about 29% and MACs about 43% of its size, yet it still leads by 2.31 to 8.20 percentage points across all conditions. That is, the parallelization modification not only did not incur a capacity cost, but also simultaneously improved efficiency and robustness, with the lead increasing as noise becomes heavier.

### Key Technical Innovation 2: TFCA—Transferring Coordinate Attention to the Time-Frequency Plane

TFCA transfers Coordinate Attention from computer vision (CVPR 2021) to acoustic feature modeling (Section 3.4). The design motivation stems from an acoustic observation: **noise interference with speech is non-uniform across different time frames and frequency bands**—some frames are submerged by instantaneous noise, while some frequency bands are occupied by steady-state noise. The original design philosophy of coordinate attention is precisely to encode context along spatial dimensions separately. TFCA leverages this to model attention independently along time and frequency directions, guiding the model to adaptively focus on discriminative time segments and frequency regions.

Technically, TFCA uses directional pooling (Equations 9, 10) to preserve context along a single axis, concatenates them, and shares one convolution + BN + h-swish (Equation 11) for cross-axis information exchange, then splits them back into two directions to generate Sigmoid attention weights respectively (Equations 12, 13), and finally performs joint re-weighting of the original feature map via broadcast multiplication (Equation 14). Compared to SE attention which only models channel relationships, TFCA simultaneously considers inter-channel relationships and cross-position dependencies on the time and frequency axes, capable of encoding long-range dependencies along both axes, explicitly highlighting discriminative regions on the time-frequency plane—this is precisely a targeted design for "non-uniform noise interference" (end of Section 3.4). Figure 6 provides a visual verification for the keyword "stop": The Fbank original image, TFCA attention heatmap, and their superimposed image show that attention indeed concentrates on the time-frequency regions corresponding to the keyword, validating the effectiveness of the mechanism.

In terms of cost, TFCA adds only about 12.3K parameters (derived from Table 4: full model 102.861K minus TFCA-less version 90.573K), bringing gains of about 0.9 to 1.6 percentage points at low SNR (Table 4, see ablation section for details).

### Key Technical Innovation 3: Combined Training Strategy of Curriculum Learning and First-Five-Round OHEM

The training strategy addresses the problem of "multi-condition training failing to learn when SNR gaps are large." There are three notable decisions in its design, each with a clear "why":

-   **Five Stages from Easy to Hard**: Starting with clean speech allows the model to first master basic speech structures; then 0, −5, −10 dB noise samples are introduced stepwise; finally, RIR reverberation is added to half of the samples to simulate far-field. This order corresponds to the noise mixing sequence of [clean, 0], [clean, 0, −5], [clean, 0, −5, −10] in training data construction (Section 4.1.2), avoiding exposing the model to overly difficult samples in the early stages of training—this is exactly what multi-condition training is criticized for (end of Section 2).
-   **Stage Advancement Criterion crit**: Normalized validation accuracy minus validation loss (Equation 15). If there is no improvement for five consecutive epochs, the best parameters are reloaded to enter the next stage. Normalization is used to make accuracy and loss, which have different units, comparable; reloading the best parameters prevents switching to the next stage only after overfitting on the current difficulty. The paper does not specify whether the loss term is also normalized, leaving ambiguity between the text and the formula (which only gives accuracy normalization).
-   **OHEM Used Only for the First Five Epochs**: In the early stages, the model's feature representation is not yet formed; focusing on highly discriminative hard examples can accelerate convergence and improve discriminability; as training progresses, most simple samples are correctly classified, and the loss distribution tends to concentrate. At this point, continuously emphasizing hard examples would bias the optimization process towards a few outliers, increasing the risk of overfitting. The authors explicitly tried extending OHEM to 25 epochs or using it throughout, resulting in either slower convergence or worse validation performance (Section 3.5). The final choice is "hard example mining in the first five rounds + full-sample curriculum learning thereafter." The reason for selecting the top 70% hardest samples (rather than a more extreme ratio) is not reported in the paper.

### Technical Differences with Existing Methods

Placing TF-DBPResNet in the comparison coordinate system to view technical differences (Table 2 and Section 4.2 discussion):

-   **Vs. BC-ResNet**: Both belong to the broadcast residual lineage, with the difference being serial vs. parallel (detailed above). BC-ResNet-6/8 relies on enlarging network scale to improve accuracy, but its parameters and computational cost are significantly higher than the proposed model; TF-DBPResNet does the opposite—exchanging structural decoupling for efficiency, with parameters only half that of BC-ResNet-6, yet accuracy leads comprehensively.
-   **Vs. ConvMixer**: ConvMixer also combines 2D frequency-domain sub-blocks and 1D time-domain sub-blocks (and uses curriculum learning), but its feature extraction is serial, modeling time domain then frequency domain sequentially, potentially losing key information during conversion. TF-DBPResNet replaces serial with dual-branch parallel, strengthening feature retention and representation capability. A honest detail needs to be pointed out: ConvMixer has lower MACs (22.34M vs. 38.65M, Table 2); the proposed model is not the best in terms of multiply-accumulate operations, a disadvantage the paper does not discuss; its advantage lies in parameter count (103K vs. 119K) and accuracy across all conditions (leading by 1.65 to 6.21 percentage points).
-   **Vs. MatchboxNet**: Pure stacking of 1D time-domain depthwise separable convolutions, with low computation and parameters (140K, 40.11M), but accuracy lags significantly behind in far-field noise (lagging by 7.77 percentage points at 0 dB and 10.37 percentage points at −10 dB), indicating that pure 1D structures have limited feature modeling capabilities under complex noise.
-   **Vs. ResNet-15 and MHAtt-RNN**: ResNet-15 achieves 89.45% in clean conditions, but MACs are as high as 894.56M, incurring high deployment costs; MHAtt-RNN relies on multi-head attention to introduce massive parameters (784K) and computation (305.19M), with accuracy collapsing to 51.23% at low SNR, making it highly sensitive to noise. TF-DBPResNet achieves higher accuracy with less than one-twentieth of the MACs of ResNet-15.
-   **Vs. SE Channel Attention**: SE only models channel importance, while TFCA additionally captures cross-position long-range dependencies on the time and frequency axes, performing double re-weighting of "channel + position" (Section 3.4).

## Experimental Results

### Datasets Used and Their Scale

-   **Main Dataset: Google Speech Commands V2** (Section 4.1.1): 105,000 speech clips, each 1 second long, sampled at 16 kHz, covering 35 words. The task is 12-class keyword classification: 10 command words (up, down, left, right, yes, no, on, off, go, stop) plus silence (background silence) and unknown (other unselected words grouped into one class). The official predefined train/val/test split is used, with a ratio of 80%/10%/10%.
-   **Noise Library: MUSAN**: 930 noise recordings at 16 kHz, totaling about 6 hours, covering real noise types such as technical sounds (DTMF tones), natural environment sounds (thunder), and daily noises (car horns); in experiments, they are randomly selected and superimposed onto speech commands at different SNRs.
-   **Reverberation Library: BUT Speech@FIT Reverberation Database (BUT RIR)**: Reverberation impulse responses from 9 rooms of different sizes (large, medium, small), with multiple microphone-speaker configurations in each room to capture different sound propagation paths; RIR filters are convolved with speech to synthesize far-field speech with real reverberation characteristics.
-   **Cross-Dataset Test: Google Speech Commands V1**: Used only at the testing stage; the model is trained entirely without seeing V1 data and without fine-tuning, used to test generalization under distribution shift. Compared to V1, V2 increases samples, improves annotation quality, and removes noisy or ambiguous recordings, making V1 a related but different data distribution (Section 4.2).

Experimental settings worth emphasizing: Noise addition and reverberation enhancement are **used simultaneously for training and evaluation**; "Clean" in testing refers to the far-field clean condition—no additive noise but still in a reverberant far-field acoustic environment (Table notes in Section 4.2). That is, all test conditions are far-field tasks, and baseline models are all retrained using official source code in the same data environment to ensure fairness.

### Definition and Rationale for Evaluation Metrics

-   **Recognition Accuracy (%)**: 12-class classification accuracy, reported separately for five conditions: Clean, 20, 0, −5, −10 dB on far-field test commands. Rationale: This is the standard metric for the GSC 12-class task, and reporting across multiple SNR levels directly exposes the model's noise robustness curve, providing more information than single-point accuracy. Among them, 20 dB is an SNR unseen during training (training mixing only used 0, −5, −10 dB, Section 4.1.2), specifically used to evaluate generalization to novel noise conditions.
-   **Parameters (K) and Multiply-Accumulate Operations (MACs, M)**: Measure deployment costs on resource-constrained devices; parameters correspond to storage and memory, while MACs correspond to computation and power consumption.
-   **CPU Inference Latency and Memory Footprint**: Simulate embedded performance in a pure CPU environment (Intel Core i7-13620H, 2.5 GHz), reporting average single-sample inference latency and memory usage.

### Detailed Comparison with Baseline and SOTA Methods

Main comparison results are in Table 2 (far-field test command accuracy):

| Model | Parameters (K) | MACs (M) | Clean | 20 dB | 0 dB | −5 dB | −10 dB |
|---|---|---|---|---|---|---|---|
| MHAtt-RNN | 784 | 305.19 | 77.32 | 74.89 | 63.12 | 54.98 | 51.23 |
| ResNet-15 | 238 | 894.56 | 89.45 | 86.67 | 79.81 | 74.37 | 67.52 |
| MatchboxNet-6×2×64 | 140 | 40.11 | 87.54 | 85.28 | 74.91 | 70.68 | 61.56 |
| BC-ResNet-6 | 206 | 58.25 | 89.78 | 86.72 | 77.02 | 69.82 | 64.92 |
| BC-ResNet-8 | 353 | 90.57 | 89.86 | 86.85 | 77.68 | 70.64 | 66.32 |
| ConvMixer | 119 | 22.34 | 90.52 | 87.54 | 78.49 | 72.63 | 67.02 |
| **TF-DBPResNet** | **103** | **38.65** | **92.17** | **90.82** | **82.68** | **78.84** | **71.93** |

Key readings (all calculated from original values in Table 2): TF-DBPResNet achieves the highest accuracy under all SNR conditions; Compared to BC-ResNet-6, which belongs to the same broadcast residual lineage, it has 50% fewer parameters and about 33.6% fewer MACs, with accuracy gains of 2.39, 4.10, 5.66, 9.02, and 7.01 percentage points across the five conditions. The gap widens as noise increases, directly supporting the core argument that "parallel decoupling improves noise robustness"; Compared to ConvMixer with similar parameters (119K vs. 103K), it gains 1.65 to 6.21 percentage points in accuracy, but has higher MACs (38.65M vs. 22.34M); At the extremely low SNR of −10 dB, it maintains 71.93%, at least 4.91 percentage points higher than all baselines.

**Cross-Dataset Generalization** (Table 3): Models trained only on GSC V2 are evaluated directly on the GSC V1 test set. Accuracies across the five conditions are 92.65, 91.21, 82.93, 78.79, 72.11, which are almost持平 (持平 means持平/consistent) or slightly higher than the V2 test set (92.17, 90.82, 82.68, 78.84, 71.93), indicating good generalization of the model to dataset-level distribution shifts. The paper does not explain why V1 results are slightly higher than V2 (V1 has lower annotation quality yet yields higher scores, which is slightly counter-intuitive and worth questioning).

**Deployment Simulation**: On a pure CPU platform, the average single-sample inference latency is 12.4 ms and memory footprint is 4.8 MB (Intel Core i7-13620H, 2.5 GHz). The paper judges that the model has the potential for embedded real-time deployment based on this (Section 4.2). For a 1-second command sample, a 12.4 ms latency corresponds to a Real-Time Factor (RTF) of approximately 0.012.

### Findings from Ablation Experiments

**Module Ablation** (Table 4, all using the same curriculum learning multi-condition training strategy): The full model has 102.861K parameters, with five-condition accuracies of 92.17/90.82/82.68/78.84/71.93. Removing DBBR (replaced with a comparable serial structure of 102.705K parameters to ensure performance changes come from the mechanism rather than capacity), accuracy drops to 91.39/89.25/81.57/76.98/70.02, a decrease of 0.78 to 1.91 percentage points; Removing TFCA (parameters drop to 90.573K), accuracy drops to 91.75/89.67/81.76/77.23/70.82; Removing both (90.417K), accuracy drops to 90.89/88.72/81.05/76.46/69.86. Three findings: (1) In clean conditions, removing either module causes only slight degradation, which significantly amplifies at low SNR—the value of both modules is mainly reflected in anti-noise capabilities; (2) The drop from removing DBBR is larger than from removing TFCA, indicating that dual-branch parallel time-frequency feature extraction is more critical for noise robustness and serves as the backbone of the model; (3) Removing both performs worse than removing either alone, and the gains of the two modules are basically additive. Worthy of praise is the fairness of the ablation design: the serial replacement structure differs from the original model by only 0.156K in parameters (102.705K vs. 102.861K), excluding the confounding explanation of "capacity reduction causing performance drop."

**OHEM Timing Ablation** (Table 5): OHEM cross-entropy loss is introduced at epochs 0 (not used), 5, 10, 15, and 25 respectively. Accuracies in clean conditions are 91.65, 92.17, 92.09, 91.88, 91.79—using OHEM in the first five epochs is optimal, and extending its use leads to degradation, validating the judgment that "long-term hard example mining leads to overfitting." The paper text states an improvement of about 0.57% compared to not using OHEM; calculated from the table values, it is 92.17 − 91.65 = 0.52 percentage points, a slight discrepancy between text and table. This table only reports the Clean column; the paper does not report OHEM gains under noise conditions.

**Training Strategy Ablation** (Table 6): The gain of curriculum learning over multi-condition training expands as noise increases—Clean condition 92.17 vs. 91.09 (+1.08), 20 dB 90.82 vs. 88.26 (+2.56), 0 dB 82.68 vs. 79.85 (+2.83), −5 dB 78.84 vs. 75.91 (+2.93), −10 dB 71.93 vs. 68.76 (+3.17). The conclusion is clear: The progressive guidance from easy to hard in curriculum learning yields the greatest benefits in the low SNR range, which is exactly the main battlefield this strategy aims to address.

## Main Contributions

1.  **Proposed TF-DBPResNet, a lightweight noise-robust KWS model**: Using residual convolutions and depthwise separable convolutions as the encoder backbone, and time-frequency dual-branch parallelism as the core structure, it significantly outperforms multiple comparison models on the far-field task of GSC V2-12 with approximately 103K parameters (Table 2), achieving a balance between performance and parameter efficiency.
2.  **Designed the DBBR module, transforming broadcast residual learning from serial to parallel**: Two branches extract time and frequency features respectively, achieving cross-dimensional information fusion and reconstruction through broadcast learning, avoiding potential information loss from serial stacking, and enhancing information flow and multi-scale feature fusion.
3.  **Proposed the TFCA module, introducing coordinate attention to acoustic feature modeling**: Modeling attention along time and frequency axes respectively, capturing long-range dependencies and significant responses in the time-frequency space, guiding the network to focus on key information regions.
4.  **Designed a combined training strategy of curriculum learning and OHEM**: Organizing training samples from easy to hard, dynamically focusing on the hardest inputs in each stage, significantly improving discriminative ability and generalization under complex noise conditions, while maintaining efficient convergence.

## Limitations and Future Work

### Technical Limitations of the Method

-   **Supports Only Predefined Wake Words**: The model can only recognize the fixed set of command words defined during training and does not support user-defined wake words. The paper itself acknowledges this in the conclusion and lists it as a future direction.
-   **Insufficient Adaptability to Accents and Low-Resource Languages**: The paper admits that the model's adaptability to specific accents or low-resource languages may be insufficient, with robustness mainly validated on English command words.
-   **Room for Improvement Under Extreme Noise**: At −10 dB, 71.93% (Table 2) leads the baselines, but the absolute value is still far from practical application. The paper acknowledges that robustness under extreme noise conditions needs further improvement.
-   **MACs Are Not Pareto Optimal**: MACs are 38.65M, higher than ConvMixer's 22.34M (Table 2). The paper does not discuss this disadvantageous dimension; for power-sensitive deployment scenarios, this is a practical constraint.
-   **Lack of Explanation and Ablation for Several Design Details**: The motivation for sub-band normalization in the frequency branch (BN is used in the time branch, why sub-band normalization in the frequency branch) is not explained; the design rationale for the stepwise decrease in frequency channels (64→32→16→8) is not reported; the value of the TFCA dimensionality reduction ratio r is not reported; ablation for activation function selection (Swish/h-swish) is not performed; the basis for choosing seven layers of 3×3 in the SE-Res front end is not reported.
-   **Lack of Explicit Denoising Mechanism**: The model relies on structural robustness to withstand noise, without a preceding adaptive noise suppression module. The paper lists this as an improvement direction in the outlook.

### Deficiencies in Experimental Design

-   **Evaluation Based Entirely on Simulated Noise and Reverberation**: Noise is superimposed from MUSAN, and reverberation is synthesized by convolving BUT RIR (Section 4.1.1). There is no evaluation on real far-field recordings or data collected from real devices. There is still a gap between simulated noise and real acoustic scenarios.
-   **Hardware Verification Stays at Laptop CPU**: The so-called embedded performance simulation uses an Intel Core i7-13620H (2.5 GHz, 45 W class mobile processor), not an MCU or DSP; a 4.8 MB memory footprint is large for typical microcontrollers (SRAM is often hundreds of KB). The conclusion that "resource-constrained devices can be deployed" still requires verification on real embedded platforms.
-   **Lack of Key KWS Deployment Metrics**: Only classification accuracy is reported. Key metrics for KWS deployment such as False Alarm Rate (FAR) / Missed Detection Rate (MDR), DET curves, streaming inference latency, and memory bandwidth are not reported. The task is essentially 12-class classification of 1-second clips, rather than open-set wake-word detection on continuous streams.
-   **Statistical Robustness Not Reported**: All results are point values from single runs, without mean and variance from multiple random seeds. Ablation gains of 0.5 to 1 percentage points (e.g., 0.52 for OHEM) are difficult to distinguish from random fluctuations.
-   **Table 5 Only Reports Clean Column**: OHEM ablation does not provide results under noise conditions, making it impossible to judge the contribution of hard example mining at low SNR.
-   **Minor Numerical Discrepancies Between Text and Tables**: OHEM gain is stated as approx. 0.57% in the text, calculated as 0.52 percentage points from Table 5.
-   **Cross-Dataset Results Unexplained**: GSC V1 test results are slightly better than V2 across all conditions (Table 3). V1 has lower annotation quality yet scores higher; this counter-intuitive phenomenon is not discussed in the paper.
-   **Missing Transformer Baselines in Comparison Set**: Related work cites AST and Keyword Transformer, but they are not included in Table 2; ablation also does not cover sensitivity to training strategy hyperparameters such as curriculum learning stage order, number of stages, and crit criteria.

### Possible Future Improvement Directions

Directions given by the paper itself: Introduce adaptive noise suppression mechanisms to enhance stability in complex acoustic environments; Fuse multi-modal information (visual or other sensor data) to further improve accuracy; Use transfer learning or cross-accent training to enhance generalization to low-resource languages and diverse accents; Explore custom wake-word algorithms to optimize user experience.

Based on the analysis in this note, the following can be supplemented: Complete real-world benchmark measurements of latency, power consumption, and memory on actual MCU/DSP hardware; Supplement false alarm/missed detection and streaming evaluations to push the "classifier" towards a true "detector"; Orthogonally combine with compression techniques such as quantization and distillation to further reduce parameters and MACs (especially targeting the disadvantage of higher computational cost than ConvMixer); Supplement sensitivity experiments for un-ablated designs such as sub-band normalization, channel configuration, and TFCA dimensionality reduction ratio; Generalize the dual-branch parallel idea to new paradigms such as streaming attention to test its universality.
