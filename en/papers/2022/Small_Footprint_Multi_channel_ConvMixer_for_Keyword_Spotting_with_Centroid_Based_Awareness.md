# Small Footprint Multi-channel ConvMixer for Keyword Spotting with Centroid Based Awareness

- **Authors/Affiliations**: Dianwen Ng, Jin Hui Pang, Yang Xiao, Biao Tian, Qiang Fu, Eng Siong Chng (Alibaba Group, Beijing; Nanyang Technological University, Singapore)
- **Date**: April 2022
- **Link**: https://arxiv.org/abs/2204.05445
- **Keywords**: keyword spotting, multi-channel, noisy far-field, centroid aware, small footprint, ConvMixer, microphone array

## Problem Statement

### Problem Background and Domain Pain Points
Far-field speech interaction scenarios (such as voice control in smart homes) face the compound effects of multiple signal degradation factors: room reverberation (typical T60=0.3-1.0 seconds, where multipath propagation causes time-frequency blurring, "smearing" the transient features of keywords), environmental noise (appliance noise, traffic noise, interference from other speakers), and "replay attacks" from TVs/radios (devices may mistakenly recognize keyword sounds from the TV as user commands).

Multi-channel microphone arrays (arrays composed of multiple spatially distributed microphones, with typical configurations of 2-8 microphones spaced 1-5 cm apart) provide the hardware foundation for solving these problems. By leveraging spatial information (differences in signals received by different microphones—Time Difference of Arrival (TDOA), energy differences, coherence differences), it is possible to infer the direction of the sound source, perform beamforming (enhancing signals from the target direction while suppressing noise from other directions), and suppress noise.

### Specific Shortcomings of Existing Methods
- **Cascading errors in two-stage schemes of front-end signal processing + back-end KWS**: Traditional schemes first use front-end signal processing (such as MVDR beamforming, WPE dereverberation) to enhance speech, and then use a single-channel KWS model for recognition. The fundamental problem with this cascaded scheme is: (1) Errors in front-end enhancement (such as speech distortion caused by beamforming, residual noise) propagate irreversibly to the back-end and cannot be corrected by the back-end—the front-end may incorrectly suppress signals from the target direction (e.g., when the target speaker moves), causing the back-end to fail to recognize them; (2) Front-end and back-end are optimized independently—the optimization goal of the front-end is "signal quality" (e.g., SNR), which does not necessarily align with "KWS accuracy" (sometimes front-end enhancement actually reduces KWS accuracy because the enhancement process removes certain spectral features relied upon by the KWS model).
- **Lack of spatial information in single-channel models**: Small KWS models typically process only single-channel audio (the output after beamforming), completely ignoring spatial clues in multi-channel data. In multi-speaker scenarios (e.g., TV sound + user sound present simultaneously), spatial information is crucial for distinguishing the target speaker from interference sources—the two may have highly overlapping spectra but come from different spatial directions.
- **High computational overhead of large multi-channel models**: Some end-to-end multi-channel KWS models (such as those based on attention or large RNNs, e.g., SINC-Net + LSTM) have parameter counts in the multi-Million range, resulting in excessive computational overhead, making them unsuitable for resource-constrained edge devices.

### Key Challenges Addressed by This Paper
Design a small multi-channel KWS model with only hundreds of thousands of parameters that can learn joint spatial-spectral features directly from raw multi-channel microphone array data (rather than relying on pre-processed front-end enhancement), achieve robust keyword recognition under far-field noisy conditions, and remain compatible with traditional front-end speech enhancement techniques (in competition scenarios, front-end enhancement can serve as an additional performance boost).

## Methodology

### Overall Architecture Design
The proposed "Multi-channel ConvMixer" architecture is a natural extension of the single-channel ConvMixer (previous work by the same team)—adding a "microphone channel dimension" mixing on top of the existing time and frequency dimension mixing. The overall architecture consists of three parts: a convolutional encoder, multi-channel ConvMixer blocks (repeated N=4 times), and a post-convolutional encoder. The innovations lie in: (1) introducing microphone channel mixing in the ConvMixer blocks; (2) introducing a centroid-based awareness mechanism in the classification stage.

### Mathematical Principles of Core Algorithms

**Core Technology 1: Multi-channel ConvMixer**

**Motivation for Designing Three Mixing Dimensions**:
The spectrogram input is a 3D tensor $X \in \mathbb{R}^{T \times F \times C}$, where $T$ is the number of time frames, $F$ is the number of frequency channels, and $C$ is the number of microphone channels. Each dimension carries different physical information:
- Time dimension $T$: Encodes the dynamic changes of speech (temporal order of phoneme sequences)
- Frequency dimension $F$: Encodes spectral envelope features (formant positions and energy distribution)
- Channel dimension $C$: Encodes spatial information (differences in signals received by different microphones—phase differences and energy differences)

**Specific Structure of the Multi-channel Convolutional Mixing Block**:
Input: $X \in \mathbb{R}^{T \times F \times C}$

1. **2D Depthwise Separable Convolution in the Frequency Direction**: Performs 2D DWSConv (3x3 kernel) on the $(T, F)$ plane to capture local time-frequency patterns (such as the time-frequency trajectories of formants). Weights are shared across all $C$ channels because the time-frequency patterns are similar across different channels (only differing in phase and energy).

2. **1D Depthwise Separable Convolution in the Time Direction**: Performs 1D DWSConv (kernel size 3) along the time dimension to further capture temporal dynamics.

3. **MLP Mixing Layer for Microphone Channels**: This is the core innovation of the multi-channel extension. It performs global mixing of features along the channel dimension:

$$X_{ch\_mix} = X + \text{MLP}_C(\text{LayerNorm}(X))$$

where $\text{MLP}_C$ performs a global linear transformation along the channel dimension $C$: flattening $X$ along the channel dimension, passing it through a fully connected layer $\mathbb{R}^{C \cdot d} \to \mathbb{R}^{C \cdot d}$, and reshaping it back to the original dimensions.

**Physical Meaning of Channel Mixing**:
- Differences in signals received by different microphones (phase differences, energy differences) encode the spatial position information of the sound source.
- Channel mixing "learns" how to extract spatial information from these differences through global linear transformations—for example, if the target speaker is on the left, the signal energy received by the left microphone is higher and arrives earlier; channel mixing can learn the pattern "strong left, weak right = target on left".
- Compared to simple channel concatenation or channel averaging, MLP mixing can learn arbitrarily complex inter-channel relationships.

4. **MLP Mixing Layers for Time and Frequency**: After channel mixing, further global mixing is performed for the time and frequency dimensions (same as in single-channel ConvMixer).

**Deep Analysis of Key Design Decisions**: Instead of simply concatenating multi-channel data as an additional feature dimension ($C$ channels $\to$ $C \times F$ frequency channels), a dedicated mixing layer is designed for microphone channels. The reasons are: (1) The physical meanings of the channel dimension and frequency dimension are completely different—channel differences encode spatial information, while frequency differences encode acoustic information; mixing them together would confuse these two different physical quantities; (2) A dedicated channel mixing layer allows the model to explicitly learn "how to extract spatial information from feature differences across different microphones," making the learning objective clearer.

**Core Technology 2: Centroid-Based Awareness**

**Design Motivation**:
Standard KWS classifiers (fully connected layer + softmax) assume that each class in the feature space can be separated by a hyperplane. For multi-channel data, the feature space contains spatial position information—different speaker positions (e.g., standing on the left or right side of the room) for the same keyword produce different feature vectors. A linear classifier might treat "the same keyword at different positions" as two different classes, leading to poor generalization.

The centroid-aware mechanism provides awareness of the geometric structure of the feature space by introducing "keyword centroids."

**Mathematical Formulas**:
1. **Definition of Keyword Centroid**: In the latent projection space, a trainable "centroid vector" $c_k \in \mathbb{R}^d$ is defined for each keyword class $k$.

2. **Distance Calculation**: For the latent representation $z_i$ of each input sample (the $d$-dimensional vector after projection of the ConvMixer block output), the L2 norm Euclidean distance to all keyword centroids is calculated:

$$d_i^k = \|z_i - c_k\|_2, \quad k = 1, 2, ..., K$$

3. **Distance Feature Fusion**: The distance vector $[d_i^1, d_i^2, ..., d_i^K]$ is concatenated with the original feature $z_i$ to serve as the input to the final classifier:

$$\hat{y}_i = \text{FC}([z_i; d_i^1, d_i^2, ..., d_i^K])$$

4. **Centroid Optimization**: The centroid vectors $c_k$ are jointly optimized via MSE loss:

$$L_{centroid} = \sum_{k=1}^{K} \frac{1}{|B_k|} \sum_{z_i \in B_k} \|z_i - c_k\|_2^2$$

where $B_k$ is the set of samples for class $k$. The MSE loss pulls the latent representations of samples in the same class toward the corresponding centroid.

**Intuitive Explanation**:
The centroid-aware mechanism provides "spatial geometric priors" to the classifier—not only knowing "what this feature looks like" ($z_i$ itself), but also knowing "how far this feature is from the 'typical position' of each keyword" (distance vector). The distance vector encodes "relative position" information, which is robust to changes in absolute position—when a speaker moves around the room, the absolute values of $z_i$ may change, but the relative distance pattern of $z_i$ to each centroid may remain stable (e.g., always being closest to the centroid of "yes").

**Connection to Prototypical Networks**:
The centroid-aware mechanism is formally similar to Prototypical Networks—both use the "center" of a class to assist in classification. However, the key differences are: (1) The centroids in Prototypical Networks are dynamically calculated from support samples during inference, whereas the centroids in this paper are trainable parameters; (2) Prototypical Networks rely entirely on distance-based classification (not using original features), whereas this paper supplements the distance information as additional features to a standard classifier.

### Compatibility with Front-end Enhancement
Multi-channel ConvMixer can be cascaded with traditional beamforming front-ends (such as MVDR + WPE):
- The front-end provides preliminary speech enhancement (noise reduction, dereverberation) -> outputs enhanced multi-channel data.
- Multi-channel ConvMixer further learns spatial-spectral features on the multi-channel data processed by the front-end.
- They are complementary: front-end processing handles significant noise (SNR improvement of 5-15dB), while the back-end handles residual degradation.

## Main Contributions

1. **Multi-channel ConvMixer Architecture**: First extends ConvMixer to multi-channel KWS, achieving explicit modeling of spatial information through the newly added microphone channel mixing dimension. This extension maintains the lightweight nature of ConvMixer (473K parameters) while empowering the model to handle multi-channel data. The design principle of the channel mixing layer—designing dedicated mixing operations for dimensions with different physical meanings—also has reference value for other multi-sensor fusion tasks (such as audio-video fusion in multimodal learning).

2. **Centroid-Based Awareness Mechanism**: Innovatively injects the spatial distance information of keyword centroids into the classifier, providing the model with awareness of the geometric structure of the feature space. The centroid-aware mechanism significantly improves robustness to "the same keyword at different spatial positions"—which is particularly important in far-field scenarios where speaker positions are uncontrollable.

3. **Front-end Compatibility**: Multi-channel ConvMixer can process raw microphone array data or be cascaded with front-end beamforming. This "optional front-end enhancement" design demonstrates flexibility in competition scenarios—utilizing its gains when front-end enhancement is available, and still functioning independently when it is not.

4. **Competition Validation**: Achieved excellent results in the KWS track of the MISP Challenge 2021 (Multimodal Information Speech Processing Challenge) (score reduced from the baseline of 0.338 to 0.126, a 63% improvement), proving the effectiveness of the method in real multi-channel far-field scenarios.

## Experimental Results

### Datasets Used and Their Scale
- **MISP Challenge 2021** (Multimodal Information Speech Processing Challenge): Far-field KWS dataset in real home environments
  - 6-channel microphone array (circular arrangement, spacing approx. 4cm)
  - Real room reverberation (T60 approx. 0.3-0.8 seconds)
  - Real environmental noise (appliances, traffic, TV sound)
  - Multi-speaker scenarios (target speaker + interfering speaker)
  - Training set approx. 20,000 6-channel audio clips, test set approx. 5,000 clips

### Definition and Rationale for Evaluation Metrics
- **Challenge Score**: The official scoring standard of the MISP Challenge, comprehensively measuring the weighted cost of false alarms and misses. Lower is better. The reason for choosing the challenge score is that it directly reflects comprehensive performance in actual far-field scenarios and considers the cost differences of different error types.

### Core Performance Data

| Method | Challenge Score | Improvement |
|:---|:---:|:---:|
| Official MISP Baseline | 0.338 | - |
| Multi-channel ConvMixer (Raw Microphone Input) | 0.152 | 55% |
| Multi-channel ConvMixer + Front-end Enhancement (MVDR+WPE) | 0.126 | 63% |

### Key Findings

**Multi-channel vs. Single-channel**:
- Multi-channel ConvMixer (6-channel input) outperforms single-channel ConvMixer (using single-channel signal after beamforming) by approx. 10-15% in challenge score.
- The advantage mainly comes from spatial information modeling—the multi-channel model can distinguish "speech from the target direction" from "interference from other directions."

**Contribution of Centroid Awareness**:
- Removing the centroid-aware mechanism (using only a standard fully connected classifier) -> score degradation of approx. 5-8%.
- The advantage of centroid awareness is more pronounced in test scenarios with large speaker position variations—because the centroids provide robustness to position changes.

**Additive Effect with Front-end Enhancement**:
- Front-end enhancement (MVDR+WPE) provides an additional improvement of approx. 20%.
- The effects of front-end enhancement and multi-channel ConvMixer are additive (non-redundant)—front-end processing handles "obvious" noise and reverberation, while the back-end handles "residual" degradation.
- Optimal combination: Front-end Enhancement + Multi-channel ConvMixer + Centroid Awareness = 0.126

**Model Size**:
- Multi-channel ConvMixer: Only 473K parameters, competitive with models having much larger parameter counts (multi-Million parameters) on the competition leaderboard.
- It lies on the Pareto frontier in terms of parameter efficiency.

## Limitations and Future Work

### Technical Limitations of the Method
- **Dependence on Multi-channel Hardware**: The model requires a 6-channel microphone array as input, which adds constraints in terms of hardware cost and device design. This method cannot be directly applied to low-cost devices with only 1-2 microphones (such as cheap smart bulbs). Exploration of a "few-channel" version of ConvMixer is needed.
- **Training Complexity of Centroid Optimization**: Centroid vectors need to be jointly optimized with model parameters, and the MSE loss introduces additional hyperparameters (weight of the centroid loss). In the early stages of training, when centroid positions have not yet converged, the distance features may be unreliable—requiring warm-up strategies or staged training.
- **Fixed Number of Channels**: The model is designed for 6 channels and requires retraining for different numbers of microphone channels. In practical products, different devices may have different microphone configurations (2 channels, 4 channels, 7 channels, etc.), lacking a unified model.
- **Linearity Limitations of Channel Mixing**: The MLP mixing layer is a global linear transformation, which may have insufficient modeling capability for complex spatial patterns (such as multipath effects caused by reverberation—signals undergoing multiple reflections produce complex interference patterns on different microphones).

### Shortcomings in Experimental Design
- **Evaluation Only on MISP Challenge Dataset**: Although MISP is real far-field data, evaluation on only one dataset limits the generalizability of the conclusions. Validation on other multi-channel KWS datasets (such as the AMi Meeting Corpus, DIRHA dataset) was not performed.
- **Lack of Systematic Comparison with End-to-End Multi-channel Methods**: No systematic comparison of parameter count vs. performance was conducted with other end-to-end multi-channel KWS methods (such as attention-based or RNN-based methods, end-to-end methods based on beamforming learning).
- **Unquantified Efficiency of Spatial Information Utilization**: It was not analyzed whether the spatial features learned by the model are interpretable (e.g., whether it learned spatial clues similar to TDOA).

### Possible Directions for Future Improvement
- **Channel Number Adaptivity**: Design models capable of handling any number of microphone channels. Solutions: (1) Use channel masking during training (randomly mask some channels during training, forcing the model to work on any subset of channels); (2) Use MLP mixing layers with variable dimensions (e.g., adapting to different channel numbers via interpolation).
- **Online Update of Centroids**: Update centroids online during inference based on detected keywords, enabling rapid adaptation to new environments. For example, in a new room, the system can first update centroid positions using a small amount of labeled data to adapt to the acoustic characteristics of that room.
- **Interpretable Spatial Features**: Analyze the spatial features learned by the channel mixing layer through visualization or probe experiments to verify whether it has learned meaningful acoustic spatial clues (such as TDOA, coherence, etc.).
- **Joint Optimization with Front-end Learning**: Perform end-to-end joint optimization of the front-end beamforming (weights of MVDR) and the back-end ConvMixer, aligning the enhancement goal of the front-end with the KWS goal of the back-end.
- **Multimodal Fusion**: Combine multi-channel audio with video (e.g., speaker lip movements) to leverage multimodal information to further improve the robustness of far-field KWS.
- **Inspiration for the KWS Field**: Multi-channel ConvMixer proves that "processing multi-channel data directly within the model architecture" is more effective than "reducing channels first and then processing." This principle—retaining the multi-dimensional information of raw sensors until fusion within the model—provides important reference for the design of multi-sensor KWS systems.
