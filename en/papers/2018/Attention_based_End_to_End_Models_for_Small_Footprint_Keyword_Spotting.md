# Attention-based End-to-End Models for Small-Footprint Keyword Spotting

- **Authors/Affiliations**: Changhao Shan, Junbo Zhang, Yujun Wang, Lei Xie (School of Computer Science, Northwestern Polytechnical University; Xiaomi AI Lab)
- **Date**: 2018.03 (arXiv:1803.10916)
- **Link**: https://arxiv.org/abs/1803.10916
- **Keywords**: attention mechanism, end-to-end, keyword spotting, CRNN, LSTM, GRU, PCEN, Xiaomi

## Problem Statement

Keyword spotting systems have gone through three generations of technical evolution: (1) post-processing methods based on LVCSR — running full ASR and then searching for keywords, which demands high computational resources; (2) Keyword/Filler HMM approaches — which require a pre-trained acoustic model for phoneme-level alignment and involve complex systems; (3) Deep KWS — based on DNN posterior probability processing, but still requiring a pre-trained acoustic model and a complicated post-processing pipeline. Each of these three approaches has its own limitations, and a cleaner end-to-end solution is urgently needed.

**Domain Pain Points**
- Existing KWS systems either require heavy computational resources (the LVCSR approach) or complex pre-training and alignment pipelines (the HMM approach and Deep KWS)
- Although Deep KWS simplifies the system, it still requires a separate acoustic model pre-training step, and its post-processing (posterior smoothing, confidence computation) adds system complexity
- Small-footprint deployment requires the model to stay under 100K parameters while maintaining low latency and high accuracy

**Key Challenges This Paper Addresses**
- How to design a fully end-to-end keyword spotting model — with no pre-trained acoustic model, no alignment information, and no complex post-processing pipeline
- How to achieve detection performance surpassing the Deep KWS baseline at an extremely small model size (about 80K parameters)
- How to adapt the attention mechanism from the ASR domain to the keyword spotting task

## Methodology

### Overall Architecture Design

The paper proposes attention-based end-to-end keyword spotting models consisting of two core components: an encoder and an attention mechanism.

**Encoder**

The encoder converts a variable-length input feature sequence into a fixed-length, high-level semantic representation. The paper evaluates three encoder architectures:

1. **LSTM encoder**
   - Multi-layer long short-term memory (LSTM) network
   - The LSTM forget gate $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$ controls the retention of historical information
   - The input gate $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ controls the writing of new information
   - Configuration: 1 layer, 128 hidden units, about 86.8K parameters

2. **GRU encoder**
   - The gated recurrent unit (GRU) is a simplified version of the LSTM
   - Update gate $z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$ and reset gate $r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$
   - Fewer parameters than the LSTM, faster training and inference
   - Configuration: 1 layer, 128 hidden units, about 73.3K parameters

3. **CRNN encoder (best)**
   - Convolutional recurrent neural network, adding convolutional layers before the RNN
   - The convolutional layers extract short-term time-frequency patterns (similar to FIFO feature extraction)
   - The RNN layers model long-term temporal dependencies
   - Architecture: 1 convolutional layer (64 3x3 kernels) + 1 GRU layer (128 units)
   - Total of about 84K parameters

**Attention Mechanism**

The attention mechanism compresses the variable-length feature sequence output by the encoder into a fixed-length vector:

$$c = \sum_{t=1}^{T} \alpha_t \cdot h_t$$

where $h_t$ is the encoder output at time step $t$, and $\alpha_t$ is the attention weight:
$$\alpha_t = \frac{\exp(e_t)}{\sum_{s=1}^{T} \exp(e_s)}$$
$$e_t = v^T \tanh(W h_t + b)$$

Key roles of the attention mechanism:
- Automatically learns "which parts of the keyword matter most"
- Compresses the variable-length sequence into a fixed-dimensional vector, simplifying subsequent classification
- Requires no explicit alignment information — the attention weights implicitly learn the temporal position of the keyword

**Classifier**
- Fully connected layer + sigmoid activation
- Output: the probability of keyword presence $P(kw | x_{1:T})$
- Loss function: binary cross-entropy

### Input Features: PCEN

The paper uses Per-Channel Energy Normalization (PCEN) instead of conventional log-mel features:
$$\text{PCEN}(t, f) = \left(\frac{x(t, f)}{(\epsilon + S(t, f))^{\alpha}}\right)^{\delta} + \beta$$

Advantages of PCEN:
- Adaptive normalization: dynamically adjusts the gain for inputs at different volumes
- Differentiable: all parameters ($\alpha$, $\delta$, $\beta$) can be trained end-to-end via backpropagation
- Better suited to far-field and noisy speech: compared with fixed log compression, PCEN provides better dynamic range control

### Training and Decoding

**Training (sequence-to-one)**
- Input: a fixed 189-frame audio window (about 1.89 seconds)
- Output: a single keyword/non-keyword label
- The entire window is labeled with one label (contains keyword = 1, otherwise = 0)
- No frame-level alignment information required

**Decoding (sliding window)**
- A 100-frame sliding window is applied on the continuous audio stream
- The window slides forward every few frames
- Each window independently passes through the model to obtain a keyword probability
- Detection is triggered when the probability exceeds a threshold

## Experimental Results

### Dataset and Evaluation Setup
- Target keyword: the Xiaomi wake word (Chinese)
- Evaluation data: real-world wake-up scenario data from Xiaomi
- Evaluation metric: FRR @ 1.0 FA/hour
- Baseline: a Deep KWS system

### Core Performance Comparison

| Model | FRR (%) @ 1.0 FA/hr | Parameters |
|------|---------------------|--------|
| CRNN | 1.02 | 84K |
| GRU | 1.57 | 73.3K |
| LSTM | 2.31 | 86.8K |
| Deep KWS baseline | ~5.0 | larger |

- CRNN reduces FRR by about 80% (relative) compared with the Deep KWS baseline
- All attention models substantially outperform Deep KWS
- CRNN > GRU > LSTM; the short-term feature extraction capability of the convolutional layer is crucial for KWS

### Attention Weight Visualization
- The attention weights concentrate on the latter part of the keyword
- This indicates that the model mainly relies on the acoustic features at the end of the keyword for discrimination
- This finding also hints that the "focusing" capability of attention may be limited — the model fails to fully exploit all parts of the keyword

## Main Contributions

1. **Fully end-to-end KWS**: achieves end-to-end keyword spotting requiring no alignment information, no graph search, no pre-trained acoustic model, and no complex post-processing. The entire system consists of only three components — encoder + attention + classifier — greatly simplifying KWS system design and deployment.

2. **Adaptation of the attention mechanism to KWS**: the first successful adaptation of the attention mechanism from the speech recognition (seq2seq ASR) domain to the keyword spotting task. The "focusing" capability of attention enables the model to automatically locate the keyword's position in the audio without explicit temporal boundary annotation.

3. **Optimal performance of the CRNN encoder**: the CRNN encoder achieved the best performance across all evaluated configurations (FRR of 1.02% at 1.0 FA/hour), demonstrating the complementarity of the convolutional layers (short-term time-frequency pattern extraction) and the recurrent layers (long-term temporal dependency modeling).

4. **Small-footprint deployment verification**: all models have parameter counts in the 73K–87K range, far smaller than conventional Deep KWS systems, verifying the feasibility of end-to-end attention models on resource-constrained devices.

5. **Substantial improvement over Deep KWS**: all three attention models significantly outperform the Deep KWS baseline, with the CRNN's FRR dropping from about 5% at baseline to 1.02%.

## Limitations and Future Work

### Technical Limitations of the Method
- **Training–decoding configuration mismatch**: training uses a fixed 189-frame window while decoding uses a 100-frame sliding window. This mismatch in window size may cause the attention distribution to differ between training and decoding, hurting actual performance.
- **Latency from the fixed window**: the 100-frame sliding window used at decoding introduces about 1 second of latency (100 frames * 10 ms/frame), which may be too high for applications that require an instant response.
- **Attention focusing problem**: the attention weights concentrate at the end of the keyword and fail to fully exploit the earlier parts of the keyword. This may be because the acoustic features at the end are the most discriminative, but it may also limit the model's robustness when parts of the pronunciation are masked by noise.
- **Single-keyword testing**: validated on only one Chinese wake word; multi-keyword scenarios were not explored.

### Shortcomings in Experimental Design
- No comparison with other end-to-end KWS methods (e.g., CTC-based ones)
- No analysis of the effect of different window sizes on training–decoding consistency
- No evaluation of performance under far-field and high-noise conditions

### Future Improvement Directions
- Replace sequence-to-one window training with sequence-to-sequence frame-level training to eliminate the window-size dependency (as in the subsequent Seq2Seq KWS work of Zhang et al., 2018)
- Explore multi-head attention mechanisms that simultaneously attend to different parts of the keyword
- Introduce transfer learning, using large-scale ASR data to pre-train the encoder
- Incorporate multi-channel information (e.g., the multi-microphone array of Xiaomi smart speakers)

### Implications for the KWS Field
- The attention mechanism provides a clean and elegant solution for end-to-end KWS, without complex pipeline design
- The CRNN (convolution + recurrent) architecture combination performs excellently in KWS and has become the standard choice in much subsequent work
- The adaptive normalization capability of PCEN features has made them a standard for KWS tasks
- Xiaomi's research demonstrates the great potential of end-to-end deep learning in industrial-grade KWS systems
- This work is the beginning of Xiaomi's KWS research series; the subsequent Seq2Seq and auditory attention methods improved further on this foundation
- The industry–academia collaboration model between Northwestern Polytechnical University and Xiaomi provides a good example for KWS technological innovation
