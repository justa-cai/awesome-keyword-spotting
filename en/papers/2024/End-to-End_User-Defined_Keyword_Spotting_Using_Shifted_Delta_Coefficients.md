# End-to-End User-Defined Keyword Spotting using Shifted Delta Coefficients

- **Authors/Affiliations**: Kesavaraj V, Anuprabha M, Anil Kumar Vuppala (IIIT Hyderabad, Speech Processing Laboratory)
- **Date**: May 2024 (arXiv:2405.14489v1, submitted May 23, 2024)
- **Link**: https://arxiv.org/abs/2405.14489
- **Keywords**: user-defined keyword spotting, shifted delta coefficients, long-term temporal information, cross-attention, feature engineering, audio-text cross-modal matching, LibriPhrase

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) is a persistent low-power frontend for voice interfaces in smart speakers, mobile phones, and wearable devices: the device continuously listens to the audio stream and only wakes up the heavier, full Automatic Speech Recognition (ASR) system behind it when a wake-word (e.g., "Okay Google", "Hey Siri") is heard, thereby minimizing the invocation of expensive computation. However, traditional KWS is a closed-set task—keywords are fixed at the factory, and the model is essentially a small classifier for a few predefined categories. With the growing demand for personalized voice assistants, User-Defined Keyword Spotting (UDKWS, also known as custom keyword detection or open-vocabulary keyword spotting) has become a hotspot: users can specify any word, even one the model has never seen during training, as a wake-word after deployment.

This is the fundamental difficulty of UDKWS: closed-set KWS can assign an output node to each keyword, but an open-set task cannot exhaustively enumerate all candidate words. The system must learn to judge "whether this audio segment and this text segment refer to the same word," i.e., measure the matching degree between the audio modality and the text modality in a shared representation space. A sharper pain point is the distinction between similarly pronounced words: the paper cites the example of "madame" and "modem." The difference between such audio-text negative sample pairs in static spectra is extremely small; what truly distinguishes them is the temporal dynamics of transitions between phonemes, rather than any single-frame spectral snapshot. Prior to this work, the UDKWS field almost exclusively used short-term spectral features such as MFCCs and mel spectrograms—where each frame's feature reflects only the local spectral envelope within a ~25 ms window, and dynamic information between frames (how formants move, how phonemes connect) is largely lost.

### Specific Shortcomings of Existing Methods

The paper systematically reviews four generations of UDKWS solutions in the introduction and points out their shortcomings one by one:

- **LVCSR Retrieval Scheme**: First, use a large-scale continuous speech recognition system to decode speech into word lattices, and then search for keywords within the lattice. Theoretically, any word can be searched, but the decoding itself incurs huge computational overhead, which contradicts the positioning of KWS as a lightweight, persistent frontend.
- **Keyword/Filler HMM Scheme**: Train Hidden Markov Models (HMMs) for keywords and non-keywords separately. Keyword customization is achievable by modifying the decoding graph. However, the computational demand remains significant, and the engineering pipeline is heavy.
- **Query-by-Example (QbyE) Scheme**: The user records a segment of keyword audio as a template, and the system compares the similarity between the test speech and the template. It does not require text, but performance heavily depends on the similarity between the registration speech and the test speech—timbre differences between different users and background noise in different environments will significantly destroy consistency. More importantly, QbyE is inherently difficult to generalize to unfamiliar words: the experimental section of the paper (Table 3) shows that an attention-based QbyE method achieves an EER as low as 14.75% on the Google Commands dataset (where keywords are within the training set), but the EER collapses to 49.13% on the Qualcomm dataset (where keywords are unfamiliar), fully exposing its generalization短板 (shortcomings).
- **Text-Registered End-to-End Scheme**: This is the mainstream approach in recent years. The user inputs the text of the keyword rather than audio, and the system learns the matching between audio embeddings and text embeddings. Representative works include: ASR-free end-to-end systems (audio encoder + keyword encoder + Multi-Layer Perceptron), CMCD cross-modal matching (learning utterance-level audio-text consistency, also the proposer of the LibriPhrase dataset), PhonMatchNet (learning audio-phoneme relationships via phoneme-level detection loss), and dynamic sequence partitioning methods (optimally partitioning the audio embedding sequence to be of equal length to the text sequence). These methods place audio and text representations into a shared latent space for comparison and have achieved good results.

The key gap identified by the paper is: the aforementioned end-to-end works have focused almost entirely on the evolution of model structures and training strategies, **while feature engineering—a crucial link for the performance of any speech application—has been relatively neglected**. All these works default to using short-term spectral features, and no one has systematically answered the question "how important is long-term temporal information for UDKWS."

### Key Challenges to Be Solved by This Paper

The core problem this paper aims to solve can be summarized in three layers:

1. **How to inject long-term temporal information at the feature level**, allowing the system to capture the pronunciation variability of keywords (transitions between connecting phonemes), thereby distinguishing similar pronunciation pairs like "madame/modem" that are almost inseparable in static spectra;
2. **What is the appropriate length for temporal context**—SDC features have four configuration parameters, and the shift amount and number of stacked blocks directly determine the temporal receptive field. Ablation experiments are needed to find the sweet spot, avoiding the naive assumption that "the longer the context, the better";
3. **Fair validation under a unified experimental framework**: Changing features should not involve changing models, otherwise it is unclear whether performance improvements are due to features or models; meanwhile, comparisons with SOTA end-to-end UDKWS methods must be made on the same datasets, and generalization capabilities to keywords unseen during training must be verified.

## Methodology

### Overall Architecture Design and Design Motivation

The entire system (Fig. 1) follows the authors' team's previous work (open-set KWS transferred from speech synthesis, arXiv:2404.03914), consisting of four sub-modules: **audio encoder, text encoder, pattern extractor, and pattern discriminator**. The core motivation of this design is: since this paper studies features rather than models, the model side remains consistent with previous work, only replacing the frontend features. This ensures that all feature comparisons are conducted within the same experimental framework, yielding clean attribution.

**Audio Encoder**: The input is the acoustic features to be evaluated (any one of the five features discussed in Section 3). It first passes through two layers of 2-D convolution (32 filters per layer, kernel size 3). The first layer uses a stride of 2 to skip processing of adjacent frames to improve computational efficiency. Batch normalization is applied after each convolution operation to ensure training stability. This is followed by two Bidirectional Gated Recurrent Units (Bi-GRUs) with a dimension of 64. Finally, a fully connected layer outputs a 128-dimensional audio embedding. The output is denoted as $E_a \in \mathbb{R}^{m \times D}$, where $m$ is the number of audio frames and $D$ is the embedding dimension. The audio path uses a combination of "convolutional downsampling + bidirectional RNN." The motivation is that convolutions aggregate local time-frequency patterns at a very low cost and shorten the sequence, while Bi-GRUs encode sequential dependencies in bidirectional contexts.

**Text Encoder**: The input is a character sequence. Here lies a design significantly different from other works: the text encoder embeds a **pre-trained Tacotron 2** (a recurrent seq2seq text-to-speech system), taking the 512-dimensional intermediate representation of its encoder LSTM block as text features. This passes through a 64-dimensional Bi-GRU and a 128-dimensional fully connected layer, outputting $E_t \in \mathbb{R}^{n \times D}$ ($n$ is the number of characters). The design motivation stems from the authors' previous work: the intermediate representation of a TTS model naturally "knows what the audio looks like"—it has learned how to map characters to acoustic outputs during pre-training. Therefore, these representations are **audio-aware text embeddings**, which are more suitable for cross-modal matching than pure character embeddings (the paper explicitly cites previous conclusions: representations of the Tacotron 2 encoder LSTM block as text features outperform character embeddings). The paper does not specify whether Tacotron 2 is frozen or fine-tuned during UDKWS training.

**Pattern Extractor**: Uses a cross-attention mechanism to capture temporal correlations between audio embeddings and text embeddings. The specific setting is: **audio embedding $E_a$ serves as both key (key) and value (value), while text embedding $E_t$ serves as query (query)**. Furthermore, all hidden states of the audio and text encoders (rather than pooled single vectors) are fed into the attention layer to fully preserve temporal information. This directional choice has clear logic: the audio sequence ($m$ frames) is usually much longer than the text sequence ($n$ characters). Having each character act as a query to perform soft alignment on the audio frame sequence is equivalent to letting the model learn "which part of the audio corresponds to this character," which is precisely the temporal alignment capability needed for keyword spotting. The output context vector encodes consistency information between audio and text.

**Pattern Discriminator**: A 128-dimensional Bi-GRU layer receives the context vector. The output of the last frame passes through a fully connected layer with sigmoid activation, outputting the binary classification probability of whether the audio and text contain the same keyword.

### Mathematical Principles of Core Algorithms

The mathematical core of this paper is the calculation of Shifted Delta Coefficients (SDC). SDC is controlled by four parameters N-d-p-k:

- $N$: The number of cepstral coefficients per frame (calculated on mel spectrogram in this paper, corresponding to 40 mel filter bank channels, so N=40);
- $d$: The shift amount (delay) relative to the current frame, determining how wide a time interval each delta spans;
- $p$: The shift step between adjacent delta blocks;
- $k$: The number of delta blocks to concatenate, determining the length of the stacked temporal context.

The delta feature for the $t$-th frame in the $i$-th iteration is calculated according to the original formula (1):

$$\delta c(t, i) = c(t + ip + d) - c(t + ip - d), \quad 0 \le i \le k-1 \tag{1}$$

where $c(\cdot)$ is the static mel spectral feature frame. These $k$ delta vectors are stacked into a $k \times N$-dimensional shifted delta coefficient according to formula (2):

$$SDC(t) = \begin{bmatrix} \delta c(t, 0) \\ \delta c(t, 1) \\ \vdots \\ \delta c(t, k-1) \end{bmatrix} \tag{2}$$

Finally, the stacked delta features are concatenated with the static mel spectral features to obtain the final SDC feature vector fed into the audio encoder.

The physical meaning of this set of formulas is worth dissecting: $\delta c(t,i)$ is a first-order difference at time position $t + ip$ with an interval of $\pm d$ frames—i.e., derivative sampling of the spectral trajectory at multiple delay positions. The difference operation naturally encodes "in which direction and at what speed the spectrum is changing," which is the signal for co-articulation and phoneme transitions; a single static frame only tells you "what the spectrum looks like at this moment" and cannot distinguish the direction of change. Stacking deltas from $k$ different offsets is equivalent to equipping each frame with a summary of the **future trajectory**. The paper does not elaborate on the mathematical details of the cross-attention part; according to the standard form cited (Vaswani et al.), it is scaled dot-product attention $\text{Attention}(Q,K,V)=\text{softmax}(QK^\top/\sqrt{D})V$, where $Q=E_t$, $K=V=E_a$.

Calculating with the optimal configuration 40-1-3-8 (the paper body only gives configuration values, not dimension calculations): the stacked delta part is $8 \times 40 = 320$ dimensions. After concatenating 40 dimensions of static mel spectrum, each frame totals 360 dimensions, which is 9 times the original 40-dimensional mel spectrum. Regarding the temporal receptive field, when $i$ ranges from 0 to 7 and $p=3$, the deepest frame index touches $t + 7\times3 + 1 = t+22$, and the shallowst goes back to $t-1$. That is, each frame's feature aggregates context from approximately 24 frames, which, with a 10 ms frame shift, corresponds to about 230 ms of temporal context—far exceeding the single-frame 25 ms window and on the same order of magnitude as the transition duration of one or two phonemes.

### Key Technical Innovation 1: Transferring SDC from Language Identification to UDKWS (Feature-Level Long-Term Context)

SDC is not a new invention; it has been famous in the field of Language Identification (LID)—capturing long-term temporal information by stacking delta features across multiple frames was a classic configuration for LID systems in the GMM era (the paper cites Torres-Carrasquillo et al. 2002 work and the authors' own 2018 stacked SDC + Residual Network LID work). The innovation of this paper lies in **being the first to pose "whether feature-level long-term information is important for UDKWS" as a research question** (the authors declare this is the first exploration in this direction) and providing an affirmative answer. The logic chain for transfer is: LID relies on SDC because different languages differ in the long-term statistics of phoneme sequences; UDKWS needs SDC because the discriminative information of keywords largely exists in the dynamics of phoneme connections—the static vowel spectra of "madame" and "modem" are similar, but the transition trajectories between vowels are different. The common essence of both is "sequential patterns are encoded in cross-frame statistics." This cross-task migration of insight is the core idea of this paper.

It is worth noting that SDC is essentially a **training-free, fixed-weight temporal encoder**: it is equivalent to a multi-branch 1-D dilated convolution with coefficients fixed at ±1. Comparing this with the FSMN memory blocks deconstructed in this series is enlightening—both use low-cost means to expand the temporal receptive field. The difference is that FSMN memory block weights are learned, while SDC delta weights are manually fixed. The advantage of SDC is that it introduces no learnable parameters and does not increase training difficulty; the cost is that the receptive field and aggregation method are entirely determined by hyperparameters, lacking task-adaptive capabilities.

### Key Technical Innovation 2: Computing SDC on Mel Spectrogram as Base

Classic SDC is usually calculated on cepstral coefficients (such as MFCCs). This paper makes a pragmatic choice: **compute SDC on mel spectrogram**, with the reason being that mel spectrogram has the best comprehensive performance in short-term feature horizontal comparisons (the paper explicitly states "it performs better than all other short-term spectral features"). This choice decouples two variables: "which static spectrum serves as the base" and "whether to add long-term deltas." The improvement from SDC cannot be attributed to switching to a better static base, because the base is the strongest baseline itself; the increment comes entirely from the stacking of long-term deltas. Additionally, static features are preserved during concatenation (static + delta), ensuring the model receives both absolute spectral information and dynamic information—this is同源 (homologous) to the convention in the MFCC field of "static + $\Delta$ + $\Delta\Delta$," but $\Delta$ only looks at adjacent frames, while SDC pulls the receptive field to the hundreds of milliseconds scale.

### Key Technical Innovation 3: Audio-Perceived Text Encoding via TTS Transfer

The text encoder borrows the intermediate representation (512 dimensions) of the pre-trained Tacotron 2 encoder LSTM, rather than training character embeddings from scratch. The motivation is the "translation" difficulty of cross-modal matching: character embeddings live in a pure symbol space with no pre-alignment to the acoustic space, relying entirely on the UDKWS task's own supervision signals to learn alignment from scratch; whereas the intermediate representation of the TTS encoder has already been optimized to "predict acoustic output from characters" during pre-training, effectively bringing a prior from characters to sound for free. This design is orthogonal to the feature innovation—reducing matching difficulty on the text side and increasing discriminative information on the audio side. The two improvements do not interfere with each other and can be attributed separately.

### Technical Differences with Existing Methods

- **With CMCD (Strongest Baseline)**: Both are highly consistent in model paradigm (both are audio-text cross-modal matching + attention alignment). The difference in this paper is **entirely in the frontend features**—replacing short-term mel spectrogram with SDC carrying 230 ms of context. This comparison is therefore particularly convincing: in Table 3, on the LPH dataset, EER drops from 32.9% to 21.48%, and AUC rises from 73.58% to 85.9%. The improvement can only be attributed to features. The paper does not report whether the CMCD reproduction also switched other features, but the setting in the same table comparison is that each model uses its original configuration.
- **With QbyE Methods**: Different registration modalities (text vs. audio). Text registration is naturally immune to speaker differences and environmental noise because text is deterministic; audio registration must withstand variations between the template and the test speech. Experimentally, this manifests as QbyE performing best on the dataset with seen keywords (G) but collapsing on datasets with unfamiliar keywords, whereas this paper's method leads comprehensively except on G.
- **With LVCSR/HMM Traditional Schemes**: No need for decoding graphs, no need to model keywords for customization. The customization cost equals the user typing a line of text.
- **With Deep Model Improvement Routes (PhonMatchNet, dynamic sequence partitioning, etc.)**: Those works modify the matching structure and loss functions, while this paper modifies the information entry point. Feature-level improvements and model-level improvements are orthogonal; theoretically, they can be stacked—this is the smartest part of this paper's positioning and the reason the authors place their conclusion on the transferable judgment that "long-term information is important."

## Experimental Results

### Datasets Used and Their Scales

- **LibriPhrase** (Main dataset, training + evaluation): A phrase dataset derived from the LibriSpeech corpus, with phrase lengths of 1 to 4 words. The training set is generated from the train-clean-100 and train-clean-360 subsets (the number of training episodes is not reported in the paper); the evaluation set is derived from the train-others-500 subset, with 4391, 2605, 467, and 56 episodes for each word length respectively (totaling 7519, calculated by the author). Each episode contains three pairs of positive samples and three pairs of negative samples. Samples are denoted as triplets (audio, text, target), with positive pairs having a target of 1 and negative pairs having a target of 0. Negative samples are further divided into easy and hard categories based on the Levenshtein distance between texts, forming two evaluation sets: LibriPhrase-Easy (LPE) and LibriPhrase-Hard (LPH)—LPH specifically houses difficult negative pairs with similar pronunciations and is the key testbed for examining the ability to distinguish similar pronunciations.
- **Google Speech Commands V1 (G)**: 1881 speakers, 30 small keywords. The validation split corresponding to the 30 keywords is taken for evaluation.
- **Qualcomm Keyword Speech (Q)**: 4 keywords, 50 speakers, totaling 4270 utterances. Each speaker contributes approximately 22-23 instances for each keyword.
- **Zero-Shot Setting**: The model is trained only on LibriPhrase and evaluated directly on G and Q without any fine-tuning, used to examine generalization capabilities to keywords unseen during training.

### Definition and Rationale for Evaluation Metrics

- **EER (Equal Error Rate)**: The error rate when the False Rejection Rate (FRR) equals the False Acceptance Rate (FAR). KWS is a system that trades off between misses and false alarms; EER provides a single scalar at the intersection of the two error rates, independent of the operating point selection.
- **AUC (Area Under the ROC Curve)**: The average discriminative power across all decision thresholds, measuring the separability of the positive pair score distribution from the negative pair score distribution.
- **F1 Score**: The harmonic mean of precision and recall, used only as a supplement in the word length analysis (Table 2).
The rationale for selection lies in the fact that UDKWS is an open-set task: keywords are arbitrarily specified by users, and the cost ratio of false wake-ups and missed wake-ups varies by scenario in actual deployment. Accuracy at a fixed operating point is meaningless; threshold-independent EER/AUC allows for fair comparison of the intrinsic discriminative power of different methods.

### Detailed Comparison with Baseline and SOTA Methods

**Feature Horizontal Comparison (Table 1)**: Comparing six frontend features under the same end-to-end framework. EER (%, lower is better) / AUC (%, higher is better) are as follows:

| Feature | EER-G | EER-Q | EER-LPE | EER-LPH | AUC-G | AUC-Q | AUC-LPE | AUC-LPH |
|---|---|---|---|---|---|---|---|---|
| MFCC | 32.24 | 12.59 | 7.99 | 29.8 | 73.95 | 91.2 | 97.8 | 77.21 |
| MFCC+Δ+ΔΔ | 30.28 | 11.8 | 6.8 | 27.01 | 76.5 | 93.29 | 98.06 | 78.54 |
| Mel Spectrogram | 27.91 | 16.7 | 5.89 | 26.45 | 79.13 | 90.97 | 98.21 | 79.81 |
| PLP | 28.43 | 15.37 | 6.58 | 25.22 | 77.65 | 90.47 | 97.88 | 78.81 |
| RASTA-PLP | 27.4 | 14.32 | 6.42 | 25.84 | 78.05 | 91.24 | 97.82 | 79.7 |
| **SDC** | **23.54** | **9.61** | **3.84** | **21.48** | **83.56** | **96.73** | **98.34** | **85.90** |

Several readings worth expanding on:

1. **SDC ranks first in all four datasets and both metrics**, without exception. The most convincing evidence is LPH: EER 21.48% vs. MFCC's 29.8% (absolute drop of 8.32 percentage points), AUC 85.90% vs. 77.21% (absolute rise of 8.69 percentage points)—the improvement on the difficult similar pronunciation set is much larger than on LPE (EER 3.84% vs. 7.99%), indicating that the long-term deltas captured by SDC are precisely the information needed to distinguish similar pronunciations.
2. **Errata regarding swapped numbers in Abstract and Body**: The Abstract writes "8.32% in AUC and 8.69% in EER," while Section 5.1 writes "8.69% in AUC and 8.32% in EER," with the two being inverted. Calculating from the original numbers in Table 1 (85.90−77.21=8.69; 29.8−21.48=8.32), the body's expression is consistent with the table, while the Abstract has the two numbers reversed. For close reading, rely on the body and table.
3. **Consistent improvement of MFCC+Δ+ΔΔ over pure MFCC** (e.g., LPE EER 7.99→6.8, LPH EER 29.8→27.01) is itself evidence that "temporal dynamics are useful"—simply concatenating first and second derivatives into the features can stably boost performance, indicating that MFCC lacks not spectral quality but dynamic information; SDC pushes this idea to a long-term scale, multiplying the gain (LPE EER 3.84% vs. 6.8%).
4. **PLP, RASTA-PLP, and mel spectrogram have mixed wins but are generally close** (e.g., on LPH, PLP's EER of 25.22% is actually better than mel spectrogram's 26.45%, while on LPE, mel spectrogram's EER of 5.89% is better than PLP's 6.58%). The paper categorizes them as "competitive alternatives"; MFCC ranks last on all datasets. Notably, the paper claims mel spectrogram was chosen as the SDC base because it "performs better than all other short-term features," but in Table 1, on the Q dataset, MFCC+Δ+ΔΔ's EER (11.8%) and AUC (93.29%) are both better than mel spectrogram (16.7%, 90.97%). This statement has a slight discrepancy with its own data.

**Comparison with SOTA Methods (Table 3)**:

| Method | EER-G | EER-Q | EER-LPE | EER-LPH | AUC-G | AUC-Q | AUC-LPE | AUC-LPH |
|---|---|---|---|---|---|---|---|---|
| CTC (DONUT) [12] | 31.65 | 18.23 | 14.67 | 35.22 | 66.36 | 89.69 | 92.29 | 69.58 |
| Attention (QbyE) [11] | 14.75 | 49.13 | 28.74 | 41.95 | 92.09 | 50.13 | 78.74 | 62.65 |
| Triplet [4] | 35.6 | 38.72 | 32.75 | 44.36 | 71.48 | 66.44 | 63.53 | 54.88 |
| CMCD [13] | 27.25 | 12.15 | 8.42 | 32.9 | 81.06 | 94.51 | 96.7 | 73.58 |
| **This Paper** | 23.54 | **9.61** | **3.84** | **21.48** | 83.56 | **96.73** | **98.34** | **85.9** |

- **Attention QbyE stands alone on G** (EER 14.75%, best in the field, even better than this paper's 23.54%). The paper explains that its similarity scoring mechanism is particularly effective when keywords appear in the training set; however, the same method suffers catastrophic degradation on Q with an EER of 49.13% and on LPH with 41.95%—the audio template matching paradigm is almost unusable for unfamiliar keywords, highlighting the necessity of the text registration route.
- **CMCD is the strongest baseline**. The improvement of this paper's method relative to it: LPE EER 8.42%→3.84% (paper reports improvement of 4.58%), LPH EER 32.9%→21.48% (paper reports improvement of 11.42%), LPH AUC improvement of 12.32 percentage points (73.58→85.9, calculated by the author from the table).
- **Zero-shot on G and Q**: This paper's method is stably better than CMCD by about 2% in AUC and 3% in EER (approximate figures given by the paper), proving that the matching ability learned on LibriPhrase can be transferred to completely unfamiliar keyword sets—this means for real products, one training allows deployment with arbitrary custom keywords.
- The only battlefield where this paper did not take first place is G (losing to QbyE's 14.75%), which is related to the keywords in G being in the training set, falling exactly into the sweet spot of the QbyE paradigm.

**Training Configuration** (Implementation Details): 25 ms window, 10 ms frame shift, 0.97 pre-emphasis, Hamming window, zero-padding in the time dimension for alignment; 0.2 dropout after each layer of the two encoders; binary cross-entropy loss; Adam with default parameters, batch size 128, fixed learning rate $10^{-4}$; best model selected according to the validation set; training hardware consists of four NVIDIA GeForce RTX 2080 Ti GPUs. The number of training epochs and the number of training sample pairs are not reported in the paper.

### Findings from Ablation Experiments

**SDC Configuration Ablation (Fig. 2)**: Fixing other parameters and varying $d$ and $k$ individually.

- **Shift amount $d$ (Fig. 2a, 2b)**: As $d$ increases from 1 to 4, performance on all datasets monotonically decreases. The reason can be understood from the definition of delta: the larger $d$ is, the farther apart the two sampling points of the delta $c(t+ip+d) - c(t+ip-d)$ are. The calculated result is a coarse-grained trend over a wider span rather than a fine transition, equivalent to temporal downsampling of the spectral trajectory derivative. Meanwhile, large $d$ is more likely to cross phoneme boundaries, mixing the spectra of different phonemes into a single delta, diluting the discriminative information of a single transition event. The optimal configuration takes $d=1$, i.e., the finest delta granularity.
- **Number of stacked blocks $k$ (Fig. 2c, 2d)**: As $k$ increases from 5 to 8, AUC and EER consistently improve on all datasets; after exceeding 8, performance saturates or decreases. This answers the question "is the longer the context, the better?"—no. There are two layers of reasonable explanation: first, the phoneme transition information of keywords is concentrated within a few hundred milliseconds. When $k$ exceeds 8, the newly added delta blocks extend into adjacent, unrelated speech, introducing noise rather than signal. Second, adding one $k$ adds 40 dimensions of features ($k=8$ reaches 360 dimensions per frame), and dimension expansion increases the optimization burden. **Optimal configuration 40-1-3-8**: 40-dimensional mel base, minimum shift, medium step, 8 blocks stacked. It strikes a balance between "sufficient information" and "precise context"—the paper's original words are that reliable recognition of keywords requires precise rather than lengthy context.

**Word Length Analysis (Table 2)**: Stratified by word length on the LibriPhrase evaluation set, comparing mel spectrogram (strongest baseline) with SDC:

| Feature | Word Length | EER (%) | AUC (%) | F1 (%) |
|---|---|---|---|---|
| Mel Spectrogram | 1 | 7.67 | 97.01 | 91.11 |
| Mel Spectrogram | 2 | 8.55 | 96.57 | 90.98 |
| Mel Spectrogram | 3 | 9.05 | 95.6 | 89.76 |
| Mel Spectrogram | 4 | 9.25 | 95.34 | 88.30 |
| SDC | 1 | 5.37 | 98.25 | 94.34 |
| SDC | 2 | 6.35 | 97.87 | 93.31 |
| SDC | 3 | 7.24 | 96.91 | 91.57 |
| SDC | 4 | 8.29 | 96.31 | 90.32 |

- The absolute F1 improvement of SDC over mel spectrogram by word length 1/2/3/4 is 3.24%, 2.33%, 1.81%, 2.02% (values reported by the paper), with stable gains in every word length category.
- Both features degrade monotonically as word length increases (mel EER 7.67→9.25, SDC EER 5.37→8.29). The reason is easy to understand: the longer the word, the longer the character sequence and audio, the more complex the alignment the cross-attention must complete, and the more opportunities for confusion in partial matching; errors accumulate over longer sequences. SDC mitigates but does not eliminate this trend—feature-level temporal enhancement cannot replace model-level alignment capability. Also note that the evaluation set for word length 4 has only 56 episodes, so the statistical confidence of conclusions for this category is limited.

## Main Contributions

1. **First systematic demonstration of the value of feature-level long-term information for UDKWS** (authors declare this as the first study in this direction): Using SDC to pull the temporal receptive field of each frame's feature from the 25 ms scale to about 230 ms, comprehensively outperforming short-term spectral features on four datasets. The most convincing evidence is the AUC +8.69%, EER −8.32% on the LPH similar pronunciation set (Table 1, body口径).
2. **Horizontal comparison of five features under a unified framework**: MFCC, MFCC+Δ+ΔΔ, mel spectrogram, PLP, RASTA-PLP, and SDC compete under the same model and same training configuration, excluding interference from model differences and providing a clean feature selection reference for subsequent work.
3. **Providing the SDC configuration sweet spot 40-1-3-8**: Through ablation of dual parameters $d$ and $k$ (Fig. 2), proving that there is an optimal point for context length, responding to the question "long-term information is not necessarily better the longer it is," which is easily biased by intuition.
4. **Stratified analysis in the word length dimension (Table 2)**: Revealing the universal law of UDKWS performance degradation with word length, and the consistency of SDC gains across word lengths.
5. **Comprehensive superiority over SOTA**: Outperforming four representative methods (CTC, Attention QbyE, Triplet, CMCD) on Q, LPE, and LPH (Table 3), and stably exceeding the strongest baseline CMCD by about 2% AUC / 3% EER in zero-shot settings on G and Q, verifying generalization capabilities to unfamiliar keywords.

## Limitations and Future Work

### Technical Limitations of the Method

- **Feature Dimension Explosion and Unassessed Edge-Side Cost**: Under the optimal configuration, each frame is 360 dimensions, which is 9 times the 40-dimensional mel spectrogram. The number of input channels for the first layer of convolution in the audio encoder increases by 9 times accordingly, and the number of parameters and computation per frame amplify synchronously. The paper does not report the number of parameters, FLOPS, inference latency, or model size throughout, while KWS is a typical persistent low-power load—if the accuracy gain from feature engineering comes at the cost of 9 times the frontend computation, the net benefit on the edge side is questionable. This is the biggest unanswered question between this paper and the "small footprint KWS" mainline.
- **Fixed Receptive Field, Non-learnable**: The delta weights of SDC are fixed at ±1, and the receptive field is locked once and for all by the N-d-p-k hyperparameters. It can capture "the spectral trajectory within 230 ms near this frame," but it cannot automatically adjust the aggregation method for the task like a learnable temporal layer. Theoretically, the optimal receptive field differs for different word lengths and speaking speeds; a fixed configuration can only take a global compromise.
- **Long-word Performance Degradation Not Fundamentally Solved**: Table 2 shows that SDC still has an EER of 8.29% at word length 4, following the same degradation trend as mel spectrogram, indicating that feature-level patches cannot solve the fundamental problem of long-sequence alignment.
- **Not Advantageous in Seen-Word Scenarios**: On the G dataset where keywords are seen in the training set, this paper's method has an EER of 23.54%, significantly inferior to the Attention QbyE's 14.75% (Table 3), suggesting that the text registration route is not optimal in scenarios with "popular fixed wake-words + available templates."
- **Absolute Performance is Far from Practical**: The EER on the hardest LPH is still 21.48%, meaning about one-fifth of similar pronunciation negative pairs will be falsely accepted. The false wake-up rate is unacceptable in real products.
- **Robustness Not Verified**: The training and evaluation data (LibriSpeech audiobook reading speech, Google Commands, Qualcomm) are basically clear, close-talk speech. The performance under noise, far-field, reverberation, and accent conditions is not experimented with in the paper (not reported).

### Shortcomings in Experimental Design

- **Lack of Control for "Model-Side Long Context"**: The paper proves that SDC is superior to short-term features, but there is no control for "short-term features + larger model receptive field" (e.g., deeper Bi-GRU, larger multi-kernel convolutions, or FSMN-style memory blocks). Strictly speaking, to fully attribute whether the gain comes from "the information entry having long-term deltas" or simply "the model receiving more information," a model-side expansion control group is needed. The MFCC+Δ+ΔΔ partial control (short-term deltas also boost performance) supports the temporal information hypothesis, but the control group is not thorough enough.
- **Ablation Only Scanned $d$ and $k$**: Among the four parameters, $p$ (step between blocks) and $N$ (base dimension) were not ablated. The rationality of $p=3$ has only results without argumentation (paper does not report scanning of $p$); Fig. 2 only presents trends in curves without attaching specific numerical tables.
- **Training Set Scale is Opaque**: The number of episodes in the LibriPhrase training split and the total number of positive/negative pairs are not reported (not reported in the paper). The number of training epochs and whether Tacotron 2 is frozen are also not specified, affecting reproducibility.
- **Single Run, No Significance Test**: No reports of multiple random seeds or confidence intervals are seen for all results; the word length 4 category has only 56 episodes (the sample size per pair type in this category is extremely small), making conclusions for this category particularly fragile.
- **Missing Baseline Reproduction Details**: It is not explicitly stated in the paper whether the four baselines in Table 3 were retrained on exactly the same data splits or if the original paper's numbers were cited.
- **Abstract Number Typo**: The improvement amounts for AUC/EER in the Abstract are inverted relative to Section 5.1 of the body and Table 1 (see errata in the Experimental Results section). This is a minor flaw in writing but affects citation accuracy.

### Possible Directions for Future Improvement

- **Direction pointed out by the authors themselves**: Hybrid feature extraction—instead of relying on a single feature, combine multiple complementary frontends, allowing different features to cover different aspects of spectral details and long-term dynamics.
- **Learnable Long-Term Frontend (Author's Deduction)**: Replace SDC's fixed ±1 delta kernel with learnable dilated temporal convolutions or FSMN-style memory blocks, allowing the network to learn the task-optimal temporal aggregation itself, preserving SDC's idea of "low-parameter expansion of receptive field" but freeing it from manual hyperparameter tuning.
- **Dimensionality Reduction and Efficiency Assessment**: The 320-dimensional stacked deltas of SDC have obvious redundancy (adjacent delta blocks are highly correlated). Low-rank projection, feature selection, or PCA compression in the delta domain can be explored to pull dimensions back to the mel spectrogram scale; meanwhile, reports on parameters/latency/power consumption must be supplemented to judge edge-side usability.
- **Orthogonal Combination with Model-Level Improvements**: The feature gain of this paper was achieved on the authors' own 2024 architecture. Whether transplanting the SDC frontend to stronger matching structures like CMCD, PhonMatchNet, etc., can stack gains is a natural subsequent experiment.
- **Robustness and Multilingual Extension (Author's Deduction)**: Verify the universality of long-term deltas on noisy/far-field data and non-English languages—SDC's cross-lingual background in the LID field actually suggests it may have additional value in multilingual UDKWS.
- **Specialized Assault on Long-Word Scenarios**: Targeting the degradation trend with word length, combine dynamic sequence partitioning or hierarchical matching strategies to leave the alignment problem of long keywords to the model side, forming a complement to the long-term information on the feature side.

---

*Terminology Quick Reference*: **UDKWS**, User-Defined Keyword Spotting, open-set task, registration modality is text or audio template; **SDC**, Shifted Delta Coefficients, features that encode long-term temporal dynamics by stacking first-order deltas at multiple delay positions; **EER**, Equal Error Rate, the error rate when the false rejection rate equals the false acceptance rate, lower is better; **AUC**, Area Under the ROC Curve, overall discriminative power across thresholds; **Levenshtein Distance**, the minimum number of edit operations between two strings, used by LibriPhrase to divide negative pairs into easy/hard categories; **CMCD**, Cross-Modal Consistency Detector, the audio-text matching baseline proposed by the LibriPhrase authors; **QbyE**, Query-by-Example, a paradigm that uses registered audio templates for similarity matching.
