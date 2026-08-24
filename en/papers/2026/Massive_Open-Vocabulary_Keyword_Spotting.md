# Massive Open-Vocabulary Keyword Spotting

- **Authors/Affiliations**: Leonor Barreiros, Raul Monteiro, Afonso Mendes, Gonçalo M. Correia - Priberam Labs (Lisbon) / Instituto Superior Técnico (Lisbon) / Instituto de Telecomunicações (Institute of Telecommunications, Portugal)
- **Date**: 2026.06
- **Link**: https://arxiv.org/abs/2606.11279
- **Keywords**: open-vocabulary keyword spotting, contextual biasing, embedding compression, sparsemax layer selection, Whisper, entity recall, production-grade ASR

## Problem Statement

### Problem Background and Domain Pain Points
ASR foundation models (such as Whisper) systematically underfit rare professional terms—entities in the tail of the word distribution—that are scarce in training data. This is a critical flaw in production scenarios such as medical consultation transcription and air traffic control: transcription errors of terminology directly disrupt downstream semantic extraction.

Contextual Biasing (CB) is a mitigation strategy: the Whisper decoder is essentially an audio-conditioned language model that references preceding tokens during generation; CB replaces this "historical context" with a custom prompt, injecting the biasing vocabulary directly into the decoder. When the vocabulary is long, the prompt loses focus. Therefore, Open-Vocabulary Keyword Spotting (OV-KWS) is required first to filter out the few actual keywords present in the audio from a large vocabulary, and then this small list is prompted into the decoder.

The production pain points lie in storage and latency: existing audio-to-audio OV-KWS approaches (the CB-Whisper lineage) must extract distributed representations from the "audio of the spoken keyword" and precompute them for storage, with complexity growing linearly with the number of vocabulary entries. The hard math of the paper: assuming 32-bit storage, under the baseline settings ($l=12$ layers, $f=150$ frames, $h=1024$ hidden dimensions), each keyword embedding occupies $4 \times 12 \times 150 \times 1024 \approx 7.3$ MB—a vocabulary exceeding 11,650 words cannot fit into a single 80GB A100 GPU. Existing works have vocabularies of no more than 500 words, which is two orders of magnitude away from real-world needs (the paper constructs a medical terminology list of 16,062 words).

### Specific Shortcomings of Existing Methods
- **Layer selection lacks principles**: CB-Whisper [10] aggregates representations from layers 10–21 of Whisper-medium, while the improved version [11] aggregates the last 12 layers. The paper points out that both "fail to explain the motivation for selection." The layer dimension is the largest contributor to storage (12 layers directly amplify storage by 12 times), yet it is determined by guesswork.
- **Audio-to-audio route is effective but inefficient**: [10, 11] both use distributed representations extracted from audio, preserving acoustic information. Empirical evidence proves that acoustic encoder information is crucial for correctly transcribing rare words [5, 14]; however, high-dimensional features make it impossible to process massive vocabularies within acceptable resource limits.
- **Text-to-audio route is efficient but ineffective**: [12, 13] encode keyword text and utterance audio into a shared hyperspace for matching, eliminating the need for keyword audio. However, this loses acoustic information, and there is no deterministic binding between how a word is pronounced and how it is written (homographs with different pronunciations, abbreviations), limiting detection quality.
- **Lack of production-oriented evaluation**: Existing works have not quantified latency and memory on real large vocabularies, with vocabulary sizes stagnating at hundreds of words.

### Key Challenges to Be Solved by This Paper
Without fine-tuning the ASR model, compress the keyword embeddings of audio-to-audio OV-KWS to be up to 128 times smaller and process 6 times faster than comparable baselines, while: maintaining the open-vocabulary property (any keyword can be queried at inference time); improving KWS detection quality (F1); and maintaining entity recall after CB comparable to uncompressed schemes, even holding true for languages unseen during training (Chinese).

## Methodology

### Overall Architecture Design and Design Motivation
The system follows the two-module OV-KWS skeleton of [10], inserting a compression module in the middle to form a three-stage structure:

1. **Acoustic Encoder**: The Whisper encoder encodes audio into $\mathbb{R}^{l \times f \times h}$ ($l$ is the number of layers from which representations are extracted, $f$ is the number of frames, $h$ is the hidden dimension). Keywords and utterances pass through the same encoder, ensuring their representations can be directly aligned.
2. **Compression Module (New in this paper)**: Compresses sequentially along three orthogonal dimensions—number of layers (sparsemax gated selection), hidden dimension (one-hidden-layer FFN for dimensionality reduction), and frame rate (1D CNN for temporal resolution reduction).
3. **ResNet Binary Classifier**: Keyword and utterance embeddings form $l$ cosine similarity matrices (frame axis zero-padded to fixed sizes $f^{utt}$ and $f^{kwd}$ to facilitate generalization). The classifier detects "diagonal features" on the matrix—whether the frame-level representation of the keyword aligns locally with the utterance along the time axis—outputting whether the word appears.

Why compress three dimensions instead of one? The embedding storage volume is the product of $l \times f \times h$, and each factor contributes linearly to storage; however, the compression methods for the three are different—the layer dimension requires "selection," the hidden dimension requires "projection," and the frame rate requires "downsampling." Each plays its own role, and only together do they yield a 128x reduction.

Production form: The keyword database is precomputed offline—after the vocabulary is synthesized via TTS (or cut from natural utterances), it is encoded, compressed, and stored in the database; online utterances are compared against the full database embeddings for classification, and detected words are injected into the Whisper decoder prompt. Training data mixes TTS and natural speech keyword audio ([11] proves this is a necessary condition for generalization to both sources), with hard negatives taken as words that are lexicographically close to positive examples but not present in the utterance.

### Mathematical Principles of Core Algorithms

**Layer Sparse Selection (First Dimension)**: Let the encoder have $l$ layers in total, $w \in \mathbb{R}^l$ be a trainable score vector, which yields:

$$p = \text{sparsemax}(w) \in \Delta^{l-1}$$

$\Delta^{l-1}$ is the $(l-1)$-dimensional probability simplex. Let the complete embedding aggregating all layers be $x \in \mathbb{R}^{l \times f \times h}$ (denoted as $l \times f \times e$ in the original text), with gating performed via element-wise multiplication $\tilde{x} = p \odot x$. The training objective is standard binary cross-entropy for KWS plus entropy regularization:

$$L = L_{BCE} + L_{aux}, \qquad L_{aux} = H(p)$$

$H(\cdot)$ is the entropy of the class distribution. Two mathematical choices: **Why sparsemax instead of softmax**—softmax outputs are strictly positive, always assigning non-zero weights to every layer, making it impossible to truly "remove" layers; sparsemax, as an Euclidean projection on the simplex, can output exact 0s, and the non-zero items of $p$ after training directly give the set of retained layers (denoted size $l_{comp}$). **Why minimize entropy**—entropy minimization pushes probability mass towards peaks, equivalent to soft pruning layer by layer, forcing discriminative power to concentrate on a few layers.

**Hidden Dimension Compression (Second Dimension)**: A lightweight one-hidden-layer FFN projects $h$ to $h_{comp}=64$ (number of hidden units is $h/2$ according to the original text). Keyword and utterance embeddings pass through the FFN separately, trained jointly with the classifier using BCE.

**Frame Rate Compression (Third Dimension)**: The Whisper encoder frame rate is 50 frames/second, and embedding size grows linearly with the number of frames. Downsampling along the frame direction using a 1D CNN (kernel 3 stride 1 → BatchNorm → kernel 3 stride 2 max pooling) reduces the rate by $\alpha=2$, i.e., $f^{utt}_{comp}=f^{utt}/\alpha$ and $f^{kwd}_{comp}=f^{kwd}/\alpha$.

**Total Compression Accounting** (according to the paper,折算 to comparable baselines with Whisper-medium): Layers $12 \to 3$ (4x) $\times$ Hidden dims $1024 \to 64$ (16x) $\times$ Frames $150 \to 75$ (2x) $=128$x. The experimental backbone is actually Whisper-large-v2 ($l=32$, $h=1280$); the paper states that memory/computation savings have already accounted for backbone differences; if calculated based on its own uncompressed representation, it is $32/3 \times 1280/64 \times 2 \approx 426$x (author's derivation). The engineering implication of 128x: an NVIDIA L40 with 48GB can accommodate 894,784 keywords, whereas the baseline on an 80GB A100 is limited to about 11,650 words—the capacity ceiling is raised by about 77 times (author's derivation).

### Key Technical Innovation 1: Automatic Sparse Selection of Encoder Layers Based on Sparsemax Gating
Addressing the previous reliance on manual experience for layer selection, this paper turns it into a learnable structural search: score vector + sparsemax gating + entropy regularization, end-to-end jointly optimized with the KWS classifier. There are three reasons: the layer dimension is the largest contributor to storage, and cutting layers yields direct and linear benefits; it is empirically unknown which of the 32 layers have strong discriminative power for KWS, so relying on gradients is more justified than manual specification; sparse selection also avoids discrete search over layer combinations. Training results: $w \in \mathbb{R}^{32}$ converges to select 3 layers—layers 14, 16, and 32 (32 being the final layer of the encoder). Author's note: 14 and 16 fall within the manually selected range of 10–21 in [10], and 32 falls within the last 12 layers of [11], yet only 3 layers are needed—indirectly confirming the redundancy of aggregating 12 layers.

### Key Technical Innovation 2: Hidden Dimension Projection Compression (FFN)
After cutting to 3 layers, the embedding is still on the order of $3 \times 150 \times 1280$, and ablation studies show that using raw 3-layer features directly causes discriminative power to collapse (see ablation section). The role of the FFN is not simply dimensionality reduction, but to **learn the mapping to project KWS discriminative information onto a 64-dimensional low-dimensional manifold after layer selection**—the paper's original words are "only by adding trainable parameters after minimizing the number of layers can one learn an embedding space projection with unique KWS features." This explains why the compression module must be trained jointly with the classifier rather than using post-hoc PCA: discriminative projection requires task gradient supervision.

### Key Technical Innovation 3: Time Resolution Downsampling (1D CNN)
KWS judgment of "whether a word appears" does not require 50Hz frame-level accuracy; there is significant redundancy along the time axis. 1D convolution + pooling halves the frame rate, synchronously dividing keyword and utterance frames by $\alpha$, linearly reducing storage. Using a lightweight CNN without modifying the encoder is because keeping the encoder untouched is a design red line in this paper—all compression occurs on the feature side, allowing it to be plug-and-play added to any Whisper version and WhisperX production line.

### Technical Differences from Existing Methods
- **vs [10]/[11]**: Maintains the audio-to-audio route (preserving acoustic information), but embeddings are 128 times smaller and 6 times faster; layer selection changes from manual 12 layers to learned 3 layers; vocabulary advances from ≤500 words to 16,062 words.
- **vs [12]/[13] (Text-to-Audio)**: Rejects encoding keywords via text—pronunciation and spelling have no deterministic binding, and acoustic information is irreplaceable; the cost (storing keyword audio embeddings) is eliminated by compression.
- **vs Model Compression (Pruning/Quantization)**: This paper compresses not the model but the **feature database**—the Whisper encoder and ResNet-50 weights remain intact; the product of compression is the offline keyword database; it solves "database doesn't fit, scanning is too slow," not inference compute power.
- **CB Integration**: Detected words are injected as hotwords into the WhisperX (based on Whisper-large-v2, VAD handles long audio) `transcribe` prompt parameter, with zero changes to ASR weights.

## Experimental Results

### Datasets Used and Their Scale
- **Training**: Multilingual LibriSpeech (MLS) audiobook corpus, reusing the pipeline from [11] to prepare 25 hours of data, covering six languages: English, French, German, Polish, Portuguese, and Spanish (the original text does not specify if it is 25 hours per language or in total). Training keywords are extracted as nouns and proper nouns from transcriptions using spaCy, with 12,000 words randomly selected per language, fixed during training. Keyword audio is a mix of TTS (edge-tts) and natural speech, with negatives including hard negatives lexicographically close to positives.
- **Validation**: MLS development set, 200 keywords per language, following the [11] workflow.
- **Evaluation** (Table 1): **Aishell** (Chinese) 76 minutes / 808 sentences / 400 entities, studio quiet environment, high-fidelity microphone; Chinese is a completely unseen language for KWS, specifically testing cross-lingual generalization; **ACL6060** (English) 51 minutes / 123 sentences / 200 entities, real academic conference speeches, noisy with many terms, the hardest out-of-domain test; **Internal** (Portuguese internal corpus) 103 minutes / 210 sentences / 16,062 entities, home medical consultations, vocabulary from the Portuguese Ministry of Health clinical terminology semantic directory, extracting all words, removing stopwords and numbers—deliberately not manually curated to simulate real production vocabularies.
- All validation and evaluation keyword audio are TTS-generated—close to production: domain terminology lists are almost never available with natural pronunciations.

### Definition and Rationale for Evaluation Metrics
- **KWS**: Open-source corpora use F1 at optimal threshold (threshold determined on validation set to avoid tuning on test set), accompanied by PR curves; internal corpus uses F1@5 (top-k, $k=5$). $k$ is deliberately kept low because CB amplifies hallucination risks [8]—adding one more erroneous hotword to the prompt may cause the decoder to transcribe it out of thin air. Results include bootstrap 95% confidence intervals.
- **ASR**: Normalized Mixed Error Rate MER (equivalent character error rate for non-space-separated languages, equivalent word error rate for space-separated languages; normalized = remove punctuation + lowercase) and Entity Recall R. Evaluation of transcriptions first aligns with unbiased transcriptions using Needleman-Wunsch at the character level before calculating metrics—hotwords injected by CB may cause insertion-type hallucinations, and alignment can partially mitigate their double-counting.
- **Efficiency**: Real-Time Factor (RTF) for each 30-second utterance, and keyword database memory usage (MB)—directly corresponding to production latency and VRAM budget.

### Detailed Comparison with Baseline Methods and SOTA
The backbone is Whisper-large-v2 ($l=32$, $h=1280$, distinct from the baseline original setting of Whisper-medium with 24 layers / 1024 dimensions), $f^{utt}=1500$, $f^{kwd}=150$, classifier ResNet-50, new parameter learning rate $1\times10^{-4}$, trained on a single NVIDIA L40. CB comparison involves three scenarios: No-CB, Baseline recr. [11] (16,062 word database does not fit in VRAM, so processed in batches of 125 words), and the most compressed configuration LHF-comp. The baseline recreation on Aishell has an MER of 24.9, which is worse than No-CB's 18.0—direct empirical evidence of the cost of CB hallucinations; batch processing also brings repeated GPU transfers and repeated KWS calculations, spiking RTF to 4.52.

**ASR End-to-End Results (Table 2)**:

| Setting | ACL6060: MER / R / RTF / Memory | Aishell: MER / R / RTF / Memory | Internal: MER / R / RTF / Memory |
|---|---|---|---|
| No-CB | 27.6 / 52.5 / 0.03 / 0 | 18.0 / 40.2 / 0.03 / 0 | 29.5 / 70.9 / 0.07 / 0 |
| Baseline recr. [11] | 27.0 / 54.3 / 0.18 / 1,406 | 24.9 / 59.3 / 0.29 / 2,812 | 28.7 / 72.0 / 4.52 / 112,929 |
| LHF-comp (This Paper) | 21.9 / 57.2 / 0.17 / 11 | 14.7 / 71.3 / 0.10 / 22 | 32.4 / 71.5 / 0.76 / 882 |

Readings for the three corpora: (1) **Aishell** (small and clean vocabulary + high-recall KWS): Entity recall 40.2→71.3 (+31.1 percentage points), MER simultaneously 18.0→14.7—greatest benefit, compressed embeddings outperform uncompressed baseline recreation comprehensively. (2) **ACL6060** (noise + terminology): Recall +4.7 (52.5→57.2), MER −5.7 (27.6→21.9), mild positive effect. (3) **Internal** (16,062 word uncurated vocabulary): Recall basically flat (70.9→71.5), **MER actually worsens 29.5→32.4**—CB is a negative asset on dirty vocabularies. Attribution: The vocabularies of the first two corpora are extracted from gold transcriptions, naturally aligned with audio; the Internal vocabulary is not cleaned, mixing in many common words (e.g., allergens), which unbiased ASR could originally transcribe correctly (biasing brings no gain) or creates CB false positives; and Whisper's prompt conditioning mechanism was pretrained to learn "previous transcription context," lacking robustness to irrelevant terms and easily misled by false matches [8]. In terms of efficiency, memory 112,929→882 MB (exactly 128x), RTF 4.52→0.76 (~5.9x, i.e., the paper's 6x); authors emphasize **only LHF-comp can load the entire database into VRAM to process massive vocabularies**.

**Cross-Lingual Generalization**: LHF-comp achieves F1 86±4 on Chinese Aishell (zero Chinese in training), close to [10]'s in-domain training result of 89.0 ([10] is in-domain evaluation, this paper and [11] are all out-of-domain); recall 71.3 is also higher than the baseline recreation's 59.3.

### Findings from Ablation Studies
Three configurations are constructed progressively along the three compression dimensions, forming a complete ablation chain (Table 3):

| Model | ACL6060 F1 | Aishell F1 | Internal F1@5 |
|---|---|---|---|
| Baseline [10] | − | 89.0 (in-domain) | − |
| Baseline [11] | 65±4 | 71±5 | 22±2 |
| L-comp (Layer compression only) | 37±4 | 43±6 | − (not reported in paper) |
| LH-comp (Layer + Hidden dim) | 63±7 | 84±3 | 13±1 |
| LHF-comp (Layer + Hidden dim + Frame rate) | 69±4 | 86±4 | 28±2 |

Four key findings:
1. **Layer compression cannot be used alone**: L-comp drops from 65/71 to 37/43. Raw features of 3 layers have far less discriminative power than aggregated 12 layers—cutting redundancy also cuts capacity, which must be compensated by learnable projection.
2. **FFN projection is key to recovery**: Adding 64-dimensional projection (LH-comp) brings F1 back up to 63/84, approaching the uncompressed baseline—64-dimensional manifold is sufficient to carry KWS discriminative information; the bottleneck is not dimensionality but whether the projection is trained with task gradients.
3. **Halving frame rate actually boosts performance**: LHF-comp exceeds LH-comp on both corpora (69>63, 86>84), and comprehensively exceeds uncompressed Baseline [11] (69>65, 86>71). Authors interpret this as "adding more trainable parameters after minimizing layers allows the network to learn an embedding space projection containing unique KWS features." Counter-intuitive conclusion: **if compression is done correctly, it can act as regularization**.
4. **Compression is more robust under large vocabularies**: Internal F1@5 rises from 22±2 to 28±2. PR curves (Fig. 3) show ACL6060 is the hardest—authors hypothesize its many abbreviations have non-trivial phonetic spellings, making them difficult for current TTS to synthesize correctly, compounded by speech noise; Aishell is a quiet environment with high-quality recordings, relatively easier.

## Main Contributions
1. **Automatic Sparse Layer Selection**: The first method to automatically filter out the most predictive layers for KWS from a transformer acoustic encoder using trainable sparsemax gating + entropy regularization, replacing unmotivated manual layer aggregation, practically compressing Whisper-large-v2's 32 layers to 3 (14, 16, 32).
2. **Three-Dimensional Embedding Compression Mechanism**: Orthogonal compression pipeline of Layers × Hidden Dims × Frame Rate, compressing acoustic embeddings to 1/128 while performance improves, increasing vocabulary capacity on 48GB GPU from ~11,000 words to 894,784 words.
3. **Efficiency Empirical Evidence for Real Production Systems**: Quantifies latency and memory benefits on a 16,062-word medical terminology list (6x speedup, 128x memory savings), and honestly reports the boundary conditions where CB harms ASR on dirty vocabularies. Code open-sourced at https://github.com/Priberam/Enhance-CB-Whisper .
4. Methodological Assertion (self-stated in paper conclusion): Text-to-audio schemes are "efficient but ineffective," audio-to-audio schemes are "effective but inefficient"; this paper eliminates the inefficiency of the latter, making it scalable to massive vocabularies for the first time.

## Limitations and Future Work

### Technical Limitations of the Method
- **CB is a negative asset on dirty vocabularies**: The worsening of MER 29.5→32.4 indicates the system is highly sensitive to vocabulary quality; the root cause is Whisper's prompt mechanism is not robust to irrelevant hotwords—compression only solves "can we process large vocabularies," not "should we put these words into the vocabulary."
- **Frame rate compression factor is conservative**: $\alpha=2$ is the only experimental value (50→25 frames/second); whether larger $\alpha$ maintains performance is not reported in the paper.
- **Overhead of the compression module itself is opaque**: Parameter counts, FLOPs, and training time for FFN and CNN are not reported, making it impossible to independently calculate the additional inference cost introduced by compression.
- **Lack of mechanistic analysis for layer selection**: Why layers 14, 16, and 32? The paper provides no explanation at the representation level; transferability to other backbones is unverified.
- **TTS Dependency**: Training and evaluation keywords rely entirely on TTS; authors themselves hypothesize that TTS pronunciation distortion of abbreviations on ACL6060 dragged down performance—TTS quality is the implicit upper limit of the system.

### Shortcomings in Experimental Design
- The internal medical corpus is not public; the landmark experiment with 16,062 words cannot be independently reproduced.
- Training data is only 25 hours × 6 languages (audiobook domain), scale and domain bias are narrow; cross-lingual conclusions are only tested on one unseen language (Chinese).
- MER is confounded by WhisperX VAD segmentation error propagation (authors self-state: open-source corpus utterances are short, turning off VAD avoids losing information; internal corpus whole consultation exceeds 30 seconds, so segmentation is necessary), making error attribution not clean enough.
- No direct comparison with text-to-audio schemes [12, 13]—the "efficient but ineffective" assertion is cited from literature, not verified by this experiment.
- The baseline is a "recreation" (backbone changed to large-v2, vocabulary batched); although memory scaling is claimed, strictly speaking, it is not a reproduction of the original settings.

### Possible Directions for Future Improvement
- **Vocabulary dimension compression** (explicit future work in paper): Heuristic filtering/clustering of vocabularies by domain, removing interfering words from the source—the most direct path to turning the negative CB return on the Internal corpus positive.
- **Interference-resistant hallucination suppression** (explicit future work in paper): Suppressing CB hallucinations even in the presence of interfering words, e.g., confidence-weighting hotwords in the prompt.
- Author's additions (not paper's viewpoint): Scanning larger $\alpha$, verifying layer selection across backbones, switching to lighter classifiers to reduce RTF, and stacking compressed embeddings with int8 quantization are all natural next steps.
