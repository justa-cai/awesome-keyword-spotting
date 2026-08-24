# Keyword spotting using convolutional neural network for speech recognition in Hindi

- **Authors/Affiliations**: Saru Bharti (iHub DivyaSampark, Indian Institute of Technology Roorkee); Pushparaj Mani Pathak (Department of Mechanical and Industrial Engineering, Indian Institute of Technology Roorkee)
- **Date**: April 2026 (arXiv:2605.02928v1 [cs.SD], submitted April 26, 2026)
- **Link**: https://arxiv.org/abs/2605.02928
- **Keywords**: keyword spotting, convolutional neural network, MFCC, Hindi speech recognition, on-device inference, data augmentation, negative class modeling

## Problem Statement

### Problem Background and Domain Pain Points

The paper opens with the background of the rise of voice assistant platforms like Alexa and Google Assistant, highlighting that speech recognition has become a key technical entry point in recent years. However, the authors immediately point out an asymmetry in language resources: while there are numerous speech recognition models for English, the options for Indian regional languages "narrow considerably." Specifically regarding **offline** speech recognition in Hindi, there are only a handful of available models, and these models are often "larger or less accurate" than their cloud counterparts. This statement clarifies the engineering positioning of this paper: the authors are not seeking a large cloud model, but rather an efficient system that can run locally on low-power devices and is tailored to specific user queries.

The second pain point is **data scarcity**. The paper reviews the current state of Hindi speech data in the introduction: the Hindi portion of Mozilla Common Voice contains only 17 hours of transcribed speech (Reference [11]); the ULCA-ASR corpus includes 2398.76 hours of labeled audio plus 2432.92 hours of unlabeled data, but the sources are broadcast scenarios such as DD Vigyan Prasar and various Hindi news channels; the OpenSLR Hindi data released for the MUCS 2021 challenge is approximately 100 hours, all from narrative reading (References [13][14]); Diwan et al. (Reference [20]) provided 600 hours of transcribed speech across six Indian languages. From this, the authors conclude that most existing open-source Hindi datasets are sourced from YouTube, audiobooks, tutorials, etc., **lacking standardized annotations and the distribution of "digits plus specific spoken words" truly needed for keyword tasks**. This project requires well-annotated Hindi digits and specific words, which are not available on the market and must be built from scratch.

Thirdly, the task nature of Keyword Spotting (KWS) differs from general speech recognition: it does not require transcribing an entire sentence into text, but only requires determining "whether a preset keyword appears" in a continuous audio stream. This means the system must handle three scenarios: saying the keyword, saying something else, or saying nothing. Previous work mostly covered only the first scenario; this paper explicitly introduces a negative class to cover the latter two.

### Specific Shortcomings of Existing Methods

The paper attributes the shortcomings of existing work to four lines of reasoning:

**First, HMM-based isolated word recognition in Hindi is far from practical.** Pruthi et al. (Reference [1])'s Swaranjali system is speaker-dependent, targeting only two male speakers and Hindi digits zero to nine, using Linear Predictive Cepstral Coefficients (LPCC) as features and Hidden Markov Models (HMM) for recognition. Kumar et al. (Reference [2]) created a speaker-independent system with 94.63% accuracy, but the training set consisted of only 8 people (5 men, 3 women) and a vocabulary of 30 words, still relying on the traditional pipeline of MFCC plus HMM. Saksamudre et al. (Reference [4]) conducted a comparative study in 2015 using K-Nearest Neighbors (KNN) with MFCC, achieving 89% accuracy. Saini et al. (Reference [7]) built a system using the HTK toolkit with a vocabulary of 113 words and training data from 9 speakers, recognizing only isolated words. The common problems with these works are: small speaker scale, fixed vocabulary, reliance on generative modeling that requires explicit alignment, and lack of robustness design for noisy scenarios.

**Second, existing datasets do not match the keyword task.** The text domains of the aforementioned corpora, such as Common Voice (17 hours), MUCS (100 hours), and ULCA (2398.76 hours labeled), are read phrases, news broadcasts, and narratives, not the keyword distribution of "digits plus command words plus silence negative examples." Although Bansal et al. (Reference [15]) created a transcription dictionary mapping Hindi to CMU dictionary ARPABET phoneme representations, this solves the dictionary resource problem, not the audio data problem.

**Third, general large models perform poorly on-device and for specific word detection.** The paper specifically names OpenAI's Whisper: it supports Hindi, but the model size is huge, resulting in long processing times on low-power devices using only CPU inference; more critically, the paper's empirical tests show its accuracy for "specific word detection" is very low (the original text uses the phrase "very low," without providing specific numbers), leading to the conclusion that it is a suboptimal choice for this application.

**Fourth, existing CNN-KWS works do not cover Hindi.** Nayyer et al. (Reference [18])'s CNN keyword detection system also uses MFCC for preprocessing but employs a heavier ResNet-like model and is validated on the English Google Speech Commands dataset; López-Espejo et al. (Reference [19])'s review covers various KWS applications, datasets, and architectures, but similarly lacks a focus on Hindi.

### Key Challenges to be Solved by This Paper

In summary, this paper aims to solve three things simultaneously: (1) **Data challenge** — build a standardized, annotated Hindi keyword dataset from scratch covering 21 categories, including 16 digits, 4 spoken words, and 1 negative class, totaling over 40,000 samples (Abstract, Section II); (2) **Model challenge** — design a computationally efficient CNN classifier with MFCC as input that can run on an on-device CPU, replacing existing offline solutions that are "larger or less accurate"; (3) **Robustness challenge** — ensure the model maintains high recognition rates in real noisy environments (vehicle noise, human voices, fans, etc.). The answer chain provided by the paper is: build a custom dataset plus seven types of noise augmentation and time-shift augmentation, pulling the accuracy on real test samples from 60% to 91.79% (Section IV, Part A).

## Methodology

### Overall Architecture Design and Design Motivation

The entire system follows a classic pipeline of "audio fixed-lengthing — MFCC feature engineering — CNN classification":

**Input Fixed-Lengthing.** Each audio clip is truncated to exactly 1.9 seconds. Why 1.9 seconds instead of the mainstream 1 second commonly used in KWS: the paper does not explicitly justify this, but looking at the vocabulary composition, the pronunciation duration of digits zero to fifteen and the four two-syllable words can all be covered within 1.9 seconds, and fixed-lengthing ensures a fixed CNN input dimension. At a 44kHz sampling rate, 1.9 seconds corresponds to $1.9 \times 44000 = 83600$ samples (Section III, Part B).

**Feature Extraction.** Using a Hann window for framing (window length 1024 samples, hop length 512 samples), performing Discrete Fourier Transform (DFT) frame by frame, mapping the power spectrum to the Mel frequency scale and taking the logarithm to obtain the log-Mel spectrum, and then compressing it via Discrete Cosine Transform (DCT) to extract 13 MFCC coefficients. Each audio clip yields approximately 162 frames ($44000 \times 1.9 / 512 - 1 \approx 162$, the original calculation in Section III, Part B), so the network input is a 2D "pseudo-image" of 162 frames $\times$ 13 coefficients. **Why MFCC was chosen over raw waveforms or filter bank features**: The paper states that MFCC represents the short-time power spectrum of sound, which is a compact representation after feature engineering on the original recording; from an engineering perspective, feeding the time-frequency representation as a 2D image to the CNN allows convolutional kernels to capture local patterns in both the frequency dimension (cepstral dimension) and the time dimension simultaneously, which is the structural advantage of CNNs relative to fully connected networks.

**Classification Network.** The network morphology described in Section III, Part A and Figures 1 and 2 is: a series of **convolutional layers with incrementally increasing filter sizes**, accompanied by ReLU activation, batch normalization, and max pooling, with dropout used for regularization; after the convolution stack, **two dense layers** integrate the learned features into more abstract representations; finally, a softmax layer distributes probabilities across 21 categories. The paper provides a "why" for each component: convolutional layers gradually increase the receptive field and abstract spectral features at various levels (incrementally increase the receptive field and abstract the spectral features at various levels); the two dense layers are responsible for integrating features and serving the classification task; softmax ensures multi-class discrimination for the 21 classes. Dropout and batch normalization jointly suppress overfitting. **It should be noted**: The paper text does not provide specific channel numbers, kernel sizes, pooling windows, dense layer widths, total parameters, or computational cost for each convolutional layer — these details exist only in the schematic diagrams of Figures 1 and 2 and cannot be extracted from the text, belonging to the "not reported by the paper" category, which also makes the claim of "computationally efficient" lack quantitative support (see Limitations section).

**Deployment Target.** The abstract explicitly states on-device KWS for user-specific queries, meaning the model runs locally on the device and only detects a small number of keywords of interest to the user, which is the fundamental difference from cloud ASR.

### Mathematical Principles of Core Algorithms

The paper provides three formulas for the preprocessing pipeline:

**Hann Window (Equation 1)**:
$$W[n] = 0.5 - 0.5\cos\left(\frac{2\pi n}{N-1}\right),\quad 0 \le n \le N-1$$
The paper's original statement on the motivation for windowing is "to control leakage and increase the dynamic range." The engineering implication is: directly performing a Fourier transform on a truncated signal is equivalent to multiplying by a rectangular window, which introduces spectral leakage, smearing energy to adjacent frequencies and suppressing spectral contrast; the Hann window smoothly zeros out at both ends, eliminating edge discontinuities in each frame, allowing the energy distribution obtained from subsequent Mel filtering to more closely approximate the true spectral envelope, which is fundamental for KWS tasks that distinguish word tokens based on formants and noise transients.

**Discrete Fourier Transform (Equation 2)**:
$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-i2\pi nk/N}$$
Transforms the time-domain signal to the frequency domain frame by frame to obtain the power spectrum, which is then weighted by the Mel filter bank.

**Mel Frequency Mapping (Equation 3)**:
$$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
The paper explains the Mel scale as "a pitch scale that is perceptually equidistant." Why use a logarithmic perceptual scale in KWS: The human ear has much higher resolution for low-frequency differences than for high frequencies. The Mel filter bank is dense at low frequencies and sparse at high frequencies. After taking the logarithm and performing DCT, the resulting cepstral coefficients compress the dynamic range and separate the vocal tract excitation (related to fundamental frequency) from the vocal tract filtering (formant envelope) — the latter is primarily relied upon for keyword discrimination.

**Correspondence of Frame Parameters.** The window length of 1024 samples and hop length of 512 samples given in Section III, Part B correspond roughly to the "window size 23 ms, frame period 10 ms" in Table I (TABLE I: Audio Parameters): $1024 / 44000 \approx 23.3$ ms matches the 23 ms window; $512 / 44000 \approx 11.6$ ms slightly differs from the 10 ms frame period in Table I (possibly due to rounding differences). 50% frame overlap ensures that transient features are not chopped up by frame boundaries.

**Training Objective.** The paper does not provide a loss formula, only stating that cross-entropy is used as the loss function for supervised learning (Section IV, Part B). The standard multi-class cross-entropy is $L = -\sum_{c=1}^{21} y_c \log \hat{y}_c$, combined with softmax output. The optimizer selected is Adam, with the paper's justification being that it is "known for efficiently managing sparse gradients and adaptive learning rates"; the learning rate is dynamically adjusted during training (specific initial value and scheduling strategy are not reported by the paper); the mini-batch size is 64, which the paper explicitly states is to "strike a balance between computational efficiency and generalization ability"; the training/validation split is 80:20, with the reason being to "ensure sufficient training samples while minimizing the risk of overfitting."

### Key Technical Innovation 1: Custom-Built Hindi Keyword Dataset with Negative Class

This is the most core entity contribution of this paper. The key design of the dataset and its "why":

**Category Composition (Section II)**: 21 classes = 16 classes for digits 0–15 + 4 words (ha, nhi, sambandh, vibhag, i.e., affirmative response, negation, connection, department — high-frequency words in spoken Hindi) + 1 negative class. The paper does not explicitly state the motivation for choosing digits as the main vocabulary, but from an application perspective, digits are the highest-frequency inputs for device interactions such as phone numbers, channels, and times. The four words cover the most basic affirmative/negative and query operations. **The negative class is explicitly designed**: The paper states that the negative class covers "various indoor and outdoor noises," with the purpose of "recognizing when nothing is being said." In KWS engineering, this corresponds to suppressing false triggers — without a negative class, any non-keyword speech would be forced into one of the 21 classes, causing a large number of false positives. The results in Section V echo this: the model correctly falls into the negative class for erroneous or no input.

**Recording Diversity (Section II)**: Recorders cover different demographic backgrounds, genders, native languages, and accents. Why emphasize diversity: India is a highly multi-accent region, and the native language backgrounds of Hindi speakers vary greatly. Recording from a single group would cause the model to learn specific vocal tract characteristics of that group rather than the acoustic invariants of the words.

**Recording Specifications (Table I)**: Mono, .wav format, bitrate 704 kbps, sampling rate 44kHz. 704 kbps matches $44100 \text{ Hz} \times 16 \text{ bit} \approx 705.6 \text{ kbps}$, i.e., 16-bit PCM mono. Why choose 44kHz instead of the mainstream 16kHz for speech: The paper does not provide justification. An objective consequence is that it retains the frequency band from above 16kHz to 22kHz (part of the energy of fricatives lies here), but the cost is that the number of samples for the same duration is approximately 2.75 times that of a 16kHz configuration, synchronously increasing the computational cost of feature extraction and convolution (see Limitations).

**Scale (Abstract, Section II)**: Totaling over 40,000 samples. Combined with the statement in Section IV, Part A that "approximately 35,000 samples are generated by augmentation," the original recordings amount to only about 5,000 clips, with the remaining nearly 90% coming from augmentation — this indicates that the manual scale of building the custom dataset is actually quite small, with augmentation bearing the main burden of expansion.

### Key Technical Innovation 2: Data Augmentation Pipeline of Noise Superposition and Time-Shift

The augmentation strategy described in Section IV, Part A includes three actions:

**Seven Types of Noise Superposition**: Introduce seven types of different noises (vehicle sounds, human conversation, flowing water, and other daily environmental noises; Figures 3 and 4 show original waveforms and spectra, Figures 5 and 6 show the corresponding comparisons after augmentation), carefully superimposing the noise samples onto the original audio, **followed by volume normalization to maintain consistent audio levels**. Why normalization is mandatory after superposition: The energy differences between different noise sources can reach tens of decibels. If not leveled, the network would learn the spurious rule that "loudness is related to class."

**Time-Shift Augmentation**: Change the starting position of the input word in the audio and cover "partially recorded" audio (where the word is only recorded halfway). Why this is needed: In real usage, users do not speak with precise timing; the keyword may appear at any phase within the detection window, or even be truncated. Time-shift makes the model insensitive to the position of the word, alleviating overfitting where "the keyword must appear at the beginning of the window."

**Quantitative Effect**: The paper reports that this set of augmentation technologies improved the model's accuracy on real test samples **from 60% to 91.79%** (Section IV, Part A) — an increase of 31.79 percentage points. This is the most convincing number in the entire paper, and conversely, it indicates that the model without augmentation is almost unusable on real noisy audio.

### Technical Differences from Existing Methods

Compared to various lines in the literature: **For HMM/KNN-based Hindi works ([1][2][4][7])**, this paper replaces the traditional pipeline of "feature extraction + generative model + alignment" with a discriminative end-to-end CNN, where category discrimination is directly completed by convolutional features, and for the first time, the negative class is incorporated into the category system; **For Nayyer et al. ([18])'s CNN-KWS**, the latter uses a heavier ResNet-like backbone and the English Google Speech Commands dataset, while this paper takes the route of Hindi, custom data, and a lighter network; **For general large models like Whisper**, this paper exchanges a 21-class small classifier on 1.9-second fixed-length segments for CPU on-device feasibility, at the cost of a closed vocabulary and no open-vocabulary capability; **For dataset-oriented works ([11][13][20])**, this paper's data is customized for the keyword task distribution (digits, command words, negative examples), rather than general transcription corpora. It should be noted: The paper does not perform unified controlled comparative experiments under the same test conditions; the above differences are at the design route level, not performance differences on the same test set (see Experimental Results section).

## Experimental Results

### Dataset Used and Its Scale

**Training and Validation**: The full custom dataset exceeds 40,000 samples (Abstract, Section II), of which approximately 35,000 are generated by augmentation (Section IV, Part A), split 80:20 for training and validation (Section IV, Part B). The paper does not report the per-class sample distribution for the 21 categories, nor does it explain whether the class ratio of augmented samples is balanced.

**Testing**: Section V states that the test data consists of "10 samples per person, covering 21 word classes, from another group of individuals not involved in training," ensuring testing under real-life scenarios; these samples are completely not involved in training. However, **the total number of test persons and the total number of test set samples are not reported by the paper**, so the sample base corresponding to the 91.79% number cannot be confirmed from the text.

### Definition and Rationale for Evaluation Metrics

The only metric used in the paper is **classification accuracy**: it reports training and validation accuracy curves (Figure 8), training and validation loss curves (Figure 9), 95% accuracy on the validation set, and 91.79% accuracy on the real test set (Section IV, Part B), as well as a 21-class confusion matrix (Figure 7). The paper does not explain why only accuracy was selected, nor does it report metrics commonly used in the KWS domain such as false alarm rate (false triggers per hour), true positive rate — false positive rate operating points, ROC/AUC, or DET curves. For keyword detection, the false trigger rate is an user experience metric equally important to accuracy (the cost of one false wake-up is much higher than that of one missed recognition). The absence of this metric means the conclusion that "negative class design is effective" can only remain at the qualitative level of the confusion matrix.

### Detailed Comparison with Baseline Methods and SOTA

The paper **does not set up a formal comparison table**; all horizontal numbers that can be cited come from the literature review in the introduction, and the test conditions are mutually different: Kumar et al. (Reference [2])'s HMM system achieved 94.63% under conditions of a 30-word vocabulary and 8 training speakers; Saksamudre et al. (Reference [4])'s KNN + MFCC system was 89%; this paper is 21 classes (including negative class), 40k samples, 91.79%. On the surface, this paper is lower than the 94.63% of Reference [2], but the two have different vocabularies, different numbers of categories (the latter has no negative class), different noise conditions, and different datasets, making them not directly comparable. For Whisper, the paper only gives a qualitative judgment of "very low accuracy for specific word detection" (Introduction), without attaching specific numbers; for Nayyer et al. ([18])'s ResNet-like CNN-KWS, there is no comparison of parameters or accuracy on the same data. Therefore, strictly speaking, **this paper's 91.79% is an independent result on a custom test set, not a controlled comparison with any baseline**.

### Findings from Ablation Studies

The paper **does not have a systematic ablation section**. The evidence closest to ablation is only one place: the comparison before and after turning on data augmentation — without augmentation, the real test accuracy is 60%; with seven types of noise superposition and time-shift augmentation, it is 91.79% (Section IV, Part A), indicating that augmentation contributed 31.79 percentage points and is the decisive component for the success or failure of the entire system. Apart from this, the paper does not involve the following conventional ablation dimensions: number of MFCC coefficients (whether 13 is optimal), window length and hop length combinations, convolutional depth and width, removal of batch normalization or dropout, proportion of negative class samples, number of noise categories, comparison between 44kHz and low sampling rates. The confusion matrix in Figure 7 is also not interpreted class-by-class in the main text (which digits or words are confused with each other, and what is the recall rate of the negative class, are not given in the text).

## Main Contributions

1. **A custom-built Hindi keyword dataset**: Over 40,000 samples, 21 categories, covering digits 0–15, 4 high-frequency spoken words, and 1 negative class, with diverse demographic, gender, native language, and accent backgrounds for recorders, unified specifications of 44kHz mono 1.9-second wav (Abstract, Section II, Table I). This directly fills the gap of "lack of standardized annotated Hindi digit plus specific word open-source data," which is the most solid contribution of the paper.

2. **An MFCC + CNN keyword detection pipeline oriented towards on-device deployment**: Fixed-length truncation, Hann windowing, Mel log spectrum, DCT extracting 13 coefficients, convolutional stack plus two dense layers plus 21-class softmax, ultimately achieving 91.79% accuracy on independent population real recordings and 95% accuracy on the validation set (Abstract, Section IV, Part B, Section V).

3. **Proving the decisive role of data augmentation through controlled before-and-after comparison**: The combination of augmentation with seven types of daily noise superposition, volume normalization, and time-shift (including partially recorded words) improved the real test accuracy from 60% to 91.79% (Section IV, Part A), an increase of 31.79 percentage points, which is crucial for the usability of small-scale custom datasets.

4. **Validation of the effectiveness of negative class design**: The model correctly falls into the negative class for erroneous or no input (Section V), providing a category-level mechanism for suppressing false triggers in on-device KWS.

5. **Evidence of usability under real noise conditions**: The conclusion section reports that the model maintains 91.79% accuracy under various real-world conditions ranging from "background noise with human activity" to "fan on maximum."

## Limitations and Future Work

### Technical Limitations of the Method

**"Computationally efficient" lacks quantitative evidence.** The paper repeatedly emphasizes on-device and computational efficiency (Abstract, Section III, Part A), but throughout the text, it does not report the number of parameters, floating-point operations, model file size, single inference latency, or memory usage, nor does it provide size comparison numbers with Whisper or ResNet-like models. "Lightweight" is currently only a qualitative architecture (convolution plus two dense layers), making it impossible to judge whether it can fit into the target hardware budget based on this.

**Implicit overhead of 44kHz sampling rate.** Mainstream KWS and speech front-ends use 16kHz. 44kHz significantly increases the number of samples (83,600) and frames (approximately 162 frames) for every 1.9 seconds of audio compared to 16kHz solutions, synchronously amplifying the computational cost of feature extraction and convolution, while the marginal benefit of the frequency band above 16kHz for Hindi keyword discrimination is not argued by the paper, nor is a down-sampling comparative experiment conducted.

**Adaptation issues with the fixed 1.9-second window.** All inputs are truncated to 1.9 seconds, meaning longer utterances will be cut off and shorter utterances need to be padded. The paper does not explain how sliding windows are handled for continuous audio streams during deployment, and streaming detection mechanisms (window stepping, decision smoothing) are completely undiscussed — and this is exactly the key link for KWS to transform from a "classifier" into a "detection system."

**Closed 21-class vocabulary.** Adding new keywords requires retraining the model; the "user-specific customization" claimed in the abstract is only reflected in the method as "users can select a subset of keywords of interest," without a custom wake-word mechanism that avoids retraining.

### Shortcomings in Experimental Design

**Test set scale not disclosed.** Section V only says "10 samples per person, 21 word classes, another group of people," without giving the number of people and total test samples, making it impossible to estimate the confidence interval of 91.79%, nor are there repeated experiments with multiple random seeds.

**No controlled baseline comparison.** The numbers 94.63% (Reference [2]) and 89% (Reference [4]) belong to different datasets, different vocabularies, and different conditions. The paper does not perform horizontal evaluations on any public dataset or under unified test conditions, nor is there a same-data comparison with contemporary small models (such as the classic KWS backbone DS-CNN).

**Lack of analysis of the gap between validation and test sets.** There is a drop of about 3 percentage points from 95% validation to 91.79% real test, which the paper does not discuss the source of; especially critically, **it is not explained whether augmentation was performed before or after the 80:20 split** — if multiple augmented copies of the same original recordings are assigned to the training set and validation set, the validation accuracy would be overestimated, which is the most noteworthy point to question in this paper's experimental narrative.

**Missing core KWS metrics.** No false alarm rate, no false wake-up statistics, no per-class recall and precision. The confusion matrix (Figure 7) is not numerically interpreted, and the actual interception capability of the negative class is not quantified.

**Ablation gaps beyond augmentation.** All hyperparameters such as 13 MFCC coefficients, window length 1024, hop length 512, number and width of convolutional layers, and dropout ratio are given only single values, with no sensitivity analysis.

### Possible Directions for Future Improvement

The paper itself gives two directions in the conclusion: first, expand the dataset to include more digits and more words; second, utilize next-generation neural processing devices to achieve more efficient real-time inference. Beyond this, based on this close reading, directions worth pursuing include: reducing the sampling rate to 16kHz and retesting accuracy to compress on-device computational cost; supplementing reports on parameters, FLOPS, latency, and false wake-up rate to make "on-device efficiency" verifiable; introducing streaming sliding windows and decision smoothing to upgrade the classifier into a true continuous detection system; performing cross-dataset validation on public Hindi corpora (such as the 100-hour MUCS data) to test generalization ability; performing targeted hard-example augmentation for categories that are confused with each other in the confusion matrix (near-homophonic digits are a high-incidence area); exploring few-shot custom wake-word mechanisms (such as few-shot fine-tuning on pre-trained features) to fulfill the promise of "user-specific customization" in the abstract; supplementing augmentation for far-field reverberation and multi-speaker overlap scenarios to cover real deployment forms such as smart speakers and set-top boxes.

---

*Close Reading Notes: This paper belongs to the category of applied short articles, with limited methodological depth (no formulaic derivation, no ablation matrix, no controlled comparison). Its value lies mainly in the custom-built dataset and the engineering evidence that "augmentation pulled real test accuracy from 60% to 91.79%"; all numbers in the notes are from the original paper and cited (Abstract, Sections II–V, Table I, Figures 1–9), and items not reported by the paper have been noted one by one.*
