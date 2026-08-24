# On the Efficiency of Integrating Self-Supervised Learning and Meta-Learning for User-Defined Few-Shot Keyword Spotting

- **Authors/Affiliations**: Wei-Tsung Kao, Yuan-Kuei Wu (Department of Communication Engineering, National Taiwan University); Chia-Ping Chen, Zhi-Sheng Chen, Yu-Pao Tsai (intelliGo Technology Co., Ltd.); Hung-Yi Lee (National Taiwan University)
- **Date**: April 2022 (Revised October 2022)
- **Link**: https://arxiv.org/abs/2204.00352
- **Keywords**: keyword spotting, few-shot learning, self-supervised learning, meta-learning, HuBERT, Matching network, speech encoder

## Problem Statement

### Problem Background and Domain Pain Points
User-Defined Keyword Spotting (User-Defined KWS) allows end-users to register new wake-word commands using their own voice samples without retraining the entire model. This feature is in urgent demand across multiple scenarios: smart speakers (custom wake words to avoid false activations of neighbors' devices), mobile assistants (personalized commands like "open camera"), security systems (voice passwords), and assistive technologies (customized command sounds for individuals with language impairments). The core challenge lies in the fact that users can typically provide only a very small number of registration samples (1-5 utterances, as excessive registration steps severely degrade user experience). The model must accurately identify the new keyword based solely on these samples—this is the Few-Shot Learning (FSL) problem.

From a technical perspective, the fundamental difficulty of FSL lies in "out-of-distribution generalization": the registration samples (support set) are a small number of samples recorded by the user at a specific time and in a specific environment, while the test samples (query) may be spoken in different environments or under different physical/mental states. The model needs to accurately discriminate "whether this test sample belongs to the new keyword" within a vast acoustic variation space, given such limited reference information.

### Specific Deficiencies of Existing Methods
- **Limitations of Self-Supervised Learning (SSL) Methods**: SSL models such as HuBERT, Wav2Vec2, and WavLM learn general speech representations by pre-training on large-scale unlabeled speech data (e.g., 960 hours of LibriSpeech). These representations perform well across various downstream tasks, but SSL models are not designed for "few-shot adaptation"—they may learn "general acoustic features" (e.g., distinguishing different phonemes) rather than features most discriminative for distinguishing different keywords (e.g., the homophones "yes" and "yet" are almost identical at the phoneme level, differing only in subtle variations in the last ~50ms). The representations of SSL models in FSL scenarios may be too "general," lacking task-specific discriminability.
- **Meta-Learning Methods' Dependence on Initial Encoders**: Meta-learning algorithms such as MAML, Prototypical Network, and Matching Network train models via a "learning to learn" paradigm, enabling them to quickly adapt to new tasks (with 1-5 samples). However, the effectiveness of meta-learning largely depends on the quality of the initial encoder—meta-learning optimizes "how to use the features extracted by the encoder." If the features extracted by the encoder are of poor quality, meta-learning cannot compensate.
- **Lack of Systematic Combinatorial Research**: Are SSL and meta-learning complementary? If so, which combination of SSL model and meta-learning algorithm is optimal? These questions lack systematic experimental research. Previous works typically validate only a single combination (e.g., HuBERT + MAML or Wav2Vec2 + Prototypical Network), making it impossible to draw comprehensive conclusions—different studies may reach contradictory conclusions due to the use of different SSL-meta-learning combinations.

### Key Challenges Addressed by This Paper
To systematically answer the following questions: What are the respective roles of SSL models and meta-learning algorithms in few-shot KWS? Are they complementary? How should the optimal SSL-meta-learning combination be selected to maximize few-shot KWS performance?

## Methodology

### Overall Experimental Framework
This paper designs a systematic experimental matrix covering 5 SSL models x 7 meta-learning algorithms = 35 combinations (plus various training configurations), conducting a comprehensive comparison under a unified evaluation protocol. This is the largest-scale SSL-meta-learning combinatorial study in the field of few-shot KWS to date.

### SSL Models (Encoder Selection)

All SSL models use the same training data (LibriSpeech 960 hours, approximately 28,000 audiobook recordings) and the same base architecture (12-layer Transformer encoder, approximately 95M parameters, hidden dimension 768). Using the same training data ensures fairness in comparison—performance differences can only stem from differences in pre-training objective functions.

- **CPC (Contrastive Predictive Coding)**: Learns the temporal structure of speech through contrastive learning to predict representations of future frames. The autoregressive model of CPC predicts the representation of future frames from past frames, distinguishing real future frames from random frames via negative sampling. CPC tends to learn long-timescale structures of speech (e.g., prosody, intonation patterns).
- **TERA**: Performs self-supervised training through masked speech reconstruction. Randomly masks parts of the time-frequency regions of the input spectrogram and trains the model to reconstruct the masked content. TERA tends to learn local time-frequency patterns (e.g., the shape and position of formants).
- **Wav2Vec2**: Learns speech representations through quantized contrastive learning. Quantizes continuous speech representations into discrete tokens (via Gumbel-Softmax) and distinguishes correct quantized tokens from distractor tokens via contrastive loss. Wav2Vec2 tends to learn discrete representations at the phoneme level.
- **HuBERT**: Learns speech representations through masked cluster prediction. First, clusters MFCC features using k-means (generating discrete "pseudo-labels"), then trains the Transformer to predict the cluster labels at masked positions. HuBERT's key innovation is using "offline clustering" instead of Wav2Vec2's "online quantization," avoiding quantization noise. HuBERT is particularly adept at capturing phoneme-level information.
- **WavLM**: Builds upon HuBERT by adding speaker noise augmentation—actively injecting interfering speech from different speakers into the pre-training data, training the model to accurately predict labels at masked positions even under multi-speaker conditions. WavLM typically outperforms HuBERT on speaker-dependent tasks (e.g., Speaker Verification), but its advantage in KWS tasks is not obvious.

### Meta-Learning Algorithms

**Optimization-based Methods (4 types)**:
- **MAML (Model-Agnostic Meta-Learning)**: Learns a good initialization parameter $\theta_0$ such that good performance can be achieved with only 1-5 gradient steps during few-shot adaptation. Inner and outer loop optimization: the outer loop optimizes $\theta_0$ across multiple tasks, while the inner loop performs fast adaptation on each task using $k$ samples. The advantage of MAML is strong adaptation capability (adjusting parameters via gradient updates), while the disadvantage is high computational overhead (requiring second-order gradients or first-order approximations).
- **ANIL (Almost No Inner Loop)**: A variant of MAML that updates only the classification head (the last layer) in the inner loop, keeping the feature extractor unchanged. ANIL assumes that a good initialization already provides sufficient feature representations, requiring only the learning of a classification mapping.
- **BOIL (Body Only Inner Loop)**: Opposite to ANIL, updates only the feature extractor (body) in the inner loop, keeping the classification head unchanged. BOIL assumes the classification head is simple, and the key lies in adjusting feature representations for each task.
- **Reptile**: A simplified variant of MAML that does not use second-order gradients but instead finds a good initialization through "parameter averaging across tasks": $\theta_0 \leftarrow \theta_0 + \epsilon(\theta_{task} - \theta_0)$, where $\theta_{task}$ is the parameter after training on a single task.

**Metric-based Methods (3 types)**:
- **Prototypical Network**: Computes the "prototype" (mean embedding of all support samples within a class) for each class: $p_c = \frac{1}{|S_c|}\sum_{x_i \in S_c} f_\theta(x_i)$, and classifies query samples based on the Euclidean distance to the prototype: $p(y=c|x_q) \propto \exp(-\|f_\theta(x_q) - p_c\|^2)$. Assumes that each class forms a compact cluster in the embedding space.
- **Relational Network**: Learns a relational metric function $g_\phi(\cdot)$ to compare the similarity between query samples and support samples: $r_{ij} = g_\phi(f_\theta(x_q) \circ f_\theta(x_{s_i}))$, where $\circ$ denotes feature concatenation. Does not assume a specific distance metric but learns it.
- **Matching Network**: Uses an attention mechanism to perform weighted matching between query samples and the support set: $\hat{y}_q = \sum_{i=1}^{|S|} a(x_q, x_{s_i}) \cdot y_{s_i}$, where $a(\cdot,\cdot)$ is the attention weight (based on cosine similarity or bilinear attention). The key characteristic of Matching Network is that each support sample independently calculates correlation with the query sample, without requiring classes to form compact clusters.

### Training Pipeline
1. **SSL Pre-training** (completed offline): Pre-train the speech encoder on LibriSpeech 960 hours.
2. **Meta-Learning Training**:
   - Sample N-way K-shot classification tasks from Google Speech Commands V2.
   - N=12 (12 candidate classes, including new keywords and background classes).
   - K=1 or 5 (1-sample or 5-sample settings).
   - Initialize the encoder using SSL pre-trained weights (optionally frozen or fine-tuned).
3. **Few-Shot Evaluation**: Perform N-way K-shot testing on unseen keywords.

### Key Experimental Variables
- SSL Model Selection: 5 types
- Meta-Learning Algorithms: 7 types
- Encoder Strategy: Frozen SSL parameters vs. End-to-End Fine-Tuning
- Number of Samples: K=1 (extreme few-shot) vs. K=5

## Main Contributions

1. **First Systematic SSL-Meta-Learning Combinatorial Study**: Comprehensive experiments covering 35 combinations provide the most complete analysis of SSL-meta-learning interaction effects in few-shot KWS to date. The study not only concludes the "optimal combination" but, more importantly, reveals the rules of advantages and disadvantages of different methods under different conditions—this has direct guiding value for the design of future few-shot KWS systems.

2. **Identification of HuBERT + Matching Network as the Optimal Combination**: Among all tested combinations, the combination of the HuBERT encoder and the Matching Network meta-learning algorithm achieved the best performance in few-shot KWS. Deep analysis of the reasons for this conclusion:
   - The discrete speech unit representations learned by HuBERT through masked cluster prediction are particularly suitable for phoneme-level matching of keywords—the acoustic features of keywords can be decomposed into a series of discrete speech units, and HuBERT's representations are naturally suited for this decomposition.
   - The attention matching mechanism of Matching Network can effectively utilize a small number of support samples—it does not require all samples of the same keyword to form a compact cluster in the embedding space (as assumed by Prototypical Network) but allows each support sample to independently calculate correlation with the query sample. This is particularly important when K=1 (only one reference sample is available, making it impossible to estimate the class "prototype").

3. **Proof of Complementarity between SSL and Meta-Learning**: Experiments clearly demonstrate that combining SSL pre-training with meta-learning yields better results than using either method alone. SSL provides high-quality speech representation initialization (solving the "poor feature space" problem), while meta-learning provides the ability to quickly adapt to new tasks (solving the "how to adjust with few samples" problem)—the two address challenges at different levels of few-shot learning and are indispensable.

4. **Metric-Based Methods Outperform Optimization-Based Methods**: In few-shot KWS tasks, metric-based methods (especially Matching Network) generally outperform optimization-based methods (MAML, ANIL, etc.). This finding provides clear guidance for algorithm selection in future few-shot KWS systems—when designing few-shot KWS systems, metric-based meta-learning algorithms should be prioritized.

## Experimental Results

### Datasets Used and Their Scale
- **Training/Meta-Learning Data**: Google Speech Commands V2, 35 classes, approximately 105,000 1-second audio clips.
- **Evaluation Protocol**: 12-way K-shot classification (K=1 or 5). In each evaluation, 12 classes are randomly selected (including the target keyword and background classes), K support samples are provided for each target class, and evaluation is performed on the query set.
- **Evaluation Method**: Few-shot testing is conducted on unseen keywords, with average accuracy and standard deviation reported using 5-fold cross-validation.

### SSL Model Comparison

Average performance ranking across all meta-learning algorithms (K=1 setting):
1. HuBERT (Optimal): Average accuracy approximately 87-88%
2. WavLM: Average accuracy approximately 85-86%
3. Wav2Vec2: Average accuracy approximately 83-84%
4. TERA: Average accuracy approximately 80-81%
5. CPC: Average accuracy approximately 78-79%

The advantage of HuBERT is particularly significant in the K=1 setting (approximately 2-3% accuracy advantage), which narrows in K=5 (approximately 1-2%). Reason analysis:
- The "discrete speech unit" representations learned by HuBERT through cluster prediction are particularly suitable for keyword matching—each keyword can be viewed as a specific sequence of speech units, and HuBERT's representations encode this sequence information.
- Although WavLM adds speaker noise augmentation (which helps SV tasks), it offers limited benefit for KWS tasks—KWS requires distinguishing different phoneme sequences, not different speakers.

### Meta-Learning Algorithm Comparison

Average performance ranking across all SSL models (K=1 setting):
1. Matching Network (Optimal): Average accuracy approximately 87-88%
2. Prototypical Network: Average accuracy approximately 84-85%
3. Relational Network: Average accuracy approximately 83-84%
4. MAML / ANIL / BOIL (Performance Close): Average accuracy approximately 80-82%
5. Reptile: Average accuracy approximately 78-79%

Deep analysis of Matching Network's advantages:
- The attention mechanism $a(x_q, x_{s_i})$ of Matching Network directly calculates the similarity between the query sample and each support sample, without needing to estimate the class center (as in Prototypical Network). When K=1, the "prototype" of Prototypical Network is that single support sample—this is equivalent to Matching Network. However, when K>1, Prototypical Network uses the mean as the prototype, which may be skewed by outlier samples; Matching Network's weighted matching is more robust.
- Compared to optimization-based methods like MAML, Matching Network does not require gradient updates (forward inference suffices), resulting in faster and more stable adaptation.

### Encoder Strategy Comparison

| Strategy | K=1 | K=5 |
|:---|:---:|:---:|
| Frozen SSL Encoder | Better | Slightly Worse |
| End-to-End Fine-Tuning | Slightly Worse | Better |

**Explanation**:
- When K=1, training data is extremely scarce (only 1 sample per class), and fine-tuning the SSL encoder (approximately 95M parameters) is highly prone to overfitting—the model may memorize the specific features of that 1 sample rather than learning the general features of the keyword. Freezing the encoder preserves the generalization ability of SSL pre-training.
- When K=5, there is more training data (5 samples per class), and fine-tuning can utilize this data to optimize the encoder's discriminability for keywords. At this point, the benefit of "task-specific adaptation" brought by fine-tuning outweighs the risk of overfitting.

### Performance of the Optimal Combination

- HuBERT + Matching Network + Frozen Encoder (K=1): Achieved optimal performance on the 12-way 1-shot task (approximately 88-90% accuracy).
- HuBERT + Matching Network + Fine-Tuned Encoder (K=5): Achieved optimal performance on the 12-way 5-shot task (approximately 92-94% accuracy).
- Demonstrates good robustness to changes in the number of few-shot samples (K=1 to K=5)—the accuracy improvement from 1-shot to 5-shot is approximately 4-6%.

## Limitations and Future Work

### Technical Limitations of the Method
- **Computational Resource Requirements**: SSL models (e.g., HuBERT Base) contain approximately 95M parameters and require approximately 150M FLOPs for inference. Deploying the full HuBERT encoder on resource-constrained edge devices may not be feasible (even for forward inference alone, it takes several seconds on an ARM Cortex-M4).
- **Exclusion of MetaOptNet Due to Convergence Issues**: MetaOptNet (using convex optimization as the inner-loop solver) might theoretically outperform Matching Network but was excluded from the experiments due to training instability (numerical instability of the SVM solver in few-shot settings). This may have missed a competitive method.
- **Limitations of the Evaluation Protocol**: The 12-way classification assumes the system knows "the query belongs to one of these 12 classes," but in practical applications, the system needs to determine "whether the query is the target keyword" (binary classification), while there may be thousands of background classes. The evaluation protocol may have a gap with actual application scenarios.

### Deficiencies in Experimental Design
- **English Keywords Only**: All experiments used English keywords from Google Speech Commands V2. The performance of few-shot KWS in tonal languages (e.g., Chinese—different tones of the same syllable represent completely different words) or morphologically rich languages (e.g., Turkish—affix variations produce many new words) may differ.
- **Evaluation Limited to 1-Shot and 5-Shot**: Performance changes with more samples (e.g., K=10 or K=20) were not explored. In practice, users may be willing to say 3-5 times, but might also accept saying 10 times if the system requires it. Threshold analysis of "how many samples are enough" has direct value for product design.
- **Lack of Comparison with End-to-End Methods**: No comparison was made with pure end-to-end meta-learning methods that do not use SSL pre-training (e.g., Matching Network trained from random initialization), making it difficult to quantify the absolute contribution of SSL pre-training.

### Possible Directions for Future Improvement
- **Lightweight SSL Encoders**: Compress HuBERT into a small encoder suitable for edge deployment while retaining its advantages in few-shot KWS, through knowledge distillation (distilling HuBERT's knowledge into a small encoder with 5-10M parameters) or pruning (removing unimportant attention heads and layers).
- **Cross-Lingual Few-Shot KWS**: Verify the effectiveness of SSL-meta-learning combinations in multilingual environments. Specifically, explore "cross-lingual transfer"—whether an encoder pre-trained on English SSL can perform well on Chinese few-shot KWS?
- **Online Meta-Learning**: Deploy the meta-learning process on-device, allowing real-time few-shot adaptation when users register new keywords (rather than pre-training a fixed adaptation strategy).
- **Implications for the KWS Field**: This systematic study reveals that "representation learning (SSL) + fast adaptation (meta-learning)" is an effective paradigm for solving few-shot KWS—SSL addresses the "are features good?" problem, and meta-learning addresses the "how to adapt with few samples?" problem. This paradigm can be generalized to other speech tasks requiring rapid customization (e.g., few-shot speech activity detection, few-shot acoustic event detection).
