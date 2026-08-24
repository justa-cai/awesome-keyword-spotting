# Wav2KWS: Transfer Learning from Speech Representations for Keyword Spotting

- **Authors/Affiliations**: Jeong-eun Park, Ian McLoughlin, Prashanth Sudhakar - Karlsruhe Institute of Technology (KIT)
- **Date**: 2021.05
- **Link**: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9427206
- **Keywords**: Transfer Learning, wav2vec, Speech Representations, Keyword Spotting, Pre-training, Self-supervised Learning, Fine-tuning Strategies

## Problem Statement

Training Keyword Spotting (KWS) models from scratch requires large amounts of labeled data and may not generalize well to diverse acoustic conditions (different speakers, recording environments, and noise types). In particular, for low-resource languages and custom keyword scenarios, the cost of obtaining labeled data is extremely high.

In recent years, Self-supervised Speech Representation Learning has achieved breakthrough progress. Models such as wav2vec 2.0 and HuBERT learn general speech representations by training on large amounts of unlabeled speech data, and their learned features have demonstrated strong transferability across multiple downstream speech tasks (such as ASR, speaker recognition, emotion recognition, etc.).

However, effectively transferring these large pre-trained models to the specific downstream task of KWS still faces several challenges:
1. **Model Size Mismatch**: Pre-trained models typically have a huge number of parameters (tens of millions to hundreds of millions), while KWS needs to run on resource-constrained edge devices, with model sizes usually within hundreds of thousands of parameters.
2. **Uncertainty in Fine-tuning Strategies**: Which layers of the pre-trained model should be frozen and which should be fine-tuned? How do different fine-tuning strategies affect KWS performance?
3. **Feature Selection**: Different layers of pre-trained models capture speech information at different levels (lower layers capture acoustic features, higher layers capture semantic information). Which layer features are most useful for KWS?
4. **Choice of Pre-trained Model**: How much difference do different pre-trained models (wav2vec 2.0, HuBERT, CPC, etc.) make in KWS transfer performance?

This paper systematically investigates these issues, providing a comprehensive analysis of transfer learning from self-supervised speech representations to KWS.

## Methodology

### Overall Framework
The transfer learning framework of Wav2KWS is divided into two stages: pre-trained feature extraction and downstream KWS fine-tuning.

### Pre-trained Speech Models
The paper systematically evaluates various pre-trained speech representation models as feature extractors for KWS:

- **wav2vec 2.0 Base (95M parameters)**: Pre-trained on LibriSpeech 960 hours. wav2vec 2.0 uses a convolutional encoder to extract acoustic features, then performs contrastive learning on quantized representations (predicting the correct quantized acoustic unit from masked positions). The model contains 12 Transformer encoder layers, outputting 768-dimensional features.

- **wav2vec 2.0 Large (317M parameters)**: A larger model pre-trained on LibriSpeech 960 hours, containing 24 Transformer encoder layers, outputting 1024-dimensional features.

- **HuBERT Base/Large**: A self-supervised model using masked prediction rather than contrastive learning. HuBERT uses discrete acoustic units obtained via clustering as prediction targets, avoiding the instability of the quantization module in wav2vec 2.0.

- **Comparison Baselines**: Traditional MFCC features (40 dimensions), FBank features (80 dimensions), and randomly initialized models (no pre-training).

### Transfer Learning Strategies
The paper systematically compares various fine-tuning strategies:

1. **Feature Extraction (FE)**:
   - Completely freeze all parameters of the pre-trained model.
   - Train only a lightweight classification head (1-2 layer fully connected network) on top of the frozen features.
   - Advantages: Fast training, does not require large computational resources, avoids catastrophic forgetting.
   - Disadvantages: Cannot adapt to feature requirements specific to the KWS task.

2. **Full Fine-tuning (FT)**:
   - Unfreeze all parameters of the pre-trained model.
   - Jointly fine-tune all parameters on KWS data using a small learning rate.
   - Advantages: Can adapt to the KWS task to the maximum extent.
   - Disadvantages: High computational overhead, prone to overfitting (especially when KWS training data is scarce), and may lead to catastrophic forgetting.

3. **Top-layer Fine-tuning**:
   - Freeze the lower layers (1-6 layers) and fine-tune only the top layers (7-12 layers) and the classification head.
   - Balances the pros and cons of FE and FT: lower layers retain general acoustic knowledge, while top layers adapt to the KWS task.
   - This is the default strategy recommended by the paper.

4. **Gradual Unfreezing**:
   - Start from the top layers and gradually unfreeze more layers.
   - Decrease the learning rate each time a layer is unfrozen.
   - The most granular fine-tuning strategy, but with the longest training time.

5. **Layer Concatenation Strategy**:
   - Extract features from different layers of the pre-trained model and concatenate them.
   - For example, concatenate the outputs of layer 4 and layer 8, simultaneously utilizing lower-layer acoustic features and higher-layer semantic features.
   - Then train the classification head on the concatenated features.

### Analysis of Features from Different Layers
The paper conducts an in-depth analysis of features from different layers of pre-trained models:

- **Lower-layer features (1-4 layers)**: Capture basic acoustic features (such as spectral envelope, short-term energy, fundamental frequency, etc.), which are close to signal processing-level features. These features have strong generality across different tasks but weaker discriminative power.
- **Middle-layer features (5-8 layers)**: Capture phoneme-level acoustic patterns (such as formant positions, voiced/voiceless characteristics, etc.). These features are most useful for speech recognition and keyword spotting.
- **Higher-layer features (9-12 layers)**: Capture semantic-level information (such as phoneme sequences, word-level representations, etc.). These are most useful for ASR tasks, and their contribution to KWS tasks depends on the complexity of the keywords.

### KWS Classification Head Design
On top of pre-trained features, the paper explores different classification head designs:

- **Simple Classification Head**: Global Average Pooling -> Fully Connected -> Softmax. Has very few parameters (approximately D*K, where D is the feature dimension and K is the number of classes), but limited expressive power.
- **Multi-layer Classification Head**: Global Average Pooling -> FC(256) -> ReLU -> Dropout -> FC(K) -> Softmax. Increases non-linear transformation capability.
- **Classification Head with Attention Pooling**: Uses an attention mechanism instead of simple global average pooling to adaptively weight features at different time steps.

## Main Contributions

1. **Systematically studied transfer learning from self-supervised speech representations to KWS**: Provided the most comprehensive experimental analysis of wav2vec 2.0/HuBERT to KWS transfer learning to date, covering different pre-trained models, different fine-tuning strategies, and the feature contributions of different layers. This systematic study provides valuable practical guidance for the KWS community.

2. **Proved that pre-trained speech features significantly improve KWS accuracy**: Under all experimental settings, pre-trained features (especially wav2vec 2.0 and HuBERT) consistently outperformed traditional MFCC/FBank features and random initialization, with accuracy improvements of about 2-5%. This result validates the value of self-supervised pre-training for KWS tasks.

3. **Provided guidance on which layers should be frozen and fine-tuned to achieve optimal KWS performance**: Experiments found that fine-tuning the top 3-4 layers (freezing the bottom 8-9 layers) is the best strategy, achieving an optimal balance between full feature extraction and full fine-tuning. This finding provides clear operational guidelines for practical KWS transfer learning.

4. **Proved that wav2vec-based features outperform traditional MFCC features for KWS**: Not only in classification accuracy, but also in cross-speaker generalization and noise robustness, pre-trained features demonstrated significant advantages. This provides a new direction for feature selection in KWS systems.

5. **Analyzed transfer learning effects in low-data scenarios**: A particularly important finding is that in scenarios with limited training data (10-20% of data), the relative advantage of pre-trained features is more pronounced (accuracy improvements can reach 5-10%), making transfer learning a key technology for low-resource KWS.

## Experimental Results

### Datasets and Settings
- **Google Speech Commands (GSC) v2**: 12-class and 35-class classification tasks.
- **Low-data Evaluation**: Used 10%, 20%, 50%, and 100% of the training data.
- **Evaluation Metrics**: Classification accuracy, cross-speaker accuracy.
- **Pre-trained Models**: wav2vec 2.0 Base/Large, HuBERT Base.

### Main Performance
- **Full Data (100%) Setting**:
  - wav2vec 2.0 Base + Top-layer Fine-tuning: Accuracy of approximately 96-97%, outperforming the DS-CNN baseline trained on MFCC (approximately 95%).
  - HuBERT Base + Top-layer Fine-tuning: Accuracy of approximately 97%, comparable to wav2vec 2.0.
  - wav2vec 2.0 Large: Accuracy of approximately 97.5%, but the parameter count is too large for direct deployment.

- **Low Data Setting (10% Data)**:
  - The advantage of pre-trained features is most significant: wav2vec 2.0 transfer accuracy is approximately 90-92%, while the MFCC baseline is only approximately 82-85%.
  - Pre-trained features narrowed the performance gap in low-data scenarios from 15% to 5%.

### Fine-tuning Strategy Comparison
- **Top-layer Fine-tuning (Unfreezing the last 3-4 layers)**: The best strategy across all data volume settings.
- **Feature Extraction (Completely Frozen)**: Accuracy is about 1-2% lower than top-layer fine-tuning, but training is 10 times faster.
- **Full Model Fine-tuning**: Comparable to top-layer fine-tuning with 100% data, but prone to overfitting in low-data scenarios (<20%), with performance even lower than the feature extraction mode.
- **Gradual Unfreezing**: Comparable to top-layer fine-tuning, but with longer training time.

### Feature Contribution of Different Layers
- Middle-layer features (layers 6-8) contribute the most to KWS.
- Concatenating features from layer 4 and layer 8 improves performance by about 0.5-1% compared to using any single layer's features.
- Lower-layer features (1-3 layers) contain less KWS discriminative information, being close to MFCC-level acoustic features.
- Higher-layer features (10-12 layers) contain more semantic information, which helps in detecting complex keywords.

### Impact of Classification Head Design
- The simple classification head (Global Average Pooling + FC) already achieves good performance on pre-trained features.
- The attention pooling classification head improves performance by about 0.5% over global average pooling by adaptively focusing on key time segments of keywords.
- Deeper classification heads (3-layer FC) did not bring significant improvements, indicating that the expressive power of pre-trained features is already sufficient.

### Cross-speaker Generalization
- The drop in accuracy for unseen speakers is smaller for pre-trained features than for the MFCC baseline (approximately 2% vs. approximately 5%), indicating that pre-trained features have learned more speaker-independent speech representations.

## Limitations and Future Work

### Technical Limitations
- **Pre-trained Model Size is Too Large**: wav2vec 2.0 Base (95M parameters) and Large (317M parameters) far exceed typical KWS models (<1M parameters), making direct deployment on resource-constrained devices unrealistic. Although the paper provides analysis of transfer learning, it does not solve the model size problem during the deployment phase.
- **Computational Cost**: The computational cost of extracting wav2vec features is much higher than that of extracting MFCC features, potentially increasing inference latency by more than 10 times. For KWS systems requiring real-time response, this latency may be unacceptable.
- **Sensitivity of Fine-tuning**: Fine-tuning pre-trained models requires careful learning rate scheduling and regularization to avoid catastrophic forgetting (loss of pre-trained knowledge) and overfitting (overfitting on KWS data). In particular, the choice of learning rate—if too large, it leads to forgetting; if too small, adaptation is insufficient.
- **Domain Mismatch**: Pre-trained models are primarily trained on read speech (LibriSpeech), which has domain differences from conditions potentially encountered in KWS scenarios, such as noisy environments, far-field recordings, and different microphones.

### Experimental Design Shortcomings
- Did not explore model distillation techniques to transfer the knowledge of large pre-trained models to small-footprint KWS models. This is a key step to bring the advantages of pre-trained features into edge deployment.
- Lacks a detailed analysis of inference latency and computational costs—although accuracy improvements are significant, is the computational cost worth it?
- Did not fully evaluate in more challenging scenarios such as noise, far-field, and custom keywords.
- Did not compare with contemporary small pre-trained models (such as distilled versions of wav2vec 2.0, LiteASR, etc.).

### Future Improvement Directions
- Explore knowledge distillation techniques: Use the outputs (or intermediate layer features) of large pre-trained models as teacher signals to train a lightweight student KWS model. This method can significantly reduce the model size during deployment while maintaining the advantages of pre-trained features.
- Research lightweight pre-trained models optimized for KWS—considering the needs of downstream KWS tasks (such as processing short speech segments, keyword discriminative features) during the pre-training phase, rather than directly using general ASR pre-trained models.
- Explore the effectiveness of multi-lingual pre-trained models (such as XLSR) in multi-lingual KWS transfer, especially for low-resource languages.
- Combine model compression techniques such as quantization and pruning to reduce the inference overhead of pre-trained features.
- **Insights for the KWS Field**: The general speech representations from self-supervised pre-training have significant value for KWS tasks, especially in low-data scenarios. This finding has driven a paradigm shift in the KWS community from "training from scratch" to "pre-training + fine-tuning." Future KWS systems are likely to be built based on pre-trained models rather than designed from scratch.
