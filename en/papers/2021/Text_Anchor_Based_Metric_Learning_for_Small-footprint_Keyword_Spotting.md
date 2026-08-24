# Text Anchor Based Metric Learning for Small-footprint Keyword Spotting

- **Authors/Affiliations**: Jiqing Han, Shiwen Wang, Xuyang Li, Ting Liu - Peking University ADSPLAB
- **Date**: 2021.08
- **Link**: https://arxiv.org/abs/2108.05516
- **Keywords**: Metric Learning, Text Anchor, Keyword Spotting, Small Footprint, Embedding Space, Zero-Shot, Cross-Modal Alignment

## Problem Statement

Traditional Keyword Spotting (KWS) systems use classification-based approaches where the model outputs a fixed number of classes (corresponding to predefined keywords). This classification paradigm has a fundamental limitation: whenever a new keyword needs to be added or an existing wake word changed, a large amount of training data for that keyword must be collected and the model retrained. This process is time-consuming and labor-intensive, severely limiting the flexibility and user customization capabilities of KWS systems.

Metric Learning methods offer a new solution to this problem: by learning an embedding space, speech samples of the same keyword are made close to each other in the embedding space, while speech samples of different keywords are kept far apart. During inference, detecting a new keyword only requires calculating the distance between its embedding and the stored prototype embedding, without the need to retrain the model.

However, existing metric learning-based KWS methods typically use audio anchors—i.e., audio samples of the new keyword are required to compute the prototype embedding. This still poses limitations in practical deployment: users need to record multiple audio samples of new keywords, which provides a poor user experience for those unfamiliar with technical operations.

The core innovation proposed in this paper is: Can text descriptions (rather than audio samples) be used as anchors to achieve metric learning for keywords? The advantage of text anchors is that users only need to input the text of the keyword (e.g., "Hey Siri"), and the system can convert it into an anchor in the embedding space via a text encoder, without any audio recording. Furthermore, text anchors make true zero-shot keyword customization possible—the system can detect keywords for which it has never seen audio samples during training, requiring only their text descriptions.

## Methodology

### Overall Framework: TAML (Text Anchor based Metric Learning)
The TAML framework consists of three core components: an Audio Encoder, a Text Encoder, and a Cross-modal Alignment module.

### Audio Encoder
The audio encoder maps input speech segments to vectors in a shared embedding space:

- **Input Features**: MFCC or spectrogram features extracted from 1-second speech segments.
- **Backbone Network**: A multi-layer CNN (such as DS-CNN or ResNet variants) is used to extract high-level representations from acoustic features.
- **Embedding Projection**: The output of the CNN undergoes global average pooling and linear projection, mapping it to a D-dimensional embedding space (typically D=128 or 256).
- **L2 Normalization**: The embedding vectors are L2-normalized so that they lie on the unit hypersphere, facilitating subsequent distance calculations.

The embedding vector $f_{audio}(x)$ output by the audio encoder should capture key discriminative features of the speech (such as phoneme sequences, prosodic patterns, etc.), ensuring that embeddings of the same keyword from different speakers are close to each other.

### Text Encoder
The text encoder maps the text description of a keyword to the same shared embedding space:

- **Input**: Text strings of keywords (e.g., "hello", "stop", etc.).
- **Text Representation**: Text is converted into phoneme sequences or character sequences. For the KWS task, phoneme-level representation may be more appropriate, as different spellings of the same keyword (e.g., "OK" vs "okay") are consistent at the phoneme level.
- **Encoder Architecture**: A bidirectional LSTM or Transformer encoder is used to encode phoneme/character sequences.
- **Embedding Projection**: The output of the encoder is mapped to a D-dimensional embedding space (with the same dimensionality as the audio embedding space) via linear projection.
- **L2 Normalization**: Text embeddings are also L2-normalized.

The embedding vector $f_{text}(t)$ output by the text encoder should capture the phonetic structure and pronunciation features of the keyword, ensuring that various audio pronunciations corresponding to this text are close to this text anchor in the embedding space.

### Cross-modal Alignment
The core challenge of TAML is how to align the representations of the audio and text modalities within the same embedding space:

- **Contrastive Loss**: For matched audio-text pairs (keyword audio + corresponding text), minimize the distance between them in the embedding space; for unmatched pairs, maximize the distance.

  For matched pairs $(x_{audio}, t_{text})$: $L_{match} = d(f_{audio}(x), f_{text}(t))^2$
  For unmatched pairs $(x_{audio}, t_{text}')$: $L_{non\_match} = \max(0, margin - d(f_{audio}(x), f_{text}(t')))^2$

  Where $d(.,.)$ is the Euclidean distance or cosine distance, and $margin$ is the margin parameter (typically set to 0.5-1.0).

- **Triplet Loss**: For each anchor audio sample $x_a$, select the corresponding text $t_p$ (positive sample) and an unmatched text $t_n$ (negative sample), minimizing:

  $L_{triplet} = \max(0, d(f_{audio}(x_a), f_{text}(t_p)) - d(f_{audio}(x_a), f_{text}(t_n)) + margin)$

- **Symmetric Alignment**: Not only should audio embeddings be close to corresponding text embeddings, but text embeddings should also be close to corresponding audio embeddings. Symmetric contrastive loss ensures the quality of bidirectional alignment.

### Inference Process
During inference, TAML supports two flexible keyword customization methods:

1. **Zero-shot Detection**:
   - The user provides the text description of a new keyword (e.g., "Hey Monday")
   - The text encoder computes the embedding vector $f_{text}("Hey Monday")$ for this text
   - For input audio $x$, calculate the distance between $f_{audio}(x)$ and $f_{text}("Hey Monday")$
   - If the distance is less than a threshold, it is判定 as a keyword match
   - The entire process requires no audio samples of the new keyword

2. **Few-shot Enhanced**:
   - If the user can provide a small number (1-5) of audio samples of the new keyword
   - Calculate the average of these audio embeddings as an audio anchor, and fuse it with the text anchor via weighted averaging
   - Fused Anchor = $\alpha \times \text{text\_anchor} + (1-\alpha) \times \text{audio\_anchor}$
   - The fused anchor is typically more accurate than a pure text anchor

### Negative Class Handling
KWS systems must not only correctly detect target keywords but also effectively reject non-keyword speech. TAML's negative class handling strategy includes:

- **Generic Negative Class Embedding**: A generic "negative class center" is computed using a large number of non-keyword speech samples. Any audio closer to the negative class center than to the keyword anchor is classified as non-keyword.
- **Hard Negative Mining**: During training, special attention is paid to non-keywords that are acoustically similar to the target keyword (e.g., "yes" vs "yeah"). The weights of these hard negative examples are increased to enhance the model's discriminative ability.

## Main Contributions

1. **Introduction of Text-Based Anchors for Metric Learning in Keyword Spotting**: This is the first proposal in the KWS field to use text anchors instead of audio anchors for metric learning. Text anchors eliminate the need for audio samples of new keywords; users only need to provide text descriptions to achieve keyword customization. This innovation significantly lowers the barrier to KWS customization.

2. **Realization of Zero-Shot Keyword Customization Using Only Text Descriptions**: Through cross-modal alignment, TAML can achieve detection on keywords completely unseen during training. The text-to-speech cross-modal bridge brings unprecedented flexibility to KWS systems.

3. **Proposal of a Joint Audio-Text Embedding Framework for Keyword Spotting**: A shared audio-text embedding space is constructed, where keyword representations of both audio and text modalities can be directly compared. This cross-modal framework lays the foundation for multimodal KWS research.

4. **Demonstration that Text Embeddings Can Effectively Guide Keyword Spotting**: Experiments show that the performance of the pure text anchor method is close to that of the pure audio anchor method, validating the effectiveness of text-audio cross-modal alignment in KWS.

5. **Provision of a Flexible Keyword Customization Paradigm**: The TAML framework supports a continuous spectrum from pure text zero-shot to few-shot audio enhancement, allowing users to choose different levels of customization based on actual needs.

## Experimental Results

### Datasets and Setup
- **Google Speech Commands (GSC) v2**: 12-class and 35-class classification tasks
- **Zero-Shot Evaluation**: Uses a "leave-class-out" protocol—certain keyword classes are excluded during training, and inference is performed using only their text descriptions
- **Evaluation Metrics**: Classification accuracy, AUC, Equal Error Rate (EER)
- **Embedding Dimension**: D=128

### Main Performance
- **Zero-Shot Performance**: On keywords unseen during training, using only text anchors achieves approximately 75-85% detection accuracy (depending on the phonetic complexity of the keyword and its similarity to training keywords). Although this accuracy is lower than fully supervised methods (approx. 95%), it is competitive for a zero-shot setting.
- **Comparison with Audio Anchors**: The text anchor method maintains approximately 85-90% of the performance of audio anchor metric learning methods under the same settings, indicating high quality of cross-modal alignment.
- **Keyword Customization Flexibility**: Compared to classification-based methods (which require retraining to add new keywords), TAML can instantly add new keywords without retraining.

### Advantages of Phoneme-Level Text Representation
- Using phoneme sequences as input to the text encoder (rather than character sequences) significantly improves performance, especially for keywords with inconsistent spelling and pronunciation (e.g., "colonel" pronounced as /kernel/).
- The alignment quality between phoneme-level text embeddings and audio embeddings is higher than that at the character level.

### Ablation Studies
- **Embedding Dimension**: D=128 achieves the best balance between accuracy and computational efficiency. Accuracy drops by approx. 2% when D=64, and increases by less than 1% when D=256.
- **Contrastive Loss Margin**: margin=0.5 yields the best results.
- **Cross-Modal Alignment Strategy**: Symmetric contrastive loss outperforms unidirectional contrastive loss.
- **Negative Class Strategy**: Using hard negative mining improves accuracy by approx. 2-3% compared to random negative sampling.

### Few-Shot Enhancement Effects
- Fusing 1 audio sample with the text anchor improves accuracy by approx. 3-5% (compared to pure text anchors).
- Fusing 5 audio samples brings accuracy close to that of the pure audio anchor method (gap <2%).
- alpha=0.3 (lower weight for text anchor) performs best in few-shot settings.

## Limitations and Future Work

### Technical Limitations
- **Text Representation Quality Depends on the Text Encoder**: The quality of the text encoder directly affects the effectiveness of cross-modal alignment. If the text encoder cannot accurately capture the pronunciation features of keywords (e.g., for rare words, foreign words, abbreviations, etc.), the quality of the text anchor will degrade. Using pre-trained language models (such as BERT) may improve text encoding quality but also increases model complexity.
- **Challenges with Homophones**: For keywords with the same pronunciation but different meanings (e.g., "write" and "right"), text anchors cannot distinguish them because their phonetic representations are identical. Additional contextual information is required to disambiguate in such cases.
- **Imperfection of Text-Audio Alignment**: Alignment between text embeddings and audio embeddings may not be perfect for all keywords. The text-to-audio mapping for some keywords may be more complex than for others (e.g., multi-syllabic words, words containing rare phonemes), leading to uneven alignment quality.
- **Cross-Lingual Generalization**: The current method is primarily validated on English data. For tonal languages (such as Chinese, where different tones for the same syllable correspond to different words) and morphologically rich languages, the design of text anchors may need to additionally consider tonal information and morphological variations.

### Experimental Design Shortcomings
- Evaluation is limited in large-vocabulary or continuous speech scenarios. The current evaluation mainly focuses on isolated word KWS; the performance of the text anchor method in keyword search scenarios within continuous speech remains unclear.
- The text encoder component adds extra computational overhead, but the paper does not analyze the impact of this overhead on practical deployment.
- No comparison is made with phonetic posteriorgram-based KWS methods.
- Fine-grained analysis of different types of keywords (short vs. long words, common vs. rare words) in the zero-shot setting is insufficient.

### Future Improvement Directions
- Explore the use of pre-trained multimodal models (such as the audio version of CLIP, speech-text contrastive learning models) to improve cross-modal alignment quality.
- Introduce prosodic information such as tone and stress into the text encoder, enabling text anchors to distinguish tonal differences in tonal languages.
- Investigate online learning and user feedback mechanisms—when a user's pronunciation does not match the text anchor well, the system can gradually adapt to the user's acoustic characteristics.
- Incorporate speaker information into the embedding space, allowing text anchors to be personalized to some extent (the text-to-audio mapping may differ for different speakers).
- **Implications for the KWS Field**: Text anchors provide a brand-new paradigm for keyword customization in KWS—a leap from "requiring audio data" to "requiring only text descriptions." This approach can be generalized to other speech tasks requiring flexible class customization, such as command customization in speech command recognition and event type expansion in sound event detection.
