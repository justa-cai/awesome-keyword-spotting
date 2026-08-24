# Zero-shot Keyword Spotting for Visual Speech Recognition in-the-wild

- **Authors/Affiliations**: Themos Stafylakis, Georgios Tzimiropoulos (School of Computer Science, University of Nottingham)
- **Date**: July 2018 (ECCV 2018, European Conference on Computer Vision)
- **Link**: https://openaccess.thecvf.com/content_ECCV_2018/papers/Themos_Stafylakis_Zero-shot_keyword_search_ECCV_2018_paper.pdf
- **Keywords**: visual keyword spotting, zero-shot learning, lipreading, grapheme-to-phoneme, visual speech recognition, G2P

## Problem Statement

Visual Speech Recognition (VSR), also known as lipreading, aims to recognize spoken content from mouth and lip movements alone. Visual keyword spotting is the task of detecting specific keywords in a continuous visual speech stream; it has important applications in security surveillance, speech recognition in extremely noisy environments, and assisting people with hearing impairment.

**Pain Points in the Field**
- Audio speech recognition fails completely in extremely noisy environments (e.g., factories, concerts) or where audio is unavailable (e.g., muted surveillance footage)
- Visual speech recognition does not rely on the audio signal and can serve as an important complement to audio KWS
- Collecting annotated visual speech data is extremely tedious—it requires precise time-boundary annotations and high-quality lip video

**The Need for Zero-shot Detection**
Traditional visual speech recognition systems can only detect vocabulary seen in the training set. However:
- Real applications may need to detect arbitrary keywords (e.g., new security alert words, new names)
- Re-collecting training data and retraining the model for every new keyword is impractical
- Zero-shot capability—detecting words unseen during training—is a hard requirement for visual KWS

**Key Challenges This Paper Aims to Solve**
- How to build a visual keyword spotting system that can detect entirely new words unseen during training
- How to convert a text-form keyword into a representation that can be matched against visual features
- How to achieve robust visual keyword spotting on in-the-wild video data

## Methodology

### Overall Architecture Design

The paper proposes an end-to-end zero-shot visual keyword spotting architecture with three tightly integrated components.

**Component 1: Visual Feature Extraction (Spatio-Temporal Residual Network)**

Extracts high-level visual features from mouth/lip video.

1. **Face detection and lip cropping**: a face detector locates the mouth region, which is cropped into a fixed-size image sequence
2. **Spatio-temporal residual network**: a 3D-ResNet-based architecture that jointly models spatial (lip shape) and temporal (lip motion) information
   - Spatial stream: captures lip morphology (e.g., degree of mouth opening, corner-of-mouth position)
   - Temporal stream: captures lip motion patterns (e.g., open-close rhythm, the sequence of mouth-shape changes)
3. **Output**: a fixed-dimensional visual feature vector per frame (or every few video frames)

**Component 2: Grapheme-to-Phoneme (G2P) Encoder-Decoder**

Maps a text-form keyword to its pronunciation embedding—the key to zero-shot detection.

1. **Input**: the keyword's grapheme sequence, e.g., "H-E-L-L-O"
2. **Encoder**: encodes the grapheme sequence into a fixed-length vector
3. **Decoder**: decodes the encoded vector into a phoneme sequence, e.g., "HH AH L OW"
4. **Pronunciation embedding**: the encoder's hidden state serves as the keyword's "pronunciation representation"
5. **Advantages of G2P**:
   - Different but similarly pronounced words (e.g., "their" and "there") produce similar embeddings
   - Provides a better phoneme-level representation than a plain grapheme sequence
   - Lets the system understand "what this new word sounds like"

**Component 3: Matching Network (BiLSTM RNN Stack)**

Matches visual features against the keyword's pronunciation embedding.

1. **Input**: visual feature sequence + keyword pronunciation embedding
2. **BiLSTM stack**: bidirectional LSTMs contextually encode the visual feature sequence
3. **Matching mechanism**: computes a similarity or matching score between the visual feature sequence and the keyword pronunciation embedding
   - Attention can align visual frames with phoneme positions
   - Alternatively, a sequence-to-sequence matching framework can compute an overall match score
4. **Output**: keyword presence probability at each temporal position

### Training Strategy
- Trained on the LRS2 (Lip Reading in the Wild) database
- LRS2 contains a large amount of lip video excerpted from BBC television, with text annotations
- Training uses a large vocabulary (not limited to the keywords to detect), letting the model learn general visual-to-pronunciation correspondences
- At zero-shot test time, brand-new keywords (unseen in training) are fed in and detected directly

## Main Contributions

1. **First zero-shot visual keyword spotting**: the first systematic study of visual keyword spotting for words unseen in training. Prior work either required training a dedicated model per keyword or worked only on a fixed vocabulary. Zero-shot capability lets the system detect arbitrary new keywords, greatly extending practicality.

2. **G2P-based keyword representation**: proposes using a grapheme-to-phoneme model to generate a keyword pronunciation embedding instead of a plain grapheme sequence. This representation captures the phonetic essence of words, making visual feature matching more accurate. For example, similarly pronounced words yield similar embeddings and thus may also produce similar matching patterns visually.

3. **End-to-end trainable**: the whole architecture (visual feature extraction, G2P encoding, matching network) is end-to-end trainable, without hand-crafted intermediate features or keyword boundary annotations. End-to-end training allows joint optimization of all components for better overall performance.

4. **Beats the ASR baseline**: outperforms traditional ASR-based baselines on visual keyword spotting. This shows that directly training a visual keyword spotting model is more effective than the indirect approach of "first transcribe with a VSR system, then search the text".

5. **Published at ECCV 2018**: published at a top computer vision conference, reflecting the cross-disciplinary value of the work—bringing the keyword spotting concept from speech technology into the visual domain.

## Experimental Results

### Dataset
- LRS2 (Lip Reading in the Wild) database
- Source: BBC television programs
- Scale: tens of thousands of talking-head video clips
- Characteristic: "in-the-wild"—unconstrained lighting, pose, and speaking conditions

### Core Results

**Zero-shot detection performance**
- Achieves "very promising" visual-only keyword spotting results on the challenging LRS2 database
- Can detect any keyword, including words never present in the training set
- G2P embeddings significantly outperform simple grapheme-based word representations

**Comparison with baselines**
- Outperforms ASR-based visual speech keyword spotting baselines
- Substantially surpasses previous ASR-free keyword spotting methods
- Direct visual matching is more effective than the indirect "first visually transcribe, then text-search" approach

**Effect of G2P embeddings**
- G2P embeddings provide better alignment than grapheme embeddings in visual feature matching
- Similarly pronounced words are closer in G2P space, helping with homophones and near-homophones
- The G2P model needs a phonetic lexicon for training, but once trained it generalizes to new words

### Qualitative Analysis
- The model correctly localizes the temporal position of keywords in video
- Shows a degree of robustness to different speakers and lighting conditions
- Short keywords (1-2 syllables) are harder to detect than long ones (subtler visual variation)

## Limitations and Future Work

### Technical Limitations of the Method
- **Innate limits of visual-only sensing**: purely visual keyword spotting is inherently less accurate than audio-based methods. Lip motion carries far less information than the audio signal—many phonemes are visually hard to distinguish (e.g., "p" vs. "b", "m" vs. "n"), known as the "visual homophones" problem.
- **Dependence on visual quality**: requires clearly visible lips; side angles, occlusion (e.g., face masks, a hand over the mouth), and low resolution all severely hurt performance.
- **Language dependence of G2P**: the G2P model needs a phonetic lexicon of the target language for training, which may be unavailable for low-resource languages.
- **Computational complexity**: 3D ResNet visual feature extraction is computationally heavy; real-time operation on embedded devices may be difficult.

### Shortcomings of the Experimental Design
- Evaluated only on LRS2 (English); cross-language and cross-domain generalization is unverified
- No comparison with audio-visual fusion keyword spotting methods
- The effect of the choice and number of zero-shot keywords on results was not analyzed in depth

### Future Improvement Directions
- Combine audio and visual information for multimodal keyword spotting, using visual cues to enhance audio detection in noisy environments
- Explore lighter visual feature extraction architectures for real-time inference on edge devices
- Use self-supervised pretraining (e.g., learning visual speech representations from large amounts of unlabeled video)
- Extend to multilingual zero-shot visual keyword spotting
- Explore adversarial training for robustness to lighting and angle changes

### Implications for the KWS Field
- Visual keyword spotting is an important complement to audio KWS, with unique value where audio is unavailable or extremely noisy
- Zero-shot learning lets KWS systems flexibly adapt to new keywords without retraining
- The G2P embedding idea transfers to audio KWS—encode keywords by pronunciation rather than fixed labels
- Cross-modal (visual-speech) representation learning opens a new research direction for KWS technology
- Publication at ECCV shows that KWS technology has broad cross-disciplinary application prospects
