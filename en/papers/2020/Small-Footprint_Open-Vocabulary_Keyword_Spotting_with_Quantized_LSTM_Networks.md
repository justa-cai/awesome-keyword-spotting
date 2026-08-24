# Small-Footprint Open-Vocabulary Keyword Spotting with Quantized LSTM Networks

**Authors/Affiliations**: Ivan Lopez-Espejo, Zheng-Hua Tan, Jesper Jensen (Sonos Inc., Aalborg University)

**Date**: February 2020 (arXiv:2002.10851)

**Link**: https://arxiv.org/abs/2002.10851

**Keywords**: Keyword Spotting, Open-Vocabulary, Quantized LSTM, Small-Footprint, Spoken Word Detection

## Problem Statement

Most KWS systems have a fixed set of keywords during training and can only detect these predefined keywords after deployment. This Closed-Vocabulary approach has obvious limitations:
- **Poor Flexibility**: Users cannot customize keywords
- **Multilingual Difficulties**: Different languages require training different models
- **High Update Costs**: Adding new keywords requires retraining or fine-tuning the model

Open-Vocabulary KWS allows users to specify arbitrary keywords during the registration phase without retraining the model. This is crucial for scenarios such as personalized voice interaction and smart home control.

Technical approaches to implementing Open-Vocabulary KWS:
- Train a generic phoneme recognizer (Acoustic Model)
- Keywords are defined as sequences of phonemes (e.g., "Hey Siri" -> /HH EY S IY R IY/)
- Detection is achieved by matching the phoneme recognition results with keyword phoneme templates

Challenge: How to achieve high-quality phoneme recognition and flexible keyword matching while maintaining a small footprint?

## Methodology

### LSTM Acoustic Model

**Network Architecture**:
- Multi-layer LSTM (Long Short-Term Memory) network
- Input: Audio spectral features (Mel filterbanks or MFCCs)
- Output: Frame-level phoneme posterior probabilities
- Reasons for choosing LSTM:
  - Naturally suited for sequence modeling, capable of capturing temporal dependencies between phonemes
  - Gating mechanism effectively handles long-term dependencies in speech
  - Suitable for streaming, frame-by-frame processing

**Phoneme Output**:
- Output dimension equals the size of the phoneme set (e.g., approximately 40 phonemes for English)
- Uses Softmax to output the probability distribution for each phoneme
- Predicted independently per frame, with temporal constraints considered in subsequent decoding

### Keyword Detection Mechanism

**Phoneme Sequence Matching**:
1. Run the LSTM acoustic model to obtain a sequence of frame-level phoneme posterior probabilities
2. Keywords are defined as phoneme sequence templates (e.g., "Alexa" -> /AE L EH K S AH/)
3. Use Dynamic Time Warping (DTW) or Viterbi decoding to match the phoneme posterior sequence with the keyword template
4. If the matching score exceeds a threshold, the keyword is considered detected

**Keyword Registration**:
- The user provides the text of the keyword
- The text is converted into a phoneme sequence via a Grapheme-to-Phoneme (G2P) system
- The phoneme sequence is stored as a template for subsequent detection

### Network Quantization

To achieve small-footprint deployment:
- **Int8 Quantization**: Quantize LSTM weights and activations from 32-bit floating-point to 8-bit integers
- **Quantization Methods**: Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
- **Goal**: Reduce model size (by approximately 4x) and computational load, while minimizing accuracy loss
- **Key Consideration**: The recurrent connections in LSTMs are sensitive to quantization and require special handling

## Main Contributions

1. **Open-Vocabulary KWS System**: Proposes a complete open-vocabulary KWS system where users can specify arbitrary keywords (in the form of phoneme sequences) during the registration phase without retraining the model. This solves the flexibility problem of traditional closed-vocabulary KWS.

2. **Quantized LSTM Architecture**: Systematically evaluates the quantization effects of LSTM acoustic models, demonstrating that int8 quantization can reduce model size by approximately 4x with minimal accuracy loss, making the model suitable for embedded deployment.

3. **Phoneme-Level Matching Method**: A detection method based on phoneme sequence template matching, where a single acoustic model can serve any keyword. The model is decoupled from the keywords, achieving true "train once, detect any word."

4. **Comprehensive Quantization Impact Assessment**: Detailed study of the differential impact of quantization on detection performance for different types of keywords (varying lengths and phoneme complexities).

## Experimental Results

### Experimental Setup
- English speech dataset
- LSTM Acoustic Model: Multi-layer LSTM, outputting approximately 40-dimensional phoneme posterior probabilities
- Evaluation: Detection accuracy on different keywords
- Quantization: float32 vs int8

### Main Results
- **Detection Accuracy**: The open-vocabulary system achieves competitive detection accuracy across multiple keywords
- **Int8 Quantization**: Model size reduced by approximately 75%, with only a slight decrease in detection accuracy
- **Cross-Keyword Generalization**: The same model performs consistently across all tested keywords, proving the universality of the open-vocabulary approach
- **Long vs. Short Keywords**: Longer keywords (with more phonemes) are easier to detect because they provide more matching information
- **Phoneme Confusion**: Confusion between certain phoneme pairs (e.g., /b/ and /p/) increases after quantization, but the overall impact is controllable

### Detailed Quantization Analysis
- The input and forget gates of the LSTM are the most sensitive to quantization
- Uniform quantization performs well in most cases
- Quantization-Aware Training (QAT) provides better quantization recovery than Post-Training Quantization (PTQ)

## Limitations and Future Work

### Method Limitations
- **Upper Bound of Phoneme Recognition Accuracy**: The accuracy of the open-vocabulary approach is limited by the quality of the phoneme recognizer and may be lower than that of dedicated keyword models
- **G2P Dependency**: Keywords need to be converted into phoneme sequences; errors in the G2P system directly affect detection performance
- **Phoneme Annotation Requirement**: Training the phoneme recognizer requires phoneme-level annotated data
- **Impact of Quantization on Similar Words**: Quantification may exacerbate confusion between acoustically similar keywords

### Future Directions
- Research end-to-end open-vocabulary KWS methods that do not require explicit phoneme intermediate representations
- Explore sub-word units as alternatives to phonemes to provide more flexible keyword representations
- Combine embedding learning methods to improve discriminative ability in open-vocabulary scenarios
- Research multilingual phoneme sets to support cross-lingual open-vocabulary KWS
- Explore more aggressive quantization methods (e.g., 4-bit, 2-bit) to further reduce model size
