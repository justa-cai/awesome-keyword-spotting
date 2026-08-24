# Query-by-Example Keyword Spotting Using CTC and FST

- **Authors/Affiliations**: Byeonggeun Kim, Mingu Lee, Jinkyu Lee, Yeonseok Kim, Kyuwoong Hwang (Qualcomm AI Research)
- **Date**: October 2019 (ASRU 2019)
- **Link**: https://arxiv.org/abs/1910.05171
- **Keywords**: Keyword Spotting, Query-by-Example, CTC, FST, Open Vocabulary, Keyword Registration, Phoneme Decoding

## Problem Statement

The vast majority of keyword spotting systems are designed for a **fixed set of keywords**—such as "OK Google," "Alexa," and "Hey Siri." Adding new keywords requires re-collecting data, retraining the model, or at least performing significant model adjustments. This closed-vocabulary design lacks flexibility in the following scenarios:

1. **User Customization Needs**: Users may wish to use personalized wake words or command words rather than vendor-predefined keywords.
2. **Multilingual and Dialectal Variations**: Different languages and dialects require different sets of keywords; training separate models for each language is costly.
3. **Rapid Deployment Requirements**: New products or features may need to quickly add new voice commands, and the cycle for retraining models is too long.

Therefore, the core requirement is to design an **open-vocabulary** KWS system that can detect any user-defined keyword from a few registration samples without modifying or retraining the underlying neural network model.

## Methodology

This paper proposes a Query-by-Example keyword spotting system based on **CTC (Connectionist Temporal Classification) + FST (Finite State Transducer)**.

### 1. System Architecture

The system consists of two core modules:

#### 1.1 CTC-based Phoneme Posterior Generator

- **Input**: Continuous stream of acoustic features
- **CTC Acoustic Model**: A trained CTC neural network that predicts **phoneme posterior probabilities** from the input features
- The special design of CTC allows it to handle uncertain alignment relationships between input and output—no frame-level phoneme alignment annotations are required
- **Output**: Phoneme posterior probability distribution for each frame (covering all phoneme units + CTC blank label)

Key feature: The CTC model is trained **once** and then fixed; it does not change with keyword variations.

#### 1.2 FST-based Keyword Decoder

- **FST (Finite State Transducer)**: Represents keywords as **sequences of phoneme units**
- For example, the keyword "hello" is represented as the phoneme sequence /HH AH L OW/
- The FST takes the sequence of phoneme posterior probabilities from CTC as input and uses Viterbi decoding to find the path that best matches the keyword's phoneme sequence
- When the matching score exceeds a threshold, keyword detection is triggered

### 2. Keyword Registration Process

The process for registering a new keyword is extremely concise:
1. Obtain the **phoneme transcription** of the new keyword—this can be obtained via dictionary lookup or automatic phonemization tools
2. Add the phoneme sequence to the decoding graph of the FST
3. **No need to retrain the CTC neural network**

This approach completely encodes the "knowledge" of keywords in the phoneme sequences of the FST, while the acoustic model (CTC) remains general and fixed.

### 3. Technical Details

#### 3.1 CTC Training
- The CTC model is trained on large-scale speech data to learn general phoneme recognition capabilities
- The CTC blank label handles silences and transitions between phonemes
- The model outputs frame-level posterior probabilities for all phonemes

#### 3.2 FST Decoding
- The FST encodes the phoneme sequence of keywords into a state transition graph
- Supports phoneme variant pronunciations (different realizations of the same phoneme)
- Can easily integrate language model constraints to reduce false alarms
- The Viterbi algorithm searches for the optimal matching path on the phoneme posterior grid

### 4. Comparison with Traditional Methods

| Aspect | Traditional Closed-Vocabulary KWS | CTC+FST Open-Vocabulary |
|------|----------------|----------------|
| New Keywords | Requires retraining | Only requires phoneme transcription |
| Model Updates | Retraining/Fine-tuning | Update FST graph |
| Deployment Flexibility | Low | High |
| Computational Overhead | Usually small | Additional overhead from FST decoding |

## Main Contributions

1. **CTC+FST Open-Vocabulary KWS Architecture**: Proposes a systematic framework that combines CTC phoneme posterior generation with FST keyword decoding, enabling flexible registration and detection of arbitrary keywords. This design elegantly decouples acoustic modeling (CTC) from keyword matching (FST).

2. **Keyword Registration Without Retraining**: Registration is completed simply by adding the phoneme transcription of the new keyword to the FST, without modifying or retraining the CTC neural network. This significantly reduces the deployment cost and time for new keywords.

3. **Phoneme-Level General Representation**: Utilizes phonemes as a general intermediate representation—the CTC model learns general phoneme recognition capabilities, and the FST uses phoneme sequences to define keywords. This phoneme-level abstraction allows the same CTC model to serve any keyword.

4. **Maintaining Small Model Footprint**: The size of the CTC acoustic model is kept within a range suitable for edge deployment, and the storage requirements for the FST decoding graph are also small.

5. **Published at ASRU 2019**, representing important work by Qualcomm in the field of open-vocabulary KWS.

## Experimental Results

- The system can effectively detect any registered keyword, with competitive accuracy
- The combination of CTC+FST demonstrates the feasibility of phoneme-based open-vocabulary KWS methods
- The keyword registration process is fast—only phoneme transcription is required
- Reasonable detection performance is maintained even under conditions with few registration samples

## Limitations and Future Work

### Technical Limitations
- **Bottleneck of CTC Phoneme Prediction Quality**: The overall performance of the system is limited by the accuracy of the CTC model's phoneme predictions. Under conditions such as noisy environments, accented speech, or children's speech, the phoneme prediction error rate increases, directly affecting the recall rate of keyword detection.
- **Propagation Effect of Phoneme Errors**: Errors in CTC's phoneme predictions propagate to the FST decoding stage, leading to keyword matching failures. This is particularly problematic for short keywords (1-2 syllables), where a few phoneme errors can lead to detection failure.
- **Ambiguous Phoneme Representation**: Some keywords have multiple possible phoneme representations (e.g., heteronyms, dialectal variants); choosing an inappropriate phoneme transcription may affect detection performance.
- **FST Decoding Latency**: For very large sets of keywords, the computational latency of FST decoding may become a bottleneck, affecting real-time performance.

### Future Directions
- Explore end-to-end training of CTC+FST systems, incorporating the matching objectives of FST decoding into the CTC training process.
- Research methods for automatic phoneme transcription generation to reduce reliance on manual phoneme annotation.
- Combine subword units as intermediate representations to achieve a better balance between phoneme and word levels.
- Optimize FST decoding algorithms to maintain real-time performance even under large-scale keyword sets.
- Explore methods for registering keywords directly from audio examples (without phoneme transcription) to further improve user experience.
