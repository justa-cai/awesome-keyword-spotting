# Online Keyword Spotting with Character-Level Recurrent Neural Networks and CTC

- **Authors/Affiliations**: Kyuyeon Hwang, Minjae Lee, Wonyong Sung (Seoul National University)
- **Date**: 2017
- **Link**: https://arxiv.org/abs/1706.02693
- **Keywords**: Keyword Spotting, Character-level RNN, CTC, Online Detection, Streaming Inference

## Problem Statement

Keyword spotting systems need to operate in an online (streaming) manner—i.e., the system can detect the wake-up word immediately after the user speaks it, without waiting for the complete utterance to end. Traditional offline methods typically require obtaining the complete audio segment before making a judgment, introducing unacceptable latency that fails to meet the requirements of real-time applications.

The core problem addressed in this paper is: How to construct an online keyword spotting system based on character-level recurrent neural networks and CTC (Connectionist Temporal Classification) that can detect keywords frame-by-frame with extremely low latency while audio frames are continuously input? Specifically, the system needs to: (1) process audio inputs frame-by-frame and incrementally generate character predictions; (2) monitor the target keyword character sequence within the CTC output sequence; (3) trigger immediately upon detecting the complete keyword, minimizing end-to-end latency.

## Methodology

### Overall Architecture

The proposed online KWS system consists of three core modules:

1. **Feature Extraction Frontend**:
   - Audio is collected at a sampling rate of 16kHz
   - Extract 40-dimensional log-mel filterbank energies
   - Frame length 25ms, frame shift 10ms
   - Optional: Apply Cepstral Mean and Variance Normalization (CMVN) to features

2. **Character-level RNN Acoustic Model**:
   - Uses multi-layer Recurrent Neural Networks (LSTM or GRU) as the acoustic model
   - Input: Audio feature vector per frame
   - Output: Character-level probability distribution (including a-z, space, CTC blank label)
   - The RNN processes audio frame-by-frame, maintaining hidden states, implicitly encoding the temporal context of processed frames

3. **Streaming Keyword Detection Logic**:
   - Continuously monitors the RNN's CTC output sequence
   - Uses the CTC forward variable to calculate the probability that the target keyword has been spoken up to the current moment
   - Triggers keyword detection when this probability exceeds a threshold
   - Resets state after triggering, preparing to detect the next keyword

### CTC Training Framework

1. **CTC Principles**:
   - CTC introduces a blank label (blank), allowing the network to output blanks between any two characters
   - Consecutive outputs of the same character are merged (via the CTC collapse rule: removing blanks and duplicate characters)
   - Example: "__a_pp__l_e__" → "apple"
   - The CTC loss function efficiently calculates the marginal probability of all possible alignment paths using the forward-backward algorithm

2. **Training Objective**:
   - Given an audio sequence $X$ and a target character sequence $L$, maximize $P(L|X)$
   - $P(L|X) = \sum_{\pi \in B^{-1}(L)} \prod_{t=1}^{T} p_t(\pi_t|X)$
   - Where $B^{-1}(L)$ is the set of all paths that can be mapped to $L$ via the CTC collapse rule

3. **Frame-level Prediction**:
   - After training, the RNN outputs a character probability distribution for each frame
   - Greedy decoding: Select the character with the highest probability per frame (which may be the blank label)
   - Obtain the final character sequence through the CTC collapse rule

### Online Detection Algorithm

The core challenge of streaming detection lies in making detection decisions before the audio has been fully received:

1. **Incremental CTC Forward Computation**:
   - Maintain the CTC forward variable $\alpha(t, s)$, representing the cumulative probability at frame $t$ for the $s$-th character of the keyword
   - Incrementally update the forward variable with each new audio frame received
   - Detection score = $\alpha(t, S)$, where $S$ is the position of the last character of the keyword

2. **Detection Trigger Condition**:
   - Trigger keyword detection when the detection score exceeds a preset threshold
   - The choice of threshold balances detection sensitivity and false alarm rate

3. **State Reset**:
   - After detection is triggered, reset the CTC forward variables and RNN hidden states
   - Prevent the detected keyword from affecting subsequent detections

### Key Technical Details

- **Advantages of Character-level Output**: A character-level output space allows the same model to detect any keyword (by simply changing the target character sequence) without retraining the acoustic model
- **Unidirectional RNN**: Online scenarios must use unidirectional RNNs (utilizing only historical information); bidirectional RNNs requiring future frames cannot be used
- **Latency Analysis**: Keyword detection latency depends on: (1) the temporal modeling capability of the RNN; (2) the convergence speed of the CTC forward variables; (3) threshold settings

## Main Contributions

1. **Online Character-level RNN-CTC KWS System**: Developed the first online (streaming) keyword spotting system based on character-level RNNs and CTC, achieving the ability to detect any keyword with minimal latency from a continuous audio stream. This system extends CTC from offline speech recognition to real-time keyword spotting in streaming scenarios.

2. **Flexibility of Character-level Modeling**: Demonstrated that character-level output provides a flexible framework for keyword spotting—by simply specifying the character sequence of the target keyword, the same acoustic model can detect any keyword without retraining. This flexibility is of significant value in practical deployment, allowing users to customize wake-up words.

3. **Streaming CTC Detection Algorithm**: Implemented an incremental streaming detection algorithm based on CTC forward variables, capable of calculating keyword detection scores in real-time as audio is continuously input, and triggering immediately when the threshold is reached.

4. **Systematic Analysis of Latency and Accuracy**: Systematically analyzed the latency characteristics of streaming detection, quantifying the trade-off between detection latency and detection accuracy.

## Experimental Results

### Experimental Setup
- **Dataset**: Used public speech datasets (such as WSJ, LibriSpeech) and internal KWS datasets
- **Evaluation Metrics**: Detection accuracy, false alarm rate, detection latency (latency from the end of the keyword to the trigger)
- **Comparison Methods**: Offline CTC-KWS, HMM-DNN KWS, frame-level classification methods

### Key Results
- The online detection accuracy of character-level RNN-CTC is comparable to offline methods (using complete utterances), proving that streaming processing does not significantly sacrifice accuracy
- Detection latency is controlled within a reasonable word-level range, triggering within hundreds of milliseconds after the keyword is spoken
- Character-level output allows the model to flexibly handle different keywords without specialized training for each keyword
- The model size is moderate, suitable for deployment on resource-constrained devices

### Latency Analysis
- Most keyword instances are successfully detected within 200-500ms after the keyword ends
- Detection latency mainly depends on the clarity with which the last few characters of the keyword appear in the CTC output
- Lower detection thresholds reduce latency but increase false alarms, requiring adjustment based on application needs in practice

## Limitations and Future Work

### Limitations

1. **Frame-level CTC Prediction Noise**: CTC character predictions at the frame level can be very noisy—many frames output blank labels, occasionally outputting characters, and character boundaries may be imprecise. This noisy output requires careful post-processing (CTC collapse and forward variable computation) to obtain reliable keyword detection.

2. **False Triggers from Partial Matches**: The streaming detection logic may produce high detection scores when the keyword is only partially spoken (e.g., if the target keyword is "hey siri", saying only "hey" might trigger it). Additional mechanisms are needed to prevent false triggers based on partial matches.

3. **Gap in Accuracy Compared to Dedicated Models**: The detection accuracy of a general character-level acoustic model on specific keywords may be lower than that of a dedicated model trained specifically for that keyword. There is an inherent trade-off between generality and accuracy.

4. **Limitations of Character Representation**: Keywords are specified in character (spelling) form, rather than phonetic form. For words with the same pronunciation but different spellings, or different phonetic variants of the same word, character-level representation may not be flexible enough.

5. **Performance Ceiling of Unidirectional RNNs**: Online scenarios require the use of unidirectional RNNs, which cannot utilize future audio information, potentially becoming a performance bottleneck in recognizing the acoustic patterns of certain keywords.

### Future Work

1. **RNN-Transducer Upgrade**: Use RNN-T instead of CTC, introducing dependencies between output characters via a prediction network, breaking the limitations of CTC's conditional independence assumption, which is expected to improve the accuracy of character-level online detection
2. **Phoneme-level Output**: Combining character-to-phoneme conversion, using a phoneme-level output space may better capture the acoustic characteristics of keywords
3. **Attention Mechanism**: Introduce an attention mechanism on top of the RNN, allowing the model to automatically focus on the most discriminative time segments of the keyword, improving detection accuracy
4. **Adaptive Thresholds**: Dynamically adjust detection thresholds based on current environmental noise levels and speaker characteristics, reducing sensitivity in quiet environments to minimize false alarms
5. **Multilingual Extension**: Extend the character-level framework to multilingual scenarios, using a unified character/subword output space to support multilingual keyword detection
