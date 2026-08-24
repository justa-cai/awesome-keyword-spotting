# Efficient Voice Trigger Detection for Low Resource Hardware

- **Authors/Affiliations**: Siddharth Sigtia, Rob Haynes, Hywel Richards, Erik Marchi, John Bridle (Apple Siri voice team)
- **Date**: 2018.11 (Interspeech 2018)
- **Link**: https://www.isca-speech.org/archive/Interspeech_2018/pdfs/2204.pdf
- **Keywords**: voice trigger detection, DNN-HMM, low resource hardware, embedded devices, Apple Siri, frame rate optimization

## Problem Statement

Voice trigger detection is the first gateway of intelligent voice assistants (such as Siri, Alexa, and Google Assistant), responsible for detecting a specific wake word (e.g., "Hey Siri") in a continuous audio stream. This functionality faces unique technical challenges:

**Scenario Background and Field Pain Points**
- A voice trigger system must run continuously in an "always-on" mode on battery-powered mobile devices, which means an extremely low power budget is the primary constraint
- Apple's product ecosystem includes devices such as the iPhone and the Apple Watch, where the Apple Watch has even more constrained computational resources and battery capacity, imposing stricter efficiency requirements on the voice trigger system
- The system adopts a multi-stage cascade architecture: the first-stage low-power detector runs on a dedicated audio processor and performs initial screening; once triggered, it wakes up a second-stage detector that is more accurate but also more power-hungry

**Specific Shortcomings of Existing Methods**
- In the traditional DNN-HMM architecture, each phoneme typically uses 3 HMM states (corresponding to the beginning, middle, and end segments of the phoneme), which leads to a high-dimensional DNN output layer and thus a large amount of computation
- The standard 100Hz frame rate means the DNN acoustic model must perform 100 forward inferences per second, challenging the computational capability of low-power processors
- Significantly reducing computational complexity while maintaining detection accuracy (especially a low false alarm rate) remains an optimization space that has not been fully explored

**Key Challenges This Paper Aims to Solve**
- How to reduce the computational cost of the DNN-HMM trigger detector to 1/6 of the original design without compromising the accuracy of "Hey Siri" detection
- How to optimize computational efficiency by adjusting the HMM state configuration and the frame rate

## Methodology

### Overall Architecture Design

The paper describes the main detector of the "Hey Siri" voice trigger system in Apple's production environment, which adopts the classic DNN-HMM architecture. This detector is the first stage of the multi-stage system and runs on a low-power always-on processor.

### Core Technical Solution

**1. DNN Acoustic Model**
- **Input features**: 13-dimensional MFCC features at a frame rate of 100Hz, with a context window of 5 frames before and after each frame concatenated as the input
- **Network structure**: a multi-layer fully connected DNN (the exact number of layers and widths are not disclosed due to proprietary information)
- **Output layer**: a softmax classification over the HMM states, outputting the posterior probability of each state

**2. HMM Decoder**
- A Hidden Markov Model (HMM) is used to model the phoneme sequence of "Hey Siri"
- Each phoneme corresponds to several HMM states, with self-loops and forward transitions between states
- Viterbi decoding is used to find the optimal path over the sequence of state posterior probabilities output by the DNN

### Key Optimization Strategies

**Optimization 1: HMM State Reduction (from 3 states to 1 state)**
In the traditional approach, each phoneme uses 3 HMM states (typically labeled as beginning, middle, end). The paper proposes reducing the number of target labels per phoneme from 3 to 1, i.e., all frames are uniformly mapped to a phoneme-level label. This change brings threefold benefits:
- It reduces the dimensionality of the DNN output layer (the number of output nodes is reduced to about 1/3)
- It simplifies the HMM topology and reduces the computational complexity of Viterbi decoding
- Experiments show that in keyword detection tasks, the three-state subdivision within a phoneme is not necessary, because the phoneme sequence of the keyword is fixed and does not require fine-grained state discrimination

**Optimization 2: Frame Rate Reduction**
The operating frame rate of the acoustic model is reduced from the standard 100Hz to 50Hz (i.e., one frame is computed every 20ms instead of every 10ms). This change directly halves the forward inference frequency of the DNN. The key findings are:
- In keyword detection tasks, a 10ms frame resolution provides excessive temporal precision
- A 20ms frame interval is still able to capture the acoustic characteristics of all key phonemes in "Hey Siri"
- The impact of the frame rate reduction on detection accuracy is negligible

**3. Minimum HMM State Duration Constraint**
To address the problem of short phonemes being misjudged as a result of the frame rate reduction, the paper adjusts the minimum state duration parameter of the HMM. By enforcing at the HMM level that each phoneme lasts at least a certain number of frames, it compensates for the loss of temporal resolution caused by the reduced frame rate.

## Main Contributions

1. **A systematic methodology for DNN-HMM efficiency optimization**: For the first time in an industrial-scale production system, it discloses a complete approach to optimizing the efficiency of a voice trigger detector along two dimensions—HMM state reduction and frame rate reduction. The 6x reduction in computational cost is achieved through the multiplicative effect of state reduction (about 3x) and frame rate halving (2x).

2. **Validation of the simplification from 3 states to 1 state**: It demonstrates the feasibility of reducing the number of HMM states per phoneme from 3 to 1 in keyword detection scenarios. This finding challenges the conventional belief in the speech recognition field that "three-state phoneme modeling is a necessary standard," pointing out that for the detection of fixed phrases, a simplified modeling scheme is already sufficient.

3. **A practical guideline for frame rate optimization**: It systematically evaluates the impact of different frame rates on detection accuracy, finding that a 50Hz frame rate performs comparably to a 100Hz frame rate in keyword detection, providing a practical parameter selection guideline for the design of embedded speech systems.

4. **Production system architecture insights**: It describes Apple Siri's multi-stage detection architecture—a low-power main detector that, once triggered, wakes up a more accurate second-stage detector. This cascade design has broad reference value across the industry.

## Experimental Results

### Evaluation Setup
- Target keyword: "Hey Siri"
- Evaluation platform: Apple's proprietary low-power always-on processor
- Evaluation metrics: detection accuracy (specific values are not disclosed due to proprietary information), reported as relative changes

### Core Results

**Computational Efficiency Optimization**
- **Overall**: a 6x reduction in computational cost while maintaining detection accuracy
- **HMM state reduction**: each phoneme reduced from 3 states to 1 state, substantially lowering the DNN output dimensionality and the HMM decoding complexity
- **Frame rate reduction**: reduced from 100Hz to 50Hz, halving the inference frequency while accuracy remains at a comparable level

**Accuracy Maintenance**
- After combining the two optimizations above, the detection accuracy (false alarm rate and false rejection rate) remains consistent with the unoptimized baseline system
- The minimum HMM state duration constraint effectively compensates for the potential accuracy loss introduced by the frame rate reduction

### Implied Performance Characteristics
- The system is designed for always-on processors with strict power and memory constraints
- The multi-stage architecture ensures that false alarms from the main detector can be filtered out by the subsequent, more accurate detectors

## Limitations and Future Work

### Technical Limitations of the Method
- **Proprietary information restrictions**: Due to Apple's commercial confidentiality policy, the paper does not disclose specific accuracy values, dataset sizes, or network structure details, making independent reproduction and comprehensive evaluation difficult
- **Single trigger phrase**: All optimizations are designed for the specific trigger phrase "Hey Siri" and their generalization to other keywords is not validated
- **Lack of comparison with emerging methods**: No comparison is made with keyword detection methods based on end-to-end neural networks (such as CRNNs and attention-based models)

### Shortcomings of the Experimental Design
- The ambiguity of the evaluation metrics (only relative changes are reported) makes it difficult to judge the absolute performance level of the system
- The difference in optimization effects across different speaker groups (gender, age, accent) is not analyzed
- The robustness of the optimizations under extreme noise conditions is not discussed

### Future Improvement Directions
- Explore replacing the DNN-HMM architecture with a purely neural network architecture, which could further simplify the system
- Combine quantization and pruning techniques to further compress the model on top of the existing optimizations
- Explore an adaptive frame rate mechanism that uses a high frame rate during speech-active periods and a low frame rate during silence

### Implications for the KWS Field
- The idea of HMM state reduction can be generalized to all DNN-HMM-based speech processing systems
- The multi-stage cascade detection architecture is the standard paradigm in industry for handling the accuracy-efficiency trade-off
- Frame rate optimization is a simple yet effective and often overlooked means of reducing the computational cost of embedded speech systems
- Apple's production system design experience provides valuable engineering reference for the entire KWS community
