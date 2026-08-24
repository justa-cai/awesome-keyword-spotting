# Benchmarking Keyword Spotting Efficiency on Neuromorphic Hardware

- **Authors/Affiliations**: Peter Blouw, Xuan Choo, Eric Hunsberger, Chris Eliasmith (Applied Brain Research, Inc. / University of Waterloo)
- **Date**: December 2018 (arXiv:1812.01739)
- **Link**: https://arxiv.org/abs/1811.01739
- **Keywords**: neuromorphic hardware, keyword spotting, energy efficiency, Intel Loihi, spiking neural network, SNN, CTC loss

## Problem Statement

Keyword spotting (KWS) is a core function of voice-interactive devices and must run continuously in an "always-on" mode on edge devices. This characteristic imposes extremely stringent power requirements — the device must complete real-time inference within a very tight energy budget. Although traditional CPU and GPU platforms are powerful, their power-consumption profiles make them unsuitable for keyword spotting tasks that run continuously over long periods. Even low-power embedded platforms such as the NVIDIA Jetson TX1 and the Movidius Neural Compute Stick (NCS) still leave room for improvement in terms of power consumption.

Neuromorphic hardware adopts the computing paradigm of spiking neural networks (SNNs), emulating the event-driven computation of the biological brain, and can in principle achieve extremely low-power inference. However, there had previously been no systematic benchmarking of neuromorphic hardware on keyword spotting tasks, especially no quantitative comparison in terms of energy efficiency, accuracy, and latency. Intel Loihi, as a new-generation neuromorphic research chip, provides programmable spiking neural network support, but its energy-efficiency advantage on practical tasks such as keyword spotting had not yet been validated.

The key challenges this paper aims to solve include:
- How to convert a traditional continuous-valued neural network into an equivalent spiking neural network while maintaining detection accuracy
- How to conduct a fair, systematic energy-efficiency comparison across multiple hardware platforms
- How to evaluate the scalability of neuromorphic hardware at different network sizes

## Methodology

### Overall Architecture Design

The paper designs a two-layer feedforward neural network as the base keyword spotter. The network is trained with the CTC (Connectionist Temporal Classification) loss, and the keyword "aloha" is chosen as the detection target. The overall pipeline consists of two stages:

**Stage One: Standard Network Training**
A standard continuous-valued feedforward neural network is trained in TensorFlow. The input features are MFCCs (Mel-frequency cepstral coefficients); the network consists of two hidden layers, and the output layer is trained end-to-end with the CTC loss. Introducing the CTC loss enables the model to learn sequence-level keyword discrimination, rather than merely frame-level classification.

**Stage Two: Spiking Network Conversion**
The trained standard network is converted into a spiking neural network using the Nengo deep-learning toolkit developed by Applied Brain Research (ABR). The core principle of the conversion is to map continuous-valued activations to firing rates — that is, to use the frequency of neuron spike firing to encode the activation strength of the original network. This conversion method does not require retraining; instead, the weights are transferred directly into the spiking neuron model.

### Key Technical Details

- **Input features**: 13-dimensional MFCC features at a 100 Hz frame rate, a standard acoustic feature configuration commonly used in keyword spotting
- **Spike encoding**: the LIF (Leaky Integrate-and-Fire) neuron model is adopted, with the membrane time constant and threshold tuned to match the activation distribution of the original network
- **CTC output processing**: in the spiking network, the probabilities of the CTC blank symbol and the keyword labels are computed by accumulating spike counts

### Benchmarking Methodology

The paper performs systematic benchmarking on the following five hardware platforms:
1. **Intel Loihi** — a neuromorphic research chip with 128 neuromorphic cores, each supporting up to 1024 neurons
2. **Intel Xeon E5-2630** — a server-grade CPU, representing the traditional high-performance computing platform
3. **NVIDIA Quadro K4000** — a discrete GPU, representing the graphics-accelerated computing platform
4. **NVIDIA Jetson TX1** — an embedded AI platform, representing mobile deep-learning inference platforms
5. **Intel Movidius NCS** — a dedicated neural compute stick, representing low-power inference accelerators

The measured metrics include: energy per inference (joules), inference throughput (inferences per second), operating power (watts), and detection accuracy (true positive rate and true negative rate). To ensure a fair comparison, all platforms run the same network architecture and input data.

## Main Contributions

1. **First systematic benchmark**: This is the first comprehensive benchmark of a keyword spotting task on the Intel Loihi neuromorphic processor, providing a direct energy-efficiency comparison between neuromorphic hardware and a variety of traditional computing platforms. Previous work had mostly been limited to simple pattern-recognition tasks and did not address continuous-signal scenarios such as speech processing.

2. **Validation of extreme energy efficiency**: Loihi is shown to consume only 0.00027 joules per inference — 5.6x lower than the Movidius NCS (0.0015 J), 20.7x lower than the Jetson TX1 (0.0056 J), 23.3x lower than the CPU (0.0063 J), and 110.4x lower than the GPU (0.0298 J). This result experimentally confirms the enormous potential of the neuromorphic computing paradigm in terms of energy efficiency.

3. **Accuracy preservation**: The spiking network achieves a true positive rate of 93.8%, slightly higher than the 92.7% of the non-spiking version, while the true negative rate remains identical at 97.9%. This shows that the conversion to spikes not only loses no accuracy but in fact improves slightly, due to a regularization effect introduced by the spiking neurons.

4. **Network-size scalability analysis**: It is shown that as the network size grows, Loihi maintains a stable inference rate (>100 inferences per second) thanks to its massively parallel architecture, whereas the Movidius NCS's inference speed drops sharply as the network grows, eventually falling below the threshold required for real-time processing. This finding has important implications for real deployment scenarios.

## Experimental Results

### Dataset
- Keyword: "aloha"
- Data scale: 192 utterances from 96 speakers
- Evaluation protocol: leave-one-speaker-out cross-validation

### Energy-Efficiency Comparison (Core Result)

| Platform | Operating Power (W) | Inference Rate (per second) | Energy per Inference (J) |
|----------|--------------------|-----------------------------|--------------------------|
| Intel Loihi | 0.029 | 296 | 0.00027 |
| Movidius NCS | 0.150 | 100 | 0.0015 |
| Jetson TX1 | 0.560 | 100 | 0.0056 |
| CPU (Xeon) | 12.800 | ~2000 | 0.0063 |
| GPU (K4000) | 14.900 | ~500 | 0.0298 |

### Detection Accuracy
- **Spiking network (Loihi)**: true positive rate 93.8%, true negative rate 97.9%
- **Non-spiking network**: true positive rate 92.7%, true negative rate 97.9%
- The spiking network's accuracy is slightly better than the non-spiking version, which can be attributed to the stochasticity of spike firing introducing an implicit regularization effect

### Scalability Experiments
- When the network size is increased, Loihi scales in parallel by activating more neural cores, and its inference speed is essentially unaffected
- The Movidius NCS has limited computational resources; as the network grows, its inference speed drops sharply and ultimately cannot meet real-time requirements

## Limitations and Future Work

### Technical Limitations of the Method
- **Small dataset scale**: The evaluation uses only 192 utterances from 96 speakers, far too little to represent the diversity of real deployment scenarios. Actual production environments must handle data from thousands or even tens of thousands of speakers.
- **Single-keyword restriction**: Only the single keyword "aloha" is tested; the method's performance on multiple keywords or longer keyword phrases is not validated. Differences in phonetic composition and duration across keywords may affect the effectiveness of spike encoding.
- **Power-measurement precision**: The paper adopts a conservative power-measurement methodology (reporting the highest power reading measured on the Loihi chip), which may in fact overestimate Loihi's dynamic power, since the power consumption of neuromorphic chips is highly dependent on network activity.
- **Software-stack differences**: The different platforms enjoy different degrees of software optimization (TensorFlow vs. Nengo vs. the Movidius SDK), introducing some bias into what is intended as a pure hardware energy-efficiency comparison.

### Shortcomings of the Experimental Design
- No noise-robustness testing was performed; all evaluations may have been conducted under relatively clean conditions
- No comparison with more complex KWS models based on RNNs or CNNs
- Detection performance on continuous speech streams was not evaluated

### Future Improvement Directions
- Explore training spiking neural networks directly on neuromorphic hardware (rather than converting), which may further improve performance
- Extend the method to multi-keyword detection and continuous speech recognition scenarios
- Validate on larger, more challenging datasets (such as Google Speech Commands)
- Leverage the event-driven nature of neuromorphic hardware to explore spike-based audio feature extraction

### Implications for the KWS Field
- Neuromorphic hardware offers a brand-new ultra-low-power solution for always-on KWS, particularly suitable for battery-powered IoT devices
- The approach of mapping spiking neural networks onto hardware can be generalized to other types of neuromorphic chips (such as IBM TrueNorth and BrainChip Akida)
- As neuromorphic hardware matures and enters mass production, KWS systems are expected to achieve order-of-magnitude improvements in energy efficiency
