# Federated Learning for Keyword Spotting

- **Authors/Affiliations**: David Leroy, Alice Coucke, Thibaut Lavril, Thibault Gisselbrecht, Joseph Dureau (Snips.ai, now Sonos)
- **Date**: October 2018 (arXiv:1810.05512)
- **Link**: https://arxiv.org/abs/1810.05512
- **Keywords**: federated learning, keyword spotting, wake word detection, privacy preservation, FedAvg, distributed training

## Problem Statement

Voice assistants (such as Amazon Alexa, Google Home, and Apple Siri) need to continuously listen to ambient audio in order to detect a wake word. Training and improving these systems typically requires collecting large amounts of user speech data, including wake word triggers and surrounding ambient audio. This raises serious privacy concerns:

**Privacy Background and Field Pain Points**
- Audio of users' home environments may contain private information such as sensitive conversations and personal habits
- Centralized data collection means users must trust tech companies to properly safeguard their voice data
- Privacy incidents in recent years (such as smart speaker recordings being reviewed by human listeners) have intensified public concern over voice data collection
- Privacy regulations such as GDPR impose strict legal requirements on the handling of personal data

**Shortcomings of Traditional Training Methods**
- Centralized training requires uploading all user data to cloud servers, violating the principle of data minimization
- Different users' environments (quiet/noisy, near-field/far-field, different accents) vary enormously, yet centralized training cannot personalize the model for each user
- Data transfer costs are high, especially in bandwidth-constrained environments

**Key Challenges This Paper Aims to Solve**
- How to train and continuously improve a wake word detection model without collecting users' raw voice data
- How to achieve efficient model aggregation in a heterogeneous distributed environment
- How to minimize on-device computation and communication overhead

## Methodology

### Overall Architecture Design

The paper applies Federated Learning (FL) to training the "Hey Snips" wake word detector from Snips. The core idea of federated learning: data stays on the users' devices, and only model parameter updates are shared.

**Federated Learning Framework**

**1. Global Model Initialization**
- Initialize a keyword detection neural network model on the server side
- The model structure is constrained by the computational limits of embedded devices: 200K parameters, 20 MFLOPS of computation
- Model architecture: a small CNN-based network (see Snips' earlier work for the specific structure)

**2. On-Device Local Training**
Each device participating in federated learning (called a client) performs:
1. Downloads the current global model parameters from the server
2. Runs several steps of SGD updates using locally collected speech data (containing "Hey Snips" positive samples and ambient audio negative samples)
3. Uploads the updated model parameters (or parameter deltas) back to the server
4. The number of devices participating in each round of training is a subset of all devices (simulating realistic participation rates)

**3. Server-Side Model Aggregation**
After collecting parameter updates from multiple devices, the server uses an aggregation strategy to produce a new global model.

### Adaptive Federated Averaging

The paper's key technical innovation is an adaptive averaging strategy inspired by the Adam optimizer.

**Standard FedAvg (Federated Averaging)**
$$\theta_{t+1} = \sum_{k=1}^{K} \frac{n_k}{N} \theta_t^k$$
where $K$ is the number of participating devices, $n_k$ is the amount of data on the $k$-th device, and $N$ is the total amount of data. FedAvg computes a weighted average of the devices' model parameters, weighted by data volume.

**Adaptive FedAvg (this paper's improvement)**
Standard FedAvg uses the same averaging weights for all parameter dimensions. Inspired by the adaptive learning rates in the Adam optimizer, the paper proposes using different update magnitudes for different parameter dimensions:
- More conservative updates for parameter dimensions with large variation (analogous to the second-moment estimate in Adam)
- More aggressive updates for parameter dimensions with small variation
- This amounts to adaptive step-size control in parameter space

The concrete implementation draws on Adam's core formulas:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_{t+1} = \theta_t - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$

where $g_t$ is the weighted average of the device updates, and $m_t$ and $v_t$ are the exponential moving averages of the first and second moments, respectively.

### Communication Efficiency Optimization
- Each device uploads the full model parameters every round (200K parameters * 4 bytes = 800KB)
- The total uplink communication cost per user is estimated at about 8MB (spread across multiple communication rounds)
- This communication volume is acceptable for the network connections of modern smart home devices

### The "Hey Snips" Dataset
The paper open-sourced the "Hey Snips" wake word dataset:
- Contains a large number of "Hey Snips" positive samples recorded by a large population
- Contains various ambient audio as negative samples
- Purpose: serves as a benchmark dataset for centralized training while simulating the distributed data distribution of federated learning

## Main Contributions

1. **First work to apply federated learning to speech KWS**: pioneering the extension of federated learning from image classification and language modeling to the field of speech keyword spotting, demonstrating the feasibility of FL in speech AI.

2. **Adaptive aggregation strategy**: proposed the Adam-inspired adaptive FedAvg algorithm, which significantly reduces the number of communication rounds required through parameter-level adaptive updates. Compared with standard weighted averaging, convergence speed is greatly improved.

3. **Communication cost quantification**: provided a detailed estimate of 8MB of uplink communication cost per user, showing that the communication overhead of federated learning is acceptable in smart home scenarios.

4. **Open-sourced the "Hey Snips" dataset**: released a publicly available wake word dataset, promoting transparency and reproducibility in wake word detection research. The dataset later became one of the standard benchmarks for wake word detection research.

5. **Validation of the privacy-preserving paradigm**: demonstrated a complete pipeline for training a high-quality wake word detector without ever touching users' raw voice data, laying the foundation for privacy-preserving speech AI.

## Experimental Results

### Experimental Setup
- Model constraints: 200K parameters, 20 MFLOPS
- Simulated federated environment: a crowdsourced dataset used to simulate distributed devices
- Evaluation metrics: detection accuracy, communication rounds, communication volume

### Core Results

**Convergence Speed Comparison**
- Adaptive FedAvg converges faster than standard weighted model averaging, requiring fewer communication rounds to reach the same accuracy
- By reducing oscillation and accelerating convergence, the adaptive strategy significantly lowers the overall communication cost

**Model Quality**
- The federatedly trained model matches the centrally trained model in accuracy
- This demonstrates that federated learning does not sacrifice model quality on the keyword spotting task

**Communication Overhead**
- Total uplink communication per user: about 8MB
- Per-round communication: about 800KB (200K parameters * 4 bytes)
- Communication rounds: the adaptive strategy significantly reduces the number of rounds required

**Dataset Statistics**
- The "Hey Snips" dataset was publicly released, containing positive and negative samples from a large number of speakers
- The dataset can be used for benchmark comparisons against centralized training

## Limitations and Future Work

### Technical Limitations of the Method
- **Simulated federated environment**: the experiments use crowdsourced data to simulate the federated setting rather than real distributed devices. Device heterogeneity in real scenarios (computational capability, network latency, data distribution) may be more complex.
- **Model capacity limit**: the 200K-parameter and 20 MFLOPS constraints may leave the model with insufficient capacity to handle all acoustic conditions. More advanced model architectures may require re-evaluating the efficiency of federated learning.
- **Single keyword**: evaluated only on "Hey Snips", without validating multi-keyword or long-phrase wake scenarios.

### Security Considerations
- Robustness against malicious participants (such as poisoning attacks and backdoor attacks) was not evaluated
- Stronger privacy protection mechanisms such as Differential Privacy were not discussed
- On-device model updates may leak information about users' data (membership inference attacks)

### Future Improvement Directions
- Incorporate differential privacy to provide formal privacy guarantees
- Explore asynchronous federated learning to accommodate device heterogeneity and unavailability
- Introduce personalized federated learning to customize models for different users' acoustic environments
- Extend federated learning to the entire voice assistant system (including ASR and NLU)
- Study the application of federated learning in continual learning and online adaptation

### Implications for the KWS Field
- Federated learning offers a feasible technical path toward resolving the privacy dilemma in speech AI
- The idea of the adaptive aggregation strategy can be generalized to other distributed machine learning settings
- Open-sourcing the "Hey Snips" dataset lowered the barrier to entry for wake word detection research
- As privacy regulations tighten, the application of federated learning in speech AI will become increasingly important
- Snips' work inspired a great deal of subsequent research applying federated learning to speech technology
