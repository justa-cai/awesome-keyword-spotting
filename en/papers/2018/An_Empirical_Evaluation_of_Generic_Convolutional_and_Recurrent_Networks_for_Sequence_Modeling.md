# An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling

- **Authors/Affiliations**: Shaojie Bai, J. Zico Kolter, Vladlen Koltun (Carnegie Mellon University, Computer Science Department; Intel Labs)
- **Date**: March 2018 (arXiv:1803.01271)
- **Link**: https://arxiv.org/abs/1803.01271
- **Keywords**: temporal convolutional network, LSTM, GRU, sequence modeling, causal convolution, dilated convolution, TCN

## Problem Statement

Recurrent neural networks (RNNs), especially LSTMs and GRUs, have long been regarded by deep learning practitioners as the default architecture choice for sequence modeling tasks. This preference stems from a theoretical consideration: through recurrent connections, RNNs can in theory maintain memory of unbounded length. However, recent preliminary studies suggest that convolutional architectures may outperform recurrent networks on certain sequence tasks.

**Field Pain Points**
- Sequence modeling is the foundation of many AI tasks (speech processing, time-series forecasting, language modeling, etc.); choosing a suitable architecture is critical to the performance of these tasks
- The sequential-dependence nature of RNNs makes training hard to parallelize, resulting in low training efficiency
- The gating mechanisms of LSTM and GRU increase model complexity and the difficulty of hyperparameter tuning
- The deep learning community's default assumption that "sequence modeling = recurrent networks" has never received systematic empirical validation

**Key Challenges This Paper Aims to Solve**
- Is there a systematic difference between convolutional and recurrent architectures in sequence modeling performance?
- Is this difference consistent across tasks?
- In which aspects (accuracy, training efficiency, memory length) do convolutional architectures hold advantages or disadvantages?

## Methodology

### Temporal Convolutional Network (TCN)

The paper proposes a generic Temporal Convolutional Network (TCN) architecture as the representative of convolutional sequence modeling, and systematically compares it with generic LSTM and GRU.

**Core Design of TCN**

1. **Causal Convolution**
   - The output at time step $t$ depends only on inputs at time step $t$ and earlier
   - Ensures the model does not "peek" at future information, satisfying the basic requirement of sequence modeling
   - Implemented by shifting the standard convolution along the temporal dimension

2. **Dilated Convolution**
   - Inserts holes (dilation factor) between the elements of the standard convolution kernel, enlarging the receptive field exponentially
   - The dilation factor of the $i$-th layer is $d = 2^i$ (1, 2, 4, 8, 16, ...)
   - The receptive field size is $O(2^L)$ ($L$ being the number of layers), achieving an exponential receptive field with logarithmic network depth
   - This allows TCN to cover very long temporal spans at a reasonable computational cost

3. **Residual Connections**
   - Each TCN block contains two dilated causal convolution layers + a residual connection
   - Residual connections ensure that gradients can propagate directly into deep layers, alleviating the vanishing gradient problem
   - When the input and output channel counts differ, a 1x1 convolution is used for dimension matching

4. **Weight Normalization**
   - Normalizes the convolution kernel weights to accelerate training convergence
   - Unlike Batch Normalization, weight normalization introduces no dependence on batch statistics

5. **Dropout Regularization**
   - Dropout is added after each dilated convolution layer to prevent overfitting

**Mathematical Formulation of a TCN Block**

For the $i$-th TCN block:
$$o_i = \text{ReLU}(\text{Conv1D}_{d=2^i}(x_i))$$
$$o_i = \text{Dropout}(o_i)$$
$$o_i = \text{ReLU}(\text{Conv1D}_{d=2^i}(o_i))$$
$$o_i = \text{Dropout}(o_i)$$
$$x_{i+1} = \text{ReLU}(o_i + x_i)$$  (residual connection)

### Comparative Methodology

**Comparison Principles**
- Use generic, non-customized architecture configurations rather than architectures fine-tuned for specific tasks
- TCN, LSTM, and GRU use the same number of parameters (or as close as possible)
- A unified hyperparameter search protocol (learning rate, number of layers, number of hidden units, etc.)
- Multiple runs with different random seeds averaged to ensure statistical significance

**Evaluation Tasks**

1. **Polyphonic Music Modeling**
   - Datasets: JSB Chorales, MuseData, Nottingham, PianoMidi
   - Task: given a past sequence of notes, predict the note presence probability at the next time step
   - Evaluation metric: negative log-likelihood (NLL)

2. **Word-level Language Modeling**
   - Datasets: Penn Treebank (PTB), WikiText-2
   - Task: given a past sequence of words, predict the probability distribution of the next word
   - Evaluation metric: perplexity

3. **Character-level Language Modeling**
   - Datasets: Penn Treebank (PTB), text8
   - Task: given a past sequence of characters, predict the probability distribution of the next character
   - Evaluation metric: BPC (Bits Per Character)

4. **Synthetic Stress Tests**
   - Synthetic tasks specifically designed to test the models' memory capacity
   - E.g., sequence copying/memory tasks that require looking back N steps at the input to predict correctly
   - Evaluation metric: accuracy

## Main Contributions

1. **TCN as a generic sequence modeling baseline**: proposes TCN, a clean, generic, and reproducible convolutional architecture for sequence modeling. TCN's design principles (causal convolution + dilation + residual) are simple and clear, easy to implement and tune.

2. **Systematic evidence that convolution outperforms recurrence**: across all evaluation tasks (music modeling, language modeling, synthetic stress tests), TCN consistently outperforms LSTM and GRU. This is the first systematic evidence provided over such a broad range of tasks, challenging the prevailing assumption that "sequence modeling = recurrent networks".

3. **Longer effective memory**: although RNNs theoretically have unbounded memory, in actual training TCN exhibits a longer effective memory length. The exponential growth of the dilated convolution's receptive field allows TCN to reliably "remember" long-range dependencies, whereas the long-term memory of LSTM/GRU tends to decay in practice.

4. **Greater transparency and flexibility**: TCN's architecture is simpler than RNNs — there are no complex gating mechanisms, hidden-state initialization issues, and the like. TCN's receptive field size can be precisely controlled through the number of layers and the dilation factor, rather than being implicitly learned during training.

5. **Open-source code**: released the complete code implementation at github.com/locuslab/TCN, facilitating reproduction and extension in follow-up research.

## Experimental Results

### Polyphonic Music Modeling

| Dataset | TCN | LSTM | GRU |
|--------|-----|------|-----|
| JSB Chorales | **-8.09** | -8.44 | -8.46 |
| MuseData | **-7.13** | -7.45 | -7.47 |
| Nottingham | **-3.79** | -4.05 | -4.07 |
| PianoMidi | **-7.67** | -7.86 | -7.89 |

(The metric is NLL; lower is better)

### Language Modeling

**Word-level (perplexity, lower is better)**
- PTB: TCN outperforms LSTM and GRU
- WikiText-2: TCN outperforms LSTM and GRU

**Character-level (BPC, lower is better)**
- PTB: TCN outperforms LSTM and GRU
- text8: TCN outperforms LSTM and GRU

### Synthetic Stress Tests
- On sequence tasks requiring memory of up to 600 steps, TCN's accuracy is significantly higher than LSTM and GRU
- TCN's effective memory length grows in a controlled manner with network depth and dilation factor
- LSTM/GRU degrade on extremely long-memory tasks

### Training Efficiency
- TCN's parallelizable nature makes its training significantly faster than LSTM/GRU (especially when using GPUs)
- The sequential dependence of LSTM/GRU limits the degree of training parallelization

## Limitations and Future Work

### Technical Limitations of the Method
- **Memory consumption**: when processing very long sequences, TCN — due to its ever-growing receptive field — needs to store the activations of all intermediate layers, so its memory consumption can exceed that of RNNs (which only need to store hidden states)
- **Transformer not considered**: when the paper was published, Transformers had just emerged, and self-attention mechanisms were not included in the comparison. Follow-up research (such as the success of Transformers in language modeling) indicates that attention mechanisms may further surpass TCN.
- **Sequence generation tasks**: TCN is mainly suitable for sequence-to-sequence prediction and classification tasks; for conditional sequence generation (such as machine-translation decoders), autoregressive TCN may be less efficient than RNNs.

### Shortcomings of the Experimental Design
- Speech-specific tasks such as speech recognition / keyword spotting were not evaluated (although TCN's design principles are generalizable)
- TCN's application in reinforcement learning and sequential decision-making was not explored
- No in-depth analysis of the theoretical differences between TCN and RNN in terms of gradient flow

### Future Improvement Directions
- Explore combining TCN with attention mechanisms (e.g., local attention + global convolution)
- Study adaptive dilation factors that dynamically adjust the receptive field according to the task
- Apply TCN to concrete domains such as speech recognition and time-series forecasting
- Explore TCN implementations for streaming / online scenarios

### Implications for the KWS Field
- TCN provides an RNN-alternative architecture for keyword spotting — causal convolutions naturally satisfy the requirements of streaming processing
- The efficient receptive-field expansion of dilated convolutions allows TCN to cover the temporal span required by keyword spotting
- TCN's parallelizable training enables faster model iteration
- This work inspired extensive follow-up research using 1D convolutional architectures in KWS (such as Temporal Convolution, Depthwise Separable Conv)
- The CMU research showed that convolutional architectures have potential on temporal tasks such as speech on par with or even superior to RNNs, which changed the default choice in KWS model design
