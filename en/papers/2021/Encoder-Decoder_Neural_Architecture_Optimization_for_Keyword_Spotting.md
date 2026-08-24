# Encoder-Decoder Neural Architecture Optimization for Keyword Spotting

- **Authors/Affiliations**: Mo et al. (including Shikhar Bhushan) - University of Alberta; University of Montreal
- **Date**: 2021.06
- **Link**: https://arxiv.org/abs/2106.02738
- **Keywords**: Neural Architecture Search, Encoder-Decoder, Keyword Spotting, Architecture Optimization, AutoML, Latent Space, Reinforcement Learning

## Problem Statement

Designing an optimal neural network architecture for Keyword Spotting (KWS) is a process that requires significant domain expertise and iterative trial-and-error tuning. The characteristics of different keyword tasks (keyword length, phonetic complexity, acoustic conditions) may require different network architectures to achieve optimal performance. For example, short keywords (e.g., "Hey Siri") might be better suited for shallow, wide networks, while longer phrases (e.g., "OK Google") may require deeper networks to model longer temporal dependencies.

Neural Architecture Search (NAS) can automate this process, but traditional NAS methods face two core challenges:

1. **Low Search Efficiency**: Standard NAS methods (such as RL-based NASNet or evolution-based AmoebaNet) require training and evaluating a large number of candidate architectures, resulting in extremely high computational costs (thousands to tens of thousands of GPU hours). Even with One-Shot NAS methods (such as DARTS) that significantly reduce search costs, search times still span several days. In practical KWS development, such search costs remain unacceptable.

2. **Underutilization of the Search Space**: Traditional methods typically sample architectures randomly or heuristically within a discrete search space, failing to learn and exploit structured patterns within the architecture space. For instance, if a certain combination of convolutional kernel sizes performs well across most KWS tasks, traditional NAS cannot leverage this prior knowledge to accelerate the search for subsequent tasks.

The key problem this paper addresses is: How to efficiently search for the optimal KWS architecture by learning a continuous representation (latent representation) of the architecture space, while incorporating hardware deployment constraints (latency, model size) into the search process. The core insight of this problem is that there is learnable structure in the architecture space—architectures with similar performance should be close to each other in a certain representation. Leveraging this structure can significantly improve search efficiency.

## Methodology

### Overall Framework
The paper proposes an Encoder-Decoder-based architecture optimization framework. The core idea is to transform the architecture search problem into an optimization problem in a continuous latent space, rather than a search in a discrete architecture space. The framework consists of the following components:

1. **Encoder**: Maps discrete architecture descriptions to continuous latent vectors.
2. **Decoder**: Generates/reconstructs architectures from latent vectors.
3. **Performance Predictor**: Predicts architecture performance in the latent space.
4. **Search Strategy**: Efficiently searches for high-performance architectures in the latent space.

### Graph Representation of Architectures
Each neural network architecture is represented as a Directed Acyclic Graph (DAG), where:
- **Nodes** represent feature maps (tensors).
- **Edges** represent operations (e.g., convolution, pooling, skip connections, etc.).
- The input of each node comes from the output of previous nodes, processed through the operation on the edge to obtain the feature map of that node.
- The topological structure of the graph encodes the connection patterns of the architecture.

This graph representation can fully describe arbitrary CNN architectures, including branching structures, skip connections, and different connection patterns.

### Design of the Encoder
The encoder's function is to map discrete architecture graphs into fixed-length continuous vectors (latent representations):

- **Graph Neural Network (GNN) Encoding**: Uses a Graph Neural Network to encode the architecture graph. The GNN iteratively aggregates neighbor information for each node through a message-passing mechanism:
  - For each node $v$, aggregate the representations of its neighbor nodes $\{u\}$: $m_v = \text{Aggregate}(\{h_u : u \in N(v)\})$
  - Update node representation: $h_v = \text{Update}(h_v, m_v)$
  - After $L$ rounds of message passing, each node contains structural information from its $L$-hop neighborhood.

- **Operation Embedding**: The operation type on each edge (e.g., 3x3 depthwise separable convolution, 5x5 max pooling, etc.) is mapped to a low-dimensional vector via an embedding layer, serving as edge features participating in the GNN's message passing.

- **Global Pooling**: By performing global average pooling or attention pooling on the final representations of all nodes, a fixed-length vector representation $z$ for the entire architecture is obtained.

The choice of the GNN encoder is critical—because the graph structure of the architecture contains important topological information (such as the pattern of residual connections, the location of downsampling), simple sequence encoders (such as LSTM or Transformer) cannot effectively capture these topological features.

### Design of the Decoder
The decoder reconstructs or generates architectures from latent vectors:

- **Autoregressive Generation**: Generates each component of the architecture step-by-step. Given a latent vector $z$, the decoder sequentially generates:
  1. Selection of input connections for each node (which predecessor nodes receive input from).
  2. Selection of the operation type on each edge.
  3. Determination of hyperparameters such as channel expansion factors.

- **Conditional Generation**: Architectures satisfying specific constraints can be generated given certain conditions. For example, given a target parameter count $P$ and target latency $L$, the decoder generates an architecture satisfying $\text{params(model)} \le P$ and $\text{latency(model)} \le L$.

- **Variational Latent Space**: Uses the framework of a Variational Autoencoder (VAE), where the encoder outputs the mean and variance of the latent distribution, and the decoder generates architectures from sampled latent vectors. This endows the latent space with better structured properties (continuity, completeness).

### Search Strategy in Latent Space
In the continuous latent space learned by the Encoder-Decoder, architecture search becomes efficient:

- **Predictor-Guided Search**:
  1. Train a performance predictor (such as a Multi-Layer Perceptron or Gaussian Process) on the latent space, mapping latent vectors $z$ to predicted architecture performance $p(z)$.
  2. The predictor uses existing (architecture, performance) pairs as training data.
  3. Use gradient ascent (since the latent space is continuous, gradients can be directly computed) or Bayesian optimization in the latent space to find regions with the highest predicted performance.
  4. The decoder decodes the optimal latent vector into a specific architecture description.

- **Evolutionary Search**: Perform mutation (adding Gaussian noise to latent vectors) and crossover (linear interpolation of two latent vectors) in the latent space to generate new candidate architectures. Due to the continuity and structured nature of the latent space, simple evolutionary operations can effectively explore the architecture space.

### Hardware-Aware Constraint Search
Hardware constraints can be incorporated into the search process, ensuring that discovered architectures are not only accurate but also practically deployable:

- **Latency Constraints**: Estimate the inference latency of each candidate architecture on target hardware (e.g., ARM Cortex-M series MCUs, DSPs). Latency estimation is achieved via lookup tables (predicting latency for different operations) or lightweight latency prediction models. Architectures that do not meet latency constraints are rejected.

- **Model Size Constraints**: Limit the number of parameters (typically <100K parameters) and/or multiply-accumulate operations (MADD, typically <10M).

- **Pareto Optimization**: Simultaneously optimize accuracy and hardware efficiency to find a set of Pareto-optimal architectures. Architectures on the Pareto frontier represent the highest accuracy achievable under given hardware constraints.

### Search Space Definition
The search space covers components commonly used in KWS:
- Different types of convolutions: Standard convolution, depthwise separable convolution, dilated/atrous convolution.
- Different convolution kernel sizes: 3x3, 5x5, 7x7.
- Different expansion ratios: Channel multiplication factors (1x, 2x, 4x).
- Pooling operations: Max pooling, Average pooling.
- Skip connections: Residual, None.
- Activation functions: ReLU, ReLU6, Swish.

The design of the search space follows the principle of "covering mainstream components while excluding obviously invalid options" to keep the search space size within a reasonable range.

## Main Contributions

1. **Introduction of Encoder-Decoder NAS for KWS**: First application of latent space-based architecture optimization methods to the field of keyword spotting. By learning continuous representations of architectures, the search process becomes more efficient and structured. Unlike traditional NAS methods that search in discrete spaces, the Encoder-Decoder method searches in a continuous space, leveraging gradient information to significantly improve search efficiency.

2. **Significant Reduction in Search Costs**: By replacing exhaustive search in discrete spaces with search in continuous latent spaces, the number of candidate architectures requiring training and evaluation is drastically reduced. Search costs are reduced from thousands of GPU hours in traditional NAS to tens of GPU hours, making NAS feasible in practical KWS development.

3. **Discovery of New Architectures Outperforming Hand-Designed Ones**: Architectures discovered by search on the GSC dataset surpass hand-designed models like classic DS-CNN. More importantly, architectures discovered by NAS exhibit design patterns different from human intuition (e.g., using large-kernel dilated convolutions in certain positions), broadening the思路 (thinking) for KWS architecture design.

4. **Hardware-Aware Architecture Search**: Integrating deployment constraints (latency, model size) into the search process ensures that discovered architectures are not only accurate but also practically deployable. This joint optimization avoids the sub-optimality of two-stage methods that "first search for high-accuracy architectures, then compress them to meet hardware constraints."

5. **Transferable Architecture Priors**: After training the Encoder-Decoder model on multiple KWS tasks, the learned latent space representations can be transferred to new KWS tasks, further accelerating architecture search for new tasks.

## Experimental Results

### Datasets
- **Google Speech Commands (GSC) v2**: 12-class and 35-class classification tasks.
- **Evaluation Metrics**: Classification accuracy, parameter count, multiply-accumulate operations (MADD), search time.

### Search Efficiency
- The search time of the Encoder-Decoder framework is significantly shorter than standard NAS methods (random search, evolutionary search)—reduced from typical 500+ GPU hours to approximately 20-50 GPU hours.
- The latent space predictor can accurately predict the performance of unseen architectures (rank correlation coefficient approximately 0.7-0.8), reducing the need for actual training and evaluation.
- After training the Encoder-Decoder model once, multiple architectures satisfying different hardware constraints can be generated without re-searching.

### Performance of Discovered Architectures
- **GSC 12-class Task**: Architectures discovered by NAS achieved approximately 96-97% accuracy, comparable to or slightly better than hand-designed DS-CNN-L.
- **GSC 35-class Task**: Architectures discovered by NAS achieved approximately 94-95% accuracy, consistently outperforming the DS-CNN series under the same parameter budget.
- Architectures discovered by search tend to use mixed convolution types (using different kernel sizes at different positions) and skip connection strategies (selectively using residual connections), rather than the uniform configurations found in hand-designed models.

### Results Under Hardware Constraints
- Under strict latency constraints (<5ms inference time), compact architectures maintaining approximately 95% accuracy were still discovered.
- Under strict parameter constraints (<20K parameters), the accuracy of discovered architectures was approximately 1-2% higher than DS-CNN-S.
- Pareto frontier analysis shows that the Encoder-Decoder method better explores the accuracy-efficiency trade-off space, generating smoother Pareto curves.

### Ablation Studies
- **Latent Space Dimension**: Medium-dimensional (32-64) latent representations achieve the best balance between search efficiency and prediction accuracy. Too low a dimension (8-16) creates an information bottleneck limiting expressiveness, while too high a dimension (128+) leads to insufficient training data for the predictor.
- **Encoder Architecture**: GNN encoders outperform simple sequence encoders (LSTM) and set encoders (DeepSets) because the graph structure of architectures contains important topological information (such as the pattern of residual connections).
- **Training Data Volume**: Increasing the number of initial architecture-performance pairs (from 100 to 1000) can continuously improve the quality of the latent space, but the gains begin to diminish after 500 pairs.
- **Search Strategy**: Gradient-guided search outperforms pure random search, and Bayesian optimization performs best under small evaluation budgets.

## Limitations and Future Work

### Technical Limitations
- **Limited Search Space Coverage**: The current search space is primarily based on CNN components and does not include emerging components such as RNNs (LSTM/GRU), Transformers, or self-attention mechanisms. Since 2021, attention-based KWS architectures (such as KWT, Conformer-KWS) have made significant progress; the absence of these components in the search space may have missed optimal architectures.
- **Requirement for Initial Architectures**: The Encoder-Decoder model requires a set of initial (architecture, performance) pairs for training. This introduces a dependency on the quality of the initial architecture set—if the initial set is not diverse enough, the latent space may fail to cover certain high-performance regions. Typically, hundreds to thousands of randomly sampled architectures and their performance evaluations are required to train the Encoder-Decoder.
- **Accuracy of Hardware Estimation**: Latency and power consumption estimates are based on lookup tables or lightweight prediction models, which may deviate from measurements on actual devices (error approximately 10-20%). Verification on target hardware is still necessary before actual deployment.
- **Coupling of Search Space and Task**: The search space is designed for the GSC dataset and may not be directly applicable to other KWS tasks (such as custom keywords, multi-language KWS). The search space may need adjustment when transferring to new tasks.

### Insufficiencies in Experimental Design
- Evaluation was conducted only on the Google Speech Commands dataset; the robustness of discovered architectures was not verified in noisy, far-field, or custom keyword scenarios.
- The interpretability of discovered architectures is limited—it is difficult to extract general design principles from the latent space analysis (e.g., rules such as "shallow wide networks should be used for short keywords").
- Inference performance was not verified on real edge hardware (MCU, DSP); latency estimates were based on models rather than actual measurements.
- Comparisons with other NAS methods (such as Once-for-All NAS, BigNAS) were not comprehensive.

### Directions for Future Improvement
- Expand the search space to include hybrid architectures (CNN + Attention/Transformer), enabling NAS to automatically discover optimal CNN-attention combinations.
- Explore zero-cost NAS proxies (such as SNIP, GraSP based on gradient information) to further accelerate search and reduce the overhead of training initial architecture sets.
- Jointly optimize architecture search with model compression (quantization, pruning) to directly search for quantization-friendly architectures that maintain high accuracy even at low precision.
- Investigate cross-task architecture transfer—how latent space knowledge learned in one KWS task can help accelerate search for new tasks.
- Insights for the KWS field: The Encoder-Decoder NAS framework provides a scalable foundation for the automated design of KWS models. Modeling architecture search as an optimization problem in a continuous latent space is a promising direction; in the future, it can be combined with hardware-software co-design to achieve automated architecture optimization tailored to specific chips.
