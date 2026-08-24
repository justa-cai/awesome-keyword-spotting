# Performance-Oriented Neural Architecture Search

**Authors/Affiliations**: Liam O'Neil, Simon OSuilleabhain, Md. Asifuzzaman Jishan, Rishabh Mehra, Darragh Nash, Joe Timoney (Trinity College Dublin)

**Date**: January 2020 (arXiv:2001.02976)

**Link**: https://arxiv.org/abs/2001.02976

**Keywords**: Neural Architecture Search, Performance Estimation, Keyword Spotting, Model Optimization, Surrogate Model

## Problem Statement

The core bottleneck of Neural Architecture Search (NAS) lies in the fact that evaluating the performance of each candidate architecture requires full training, which incurs extremely high computational costs. In the field of Keyword Spotting (KWS), a typical NAS workflow involves:
- A search space containing thousands to millions of candidate architectures
- Training each architecture takes several hours (even on GPUs)
- A complete search may require hundreds or even thousands of GPU hours

This "train-then-evaluate" paradigm severely limits the practicality of NAS. If the performance of an architecture could be accurately predicted without full training, the search process could be significantly accelerated.

Performance Predictors are a promising approach: they train a Surrogate Model to learn the mapping from architecture descriptions to performance metrics, thereby enabling rapid evaluation of candidate architectures without the need for training.

## Methodology

### Architecture Encoding

Neural network architectures are encoded into numerical vectors, which serve as inputs to the surrogate model:

**Encoding Scheme**:
- The information of each layer in the architecture is encoded into a fixed-dimensional vector.
- The encoding includes: operation type (convolution, pooling, etc.), parameter configuration (kernel size, number of channels, etc.), and connection patterns (skip connections, etc.).
- Discrete choices are represented using one-hot encoding or embedding encoding.
- The concatenation of encodings for all layers forms a complete architecture description vector.

### Surrogate Model

**Model Selection**:
- Simple regression models (such as MLPs, Random Forests) or neural networks are used.
- Input: Architecture encoding vector.
- Output: Predicted performance metric (e.g., classification accuracy).

**Training Data Construction**:
- A set of architectures is randomly sampled from the search space.
- These architectures are fully trained to obtain their true performance.
- Pairs of (architecture encoding, true performance) constitute the training set for the surrogate model.

**Iterative Refinement**:
- The initial surrogate model is based on a small amount of training data.
- As the search progresses, data from newly evaluated architectures are continuously added to the training set.
- The surrogate model is updated continuously, leading to improving prediction accuracy.

### Search Process

1. **Initialization**: Randomly sample N architectures, fully train them, and obtain their true performance.
2. **Surrogate Training**: Train the surrogate model using the current data.
3. **Candidate Generation**: Generate a large number of candidate architectures from the search space.
4. **Surrogate Evaluation**: Use the surrogate model to rapidly predict the performance of all candidates.
5. **Selection**: Select the top K candidates with the highest surrogate-predicted performance.
6. **Verification**: Fully train these K candidates to obtain their true performance.
7. **Update**: Add the new data to the training set and return to step 2.
8. **Termination**: Return the optimal architecture after reaching the search budget.

### KWS Search Space

A search space containing various convolutional operations is defined:
- Standard convolutions (with different kernel sizes)
- Depthwise separable convolutions
- Pooling operations
- Skip connections

## Main Contributions

1. **Surrogate Model-Based NAS**: Proposes a method to accelerate NAS search using a performance predictor (surrogate model), significantly reducing the number of architectures that need to be fully trained. This shifts the paradigm from "training all candidates" to "training a small number of candidates + surrogate prediction for a large number of candidates."

2. **Efficient Performance Prediction**: The surrogate model can predict architecture performance with high accuracy, achieving a high correlation coefficient with true training performance, making the search process more reliable.

3. **Application in KWS**: Successfully applies surrogate model-based NAS to keyword spotting tasks, discovering architectures that are comparable to or better than hand-designed ones.

4. **Significant Reduction in Search Cost**: Compared to standard NAS methods that require training thousands of architectures, this method only requires fully training dozens to hundreds of architectures.

## Experimental Results

### Experimental Setup
- Dataset: Google Speech Commands
- Search Space: Various convolutional operations and connection patterns
- Initial Training Set: 50-100 randomly sampled architectures
- Search Iterations: 5-10 rounds

### Main Results
- **Prediction Accuracy**: There is a high correlation between the surrogate model's performance predictions and the true training performance.
- **Discovered Architectures**: The search guided by the surrogate model discovered KWS architectures that are comparable to or better than hand-designed ones.
- **Search Efficiency**: The total search cost is reduced by an order of magnitude compared to brute-force search.
- **Iterative Improvement**: As search iterations proceed, the surrogate model's predictions become more accurate, and the performance of discovered architectures gradually improves.

### Surrogate Model Analysis
- MLP surrogate models perform well when architecture encodings are relatively simple.
- As the search space increases, more initial training data is required to establish an accurate surrogate model.
- The reliability of the surrogate model's predictions is lower in sparse data regions of the search space.

## Limitations and Future Work

### Methodological Limitations
- **Initial Training Data Requirement**: The surrogate model requires an initial set of "train-evaluate" data, the acquisition of which still consumes computational resources.
- **Prediction Reliability**: The surrogate model may provide inaccurate predictions in regions where training data coverage is insufficient.
- **Search Space Dependency**: The definition of the search space still requires domain expertise.
- **Limited Comparison with SOTA NAS Methods**: There is insufficient systematic comparison with state-of-the-art NAS methods from the same period (such as DARTS, ENAS).
- **Generalizability**: It has not been verified whether a surrogate model trained on one search space can be transferred to other search spaces.

### Future Directions
- Research Zero-Cost NAS methods, using performance metrics that do not require training (such as parameter count, FLOPs, gradient information) as architecture scores.
- Explore cross-search-space transfer of surrogate models to reduce the initial training overhead for each new search.
- Combine differentiable NAS and surrogate models into hybrid search strategies.
- Research multi-objective surrogate models that simultaneously predict multiple metrics such as accuracy, latency, and energy consumption.
- Extend surrogate model methods to more speech tasks and larger-scale search spaces.
