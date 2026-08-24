# Small-Footprint Keyword Spotting Using Graph Convolutional Networks

- **Authors/Affiliations**: Xi Chen, Shouyi Yin, Dandan Song, Peng Ouyang, Leibo Liu, Shaojun Wei (Tsinghua University)
- **Date**: December 2019 (ASRU 2019)
- **Link**: https://arxiv.org/abs/1912.05124
- **Keywords**: Keyword Spotting, Graph Convolutional Networks, Small Footprint, Google Speech Commands, Graph Representation Learning

## Problem Statement

Traditional CNN-based keyword spotting systems treat spectrograms as two-dimensional images, applying standard grid convolution operations. This "treating spectrograms as images" approach has fundamental limitations:

1. **Regular Grid Assumption**: CNNs assume that inputs are uniformly sampled on a regular grid (e.g., pixel matrices), but the relationship between frequency bands and time frames in audio spectrograms is not a simple spatial adjacency. Complex co-channel and resonance relationships exist between different frequency bands, which cannot be fully captured by regular square convolution kernels.
2. **Neglect of Inter-band Relationships**: The local receptive field of CNNs primarily captures patterns in adjacent frequency bands, but certain non-adjacent frequency bands in speech may have strong correlations (e.g., harmonic structures), which standard convolutions cannot directly model as long-range frequency relationships.
3. **Natural Expression of Graph Structure Relationships**: The time-frequency units in audio features essentially form a graph structure—the relationships between frequency bands, between time frames, and across time-frequency domains can be naturally represented by graph edges.

Therefore, the core question is: Can the flexible graph structure of Graph Convolutional Networks (GCNs) be utilized to more effectively model relationship patterns in audio features while maintaining a small model footprint suitable for edge devices?

## Methodology

This paper proposes a **Graph Convolutional Network (GCN)** architecture for keyword spotting.

### 1. Graph Construction

Construct a graph representation $G = (V, E)$ from audio features:
- **Vertices**: Nodes in the graph correspond to various dimensions or time-frequency units of the audio features. Each node carries information for that feature dimension.
- **Edges**: Edges capture relationships between nodes. The connectivity of edges defines the graph topology, determining how information propagates through the graph.
- **Adjacency Matrix**: The topology of the graph is encoded via the adjacency matrix $A$, where $A_{ij}$ represents the connection strength between node $i$ and node $j$.

### 2. Graph Convolution Operation

Graph convolution is a generalization of standard convolution to irregular graph structures:
- For each node, aggregate feature information from its neighboring nodes.
- Forward propagation of a graph convolution layer:

$$H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})$$

where $\tilde{A} = A + I$ is the adjacency matrix with self-connections added, $\tilde{D}$ is the degree matrix, $H^{(l)}$ is the node features at layer $l$, $W^{(l)}$ is the learnable weight matrix, and $\sigma$ is the activation function.

- Graph convolution achieves **weighted aggregation of neighbor information**, where each node obtains information from its neighbors via the graph structure to update its own feature representation.

### 3. GCN-KWS Architecture

The complete GCN-KWS system:
1. **Feature Extraction**: Extract acoustic features (e.g., MFCCs or spectrograms) from audio.
2. **Graph Construction**: Convert features into graph representations.
3. **Multi-layer Graph Convolution**: Stack multiple graph convolution layers to progressively aggregate information from higher-order neighbors.
4. **Graph-level Representation**: Aggregate node features into a graph-level representation via global pooling or readout operations.
5. **Classification Output**: A fully connected layer outputs the probability of keyword classes.

### 4. Key Differences from CNN

| Aspect | CNN | GCN |
|------|-----|-----|
| Data Structure | Regular Grid | Arbitrary Graph Structure |
| Neighborhood Definition | Fixed square window | Neighbors defined by graph edges |
| Relationship Modeling | Implicit (via convolution kernels) | Explicit (via graph topology) |
| Flexibility | Limited by grid structure | Flexible, allows definition of arbitrary relationships |

## Main Contributions

1. **Introduction of GCN to KWS Domain**: First to introduce Graph Convolutional Networks to the keyword spotting domain, providing a new paradigm beyond standard CNN/RNN methods. The flexible graph structure of GCNs can explicitly model relationship patterns in audio features, rather than relying on the implicit local receptive fields of CNNs.
2. **Graph-based Relationship Modeling**: Demonstrates that GCNs can effectively model the relational structure of audio features in KWS tasks—such as co-channel relationships between frequency bands and temporal dependencies between time frames—can be explicitly encoded via graph edges.
3. **Competitive Performance**: Achieves competitive accuracy on the Google Speech Commands dataset with a compact model size, proving the feasibility of the GCN architecture for KWS tasks.
4. **Effectiveness of Graph Representation**: Shows that graph-based representations can capture non-local relationships between frequency and time more effectively than grid-based convolutions.
5. **Published at ASRU 2019**, representing significant work by Tsinghua University in KWS architecture innovation.

## Experimental Results

- The proposed GCN-based model achieves competitive performance compared to CNN baselines on the Google Speech Commands dataset.
- The model maintains a small footprint suitable for deployment on resource-constrained devices.
- The flexibility of the graph structure enables the model to capture inter-feature relationships that standard CNNs cannot model.

## Limitations and Future Work

### Technical Limitations
- **Manual Design of Graph Construction**: The graph topology (how nodes and edges are defined) requires manual design or additional hyperparameter tuning. Different graph construction methods can lead to significant performance differences, and there is a lack of automated mechanisms for discovering optimal graph structures.
- **Limited Exploration of Graph Topologies**: There is limited systematic exploration of different graph topologies (e.g., fully connected graphs, k-nearest neighbor graphs, correlation-based graphs) and their impact on KWS performance.
- **Insufficient Analysis of Computational Overhead**: The computational overhead of graph operations (particularly sparse matrix multiplication) relative to standard convolutions has not been quantitatively analyzed for edge deployment. On hardware lacking graph computation optimization, GCNs may be less efficient than CNNs.

### Future Directions
- Research methods for automatically learning optimal graph topologies, such as adaptive graph construction based on attention mechanisms.
- Explore dynamic graph structures—adaptively adjusting graph connections based on input content, allowing the model to use sparse graphs for simple samples and dense graphs for complex samples.
- Evaluate the actual inference efficiency of GCN-KWS on hardware platforms that support sparse computation.
- Combine GCNs with CNNs to strike a balance between local feature extraction (CNN) and global relationship modeling (GCN).
- Explore the application of GCNs in multi-speaker KWS and speaker separation tasks, leveraging the inherent advantages of graph structures.
