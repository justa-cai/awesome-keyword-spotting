# Learning Efficient Representations for Keyword Spotting with Triplet Loss

- **Authors/Affiliations**: Alexey Kozlov, Ivan Novikov, Vladimir Bataev - Tomsk State University; NTR Labs
- **Date**: 2021.01
- **Link**: https://arxiv.org/pdf/2101.04792
- **Keywords**: Triplet Loss, Metric Learning, Keyword Spotting, Embedding Learning, Representation Learning, Prototypical Networks

## Problem Statement

Classification-based Keyword Spotting (KWS) models have a fixed number of output classes. Whenever a target keyword needs to be added, deleted, or modified, the entire model must be retrained. This limitation causes significant inconvenience in practical deployment: scenarios such as smart assistant vendors adding new wake words, users customizing personalized wake words, and multilingual expansion all require rapid keyword adaptation capabilities.

Metric Learning methods address this issue by learning a discriminative embedding space. In this embedding space, speech samples of the same keyword are close to each other (small intra-class distance), while samples of different keywords are far apart (large inter-class distance). When a new keyword needs to be added, one only needs to calculate the average embedding of a few samples of that keyword as a prototype, and then perform classification by comparing the distance between the input audio's embedding and all prototypes during inference. The entire process requires no retraining of the model.

Triplet Loss is one of the most classic and widely used loss functions in metric learning, first proposed in FaceNet for face recognition, where it achieved great success. However, applying it effectively to the KWS task requires solving specific technical challenges: the high variability of speech signals (significant differences for the same keyword across different speakers, speaking speeds, and noise conditions), the need for fine-grained distinction at the phoneme level (e.g., the difference between "yes" and "yeah" lies only in the final phoneme), and the proper calibration of the embedding space (to ensure reliable threshold-based decisions).

This paper systematically explores the application of metric learning methods based on triplet loss in KWS, investigating how to learn compact and highly discriminative keyword embedding representations to achieve flexible and accurate keyword detection and convenient adaptation to new keywords.

## Methodology

### Overall Framework
The framework proposed in the paper consists of three stages: embedding network training (using triplet loss) -> keyword prototype construction -> classification at inference. The embedding network maps speech segments to a D-dimensional embedding space, where distance metrics are used for keyword discrimination.

### Embedding Network Architecture
The backbone of the embedding network uses a CNN architecture to extract high-level representations from MFCC features:

- **Input Features**: 40-dimensional MFCC features, covering approximately 1 second of speech segments (approximately 49 time steps).
- **Backbone CNN**: Uses multi-layer Depthwise Separable Convolutions (DS-CNN style) as the backbone, including:
  - A first layer of standard convolution (to expand the initial receptive field)
  - Multiple layers of depthwise separable convolution blocks (each block contains depthwise convolution + pointwise convolution + BN + ReLU)
  - Global Average Pooling (maps variable-length feature maps to a fixed-length vector)
- **Embedding Projection**: The pooled features are projected into a D-dimensional embedding space through a fully connected layer (D is typically 64 or 128).
- **L2 Normalization**: The output embeddings are L2-normalized, mapping all embeddings to the unit hypersphere. Normalization makes Euclidean distance and cosine distance equivalent, simplifying subsequent distance calculations.

### Triplet Loss
The core idea of triplet loss is to learn the embedding space by comparing the relationships within triplets (anchor, positive sample, negative sample):

1. **Triplet Definition**:
   - **Anchor** $x_a$: An audio sample of a certain keyword
   - **Positive** $x_p$: A different audio sample of the same keyword as the anchor (possibly from a different speaker)
   - **Negative** $x_n$: An audio sample of a different keyword from the anchor

2. **Loss Function**:
   $L_{triplet} = \max(0, d(f(x_a), f(x_p)) - d(f(x_a), f(x_n)) + margin)$
   
   Where $f(.)$ is the embedding function, $d(.,.)$ is the Euclidean distance (equivalent to cosine distance after L2 normalization), and $margin$ is the margin parameter.
   
   The meaning of the loss is: when the distance between the anchor and the positive sample plus the margin is still less than the distance between the anchor and the negative sample, the loss for that triplet is zero (the margin requirement is already satisfied); otherwise, a positive loss is generated, pushing the model to shrink intra-class distances and expand inter-class distances.

3. **Margin Parameter**: The $margin$ controls the size of the margin in the embedding space. A larger margin (e.g., 0.5-1.0) requires a larger inter-class margin, producing a more discriminative embedding space, but also makes training more difficult. The paper's experiments found that $margin=0.2-0.5$ works best for the KWS task.

### Semi-Hard Negative Mining
The efficiency and effectiveness of triplet training heavily depend on the quality of the triplets. If negative samples are chosen randomly, most triplets will have zero loss (because random negative samples are usually far from the anchor), leading to sparse training signals. The paper adopts a semi-hard negative mining strategy:

- **Semi-hard negative**: Selects negative samples that satisfy the following conditions: $d(f(x_a), f(x_n)) > d(f(x_a), f(x_p))$ (i.e., the negative sample is farther from the anchor than the positive sample) and $d(f(x_a), f(x_n)) < d(f(x_a), f(x_p)) + margin$ (i.e., the negative sample is within the margin range).

- Semi-hard negatives provide stronger training signals than random negatives (they are "close" to violating the margin condition) without causing training instability like hardest negatives do.

- **Mining Frequency**: Every N training steps (e.g., every 100 steps), the embeddings of all samples are recalculated, and semi-hard negatives are re-mined.

### Keyword Prototype Construction
After training, a prototype embedding is constructed for each keyword class $k$:

- **Prototype Calculation**: Collect $N_k$ audio samples for that keyword and calculate their average embedding as the prototype:
  $c_k = \frac{1}{N_k} \sum f(x_i)$
  
- **Prototype Quality**: Prototypes calculated using more samples are more stable and representative. The paper suggests using at least 10-20 samples (from different speakers) for each keyword to calculate the prototype.

- **Online Update**: The prototype for a new keyword can be calculated at runtime without retraining the embedding network. This is the core advantage of metric learning methods.

### Classification at Inference
During inference, for an input audio $x$:

1. Calculate its embedding $f(x)$
2. Calculate the distance between $f(x)$ and all keyword prototypes $d(f(x), c_k)$
3. Classification Decision:
   - **Closed-set classification**: Select the class corresponding to the nearest prototype as the prediction result
   - **Open-set verification**: If the distance to the nearest prototype is less than a threshold $\theta$, it is classified as that keyword; otherwise, it is classified as "unknown"

### Embedding Space Analysis
The paper conducts an in-depth analysis of the learned embedding space:

- **t-SNE Visualization**: Shows that different keyword classes form clear clusters in the embedding space, with compact intra-cluster and well-separated inter-cluster structures.
- **Distance Distribution**: The distribution of distances between embedding pairs of the same keyword is clearly separated from the distribution of distances between embedding pairs of different keywords, validating the effectiveness of the triplet loss.
- **Embedding Dimension Analysis**: Systematically evaluates the impact of different embedding dimensions (32, 64, 128, 256) on KWS performance.

## Main Contributions

1. **Systematic application of triplet loss to learn compact keyword embedding representations**: Provides a systematic study of triplet loss in the KWS task, including a complete scheme for architecture design, training strategies, negative sample mining, and embedding space analysis.

2. **Demonstrates the flexibility of metric learning for keyword customization without retraining**: Experiments show that new keywords can be added through simple prototype calculation, without modifying or retraining the embedding network. This provides a practical solution for rapid customization of KWS systems.

3. **Shows that triplet loss embeddings are effective for both classification and verification tasks**: The learned embedding space supports not only standard classification tasks (selecting the nearest prototype) but also open-set verification tasks (threshold-based judgment), meeting the two core needs of KWS systems.

4. **Provides analysis of different embedding dimensions and their impact on KWS performance**: Systematically evaluates performance changes as embedding dimensions range from 32 to 256, finding that medium dimensions (64-128) achieve good performance, while higher dimensions yield diminishing returns and increase computational and storage overhead.

5. **Effective application of semi-hard negative mining strategy in KWS**: Validates that semi-hard negative mining is crucial for triplet training in the KWS task, improving accuracy by approximately 2-3% compared to random negative sampling strategies.

## Experimental Results

### Datasets and Settings
- **Google Speech Commands (GSC) v2**: 12-class and 35-class classification tasks
- **Evaluation Metrics**: Classification accuracy, Equal Error Rate (EER) for verification tasks, intra-class/inter-class distance ratio in the embedding space
- **Backbone Network**: DS-CNN style CNN
- **Embedding Dimensions**: D = 32, 64, 128, 256
- **Triplet Margin**: 0.2, 0.3, 0.5

### Classification Task Performance
- On the GSC 12-class classification task, the accuracy of the triplet loss embedding method is approximately 94-95%, close to that of cross-entropy-based classification methods (approximately 95-96%), but provides better flexibility for keyword customization.
- On the GSC 35-class task, the accuracy is approximately 92-93%. As the number of classes increases, the discriminative ability of the embedding space decreases slightly.

### Embedding Space Quality
- **Intra-class vs. Inter-class Distance**: After triplet loss training, the average intra-class distance is approximately 0.3-0.4, and the average inter-class distance is approximately 0.8-1.0, with a clear margin (when margin=0.3, the margin is approximately 0.4-0.6).
- **t-SNE Visualization**: Different keyword classes form clear clusters in the embedding space, with the clusters for high-frequency words like "yes", "no", "up", and "down" being the most compact. Acoustically similar keywords (e.g., "go" and "no") have clusters that are relatively close but still distinguishable.
- **Impact of Embedding Dimension**:
  - D=32: Accuracy approximately 92%, insufficient embedding space capacity
  - D=64: Accuracy approximately 94%, best cost-performance ratio
  - D=128: Accuracy approximately 95%, further improvement
  - D=256: Accuracy approximately 95.2%, less than 0.5% improvement compared to D=128

### Keyword Verification Task
- Using cosine distance + threshold, an Equal Error Rate (EER) of approximately 3-5% was achieved on the keyword verification task.
- The performance of the verification task depends on the choice of threshold—a threshold that is too high leads to an increased miss rate, while one that is too low leads to an increased false alarm rate.
- The calibration quality of the embedding space (the correspondence between distance values and semantic similarity) affects the reliability of threshold selection.

### New Keyword Adaptation
- After calculating prototypes using 5-10 samples of new keywords, reasonable detection of new keywords can be achieved (accuracy approximately 85-90%).
- The diversity of new keyword samples (different speakers, different speaking speeds) significantly affects prototype quality—more diverse samples produce more representative prototypes.

### Ablation Studies
- **Semi-hard vs. Random Negatives**: Semi-hard negative mining improves accuracy by approximately 2-3% compared to random negative sampling.
- **Margin Parameter**: margin=0.3 achieves the best balance between accuracy and training stability. margin=0.2 results in a margin that is not large enough (leading to easy confusion), while margin=0.5 makes training difficult (many triplets cannot satisfy the margin).
- **Impact of L2 Normalization**: Using L2 normalization improves accuracy by approximately 1% and makes training more stable.
- **Distance Metric**: After L2 normalization, the performance of Euclidean distance and cosine distance is almost identical (theoretical expectation, as they are equivalent on the unit sphere).

## Limitations and Future Work

### Technical Limitations
- **Sensitivity of Triplet Training**: Triplet loss training is very sensitive to learning rate, margin parameter, and negative sample mining strategy. Improper hyperparameter settings can lead to unstable training (e.g., embedding collapse to a single point) or slow convergence. Careful tuning is required in practical use.
- **Confusion of Acoustically Similar Keywords**: For keywords that are acoustically very similar (e.g., "yes" and "yeah", "on" and "off"), triplet loss may struggle to create a sufficiently large margin in the embedding space. This is a common challenge for metric learning methods in KWS.
- **Calibration of Embedding Space**: The learned distance values may not directly correspond to a probabilistic interpretation of semantic similarity, making threshold-based detection less reliable. Unlike softmax classification outputs, distance values themselves do not have probabilistic meaning.
- **Computational Cost Grows with Vocabulary Size**: During inference, the distance between the input embedding and all keyword prototypes must be calculated, with a computational complexity of $O(K \cdot D)$, where $K$ is the number of keywords and $D$ is the embedding dimension. For large vocabulary scenarios ($K>100$), this computational cost may not be negligible.
- **Dependence on Mining Strategy**: Regular re-mining of negatives is required, increasing the engineering complexity of training.

### Insufficiencies in Experimental Design
- Evaluation under noisy conditions and far-field scenarios is limited. Whether the embeddings learned by metric learning can maintain intra-class compactness and inter-class separation under noise interference is an important open question.
- No systematic comparison with other metric learning methods (such as Contrastive Loss, Center Loss, or Prototypical Networks).
- Joint training of the embedding network and classification network (e.g., using both triplet loss and cross-entropy loss simultaneously) was not explored; such joint training could further improve performance.
- Lack of deployment evaluation on real edge devices (impact of embedding dimension on storage and computation).

### Future Improvement Directions
- Explore more stable metric learning loss functions (such as Proxy-NCA, ArcFace, Circle Loss, which have been proven more effective in face recognition) applied to KWS.
- Combine triplet loss with cross-entropy loss for joint training, enjoying the advantages of both classification accuracy and embedding flexibility.
- Study adaptive embedding dimensions—dynamically adjust the embedding dimension based on the complexity of keywords, using low-dimensional embeddings for simple keywords to save computation.
- Explore hierarchical embedding spaces—first perform coarse-grained distinction of major categories (e.g., broad acoustic feature categories), followed by fine-grained keyword distinction.
- **Insights for the KWS Field**: Metric learning provides a fundamental solution for keyword customization in KWS, transforming it from a "fixed-class classification" problem into a "flexible distance matching" problem. Although triplet loss is the most basic metric learning method, its successful application lays the foundation for subsequent more complex metric learning research (such as cross-modal metric learning, hierarchical metric learning, etc.).
