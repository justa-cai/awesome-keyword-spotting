# Zero-Shot Federated Learning with New Classes for Audio Classification

- **Authors/Affiliations**: Gautham Krishna Gudur (Ericsson Global AI Accelerator), Satheesh Kumar Perepu (Ericsson Research)
- **Date**: 2021.06
- **Conference**: Interspeech 2021 (also accepted at ICLR 2021 DPML and HAET Workshops)
- **Link**: https://arxiv.org/abs/2106.10019
- **Keywords**: Federated Learning, Zero-Shot Learning, Audio Classification, New Class Recognition, Anonymized Data Impressions, Class Similarity Matrix, Keyword Spotting, Urban Sound Classification, Edge Devices, k-medoids Clustering

## Problem Statement

Federated Learning (FL) is a decentralized approach to training models on distributed user devices, protecting privacy by securely sharing model updates rather than raw user data. However, in practical FL scenarios, a series of complex challenges arise, the most prominent being the **dynamic emergence of new classes**: entirely new, unseen data categories may appear on different users' devices at any time, with data distributions completely different from existing classes. Due to the privacy constraints of FL, the global server and other users cannot access the raw data from any device, making it extremely difficult to identify and integrate new classes across devices.

Specifically, the core problems this paper aims to solve include:
1. **Zero-shot recognition of new classes**: When a brand-new audio category appears on a user's device, how can the global model identify whether this new class is the same as or different from new classes on other users' devices, without accessing the raw data?
2. **Two scenarios for obtaining class labels**: User devices may (a) report the label name of the new class, or (b) not report the label name. In the latter scenario, unsupervised differentiation of different new classes is required.
3. **Handling statistical heterogeneity**: Across different communication rounds and different user devices in FL, there are multiple challenges, including label distribution heterogeneity, data distribution heterogeneity, and model architecture heterogeneity.

Traditional FL assumes that all clients have the same set of classes and model architectures. However, in practical audio classification applications (such as personalized keyword spotting and urban environmental sound recognition), these assumptions often do not hold. This paper aims to propose a **unified zero-shot federated learning framework** that elegantly handles the dynamic emergence of new classes and multiple statistical heterogeneity issues while protecting user privacy.

## Methodology

### Overall Framework Design

The framework proposed in the paper is based on the classic federated learning paradigm, containing M nodes (devices), each holding private local data $D_i = \{x_{i,j}, y_{i,j}\}$, while all nodes and the global server share a public dataset $D_0 = \{x_0, y_0\}$. The public dataset is not exposed to local models during FL training and is used only for consistency evaluation during the testing phase. The entire framework consists of three core steps:

**Step 1: Build**: Each local user creates their own model using their private local data in the current FL iteration.

**Step 2: Local Update**: Divided into two processing strategies depending on whether a new class is reported.

**Step 3: Global Update**: The global server aggregates all local updates, adopting different strategies based on whether new classes are reported.

### Scenario 1: New Class Labels Not Reported

When users do not report new class labels, the framework reverts to the traditional FL setting. Local updates adopt a weighted alpha update mechanism:

$f^i_{D_m}(x_0) = f^I_G(x^{l_m}_0) + f^{D_m}(x_0)$

where $f^I_G(x^{l_m}_0)$ is the score of the global model on user m's label set $l_m$, and $\alpha = \text{len}(D^i_m) / \text{len}(D_0)$ controls the contribution ratio of the old and new models.

Global updates use label-level weighted averaging, where the weight $\gamma$ is determined by whether the label is unique: if the label is unique among users, $\gamma = 1/M$; if labels overlap, $\gamma$ is proportional to the test accuracy of each local model on that label.

### Scenario 2: New Class Labels Reported — Core Innovation

When users report new class labels, the paper proposes a zero-shot learning mechanism based on **Anonymized Data Impressions (DI)**. This is the core innovation of the paper, containing three key steps:

#### 2.1 Class Similarity Matrix (CSM)

Using the weights of the final fully connected layer of the model, a similarity matrix between classes is constructed:

$C(i, j) = (w_i^T * w_j) / (||w_i|| * ||w_j||)$

where $w_i$ is the weight vector connecting the second-to-last layer to class $i$. If two classes are similar, their fully connected layer weight vectors will also be similar. The CSM matrix $C \in \mathbb{R}^{K \times K}$ encodes similarity information among $K$ classes.

#### 2.2 Sampling Softmax Values from Dirichlet Distribution

Based on the CSM matrix, softmax values are sampled from a Dirichlet distribution:

$\text{Softmax} = \text{Dir}(K, C * \epsilon)$

where $C * \epsilon$ is the concentration parameter, controlling the concentration of softmax values on class labels. If classes are similar, the sampled softmax values will concentrate on these similar classes.

#### 2.3 Generating Anonymized Data Impressions

Synthetic data features are generated in reverse from the sampled softmax values by solving the following optimization problem:

$x^* = \arg\min_x L_{CE}(y^k_i, M(x))$

where $y^k_i$ is the $i$-th softmax vector for the $k$-th class, and $M(x)$ is the model output. Initializing $x$ as a random input, synthetic data features are obtained by iteratively minimizing the cross-entropy loss. This process is repeated for each class, thereby generating anonymized data impressions for each new class without accessing the raw data.

### New Class Recognition and Clustering

After generating anonymized data impressions for each user, the framework executes the following steps:
1. **Aggregation**: Average the DIs of all users with the same new class label $k$, $X_i = \sum_{m \in MS_k} X^i_m$
2. **k-medoids Clustering**: Perform unsupervised k-medoids clustering on the DIs of all users reporting new classes, with the number of clusters equal to the number of new classes $l_{new}$
3. **Update Public Dataset**: Add the anonymized data impressions $X_i$ obtained from clustering to the public dataset, $D_{new} = D_0 \cup X_i$
4. **Update Label Set**: Add the new labels $l_{new}$ to the label sets of each user and the global label set $Y$

A key advantage of k-medoids clustering is that even if users do not report the specific names of new classes, the clustering can automatically distinguish between same and different new classes based on the feature similarity of the data impressions.

### Handling Heterogeneity

The framework simultaneously handles three types of statistical heterogeneity:

**Label Heterogeneity**: In each FL iteration, 200-300 audio frames (GKWS) or 40-50 audio frames (US8K) are randomly generated for each label of each user. Labels can be unique or overlapping among users, simulating non-IID distributions.

**Model Heterogeneity**: Different users use local models with different architectures (2-layer CNN, 3-layer CNN, 3-layer depthwise separable CNN, 1-layer CNN, 3-layer ANN), and may even switch model architectures and activation functions (ReLU/Softmax) across different FL iterations.

**Data Heterogeneity**: Simulates the time-varying nature of user behavior in practical scenarios through changes in data distribution across FL iterations.

### Input Feature Processing

- **GKWS (Keyword Spotting)**: Sampling frequency 14400 Hz, extracting MFCC features, divided into 20 windows, each window 50ms
- **US8K (Urban Sound Classification)**: Uses a similar MFCC preprocessing pipeline

## Main Contributions

1. **Proposed Zero-Shot Federated Learning Framework**: Introduced zero-shot learning mechanisms into the federated learning setting for audio classification for the first time. By synthesizing anonymized data impressions (Anonymized Data Impressions), it achieves cross-device new class recognition and knowledge sharing without accessing raw user data at all. This mechanism effectively solves the long-ignored problem of "dynamic emergence of new classes" in FL.

2. **Dual Scenario Coverage**: Designed two complete new class processing schemes—using the CSM and DI-based method when users report labels, and reverting to traditional FL settings when users do not report labels. This flexibility allows the framework to adapt to different practical deployment scenarios.

3. **Unsupervised New Class Differentiation**: Innovatively used the k-medoids clustering algorithm to cluster on anonymized data impressions, enabling automatic identification of whether new classes on different user devices are the same. Even if multiple users encounter the same new class but use different label names, clustering can correctly map them together.

4. **Comprehensive Handling of Statistical Heterogeneity**: The framework simultaneously handles label heterogeneity, model heterogeneity, and data heterogeneity. Different users can have local models with different architectures, label distributions can vary, and even the same user can switch model architectures across different FL iterations.

5. **Edge Device Validation**: Conducted complete experimental validation on a Raspberry Pi 2 (900MHz quad-core ARM Cortex-A7 CPU, 1GB RAM), proving the practical feasibility of the framework on resource-constrained IoT devices.

## Experimental Results

### Datasets and Experimental Setup

**GKWS (Google Keyword Spotting)**: Selected 10 keywords (Yes, No, Up, Down, Left, Right, On, Off, Stop, Go). The initial public dataset contained 8 keywords × 300 frames = 2400 frames. Two new classes, Stop (iterations 4, 8) and Go (iteration 8), were introduced.

**US8K (UrbanSound8K)**: Contains 10 classes of urban sounds. The initial public dataset contained 8 classes × 50 frames = 400 frames. Two new classes, Siren and Street Music, were introduced.

User model architecture settings:
- User 1: 2-layer CNN (16, 32), Softmax activation
- User 2: 3-layer CNN (16, 16, 32), ReLU activation
- User 3: 3-layer depthwise separable CNN (16, 16, 32), ReLU activation
- Model sizes: 520 kB, 350 kB, 270 kB

### Scenario 1: Only New Classes (No Statistical Heterogeneity)

3 users, 10 FL iterations, introducing only new classes without heterogeneity:

**GKWS Results**:

| User | Local Update Accuracy | Global Update Accuracy | Improvement |
|------|---------------|---------------|------|
| User 1 | 89.684% | 93.166% | +3.482% |
| User 2 | 91.888% | 95.280% | +3.391% |
| User 3 | 91.517% | 94.727% | +3.211% |
| **Average** | **91.030%** | **94.391%** | **+3.361%** |

**US8K Results**:

| User | Local Update Accuracy | Global Update Accuracy | Improvement |
|------|---------------|---------------|------|
| User 1 | 76.526% | 80.214% | +3.688% |
| User 2 | 75.272% | 77.944% | +2.672% |
| User 3 | 77.610% | 81.838% | +4.228% |
| **Average** | **76.469%** | **80.000%** | **+3.529%** |

In all users and all FL iterations, the accuracy of global updates was higher than that of local updates, proving that the framework can still effectively aggregate knowledge when new classes appear.

### Scenario 2: New Classes + Statistical Heterogeneity

10 users, 30 FL iterations, simultaneously introducing label heterogeneity, model heterogeneity, and data heterogeneity:

| Update Type | GKWS Accuracy | US8K Accuracy |
|---------|------------|------------|
| Local Update | 92.500% | 78.240% |
| Global Update | 96.541% | 82.498% |
| **Improvement** | **+4.041%** | **+4.258%** |

Even in complex scenarios where new classes dynamically emerge and multiple heterogeneities coexist, the framework still achieved an average deterministic accuracy improvement of approximately 4%, proving its robustness.

### k-medoids Clustering Analysis

Visualizing the k-medoids clustering results via PCA dimensionality reduction:
- **Different Classes**: When two users encounter different new classes, DI features form clearly separated clusters in the PCA space.
- **Same Class**: When two users encounter the same new class, DI features naturally cluster together.
- The number of clusters corresponds precisely to the number of new classes, validating the effectiveness of unsupervised new class recognition.

### Class Similarity Matrix Analysis

The CSM matrix for GKWS reveals the acoustic similarity between keywords. For example, the weight similarity between "Yes" and "No" is high, reflecting their overlap in certain acoustic features. The CSM also reveals the model's misclassification patterns, providing clues for further optimization.

### Edge Device Performance

Computation time on Raspberry Pi 2:

| Process | Time |
|------|------|
| Training time per epoch in each FL iteration | 1.2 seconds |
| Inference time | 11 milliseconds |

This result indicates that the lightweight nature of the framework is sufficient for practical deployment on IoT devices.

## Limitations and Future Work

### Technical Limitations

1. **Fidelity of Anonymized Data Impressions**: DIs are synthetic data generated in reverse from model weights via optimization methods; their distribution may not fully cover the true distribution of raw data. For classes with significant acoustic variations (e.g., sound events in environmental noise), the representativeness of DIs may be insufficient.

2. **Dependency on Public Dataset**: The framework assumes that all users and the global server share a public dataset $D_0$, which may be difficult to guarantee in practical deployments. The quality and scale of the public dataset directly affect the framework's performance.

3. **Sensitivity of Dirichlet Distribution Parameters**: The conversion from the class similarity matrix to softmax sampling depends on the choice of the concentration parameter $\epsilon$, and different parameters may lead to different DI qualities. The paper does not deeply discuss parameter sensitivity analysis.

4. **Assumption of Known Number of New Classes**: k-medoids clustering requires prior knowledge of the number of clusters (i.e., the number of new classes), which may require additional automatic class number estimation mechanisms in practical scenarios.

5. **Limitations of Model Heterogeneity**: Although the framework supports local models with different architectures, all models are variants based on CNNs. Its ability to handle more fundamental architectural differences (e.g., CNN vs. RNN vs. Transformer) has not been verified.

### Experimental Design Shortcomings

1. **Limited Scale of Users and Iterations**: The maximum experimental setup contains only 10 users and 30 FL iterations, which is orders of magnitude smaller than actual large-scale federated learning scenarios (millions of users, thousands of iterations).

2. **Lack of Comparison with Baseline Methods**: The paper does not directly compare with other FL methods for handling new classes (such as incremental learning FL or meta-learning FL), only showing the framework's own local-to-global accuracy improvements.

3. **Missing Security Analysis**: The robustness of the framework against malicious attacks (such as model poisoning or backdoor attacks) was not evaluated. Whether the DI generation process can be exploited by attackers to inject malicious information is an important open question.

4. **Communication Overhead Not Detailed**: Although computation time is shown, the communication overhead of transmitting DIs and model weights is not analyzed in detail, which is crucial for bandwidth-constrained IoT devices.

### Future Improvement Directions

1. **Integration with Contrastive Learning**: Introducing contrastive learning objectives (such as SimCLR, MoCo) into the FL framework may produce more discriminative feature representations, thereby improving the quality of DIs and clustering accuracy.

2. **Adaptive Clustering**: Introducing density-based or hierarchical clustering methods (such as DBSCAN, HDBSCAN) to automatically determine the number of new classes, eliminating the dependency on preset cluster numbers.

3. **Integration of Differential Privacy**: Introducing differential privacy mechanisms (such as DP-SGD) during model updates and DI transmission to provide formal privacy guarantees for the framework.

4. **Larger-Scale Validation**: Conduct validation on larger-scale real federated learning platforms (such as TensorFlow Federated, PySyft) to evaluate the scalability of the framework.

5. **Multimodal Extension**: Extend the framework to multimodal classification tasks such as video and text to verify its generality.

### Implications for the KWS Field

This paper provides important insights for the federated learning deployment of personalized KWS systems. In the practical use of intelligent voice assistants, different users may wish to add custom wake words or command words, and the data for these new keywords should not leave the user's device. This framework achieves cross-device new keyword knowledge sharing while protecting user privacy through zero-shot learning mechanisms. The innovative design of anonymized data impressions provides a new paradigm for knowledge transfer in federated learning—transmitting not data nor model parameters, but rather the class "impression" distilled from model weights.
