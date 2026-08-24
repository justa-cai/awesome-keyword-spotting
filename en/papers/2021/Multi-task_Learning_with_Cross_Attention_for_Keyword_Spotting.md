# Multi-task Learning with Cross Attention for Keyword Spotting

- **Authors/Affiliations**: Xuechen Zhang, Yujun Wang, Zhiyong Wu, Helen Meng - Apple; The Chinese University of Hong Kong
- **Date**: 2021.07
- **Link**: https://arxiv.org/abs/2107.07634
- **Keywords**: Multi-task Learning, Cross Attention, Keyword Spotting, Voice Activity Detection, False Alarm Mitigation, Feature Sharing

## Problem Statement

Keyword Spotting (KWS) systems face a core contradiction in practical deployment: they must maintain extremely high sensitivity (low miss rate) for target keywords while maintaining an extremely low false alarm rate for non-keyword speech. Particularly in smart assistant scenarios, daily user conversations may contain segments acoustically similar to the wake-word, leading to frequent false triggers.

Traditional single-task KWS models only optimize the keyword classification objective and cannot leverage complementary information from related speech tasks. However, in practical voice assistants, multiple speech tasks are required simultaneously:
- **Keyword Spotting (KWS)**: Detects whether the wake-word is present
- **Voice Activity Detection (VAD)**: Determines whether a speech signal exists
- **Device-Oriented Speech Detection**: Determines whether the speech is directed towards the device

There is a natural correlation between these tasks: VAD can help KWS eliminate false alarms in non-speech segments; the acoustic features of KWS can also assist VAD in making more accurate judgments. How to effectively utilize this complementary information between tasks to improve KWS performance is the core problem addressed in this paper.

## Methodology

### Overall Architecture Design
The paper proposes a multi-task learning framework based on Cross Attention, with core designs including:
1. **Shared Encoder Backbone**: Extracts general speech features
2. **Task-Specific Decoder Branches**: Provides dedicated feature transformation for each task
3. **Cross Attention Module**: Facilitates information exchange between task branches

### Shared Encoder
- Uses a multi-layer CNN (or Conformer) as the shared acoustic feature encoder
- Input: Spectrogram or MFCC features
- Output: General acoustic feature representation, serving all downstream tasks simultaneously
- The shared encoder acquires richer feature representations through joint multi-task training

### Task-Specific Decoders
Each task (KWS, VAD, etc.) has an independent decoder branch:
- Each branch contains several layers of task-specific convolutional or fully connected layers
- The decoder extracts task-specific discriminative information from the shared features
- Independent classification heads output the prediction results for each task

### Cross Attention Module
This is the core innovation of the paper. Cross Attention allows information exchange between decoder branches of different tasks:

1. **Cross-Task Attention Calculation**:
   - For the query feature $Q_A$ of Task A, attention is calculated using the key $K_B$ and value $V_B$ of Task B
   - $Attention(Q_A, K_B, V_B) = softmax(Q_A * K_B^T / \sqrt{d}) * V_B$
   - This enables Task A to acquire relevant information from the feature representation of Task B

2. **Bidirectional Cross Attention**:
   - Task A attends to the features of Task B, while Task B also attends to the features of Task A
   - This forms symmetric information exchange, benefiting both tasks

3. **Gated Fusion**:
   - The output of cross attention is fused with the original task features via a gating mechanism
   - The fusion coefficient is controlled by learnable gating parameters
   - This prevents irrelevant information from interfering with the main task

### Loss Function
The multi-task joint loss is the weighted sum of individual task losses:
$L = \alpha * L_{KWS} + \beta * L_{VAD} + \gamma * L_{other}$

Task weights are determined through experimental tuning or dynamic weight adjustment strategies.

## Main Contributions

1. **Introduction of Cross Attention for Information Sharing between KWS Tasks**: Unlike simple multi-task shared encoders (which only share features at the bottom level), Cross Attention performs fine-grained information exchange between high-level decoders, allowing each task to precisely acquire the most relevant information from other tasks.

2. **Demonstration that Cross Attention Multi-task Learning Improves KWS Accuracy**: Experiments show that multi-task learning with Cross Attention significantly outperforms single-task learning and multi-task learning without Cross Attention in terms of KWS accuracy, particularly in controlling the false alarm rate.

3. **Revelation of the Complementarity between VAD and KWS**: The VAD task helps KWS reduce false alarms on non-speech segments (such as noise and silence), while the acoustic knowledge of KWS helps VAD more accurately distinguish between speech and non-speech.

4. **Effective False Alarm Mitigation Strategy**: Through multi-task inference, using the judgment of VAD as an auxiliary signal can effectively mitigate false triggers of KWS on non-target speech.

## Experimental Results

### Datasets
- Evaluation uses a combination of internal KWS datasets and public datasets
- Includes various test scenarios such as target keywords, non-keyword speech, noise, and silence

### KWS Performance Improvement
- **Accuracy**: Compared to the single-task KWS baseline, the accuracy of Cross Attention multi-task learning improved by approximately 1-2%
- **Significant Reduction in False Alarm Rate**: Under the condition of maintaining the same recall rate, the false alarm rate decreased by 20-30%
- **VAD Assistance Effect**: The VAD branch contributed most significantly to the reduction of the KWS false alarm rate

### Effect of Cross Attention
- **With vs. Without Cross Attention**: Under the same multi-task framework, adding Cross Attention further improved KWS accuracy by an additional 0.5-1%
- **Bidirectional vs. Unidirectional**: Bidirectional Cross Attention performed slightly better than unidirectional
- **Importance of Gated Fusion**: After removing the gating mechanism, the benefits of Cross Attention decreased

### Benefits for VAD Task
- The VAD task also benefited from joint training with KWS
- The accuracy of VAD after multi-task training was slightly higher than that of the separately trained VAD model

### Ablation Studies
- **Depth of Shared Encoder**: Sharing 4-6 layers of encoders yielded the best results; too shallow resulted in insufficiently rich features, while too deep increased task interference
- **Position of Cross Attention**: Inserting Cross Attention in the middle layers of the decoder yielded the best results

## Limitations and Future Work

### Technical Limitations
- **Increased Model Complexity**: The Cross Attention module introduces additional parameters and computational load. Although accuracy improved, it may be unacceptable under extremely strict resource constraints
- **Requirement for Multi-task Annotations**: Training requires data annotated for both KWS and VAD (or other tasks) simultaneously, increasing data preparation costs
- **Risk of Task Interference**: When the correlation between tasks is low, multi-task learning may lead to negative transfer, causing the performance of certain tasks to decline

### Insufficiencies in Experimental Design
- Limited analysis of which task combinations provide the greatest benefit to KWS
- Multi-task learning with more than three tasks was not explored
- The computational overhead of Cross Attention during inference lacks detailed quantification
- Insufficient evaluation under far-field and noisy conditions

### Future Improvement Directions
- Explore task similarity metrics to predict optimal task combinations
- Investigate dynamic task weight adjustment strategies
- Combine model distillation to transfer multi-task knowledge to single-task lightweight models for edge deployment
- **Insights for the KWS Field**: Multi-task learning, particularly the Cross Attention mechanism, provides an effective approach to reducing the false alarm rate of KWS. Future work can explore joint learning of KWS with more tasks (such as emotion recognition and language identification)
