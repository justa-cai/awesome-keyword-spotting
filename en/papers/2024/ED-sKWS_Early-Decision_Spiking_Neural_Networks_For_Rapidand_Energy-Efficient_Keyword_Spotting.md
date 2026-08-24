# ED-sKWS: Early-Decision Spiking Neural Networks for Rapid,and Energy-Efficient Keyword Spotting

- **Authors/Affiliations**: Zeyang Song, Qianhui Liu (Corresponding Author), Qu Yang, Yizhou Peng, Haizhou Li (National University of Singapore; School of Data Science, The Chinese University of Hong Kong, Shenzhen / Shenzhen Research Institute of Big Data)
- **Date**: June 14, 2024 (arXiv:2406.12726v1 [cs.SD], preprint in conference template format)
- **Link**: https://arxiv.org/abs/2406.12726
- **Keywords**: spiking neural network, keyword spotting, early decision, cumulative temporal loss, energy efficiency, real-time speech processing, timestamp-annotated dataset

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) identifies specific words or phrases within continuous audio streams, serving as a persistent voice entry point for edge devices such as smartphones, smart speakers, and in-car voice systems. The paper’s introduction (References [1][2]) outlines three explicit requirements for these scenarios: fast real-time response, high energy efficiency, and high accuracy. In always-on listening scenarios, these three requirements form a mutually constraining triangle—models are online 24/7, and every sample must be processed in its entirety to produce a result, causing power consumption to accumulate linearly with time; meanwhile, the latency between the user uttering a keyword and the system responding directly determines the "responsiveness" of the interaction experience.

Spiking Neural Networks (SNNs) are viewed by the paper as a candidate technical route to address this constraint. SNNs, the third generation of neural networks, transmit information using discrete spikes rather than continuous activation values. Neurons remain silent unless their membrane potential exceeds a threshold, an event-driven mechanism that is inherently power-efficient. Furthermore, the intrinsic temporal dynamics of spiking neurons, such as membrane potential integration, leakage, and reset, align with the temporal structure of speech, making them suitable for speech processing (References [12][4][13]). The paper also highlights a frequently overlooked structural advantage: the computational form of feedforward SNNs, unfolded step-by-step in time, is isomorphic to that of naive RNNs. Inputting one frame, computing one step, and producing a set of outputs is precisely the shape required for streaming real-time processing (Reference [14]).

The true pain point lies in the paradigm rather than computational power: although existing SNN KWS methods compute step-by-step, they uniformly adhere to a "late-decision" paradigm (a contrasting concept coined by the paper)—they must receive the entire input sample (usually zero-padded to a fixed 1 second) before providing a final classification at the readout layer (References [7][8][9]). This leads to two forms of waste: first, the actual pronunciation of a keyword often occupies only the first half of the sample, with the subsequent dozens of time steps being pure silence padding, yet spiking neurons still maintain their membrane potential states and continue forward computation; second, system response is artificially delayed until the end of the sample, completely failing to realize the real-time potential of SNNs, which are event-driven and predictable frame-by-frame.

### Specific Shortcomings of Existing Methods

- **The late-decision paradigm wastes the real-time capabilities of SNNs.** Methods such as [7][8][9] treat SNNs as "run-then-read" classifiers, producing no results until the sample is complete. Since feedforward SNNs output potentials at every time step, theoretically, an answer can be given early when sufficient evidence is available. However, existing methods lack both a stopping mechanism and training methods to support reliable output at intermediate time steps.
- **Early decision lacks training support.** Early stopping implies making judgments based on incomplete acoustic information, inevitably carrying the risk of information loss. The paper’s abstract explicitly states that this is a trade-off that requires specialized training techniques to resolve. Existing loss functions each have defects (analyzed one by one in Section 2.2): spike-rate loss [18] and cumulative loss [16] only compute loss at the last time step, leaving intermediate time step outputs unsupervised; TET loss [19], although supervising step-by-step, computes each time step in isolation, being effective only on static datasets where inputs are identical across all time steps. It does not model historical information on continuously changing data like speech, resulting in poor performance.
- **Datasets lack time annotations, making rigorous evaluation of early decision impossible.** The widely used Google Speech Commands (GSC) [15] zero-pads all samples to a uniform 1 second, burying the true start and end times of keywords within the padding segments. To quantify "how fast the model actually responds," one must know when the keyword ends; the paper points out that treating sample boundaries as keyword boundaries introduces significant error. The lack of time annotations means that early decision research lacks even a foundational basis for evaluation.
- **Energy efficiency comparisons lack a unified standard.** Energy consumption figures from different SNN KWS works vary drastically (from single-digit µJ to 247.52 µJ in Table 1 below). Comparisons are only meaningful when unified under the same process node and operational energy assumptions; otherwise, claims of "power saving" are baseless.

### Key Challenges Addressed by This Paper

The paper actually aims to solve four interlinked problems simultaneously: First, at the architecture level, how to align SNN time steps one-to-one with speech frames, enabling meaningful predictions at any intermediate time step, which is the prerequisite for early decision to be valid; Second, at the training level, how to design a loss function that optimizes "cumulative evidence up to the current frame" reliably at every intermediate time step, determining whether early stopping avoids early errors; Third, at the inference level, how to judge at runtime whether sufficient evidence has accumulated to stop, i.e., the stopping criterion; Fourth, at the evaluation level, how to construct a dataset with keyword start/end time annotations, turning "how much earlier" into a measurable number rather than an impression.

## Methodology

### Overall Architecture Design and Design Motivation

The overall pipeline of ED-sKWS (Fig. 1): Raw audio features (fbank) are extracted and fed into a feedforward fully connected SNN (gray box in the figure). The feature of the $t$-th frame is fed to the $t$-th time step of the SNN, processing frame-by-frame in a streaming manner; at each time step, the cumulative confidence $CS$ is calculated. Once it exceeds a preset threshold $C$, speech processing is immediately terminated, and the result is output.

**Why choose a fully connected feedforward SNN instead of a Spiking Convolutional Neural Network (SCNN)?** This is the most critical architectural decision in the entire paper (Section 2.1). SCNN methods ([8][9]) treat the entire spectrogram of the input audio as an image, requiring the network to use additional time steps to encode this static image into a spike sequence. In other words, "internal network time steps" and "external speech time" are decoupled: a sample must occupy several time steps before producing a single prediction, naturally restricting it to late decision. In contrast, a fully connected SNN processes only one spectrogram frame per time step, where the $t$-th frame corresponds to the $t$-th time step. The network time axis and speech time axis are completely synchronized, analogous to the streaming form of RNNs. Thus, each time step completes a forward propagation carrying historical state (membrane potentials of all layers), and the readout layer has an output at every time step, making early decision mathematically definable.

**Why choose adaptive LIF (adLIF) neurons** (following [16]): The leakage coefficient of standard LIF is fixed, and its time constant is not adjustable; adLIF introduces adaptive terms for the previous step's membrane potential and previous step's output spike into the synaptic current update equation. These are associated with subthreshold dynamics and spike-triggered responses by parameters $a$ and $b$, respectively, effectively allowing neurons to regulate the inflow and outflow of information based on their own history (Section 2.1 emphasizes its enhanced control over information flow within the network). The significance for KWS lies in: suppressing spurious discharges during silence segments and maintaining effective integration during speech segments, providing a mechanistic basis for retaining learned features for longer periods into later time steps—this design motivation is directly tested in the long-tail silence experiments of SC-100 later in the paper.

**Model Scale** (Section 4.1): The feedforward SNN contains two hidden layers and one readout layer, configured in two scales—128 neurons in the hidden layer (27.63K parameters) and 512 neurons in the hidden layer (306.80K parameters). This is completely isomorphic to the baseline adLIF [16], ensuring that performance differences in controlled experiments can be cleanly attributed to the loss function and inference mechanism.

### Mathematical Principles of Core Algorithms

**adLIF Neuron Dynamics** (Equations 1-3):

$$I_l[t] = \beta \sum_i w_i S_{l-1}[t-1] + a\,U_l[t-1] + b\,S_l[t-1] \tag{1}$$

$$U_l[t] = \alpha\,(U_l[t-1] - V_{th} S_l[t-1]) + I_l[t] \tag{2}$$

$$S_l[t] = H(U_l[t] - V_{th}) \tag{3}$$

Translated into plain language: Equation (1) states that the synaptic current $I$ of layer $l$ at step $t$ consists of three parts: the weighted sum of input spikes from the previous layer at $t-1$ (with $\beta$ controlling attenuation), the feedback of its own membrane potential from the previous step (term $a$, subthreshold adaptation), and whether it fired a spike in the previous step (term $b$, spike-triggered adaptation). Equation (2) states that the membrane potential $U$ performs leaky integration; if a spike is fired, the threshold $V_{th}$ is subtracted from the potential (soft reset). Equation (3) is the Heaviside step function $H$: if the potential exceeds the threshold, a spike is fired. Constants $\alpha$ and $\beta$ determine the decay rate of information within the neuron, while $a$ and $b$ control the strength of the adaptive mechanism. Compared to standard LIF, the adaptive terms make the neuron's equivalent time constant and firing behavior dependent on its recent history, which is the mathematical grounding for "enhanced information flow control."

**Cumulative Temporal (CT) Loss** (Equations 4-5):

$$O[t] = \sum_{i=0}^{t} \mathrm{softmax}(U_R[i]) \tag{4}$$

$$L_{CT} = \frac{1}{T} \sum_{t=0}^{T} L_{CE}(O[t], y) \tag{5}$$

$O[t]$ is the cumulative sum of softmax of the readout layer potentials from step 0 to step $t$, semantically representing "all historical evidence up to the current frame"; applying cross-entropy to $O[t]$ and averaging over all time steps yields the CT loss. There are two deliberate design layers here, corresponding to two failure modes of previous works: using a cumulative sum rather than an instantaneous readout explicitly incorporates historical information into the supervision target (learning from the lesson that TET does not model history); computing loss at every time step rather than only at the last step incorporates the reliability of intermediate predictions into the optimization target (learning from the lesson that spike-rate and cumulative losses only supervise the final step). From the perspective of gradient paths, the readout potential of the $i$-th frame receives supervision signals through all $O[t]$ where $t \ge i$. The representation of early frames is shaped "to be correct for cumulative judgments at all subsequent steps," thereby establishing credit assignment in the time dimension. The paper's phrasing is: optimizing $O[t]$ at each time step makes the cumulative output closer to the target distribution, thereby improving the accuracy of predictions at intermediate time steps.

**Early Decision Criterion** (Section 2.3): Confidence is defined as $CS_t = \max(\mathrm{softmax}(O[t]))$, i.e., the probability of the most likely class in the current cumulative distribution. When $CS_t$ exceeds the preset threshold $C$, the model considers the prediction sufficiently reliable, stops subsequent processing, and outputs the result; if below the threshold, it continues to absorb information from subsequent frames until confidence is sufficient. The reason for choosing $\max(\mathrm{softmax}(O[t]))$ as a proxy is straightforward: it is the model's confidence probability for its current judgment, ranging from 0 to 1, comparable across samples and classes, making it suitable for a globally unified stopping threshold. The paper does not report the specific value of threshold $C$, nor does it report sensitivity analysis for it.

### Key Technical Innovation 1: Frame-Synchronized Feedforward SNN + Confidence Threshold Early Decision Mechanism

The complete working mode of the early decision mechanism: The model advances frame-by-frame. At each time step, it first performs a forward propagation with membrane potential states. The readout layer potential enters the cumulative softmax to obtain $O[t]$, and its maximum component is taken as the confidence $CS_t$. Once it exceeds threshold $C$, inference terminates at step $t_d$, and $\argmax(O[t_d])$ is the final result. All forward computations from step $t_d$ to $T$ are skipped entirely. The source of energy savings is particularly concrete in SNNs: SNN inference energy consumption is proportional to the number of actual synaptic operations (accumulation operations). Skipping each time step means saving all spike propagations for that step. Section 4.5 visualizes with a single sample that these skipped steps are not zero-cost—precisely the high-firing region where membrane potential memory is maintained after the keyword ends. Furthermore, the mechanism is decoupled from the architecture: the same model, with early decision turned off (running fixed to step 98), becomes the late-decision baseline, enabling self-comparison within Table 1.

### Key Technical Innovation 2: Cumulative Temporal (CT) Loss

CT loss aims to simultaneously cater to two previously mutually exclusive goals: prediction quality at intermediate time steps (for early decision) and prediction quality at the final time step (for comparability with late decision). A comparison of the mathematical forms of four losses (definitions given in Section 4.3):

- spike-rate loss [18]: $L_{CE}(\frac{1}{T}\sum_{t=0}^{T} U_R[t],\, y)$, supervises the average readout of the entire segment only once, with no supervision for intermediate steps;
- TET loss [19]: $\frac{1}{T}\sum_{t=0}^{T} L_{CE}(U_R[t],\, y)$, step-by-step supervision but isolated per step, using instantaneous readout without accumulating history;
- cumulative loss [16]: $L_{CE}(\sum_{i=0}^{t} \mathrm{softmax}(U_R[i]),\, y)$, accumulates history but calculates loss only at the last step;
- CT loss (this paper): $\frac{1}{T}\sum_{t=0}^{T} L_{CE}(O[t],\, y)$, accumulates history and provides step-by-step supervision, representing the intersection of the previous two.

Why this intersection is important for speech: Speech is continuously changing streaming data. Acoustic information from the first few frames is objectively insufficient for classification. Isolated supervision of instantaneous outputs produces conflicting gradients (the root cause of TET's lowest final accuracy in Table 2); while supervising only the final step allows the confidence of intermediate steps to drift unconstrained (the root cause of spike-rate's worst early-decision accuracy). CT uses cumulative quantities as supervision targets, naturally expressing "which word the evidence heard so far should point to." As the number of frames increases, evidence monotonically increases, and the target becomes gradually satisfiable, with gradient directions being largely consistent over time.

### Key Technical Innovation 3: SC-100 Dataset (100k-Level Command Library with Start/End Time Annotations)

Construction Pipeline (Section 3): First, the KeywordMiner tool [20] is used to mine words from LibriSpeech [21]. It consists of an aligner (providing timestamps for each word in the sentence) and a segmenter (exporting word clips according to timestamps). The raw output cannot be used directly. The paper performs two rounds of cleaning: first, removing function words unsuitable as command words (e.g., "a", "to", whose semantics do not support the functional requirements of KWS); second, eliminating clips with pronunciation durations that are too short. The final 100 selected keywords all have durations exceeding 0.4 seconds. The word list is divided into six groups covering daily scenarios: Smart Home (change, turn, light, door, window, etc.), Entertainment (show, next, play, ready, color words, etc.), Alarm Clock (set, stop, change), Robot Assistant (go, left, right, wait, find, etc., largest word count), Office (book, time, numbers zero to nine, office, etc.), and Personal Assistant (call, send, read, morning, etc.).

Scale and Format: A total of 313,951 keyword utterances, with 1,000 to 4,000 utterances per class. Class imbalance is handled using uniform sampling weights to ensure training fairness; utterance durations range from 0.4 to 1 second. To align with the standard format of GSC, random amounts of zero-padding are added to the beginning and end to form 1-second samples. The key increment lies in the annotations: each command comes with intra-word start/end timestamps. This allows precise quantification of early decision response speed (Section 4.4) and can be externally supplied to related tasks such as Voice Activity Detection (VAD). Notably, random padding precisely creates the evaluation difficulty of "sample boundary does not equal word boundary," which is the point criticized by the paper regarding GSC. SC-100 makes up for this information via timestamps.

### Technical Differences with Existing Methods

| Dimension | SCNN Series (Yilmaz [8], sKWS [9]) | adLIF [16] | TET Training [19] | ED-sKWS (This Paper) |
|---|---|---|---|---|
| Network Time Axis | Spectrogram treated as image, requires extra time steps for encoding | Frame-synchronized | Frame-synchronized | Frame-synchronized |
| Supervision Signal | Final step | Final step (cumulative) | Isolated instantaneous per step | Cumulative history per step |
| Decision Point | Fixed T | Fixed T | Fixed T | Adaptive $t_d$, not later than T |
| Input Frontend | Spectrogram | fbank | fbank | fbank |

The relationship with each method is clarified one by one. The comparison with adLIF [16] is the cleanest: same architecture, same neurons, same parameter counts (two tiers of 27.63K and 306.80K). The differences are only in the training loss (cumulative replaced by CT) and inference mechanism (late decision replaced by early decision). The difference between the two in Table 1 can be entirely attributed to these two points, which is a clever aspect of the paper's experimental design. The difference with the SCNN series ([8][9]) lies in the organization of the time axis: they compress the entire sample into an image and spend multiple time steps encoding it, whereas ED-sKWS makes network time isomorphic to speech time, unlocking frame-by-frame prediction from the root. The difference with Spiking-LEAF [11] (another ICASSP 2024 work by the first author) lies in the frontend: Spiking-LEAF learns a learnable auditory frontend and still uses late decision; when parameters are the same (306.80K), ED-sKWS achieves comparable accuracy (93.04% vs 93.02%, Table 1) but with a decision time of 60.46 vs 98 steps and energy consumption of 23.68 vs 29.20 µJ. The difference with MSAT [26] lies in the route: MSAT improves accuracy through ANN-to-SNN conversion using multi-stage adaptive thresholds and does not touch the decision point.

## Experimental Results

### Datasets Used and Their Scales

Two datasets (Section 4.1), both with a sampling rate of 16kHz. First, Google Speech Commands V2 [15]: 105,829 1-second utterances, 35 command words, serving as the standard arena for SNN KWS. Second, the self-built SC-100: 313,951 utterances, 100 command classes, 1,000-4,000 utterances per class, durations 0.4-1 second, each with start/end timestamps. The average keyword end time is approximately 62.96 time steps (Section 4.4). Preprocessing: fbank features are extracted, with 40 filter banks and a window length of 25 milliseconds. Since sample lengths in both libraries are fixed at 1 second, a total of 98 frames are obtained, and the number of SNN time steps is also 98. Two model scales: 128 hidden neurons (27.63K parameters) and 512 hidden neurons (306.80K parameters).

### Definition and Rationale for Evaluation Metrics

- **Early Decision Accuracy $Acc_t$**: The accuracy of the prediction given at the early stopping moment $t_d$. Rationale: This is the judgment the user actually receives in real deployment, serving as the primary metric for the early decision paradigm.
- **Late Decision Accuracy $Acc_T$**: The output accuracy after running to the final time step. Rationale: Maintains comparability with traditional late-decision methods, verifying that early decision does not come at the cost of sacrificing the model's upper limit.
- **Average Decision Time $\bar{t}_d$**: The mean of stopping steps for all test samples, measuring response speed.
- **Decision Lead Time $\Delta t_d$** (SC-100 only): $\Delta t_d = \frac{1}{N}\sum (t_d - t_{end})$, i.e., the average difference between the decision time and the true end time of the keyword. A negative value indicates a decision made before the end of the word. Rationale: The essence of response speed is "how much earlier relative to the word end," not "position within the sample." Zero-padding in GSC renders the position within the sample physically meaningless. This metric can only be calculated with time annotations, which is precisely the reason for the existence of SC-100.
- **Energy Consumption**: Estimated based on 45nm CMOS process [25] following conventions [22][23][24], with MAC operations at 4.6pJ and accumulation operations at 0.9pJ. Rationale: SNN inference is primarily event-driven accumulation; fewer spikes mean lower energy consumption. This is the common SNN energy metric in the academic community, making numbers across methods in Table 1 comparable.

### Detailed Comparison with Baseline and SOTA Methods

Table 1 (Comparison of SNN KWS Methods, Parameters, Accuracy, Decision Time $t_d$, Estimated Energy Consumption):

| Method | Parameters | Acc (%) | $t_d$ | Energy (µJ) |
|---|---|---|---|---|
| Yilmaz et al. [8] | 117K | 75.20 | 98 | Not reported |
| MSAT [26] | 500K | 87.33 | 98 | Not reported |
| sKWS [9] | 86.5K | 91.7 | 98 | 247.52 |
| Spiking-LEAF [11] | 306.80K | 93.02 | 98 | 29.20 |
| adLIF [16] (128 hidden) | 27.63K | 90.46 | 98 | 5.36 |
| adLIF [16] (512 hidden) | 306.80K | 93.12 | 98 | 45.41 |
| ED-sKWS (128 hidden, early decision) | 27.63K | 90.14 | 66.07 | 2.85 |
| ED-sKWS (128 hidden, late decision) | 27.63K | 90.52 | 98 | 4.74 |
| ED-sKWS (512 hidden, early decision) | 306.80K | 93.04 | 60.46 | 23.68 |
| ED-sKWS (512 hidden, late decision) | 306.80K | 93.15 | 98 | 34.62 |

(Note: The two rows for ED-sKWS with $t_d=98$ are variants of CT loss with the early decision mechanism turned off, inferred from the table values and abstract scope; its 512-tier $Acc_T=93.15$ is completely consistent with the CT loss row in Table 2, providing cross-validation.)

Core Readings and Attribution Decomposition (Ratios are calculated from Table 1 values):

- **Against adLIF Baseline** (Scope from Section 4.2): For the 512-tier, energy consumption drops from 45.41µJ to 23.68µJ, a decrease of approximately 48%, while accuracy drops only by 0.08% (from 93.12% to 93.04%). For the 128-tier, energy consumption drops from 5.36µJ to 2.85µJ, a decrease of approximately 47%, with accuracy dropping by 0.32%. The overall scope in the abstract and conclusion of "61% time steps, 52% energy" corresponds to 60.46/98 (61.7%) and 23.68/45.41 (52.1%), respectively.
- **Gains can be decomposed into two layers**: The first layer is the contribution of the CT loss itself (early decision turned off): For the 512-tier, 34.62 vs 45.41µJ, saving 23.7%, while accuracy increases by 0.03% (93.15 vs 93.12); for the 128-tier, 4.74 vs 5.36µJ, saving 11.6%, with accuracy increasing by 0.06%. The mechanism is that CT loss reduces the spike firing rate (explicitly stated in the conclusion), and SNN energy consumption is proportional to the number of synaptic operations. The second layer is the contribution of the early decision mechanism: For the 512-tier, it drops further from 34.68 to the 23.68µJ range, saving another ~30%, at the cost of a 0.11% accuracy rollback; for the 128-tier, it drops from 4.74 to 2.85µJ, saving another ~40%, at the cost of 0.38%. The two layers stack and are independent of each other, which is the most solid part of the paper's energy story.
- **Response Speed**: The 512-tier gives predictions on average about 36 time steps (also 36 frames) before the end of the sample, reducing inference latency by approximately 38% (Section 4.2 original text; calculated as 98 minus 60.46 equals 37.5 steps, 38.3%).
- **Horizontal Positioning**: The ED-sKWS 512-tier achieves 93.04% accuracy with 23.68µJ, representing the best balance of accuracy and energy efficiency in Table 1; the small model at 2.85µJ/90.14% is the lowest energy consumption in the entire field. The huge gap between sKWS [9]'s 247.52µJ and Spiking-LEAF's 29.20µJ also illustrates the dominant role of internal firing rate differences on energy consumption within SNNs, making unified standard comparisons essential.

### Findings from Ablation Experiments

**Loss Function Ablation (Table 2, GSC v2, 512-tier)**:

| Loss Type | $Acc_t$ (%) | $\bar{t}_d$ | $Acc_T$ (%) |
|---|---|---|---|
| Spike-rate [18] | 88.18 | 65.52 | 92.73 |
| TET [19] | 91.24 | 61.24 | 91.37 |
| Cumulative [16] | 93.02 | 66.23 | 93.12 |
| CT loss (This Paper) | 93.04 | 60.46 | 93.15 |

Three findings, each traceable to the loss form: First, CT is optimal or tied for optimal in all three metrics. The early decision accuracy of 93.04% is on par with Cumulative's 93.02%, but the average decision time of 60.46 steps is 5.77 steps faster than Cumulative. Why CT decides earlier: Cumulative only supervises the final cumulative output, leaving intermediate step confidence unconstrained and slow to rise; CT pushes the cumulative distribution toward the target at every step, causing confidence to cross the threshold earlier, triggering early stopping sooner, and resulting in lower energy consumption. Second, TET's final accuracy of 91.37% is actually lower than spike-rate's 92.73%: Isolated supervision of instantaneous outputs produces conflicting gradients in speech—information in the first few frames is objectively insufficient, and forcing instantaneous outputs at every step to align with labels drags down the representation of the final step; on static datasets where inputs are identical at every step, this conflict does not exist, which precisely validates the design argument in Section 2.2. Third, spike-rate has the worst early decision accuracy (88.18%): Intermediate step outputs are never supervised, leading to poor calibration of confidence $CS_t$, causing both the timing and reliability of early stopping to be out of control.

**SC-100 Experiments (Table 3)**:

| Loss Type | $Acc_t$ (%) | $\bar{t}_d$ | $\Delta t_d$ | $Acc_T$ (%) |
|---|---|---|---|---|
| Spike-rate [18] | 90.72 | 70.32 | +7.33 | 91.76 |
| TET [19] | 91.17 | 64.21 | +1.22 | 91.20 |
| Cumulative [16] | 93.20 | 68.28 | +6.32 | 93.07 |
| ED-sKWS (CT) | 93.21 | 59.11 | −3.85 | 93.16 |

CT is the only group with a negative $\Delta t_d$: On average, it makes a decision 3.85 time steps before the keyword ends, while other losses need to wait 1 to 7 steps after the word end to gather enough confidence. Self-consistency check: 59.11 plus 3.85 equals 62.96, which fits perfectly with the paper's reported average word end time of 62.96 steps for SC-100 (inferred), indicating that the definition and implementation of $\Delta t_d$ are consistent. Assuming 98 frames correspond to 1 second, the frame shift is approximately 10 milliseconds, so 3.85 steps translate to a real lead time of approximately 39 milliseconds.

An counter-intuitive phenomenon is exposed on SC-100 (Section 4.4): Early decision accuracy of 93.21% is slightly higher than late decision's 93.16%, opposite to the relationship on GSC (93.04% vs 93.15%). The paper's explanation: The average word end in SC-100 is at 62.96 steps, with the sample tail hanging about 35 steps of pure silence. SNNs have to force learned features to persist in the membrane potential until step 98. Long silence constitutes a memory burden, and late decision actually suffers from information decay; early decision allows the model to flexibly produce results near the keyword endpoint, avoiding the risk of information loss from long silence. This indirectly validates the motivation for selecting adLIF adaptive neurons from the data side, and also shows that early decision is not just for saving energy, but in certain data distributions, it is also a means to preserve accuracy.

**Energy Mechanism Visualization (Fig. 2, Section 4.5)**: A superimposed plot of the raw waveform and step-by-step spike firing rate for a single SC-100 sample. The yellow dashed line marks the true word start/end, the red dashed line marks the early decision time $t_d$, and the orange dashed line marks the late decision time $T$. Three observations: Firing rate is significantly lower before the keyword starts, reflecting the event-driven characteristic—neurons are not activated without audio input, making this silence segment nearly free; after the keyword starts, the firing rate clearly rises to 0.2, and the network enters a working state; after the word end, the firing rate does not drop immediately but maintains at a level of approximately 0.175 for several more time steps before slowly decreasing. This is because historical information exists in the membrane potential $U$, and spikes may continue even without new input. The third point is precisely the hidden cost of late decision: real synaptic operation energy is still being generated in the tail silence. Early decision cuts off everything at step $t_d$, zeroing out all computational operations between $t_d$ and $T$. This is direct evidence of the energy saving mechanism provided in Section 4.5.

## Main Contributions

1. **ED-sKWS Framework**: Frame-synchronized feedforward SNN combined with confidence threshold early decision. On GSC v2 and SC-100, it maintains accuracy comparable to late-decision SNNs with approximately 61% of time steps and 52% of energy consumption (Abstract, Conclusion), while improving both response speed and energy efficiency.
2. **CT Loss**: A cumulative supervision loss designed for speech early decision. It combines two previously mutually exclusive designs—"accumulation of historical evidence" and "step-by-step supervision"—into a single target, catering to both intermediate and final step accuracy, while incidentally reducing spike firing rate to save further energy.
3. **SC-100 Dataset**: A large-scale KWS evaluation library with 313,951 utterances, 100 commands, and start/end timestamps for each entry. It fills the gap where early decision research could not quantify response speed, and the annotations can be externally supplied to tasks such as VAD.
4. **Methodologically Clean Comparison**: Compared to the adLIF baseline, it shares the same architecture and parameter count, attributing all gains to the loss function and early decision mechanism. It quantifies the two layers of gains separately through variants with early decision turned off (inferred from Table 1).

## Limitations and Future Work

### Technical Limitations of the Method

- **Accuracy Ceiling of Fully Connected Shallow Architecture**: The model has only two hidden layers. The highest accuracy among all SNN methods in Table 1 is 93.15%. The paper does not compare with any non-SNN baselines, making it impossible to judge the true gap with mainstream KWS methods from within the paper alone. Whether convolutional or residual frontends (such as residual blocks in [9]) are compatible with the frame-synchronized design is not discussed.
- **Early Decision Threshold $C$ is a Black Box**: The specific value is not reported by the paper, nor is there a three-way scan curve of threshold-accuracy-energy. Users cannot select a working point based on this, and the sensitivity of $C$ and its stability across words are unknown.
- **Limited Lead Time and Lack of Variance Information**: On SC-100, $\Delta t_d$ is negative 3.85 steps (approximately 39 milliseconds, inferred from 10ms frame shift). The magnitude of the lead is not large. The paper does not report the variance of this quantity, nor does it perform confusion analysis on prefix-confusable words (word pairs with identical initial segments). The risk of misjudgment in conservative scenarios for early stopping is not quantified.
- **Energy Consumption is Estimated, Not Measured**: The operation counting standard of 4.6pJ/0.9pJ under 45nm CMOS has not been verified on real neuromorphic hardware. The energy consumption of fbank feature extraction is not included, and the extent to which total edge-side energy consumption is underestimated is unclear. Specific conversion details from spike counts to energy are not reported by the paper.

### Shortcomings in Experimental Design

- **Missing Training Details**: Optimizer, learning rate, number of training epochs, batch size, form of surrogate gradient functions, and the train/test split for SC-100 are not reported by the paper, limiting independent reproduction.
- **Missing Key Deployment Metrics**: The evaluation is closed-set multi-classification (35 classes and 100 classes). There is no unknown word rejection, nor is there an hourly false trigger rate. Controlling false wake-ups is precisely a veto metric for real KWS products. Early decision amplifies the risk of confidence crossing the line too early, and analysis of such errors is particularly absent.
- **Narrow Comparison Scope**: Comparisons are only made with internal SNN methods, without comparison to early-exit or adaptive computation depth methods on the ANN side. The relative position of early decision gains is unclear. All results are from a single run, with no variance across multiple seeds.
- **Domain and Construction Noise in SC-100**: Derived from LibriSpeech audiobook reading, the domain difference from near-field or far-field command speech is not discussed. KeywordMiner cuts words according to alignment boundaries; coarticulation at cut points may produce unnatural word boundaries, the impact of which is not evaluated. The six-scenario word list contains cross-group repetitions (e.g., set, white, second appear in multiple groups). The exact composition of the 100 classes and the deduplication criteria are not fully explained by the paper.

### Possible Directions for Future Improvement

- **Streaming Structure Frontend**: Transform convolutional or residual SNNs into a causal streaming form combined with a frame-synchronized time axis to seek higher accuracy ceilings. Replace fixed fbank with a learnable auditory frontend (such as Spiking-LEAF [11]), incorporating frontend energy consumption into the optimization.
- **Upgraded Stopping Criteria**: Perform confidence calibration (e.g., temperature scaling) on $CS_t$ to give the threshold probabilistic semantics. Use adaptive thresholds based on words or SNR. Replace the heuristic max softmax with a theoretical framework such as optimal transport or evidence accumulation (e.g., Drift Diffusion Model).
- **Hardware and System Verification**: Measure energy consumption and wall-clock latency on real event-driven chips. Include feature extraction overhead to form an end-to-end energy efficiency ledger.
- **Cascading with VAD**: The timestamps in SC-100 can directly train SNN VAD (similar to SVAD [13] in the same group), constructing a two-stage always-on pipeline of "VAD gating plus KWS early decision," further institutionalizing the free event-driven characteristic of silence segments.
- **Robustness Expansion**: Evaluate early decision behavior under far-field reverberation, noise, and multi-speaker conditions. Explore transfer to custom wake words and open vocabulary directions.
