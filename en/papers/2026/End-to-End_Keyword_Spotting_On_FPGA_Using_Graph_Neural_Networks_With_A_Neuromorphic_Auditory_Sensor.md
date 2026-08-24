# End-to-End Keyword Spotting on FPGA Using Graph Neural Networks with a Neuromorphic Auditory Sensor

- **Authors/Affiliations**: Wiktor Matykiewicz, Piotr Wzorek, Kamil Jeziorek, Tomasz Kryjak (AGH University of Krakow, Embedded Vision Systems Group / Computer Vision Laboratory, Poland); Tomás Muñoz, Antonio Rios-Navarro, Angel Jiménez-Fernández (University of Seville, Robotics and Computer Technology Laboratory, Spain)
- **Date**: May 2026 (arXiv v1 submitted on 2026-05-10)
- **Link**: https://arxiv.org/abs/2605.09570 (Code/Data/Hardware modules open source: https://github.com/vision-agh/NAS-GNN-KWS)
- **Keywords**: Neuromorphic Auditory Sensor (NAS), Graph Neural Network (GNN), FPGA Edge Deployment, Keyword Spotting (KWS), Event-Based Processing, Address-Event Representation (AER), Hardware-Aware Design

## Problem Statement

### Problem Background and Domain Pain Points

Speech entry (Keyword Spotting, KWS) on mobile robots and IoT devices faces a set of conflicting constraints: high prediction accuracy, low energy consumption, low latency, and always-on capability. Traditional KWS pipelines combine "fixed sampling rate + heavy DSP + frame-level features (e.g., MFCC) + quantized neural networks." Their underlying assumption is that data is uniformly dense in time and frequency dimensions. This means that even when most of the acoustic scene is silent or has a steady background, the system still transports and computes full tensors at a fixed rate, infinitely amplifying redundant computation in always-on scenarios.

Neuromorphic sensors offer an alternative path: mimicking the asynchronous operation of biological auditory systems, they generate sparse pulses (events) only when spectral energy changes, making data naturally sparse in both spatial (frequency channels) and temporal dimensions. If this sparsity is maintained in subsequent processing, the computational load scales with actual acoustic activity rather than consuming resources fixed by time. Neuromorphic Auditory Sensors (NAS) already have mature fully digital FPGA implementations (specifically [16] used in this paper), which convert raw digital audio streams directly into event streams, replacing complex arithmetic operations with simple logic gates and counters (the Spike Signal Processing, SSP paradigm). This serves as the physical starting point for the entire technical route of this paper. The value of the SSP paradigm deserves special emphasis: although traditional digital cochlea implementations (FPGA filter banks) are scalable, they still retain significant arithmetic overhead. In contrast, SSP moves the entire information representation into the pulse domain, where addition becomes counting and filtering becomes gate logic, enabling fully parallel auditory processing at extremely low power—the sensor-side power consumption of only 29.7 mW (cited from [16]) confirms this. This also explains why the true power bottleneck in the entire pipeline (approximately 0.5 W dynamic power) lies in the GNN inference side rather than the sensing side, directing the architectural optimization efforts entirely toward the network module in the following sections.

### Specific Deficiencies of Existing Methods

The paper identifies gaps in existing work from three perspectives:

1.  **Event-based audio processing methods rely on synthetic data.** SNNs (convolutional [27,11,28], recurrent [8,24,35,9,4]), State Space Models (SSM) [29,13], and event-graph methods [26,23] are three mainstream approaches, but evaluations are generally based on **synthetically generated** pulse datasets like SHD/SSC (each with 700 audio channels). The paper explicitly points out that these benchmarks "may not fully cover the variability, noise characteristics, and temporal irregularities in real NAS outputs"—the statistical distribution of synthetic events (channel uniformity, temporal jitter, event rate fluctuations) differs from the output of real silicon sensors. Whether models tuned on synthetic data still work on real sensors remains an unanswered question.
2.  **Sensor and network are deployed separately.** Previous GNN-KWS hardware work (the baseline [14] in this paper, implemented on SoC FPGA) only deployed the network side, with event data coming from offline simulation. The sensor, filtering, graph construction, and inference are not in the same physical closed loop; "end-to-end" only extends to the network input. Cross-device data transfer introduces additional latency, power consumption, and system complexity.
3.  **Event rates are too high and channel distribution is extremely unbalanced.** Real NAS produces a large volume of events (e.g., 128-parallel configuration averages 92.18 ± 21.03 kEv/s, with a peak of 518.69 kEv/s, see Table 1 in the paper), and low channel numbers (high frequency) have far more events than high channel numbers (low frequency). Feeding the raw event stream directly into graph construction incurs high computational overhead, and the graph structure is biased by the most active channels.

### Key Challenges Addressed by This Paper

How to integrate a real NAS and GNN inference into an end-to-end system on a **single FPGA**: raw audio in, prediction output every 10 ms, without CPU intervention or conventional signal preprocessing. This breaks down into four sub-problems: (1) How to select the sensor topology and number of channels to balance frequency resolution, event throughput, and FPGA resource/power consumption; (2) How to reduce the event rate to within the network's throughput capacity without harming (or even improving) accuracy; (3) The baseline GNN architecture [14] was designed for synthetic data and high resource headroom; how to modify the architecture to fit the target device and meet real-time throughput requirements; (4) How to define evaluation metrics that reflect "when a word ends," rather than just looking at classification accuracy. The final deliverable is a measured system with 87.43% post-quantization accuracy (32-parallel configuration), 25–35 µs end-to-end latency, and 1.12 W average power consumption.

## Methodology

### Overall Architecture Design and Design Motivation

The system features a three-stage data flow: **NAS Sensor → Event Filtering → Graph Construction + GNN Inference**, all implemented within the programmable logic (PL) of a single FPGA. Two top-level design decisions warrant elaboration on "why":

-   **All logic in PL, completely bypassing the Processing System (PS/CPU).** The target is mobile robots and edge scenarios, where energy consumption is listed as one of the primary constraints. Placing the entire pipeline in reconfigurable logic, with zero CPU participation from raw audio capture to prediction output, eliminates the power consumption and uncertain latency introduced by CPU intervention. This also explains why all modules must be "hardware-friendly"—downstream constraints such as shift-attenuation in the filtering algorithm and BRAM alignment for layer-wise feature widths stem from this decision.
-   **Using GNN instead of SNN to process event streams.** Event streams are sparse, asynchronous, and temporally irregular, naturally mismatching fixed-grid tensors; modeling events as nodes and encoding spatiotemporal adjacency relationships with edges allows locality and interaction patterns to be written directly into the graph structure. Meanwhile, event-driven computation aligns highly with FPGA architecture: sparse activity means fewer flips and memory accesses, directly translating to energy efficiency and latency gains.

**Sensor Level (Sec 3.1)**: Analog audio enters the FPGA via a standard jack interface and an external audio codec, delivered via I2S digital audio interface. On-chip, PCM samples are first converted into high-frequency pulse trains using Pulse Frequency Modulation (PFM), then decomposed into multiple frequency channels by a set of digital pulse-domain filters. Outputs are transmitted via an asynchronous AER interface. Native events are (c, p) pairs (channel number + polarity). The paper additionally designs an FPGA timestamp module that timestamps each event with resolution of 1 µs, forming the final (t, c, p) triplet—this step is the prerequisite for the temporal neighborhood in subsequent graph construction to function.

The paper implements and compares two filter bank topologies, each with three channel counts (32/64/128, totaling six configurations):

-   **Cascade Architecture**: Following the approach of the bionic cochlea [16], pulse low-pass filters (SLPF) are cascaded, with each stage filtering out higher frequencies to pass to the next. There are two issues: cumulative delay grows linearly with the number of channels; non-ideal filtering characteristics compound stage by stage, causing signal attenuation and suppressing the event rate.
-   **Parallel Architecture**: The input pulse train is fed simultaneously to a set of independent pulse band-pass filters (SBPF). This eliminates cumulative delay and avoids composite degradation, resulting in similar delays across frequency channels and higher event density.

Measured statistics (Table 1, full GSCv2 passed through the sensor) show that parallel events are approximately twice those of cascade: 64-channel cascade averages 24.63 ± 11.15 kEv/s vs. parallel 54.01 ± 16.14 kEv/s; 128-parallel reaches an average of 92.18 ± 21.03 kEv/s and a peak of 518.69 kEv/s. Under all configurations, low channel numbers (high frequency) have significantly more events than high channel numbers (low frequency) (Fig. 1)—this imbalance directly drives the subsequent per-channel threshold design. Increasing the number of channels improves frequency resolution, but at the cost of increased LUT/Slices resources and power consumption.

### Mathematical Principles of Core Algorithms

**Event Filtering (Algorithm 1)**: Mimics the Leaky Integrate-and-Fire (LIF) neuron model [1]. Each channel maintains two states: the timestamp of the last event $t_{\text{last}}[c]$ and the discrete potential $v[c]$. For each arriving event $(t_i, c_i, p_i)$:

$$\Delta t = t - t_{\text{last}}[c]$$

$$v[c] \leftarrow \max\left(0,\; v[c] - \left\lfloor \frac{\Delta t}{q} \right\rfloor + w\right), \qquad q = 2^{\text{div\_factor}}$$

If $v[c] < \theta_c$, the event is discarded; otherwise, the event is accepted into subsequent graph construction, and $v[c]$ is reset to 0. Three parameters: div_factor determines the time quantization granularity of attenuation, $w$ is the potential increment injected per event, and $\theta_c$ is the per-channel threshold.

This form features two careful hardware-oriented designs. First, the attenuation term $\lfloor \Delta t/q \rfloor$ becomes a simple right-shift operation when $q = 2^{\text{div\_factor}}$, completely avoiding dividers; the entire filtering requires only one counter, one comparator, and one adder per channel. Second, the "integrate-leak-threshold-reset" state machine has a constant number of states, independent of the total number of events, allowing parallel instantiation per channel. It is also worth noting: the algorithm reads the polarity $p$ but never uses it—filtering is insensitive to polarity, and polarity information is only used in the GNN input feature layer.

The generation of per-channel thresholds $\theta_c$ considers three simple strategies: constant, linear, and exponential, with their effects evaluated in ablation studies (see Experimental Results section). The design intuition comes from the channel imbalance in Fig. 1: low channel numbers (high frequency) have many events and require stronger suppression; high channel numbers (low frequency) have few events and require weaker suppression.

**Graph Construction and PointNetConv (Eq. 1)**: Each input event is first registered as a vertex in the graph; for each newly registered event, edges are built to **previously recorded** events (a time-directed graph ensuring causality). Neighbors must fall within a "hemisphere" search area defined by channel radius $R_c$ and temporal radius (lower support radius $R_t^{\text{low}}$ and upper support radius $R_t^{\text{high}}$). Feature extraction consists of four consecutive PointNetConv modules [25]:

$$x_i' = \max_{j \in \mathcal{N}(i) \cup \{i\}} \phi_\Theta\left(x_j,\; p_j - p_i\right)$$

where $x_i$ is the input feature vector of node $i$, $p_i$ is its position (channel + timestamp), $\mathcal{N}(i)$ is the neighborhood, and $\phi_\Theta$ is a learnable function. Point-wise transformation followed by symmetric max aggregation naturally adapts to neighbor sets of arbitrary size (each event has an uncertain number of neighbors)—a property required for event streams. The first layer uses two input features (neighbor average position: channel and timestamp). In hardware implementation, polarity is added as a third input feature. The number of output features per layer was selected via ablation: 72 output features pack exactly into 9 8-bit features = 72 bits, aligning with BRAM storage word width with zero waste—an example of "hardware-aware design" applied to storage word granularity.

**Temporal Aggregation and Network Head**: The MaxPool module aggregates event features within a 10 ms time window. At the end of the window, the aggregated features are sent to the network head (4 linear layers + GRU memory unit). Every 10 ms, two outputs are produced: **class** (target keyword set + one 'unknown' class aggregating all other words/silence/noise) and **conf** (a single-value output indicating the confidence that a keyword has ended within that time window). The 'conf' output for "end-of-word localization" is a key design for streaming applications: the downstream consumer cares about "when the word finished," not a classification label once per second.

### Key Technical Innovation 1: Neuromorphic GSCv2 Dataset Recorded with Real Sensors and Sensor Topology Selection

The paper records GSCv2 (over 100,000 1-second utterances) **entirely through physical NAS hardware** as an event dataset, rather than using software simulation. The acquisition pipeline: audio flows into the NAS via I2S codec, events are captured by the Opal Kelly XEM6310 FPGA module, transmitted to the host via a high-bandwidth USB 3.0 link using the FrontPanel SDK, and on-chip logic timestamps AER packets at the microsecond level. A Python script automatically synchronizes playback and capture, converting the data to .aedat format. The recorded dataset is 15–30% longer than the original corpus—this is **intentional padding**: pre-sampling delay ensures error-free ordered capture, and post-sampling delay prevents keywords from being compressed at the end of the recording window (experiments found this harms the learning process). Why insist on physical recording: to capture the sensor's real noise characteristics and temporal dynamics, which synthetic benchmarks (SHD/SSC) cannot provide; simultaneously, the six configuration datasets (double topology × triple channel count, complete statistics in Table 1) provide the first empirical basis for "how to choose a topology"—the conclusion is that parallel has higher accuracy (twice the events but more complete information, no frequency-dependent delay), at the cost of higher event throughput pressure.

### Key Technical Innovation 2: Hardware-Friendly LIF-like Event Filtering

The motivation was elaborated in the Problem Statement: excessive event rates + channel imbalance will bias graph construction. The solution is the per-channel LIF-style filtering in Algorithm 1. There are three translatable design judgments: (1) Filtering also serves as **noise reduction**—ablation shows accuracy rises from 74.36% without filtering to 84%+ with filtering, indicating that most filtered events are noise/redundancy rather than information; (2) Attenuation uses shift quantization ($q = 2^{\text{div\_factor}}$), allowing the entire algorithm to be directly synthesized into a lightweight state machine per channel; (3) Per-channel thresholds turn "suppress where there are many events" into a tunable parameter. Overall effect (as stated in the paper abstract): accuracy increases while the number of events is reduced by approximately 47%. Checking against Table 2b numbers: under 64-cascade, $R_c$=10/skip 1 configuration, average without filtering is 24.63 kEv/s, dropping to 11.2 kEv/s with div_factor=8, i.e., a reduction of approximately 54.5%; "approximately 47%" should be a generalized value across configurations (actual reduction rates range from 33% to 55% depending on configuration).

### Key Technical Innovation 3: GNN Architecture Modification for Single-Chip Deployment

This is the part with the highest engineering content in this paper, with four modifications each having a clear causal chain:

1.  **Polarity as Input Feature.** The baseline [14] used SHD/SSC data which lacks the concept of polarity; NAS natively provides polarity. Adding it to the input features of the first convolutional layer effectively introduces a discriminative dimension distinguishing energy increase/decrease direction for free.
2.  **Parallel Multipliers reduced from 4 groups to 2.** LUTs are primarily occupied by vector multipliers in PointNetConv. The baseline supports 4-way parallel feature vector multiplication per graph convolution (each path 74 8-bit elements); this paper derives explicit throughput requirements based on measured post-filtering event rates (Table 3 averages/peaks), reducing parallelism to 2 (two 72-element vectors concurrently). The saved LUTs/DSPs make it feasible to fit the entire system (including NAS) into the target device.
3.  **Back-pressure scheduling + inter-module buffering.** The computation volume of a single PointNetConv varies with the number of neighbor edges; the baseline sets a fixed event acceptance interval based on the worst-case neighbor count to ensure "no congestion," at the cost of constant throughput locked by the worst case, introducing unnecessary latency during normal operation. This paper changes it to: each processing level has local buffering, sends a ready signal after completing the current convolution, and accepts events dynamically when downstream resources are available. Latency becomes event-dependent, and input rate adapts to instantaneous load—throughput remains sufficient even with halved multipliers. A FIFO is added between the filtering and graph construction modules to absorb short-term bursts.
4.  **Timestamp propagation mechanism.** After the scheduling mechanism broke the fixed throughput assumption, MaxPool's original assumption of "triggering every 10 ms based on a real-time counter" became invalid. It is modified to monitor NAS output timestamps, propagating the timestamp of the last event in each time window to MaxPool. MaxPool only sends features to the network head after processing the last event of the corresponding 10 ms window—ensuring correct window semantics rather than guessing by clock.

Supporting storage organization: BRAM usage is decoupled from the number of channels (each entry stores 9 8-bit features = 72 bits, 128-channel configuration requires depth 1024). This is also the reason why the 72-feature width was "hard-selected" in ablation.

### Technical Differences from Existing Methods

Comparison with three categories of objects: (1) **SNN Route** (SpikGRU, Recurrent SNN, DCLS-Delays, TSkips, SE-adLIF, PfA SNN, etc.) reports 72.0–80.7% accuracy on SSC, with 0.1–3.9M parameters, all relying on synthetic events; (2) **SSM Route**'s Event-SSM reports 85.3/88.4% on SSC (0.1/0.6M parameters), the only method with peak accuracy slightly higher than this paper's; (3) **This paper's direct predecessor [14]**: Also GNN-on-FPGA, but evaluated on synthetic SHD/SSC (700 channels), 4-way parallel multipliers, fixed acceptance, no sensor, no polarity, reporting 78.4–84.3% (8.6k–272k parameters). The difference in this paper is systematic rather than point-specific: real sensor data, single-chip end-to-end closed loop including sensor, polarity features, halved multipliers + back-pressure scheduling + timestamp propagation, 59.84k parameters. Compared to **conventional FPGA KWS** (pre-computed features + prediction output every 1 s, e.g., BNN accelerator on VC707 97.29%/536 clock cycles [36], configurable TENN 95.36% [12]), this paper's accuracy is significantly lower, but the differentiated claim is: direct processing of raw audio (no overhead/latency/power of pre-computed features) + 10 ms time resolution for prediction output, which are attributes truly needed for streaming interaction scenarios (robot HMI).

## Experimental Results

### Datasets Used and Their Scale

Training uses GSCv2 (over 100,000 1-second utterances), generating six neuromorphic versions (32/64/128 channels × cascade/parallel), all recorded via physical NAS. Table 1 provides complete statistics for each configuration (mean ± standard deviation + max, unit: kilo-events): 32-cascade average 17.22 ± 9.58 kEv/sample (average 15.03 ± 8.21 kEv/s), 32-parallel 39.19 ± 12.65 (28.08 ± 9.12 kEv/s), 64-cascade 31.67 ± 13.96 (24.63 ± 11.15 kEv/s), 64-parallel 75.40 ± 21.41 (54.01 ± 16.14 kEv/s), 128-cascade 61.27 ± 20.94 (45.49 ± 16.01 kEv/s), 128-parallel 126.69 ± 24.11 (92.18 ± 21.03 kEv/s). The highest single-sample peak event count reaches 265.96 kEv (128-parallel), and the highest peak event rate is 518.69 kEv/s. These numbers indicate two things: first, the standard deviation of event rates generally reaches about 40% of the mean, with large fluctuations, meaning hardware must design throughput or configure buffers based on peaks rather than averages; second, when the number of channels doubles, the event volume approximately doubles, meaning the cost of frequency resolution linearly translates to backend computational load. The recorded corpus is 15–30% longer than the original (intentional padding). The dataset, software model, and hardware modules are all open source.

### Definition and Rationale for Evaluation Metrics

-   **Acc. (Classification Accuracy) and macro F1**: The paper does not explicitly argue why macro F1 was chosen; inferring from the task structure (inference, not from the paper text), the 'unknown' class aggregates all other words + silence + noise, leading to natural class imbalance. Macro averaging prevents large classes from drowning out per-keyword performance.
-   **Ts-acc$_k$ (Timestamp Conditional Accuracy)**: Counted as correct only if the class is correct **and** the predicted word end time falls within ±k 10 ms bins. The rationale is clearly stated in the paper: end-of-word time localization "significantly improves practical performance"—in streaming KWS, correct class but wrong time is equivalent to false alarm/missed detection. This metric incorporates the quality of the 'conf' output into the evaluation.
-   **Event Rate Statistics (kEv/s, edges/Ev)**: Directly correspond to hardware throughput requirements, serving as a bridge between "algorithm metrics" and "hardware metrics."
-   Ablation stages all use FP32; quantized models are evaluated separately in the hardware section.

### Detailed Comparison with Baseline Methods and SOTA

Training setup: FP32 trained for 50 epochs (Adam, lr 1e-3, weight decay 1e-4, cosine annealing), then fine-tuned for 5 epochs with 8-bit Quantization Aware Training (QAT) (constant lr 1e-4), batch size 16, NVIDIA GH200, PyTorch, checkpoint selected by minimum validation loss.

**Table 3 (Optimal combination after ablation for each of the six configurations)**: Time radius, div factor, and threshold are unified to $R_t^{\text{low/high}}$=0/5000, div 8, exponential threshold 64→32. Only $R_c$/skip varies by configuration (32 channels 5/1, 64 channels 10/1, 128 channels 20/2). Results: 32-parallel achieves highest accuracy 88.03% (F1 79.36); 64-parallel achieves highest Ts-acc₁ 75.12% (Ts-acc₃ 81.85%, Acc 87.87%); parallel outperforms cascade across the board—despite parallel having more than twice the events of cascade. Cascade's frequency-dependent delay directly drags down the Ts-acc metric, providing empirical evidence that "sensor topology affects time localization accuracy."

**Table 4 (Comparison with SOTA, note baseline is on SSC, this paper is on its own NAS dataset)**: This paper's six configurations range from 83.15–88.03%, all with 59.84k parameters, exceeding all SNN methods (72.0–80.7%) and the GNN predecessor [14] (78.4–84.3%). The only exception is Event-SSM's peak of 88.4%, slightly higher than this paper's best 88.03%, but this paper's model is smaller and targeted for single-FPGA end-to-end deployment. Cross-dataset comparisons can only serve as directional references, as the paper itself makes this limitation clear.

**Table 5 (Hardware measurements, AMD Zynq UltraScale+ ZCU104, 200 MHz, no timing violations)**, comparing with baseline GNN [14]:

| Metric | GNN [14] | 64-parallel (This Paper) | 32-parallel (This Paper) |
|---|---|---|---|
| LUT | 125,130 | 88,370 | 82,739 |
| FF | 82,372 | 84,031 | 77,457 |
| BRAM | 75.5 | 55.5 | 55.5 |
| DSP | 140 | 83 | 83 |
| Latency (µs) | 10.53 | 25 (42) | 25 (35) |
| Throughput (keps) | 555 | 245 | 440 |
| Power (W) | 1.18 | 1.16 | 1.12 |
| Accuracy | 73.5% | 86.9% | 87.43% |

Significant resource reduction (LUT −29.5%/−33.9%, DSP −40.7%, BRAM −26.5%) comes at the cost of increased latency (10.53 → 25–42 µs)—but the absolute values are still more than two orders of magnitude smaller than the 10 ms decision window. Note that the accuracy comparison in the table involves different datasets (baseline 73.5% corresponds to its deployment configuration; [14] reports 84.3% in Table 4).

**Latency Breakdown**: Prediction occurs every 10 ms; effective GNN latency (from window end to prediction) typically equals network head latency of 2.11 µs, worst-case 18.62 µs; NAS input to event latency is 23 µs; end-to-end total is 25–42 µs (32 channels 25–35 µs). Event acceptance interval is 0.47–4.07 µs (depending on edge count), throughput is 245 kEv/s–2.1 MEv/s, all higher than the measured event rates for each configuration (highest 64-parallel peak 147.69 kEv/s, margin approx. 1.66x).

**Power Breakdown**: Vivado post-implementation simulation, flip rate estimation under 40 kEv/s load—64 channels (average 18 edges/Ev) dynamic 0.539 W + static 0.595 W; 32 channels (9.8 edges/Ev) dynamic 0.494 W + static 0.595 W; NAS included at [16] reported value of 29.7 mW, totaling 1.16 W / 1.12 W.

**Quantization Loss**: After 8-bit QAT, 32-parallel drops from 88.03% to 87.43%, 64-parallel from 87.87% to 86.9%, i.e., a loss of approximately 0.6–0.97 percentage points.

### Findings from Ablation Experiments

Ablation baseline (64-cascade test set): 72 channels/layers, $R_c$=20 skip 2, $R_t$=2000/10000, div 8, weight 32, exponential threshold 64→32, accuracy 78.70%.

1.  **Model Width**: {18,36,54,72,90,108,126} channels/layers, 126 is optimal (79.45%/F1 65.51%) but parameters reach 179.56k; 72 channels achieves 78.70%/62.29% with 59.84k parameters, with diminishing returns, hence 72 is chosen—a typical sweet-spot selection driven by hardware budget.
2.  **Time Radius**: Increasing $R_t^{\text{low}}$ from 0 to 500/1000 causes average F1 to drop by 1.93%/5.53% respectively—information from immediately past events is most valuable, excluding neighbors causes significant harm; optimal $R_t^{\text{low}}$=0, $R_t^{\text{high}}$∈{2500,5000} (84.5%), indicating distant historical events also have limited contribution. Conclusion: the model benefits from **restricted spatiotemporal context**, and wider neighborhoods introduce more irrelevant events.
3.  **Channel Radius/Skip**: Under paired settings (10/1, 20/2, 30/3, 40/4, 50/5) keeping the maximum neighbor count constant (20 + self-loop), increasing radius or skip both degrade performance; optimal $R_c$=10, skip 1 (85.89%/76.99%), i.e., **small and dense neighborhoods** outperform large and sparse ones.
4.  **Filtering div factor**: Without filtering 74.36%, 24.63 kEv/s; div 6→10 accuracy monotonically rises from 83.59% to 84.93% but event rate also rises (7.9→12.1 kEv/s), reflecting the trade-off between noise reduction and mis-deletion of information events; div 8 is selected (84.46%, event rate approx. halved).
5.  **Threshold Strategy**: Linear 48→16 is optimal (85.22%/75.61%), slightly better than constant 48 (84.29%); filtering effectively fails when threshold ≤ w. This validates the intuition that "low channels (high frequency) need strong suppression, high channels (low frequency) need weak suppression." Notably, the final six configurations in Table 3 uniformly use exponential 64→32 rather than the ablation-optimal linear 48→16—the paper does not explain this choice discrepancy.

## Main Contributions

1.  **First (to authors' knowledge) single-FPGA end-to-end event-based audio KWS**: Physical NAS and GNN integrated on the same chip, raw audio in, class + conf output every 10 ms, zero CPU participation, no conventional signal preprocessing.
2.  **Public real neuromorphic recording dataset**: Six NAS configuration versions of GSCv2 (with complete statistics), providing infrastructure to pull this direction out of reliance on synthetic data.
3.  **Hardware-friendly event filtering method**: LIF-like, shift-attenuation, per-channel thresholds, reducing events by approx. 47% while accuracy increases.
4.  **A set of ablation-driven GNN architecture modifications**: Polarity features, halved multipliers, back-pressure scheduling + inter-module buffering, timestamp propagation—saving resources (LUT/DSP/BRAM) and power while maintaining throughput and accuracy, a methodology transferable to other event-driven accelerator designs.

## Limitations and Future Work

### Technical Limitations of the Method

-   **Obvious accuracy ceiling.** The best configuration 88.03% (FP32) is still significantly lower than the 95–97% level of conventional FPGA KWS on the original GSCv2. The paper's differentiation defense of "raw audio direct input + 10 ms time resolution" is valid, but for product scenarios where false alarm rate is a hard metric, this gap is a real cost.
-   **Power consumption is still in the watt range.** Static power accounts for 0.595 W in the 1.12 W total—this is the physical baseline of UltraScale+ devices, unrelated to algorithms; for always-on KWS, it is not on the same order of magnitude as MCU-level milliwatt solutions. The dynamic part (approx. 0.5 W) is also much higher than the sensor itself (29.7 mW), with the bottleneck at the network rather than the sensing end.
-   **Limited throughput margin.** Worst-case throughput of 245 kEv/s provides only about 1.66x margin for 64-parallel peak of 147.69 kEv/s, relying on FIFO to absorb bursts; the paper does not report event loss behavior under sustained overload.
-   **Limited graph storage depth** (128-channel configuration depth 1024 entries), indirectly limiting long-neighborhood context due to storage constraints.

### Deficiencies in Experimental Design

-   **SOTA comparison on different datasets**: In Table 4, baseline methods are on SSC, while this paper is on its self-built NAS dataset. Although the paper makes a limitation statement, strictly speaking, this does not constitute a same-benchmark comparison.
-   **Numerical consistency issues**: Numbers for the same-named configurations in Table 3 and Table 4 differ slightly (32-cascade 84.28 vs 84.61, 128-cascade 85.13 vs 85.03); Section 5 summary attributes 87.43% to the "64-channel configuration," while Section 4.4 and Table 5 clearly state this number belongs to 32-parallel; the abstract's "approx. 47% event reduction" differs in scope from the approx. 54.5% implied by Table 2b (and the 33%–55% range for different configurations); the ablation-optimal threshold (linear 48→16) was not adopted in the final configuration and lacks explanation. These do not affect the direction of conclusions but weaken the rigor of the reproducibility narrative.
-   **Key details not reported in the main text**: Target keyword set and number of classes, training/validation/test split methods, energy per FLOPS/per inference are not reported (power is a simulation estimate under fixed 40 kEv/s load, not a physical measurement).
-   **Absence of robustness evaluation**: The dataset is a single-condition recording of clean GSCv2 passed through the sensor, with no added noise/long-range/multi-speaker/real acoustic scene evaluations—this is precisely the mirror image of the synthetic data shortcomings pointed out by the paper itself, and generalization ability in real environments remains unverified.

### Possible Directions for Future Improvement

The paper mentions two directions: validating the system in real robot human-machine interaction scenarios; exploring ASIC implementation to break through the FPGA static power floor and achieve better energy efficiency and area. Combining with the gaps exposed in this paper (analytical inference): perform noise-robust training and data augmentation in the event domain to supplement generalization evidence; make filtering thresholds/parameters learnable and jointly train with the network (current filtering parameters are purely manually ablated); expand the keyword vocabulary and support user-defined keywords; use lower bit-width (e.g., binarization) features and weights to further compress LUT/DSP; utilize the NAS's existing binaural implementation for joint modeling of sound source direction, serving the spatial auditory needs of robot HMI.
