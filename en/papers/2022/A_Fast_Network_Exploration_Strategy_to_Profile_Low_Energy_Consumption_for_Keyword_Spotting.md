# A Fast Network Exploration Strategy to Profile Low Energy Consumption for Keyword Spotting

- **Authors/Affiliations**: Arnab Neelim Mazumder, Tinoosh Mohsenin (University of Maryland, Baltimore County, USA)
- **Date**: February 2022 (tinyML Research Symposium 2022)
- **Link**: https://arxiv.org/abs/2202.02361
- **Keywords**: keyword spotting, neural architecture search, MFCC, neural networks, FPGA, energy profiling, hardware-aware design

## Problem Statement

### Problem Background and Domain Pain Points
The deployment of keyword spotting (KWS) systems on battery-powered edge devices (such as smartwatches, wireless earbuds, and IoT sensor nodes) faces the most stringent energy consumption constraints. In these devices, KWS systems need to run continuously (always-on), with an inference interval of approximately 10-30ms, and the energy consumption per inference needs to be in the microjoule (uJ) range. For example, a CR2032 coin cell battery (capacity ~225mAh, 3V) has a total energy of about 2.4 Joules. If the KWS consumes 100uJ per inference and infers once per second, the battery would last only about 6.7 hours—far short of the days or months of battery life required.

The design parameters of neural networks have significant and complex impacts on both accuracy and energy consumption: the quantization bit-width $q$ (e.g., 2-bit, 4-bit, 8-bit) simultaneously affects model storage (Flash read energy), computation energy (bit-width of MAC operations), and accuracy; the scaling factor $s$ (which controls the number of filters per layer relative to a baseline) simultaneously affects total computation, parameter count, and accuracy. The effects of these parameters are non-linear and exhibit interaction effects—the combined effect of $q$ and $s$ is not a simple superposition of their individual effects.

### Specific Shortcomings of Existing Methods
- **Trial-and-error network design**: Traditional KWS network design relies on empirically selecting several fixed configurations (e.g., {8-bit, s=1.0}, {4-bit, s=0.5}, etc.), training them one by one, deploying them to hardware, measuring energy consumption, and then comparing. This exhaustive method is extremely costly when the configuration space is large: each configuration requires full training (hours to days) + FPGA synthesis and deployment (tens of minutes to hours) + energy measurement. When the configuration space contains hundreds to thousands of candidate configurations, a comprehensive search is infeasible.
- **Accuracy-oriented design ignores energy consumption**: Most NAS methods aim to maximize accuracy (e.g., DARTS, ENAS), treating energy consumption merely as a post-hoc constraint or considering it only at the final selection stage. This design may find high-accuracy configurations that consume too much energy, failing to meet strict battery budgets.
- **Lack of systematic joint optimization**: Existing works usually explore only a single dimension—such as tuning only the quantization bit-width (Quantization-Aware Training studies) or only the network width (width multiplier studies). There is a lack of systematic methods for joint optimization considering both the scaling factor $s$ and the quantization bit-width $q$. However, the interaction effect between $s$ and $q$ has a significant impact on the selection of optimal configurations—for example, when $s=0.5$, the accuracy loss from dropping from 8-bit to 4-bit may be larger than when $s=1.0$ (because narrower networks have lower tolerance for quantization).

### Key Challenges Addressed by This Paper
How to quickly and accurately predict the KWS accuracy and FPGA energy consumption for different $(s, q)$ configurations, thereby finding Pareto-optimal configurations without extensive actual training and hardware measurements (i.e., maximizing accuracy under a given energy constraint, or minimizing energy under a given accuracy constraint).

## Methodology

### Overall Design Philosophy
This paper proposes a fast network exploration technique based on polynomial regression, modeling the mapping of KWS network configurations $(s, q)$ to performance (accuracy + energy consumption) as a mathematical function. The core idea is: train models and measure actual energy consumption on a small number of "anchor" configurations, then fit a regression model using this data, and finally use the regression model to predict the performance of the entire configuration space—reducing the evaluation time for each new configuration from "hours (training + deployment + measurement)" to "milliseconds (querying the regression model)."

### Mathematical Principles of the Core Algorithm

**Accuracy Prediction Model**:

Input: Configuration parameters $(q, s)$, where $q$ is the quantization bit-width (discrete values, e.g., 2, 4, 8), and $s$ is the scaling factor (continuous values, e.g., 0.25, 0.5, 0.75, 1.0).

Output: Predicted KWS accuracy $\hat{A}(q, s)$.

Model Form: Uses a custom polynomial equation, specifically a polynomial containing $q$, $s$, and their interaction terms:

$$\hat{A}(q, s) = \beta_0 + \beta_1 q + \beta_2 s + \beta_3 q^2 + \beta_4 s^2 + \beta_5 qs + \beta_6 q^2 s + \beta_7 qs^2 + \beta_8 q^2 s^2$$

This is a 9-parameter second-order polynomial model (containing first-order, second-order, and interaction terms for $q$ and $s$). The choice of polynomial order is based on the bias-variance trade-off: a first-order polynomial (linear model) underfits and cannot capture the non-linear jumps in accuracy relative to quantization bit-width (e.g., the sharp drop from 4-bit to 2-bit); third-order and higher-order polynomials overfit, producing unreasonable predictions when training data is sparse.

**Energy Estimation Model**:

Input: Configuration parameters $(q, s)$.

Output: Predicted FPGA inference energy $\hat{E}(q, s)$ (unit: microjoules/uJ).

The energy model considers the impact of quantization bit-width and scaling factor on three energy components (computation energy, memory access energy, and control logic energy). Since energy consumption on FPGAs is dominated by dynamic power (proportional to the toggle rate):

$$\hat{E}(q, s) \approx \alpha \cdot q \cdot s^2 + \beta \cdot s + \gamma$$

where the term $\alpha \cdot q \cdot s^2$ approximates computation energy (proportional to bit-width and computation volume), the term $\beta \cdot s$ approximates memory access energy, and $\gamma$ is the fixed overhead.

The model is fitted based on actual FPGA-measured energy data, thereby implicitly including various practical factors in FPGA implementation (such as routing delay, clock tree power consumption, BRAM access latency, etc.).

**Model Training Data Collection**:

Select $n$ "anchor" configurations (e.g., $n=15-20$) covering representative regions of the $(q, s)$ space:
- Quantization bit-width $q \in \{2, 4, 8, 16\}$ (covering the quantization space)
- Scaling factor $s \in \{0.25, 0.5, 0.75, 1.0\}$ (uniformly spaced)
- At each anchor: (1) Train the DS-CNN model (full training, approx. 30 epochs); (2) Quantize to the target bit-width; (3) Deploy to FPGA and measure energy consumption.

### Pareto Optimal Analysis

Plot the Pareto Front in the 2D space of (Accuracy, Energy):
- Each configuration on the Pareto Front represents an optimal trade-off point—no other configuration exists that has both higher accuracy and lower energy consumption.
- Designers can select configurations on the Pareto Front based on application requirements (e.g., "minimize energy consumption while ensuring accuracy is no less than 90%").
- The shape of the Pareto Front reveals the "intrinsic trade-off structure" of the configuration space—for example, if the front is particularly steep in certain regions, it indicates that the cost of accuracy improvement in that region is high (energy consumption increases sharply).

### Parameterized FPGA Accelerator Design

To enable rapid evaluation of energy consumption for different configurations on FPGA, this paper designs a parameterized hardware accelerator:
- **Scalable Processing Engine (PE) Array**: The number of PEs is configurable (1-64), supporting different levels of parallelism. Each PE executes a set of MAC operations, and PEs are interconnected via an on-chip network.
- **Flexible Precision Support**: Supports arbitrary quantization bit-widths (2-bit to 16-bit), with computing units that can be dynamically configured. At low precision, multiple weights/activations are packed into a single DSP unit.
- **Dataflow Optimization**: Optimizes the dataflow for the computational pattern of KWS DNNs (large amounts of 1x1 convolutions and depthwise separable convolutions) to maximize data reuse—specifically adopting an output-stationary dataflow to reduce accesses to off-chip memory.
- **Target Platform**: Xilinx AC 701 (Artix-7 FPGA), providing approximately 75K logic cells, 4.9Mb BRAM, and 240 DSP units.

### Feature Extraction and Network Architecture
- **Input Features**: MFCC (Mel Frequency Cepstral Coefficients), 13-dimensional coefficients + delta (first-order difference) + delta-delta (second-order difference) = 39-dimensional features.
- **Baseline Network**: A DS-CNN-based architecture (Depthwise Separable Convolutional Neural Network), where the number of filters per layer is controlled by the scaling factor $s$.

## Main Contributions

1. **Fast Network Exploration Method Based on Regression**: For the first time, polynomial regression is applied to model the KWS network configuration space, transforming the selection of network configurations from a time-consuming process of "training + measurement" (hours per configuration) to an instantaneous process of "querying the prediction model" (milliseconds per configuration). This method is highly practical—the total time for the entire exploration process (data collection + model fitting + prediction) is equivalent to the training + measurement time of about 20-30 configurations (approx. 1-2 days), but it can predict the performance of hundreds of configurations, effectively improving exploration efficiency by approximately 20-50 times.

2. **Joint Accuracy-Energy Optimization**: Unlike traditional NAS methods that optimize only accuracy, this paper models both accuracy and energy consumption as objectives, providing a complete configuration selection framework through Pareto analysis. Designers can make informed decisions intuitively between "higher accuracy" and "lower energy consumption" without repeatedly balancing the two metrics.

3. **Parameterized and Scalable FPGA Accelerator**: Provides an FPGA accelerator design supporting arbitrary precision (2-bit to 16-bit) and arbitrary parallelism (1-64 PEs), enabling fair comparison of energy consumption for different configurations on a unified hardware platform. This accelerator design itself is also an effective reference implementation for deploying KWS on FPGAs—its output-stationary dataflow and flexible precision support can be reused in other KWS hardware implementations.

4. **Significant Energy Efficiency Improvements**: Compared to recent KWS hardware implementations, it achieves at least a 2.1x improvement in energy consumption and a 4x improvement in energy efficiency (Accuracy/Energy ratio). The optimal configuration maintains 90.1% accuracy while keeping energy consumption in the microjoule range.

## Experimental Results

### Datasets Used and Their Scale
- **Google Speech Commands V1-12**: Standard 12-class KWS benchmark, with approximately 65,000 1-second speech clips. Uses MFCC feature extraction (39-dimensional features).
- **Training Configuration**: Standard training/validation/test split.

### Definition and Rationale for Evaluation Metrics
- **Accuracy (%)**: Classification accuracy, measuring model performance.
- **Energy per Inference (uJ/inference)**: Actual single-inference energy consumption measured on FPGA. Energy is chosen over latency because, on battery-powered devices, total energy consumption (rather than single-inference latency) determines battery life.
- **Energy Efficiency (Accuracy/Energy)**: The ratio of accuracy to energy consumption, measuring performance per unit of energy.
- **RMSE (Root Mean Square Error)**: Prediction accuracy of the regression model.

### Performance of the Accuracy Prediction Model
- The RMSE of the polynomial regression equation is 0.9 (within the 0-100% accuracy range), meaning the prediction error is approximately plus/minus 0.9 percentage points.
- Main sources of prediction error: (1) Non-linearity of quantization effects—the accuracy jump from 4-bit to 2-bit (which may drop sharply by 5-10%) is difficult to capture precisely with polynomials; (2) Training randomness—accuracy fluctuations between different training runs are about 0.3-0.5%.
- The model predicts reliably within the region covered by training data ($q \in [2, 16], s \in [0.25, 1.0]$), but predictions may be inaccurate in extrapolation regions (e.g., $q=1$ or $s=2.0$).

### Performance of the Energy Prediction Model
- The relative error of energy prediction is <10% (within the range of configurations covered by training data).
- The energy model is easier to fit than the accuracy model because energy consumption is mainly determined by bit-width and computation volume, resulting in more regular variation patterns.

### Pareto Front and Optimal Configuration

The Pareto Front reveals several key insights:

**Impact of Quantization Bit-Width**:
- **8-bit to 4-bit**: Energy consumption decreases by approximately 40-50%, while accuracy drops only by 1-2%. This is the "best cost-performance" optimization point—exchanging a small accuracy cost for nearly half the energy savings.
- **4-bit to 2-bit**: Energy consumption further decreases by approximately 25-30%, but accuracy may drop sharply by 5-10%. This indicates that 2-bit quantization is too aggressive for the DS-CNN architecture—the model's expressive power is insufficient to maintain performance at such low precision.

**Impact of Scaling Factor**:
- **$s$ from 1.0 to 0.5**: Computation volume (and energy consumption) decreases linearly by approximately 75%, while accuracy drops by approximately 3-5%. The magnitude of energy reduction is greater than the reduction in computation volume (because memory access energy also decreases).
- **$s$ from 0.5 to 0.25**: Computation volume further decreases by approximately 75%, but accuracy drops by approximately 8-12%, and Pareto efficiency drops sharply (the rate of accuracy loss exceeds the rate of energy savings).

**Optimal Configuration Interval**:
- Optimal configurations on the Pareto Front are concentrated in the range of $(q=4, s=0.5)$ to $(q=8, s=0.75)$.
- Best trade-off configuration: $(q=4, s=0.5)$, achieving 90.1% accuracy, with energy consumption approximately 25% of the baseline configuration $(q=8, s=1.0)$.

### Comparison with Existing Hardware Implementations

| Comparison Item | This Paper's Optimal Config | Recent KWS FPGA Implementations | Improvement Factor |
|:---|:---:|:---:|:---:|
| Energy (uJ/inference) | ~3-5 | ~10-15 | 2.1-5x |
| Energy Efficiency (Accuracy/Energy) | ~18-30 | ~5-8 | 3-4x |

Energy improvements stem from: (1) 4-bit quantization reducing memory access and computation power; (2) Scaling factors of 0.5-0.75 reducing total computation volume; (3) Optimized FPGA dataflow reducing off-chip memory accesses.

## Limitations and Future Work

### Technical Limitations of the Method
- **Extrapolation Risk of Regression Models**: Polynomial regression performs well within the configuration space covered by training data (RMSE=0.9), but predictions for configurations outside the training data range (e.g., 1-bit quantization, extended configurations like $s=2.0$, or non-standard scaling factors like $s=0.37$) may be inaccurate. Extrapolation with polynomials tends to produce unreasonable predictions (e.g., accuracy >100% or <0%) because polynomial functions tend toward positive or negative infinity outside their boundaries.
- **Limited to CNN Architectures**: The regression model is fitted based on training data from the DS-CNN architecture. For RNN architectures (such as LSTM, GRU) or Transformer architectures, the structure of the configuration space may be completely different (e.g., parameters such as the number of recurrent layers, hidden state dimensions, etc.), requiring new data collection and model fitting.
- **Limited Dimensionality of Configuration Space**: Only two dimensions, the scaling factor $s$ and the quantization bit-width $q$, are considered, excluding other important factors such as network depth (number of layers), convolution kernel size, and attention mechanisms. The actual configuration space may be 5-10 dimensional, and polynomial regression requires exponentially increasing training data in high-dimensional spaces (the curse of dimensionality).
- **Expressiveness Limitations of Polynomial Models**: Polynomial regression cannot precisely capture sharp non-linearities in the configuration space (such as the accuracy jump from 4-bit to 2-bit). Higher-order polynomials can fit more complex patterns but require more training data to avoid overfitting.

### Shortcomings in Experimental Design
- **FPGA-Specific Results**: Energy data was measured on the Xilinx AC 701 (Artix-7 FPGA). The energy model for FPGAs differs significantly from other hardware platforms (such as ARM MCUs, dedicated ASICs, RISC-V processors): (1) FPGA power consumption is dominated by static power (transistor leakage) and routing power, whereas MCU power consumption is dominated by dynamic power (toggle rate); (2) FPGA memory access uses BRAM (on-chip block RAM), while MCUs use SRAM and Flash, with access energy differing by several times. Therefore, the optimal configuration on an FPGA may not be the optimal configuration on an MCU.
- **Limitations of Model Complexity**: While polynomial regression is simple and efficient, it may not capture complex non-linear relationships in the configuration space. More flexible models (such as Gaussian Process Regression, Bayesian Neural Networks, or Random Forest-based surrogate models) may provide more accurate predictions, especially in regions with sparse training data.
- **SRAM Constraints Not Considered**: Deployment of KWS on MCUs is constrained not only by energy but also by SRAM (runtime storage of activations). The paper does not include SRAM usage in the multi-objective optimization.

### Possible Directions for Future Improvement
- **Richer Configuration Space**: Incorporate dimensions such as network depth, convolution kernel size, and activation function selection into the exploration space. Use more advanced search strategies such as Bayesian Optimization to efficiently handle high-dimensional spaces—the acquisition function of Bayesian optimization can balance exploration and exploitation, finding optimal configurations with few evaluations.
- **Cross-Hardware Platform Energy Modeling**: Establish a unified energy model capable of predicting the energy consumption of the same network configuration on different hardware platforms (FPGA, MCU, ASIC). This can be achieved through analytical energy modeling or transfer learning.
- **Online Adaptive Configuration**: Dynamically switch network configurations based on the device's real-time resource status (remaining battery, temperature, CPU load, etc.) to achieve runtime accuracy-energy adaptation. For example, automatically switching to a low-power configuration $(q=4, s=0.25)$ when battery level drops below 20%.
- **Alternative Regression Models**: Explore the use of neural networks as surrogate models (Surrogate Neural Networks), such as Multi-Layer Perceptrons or small Transformers, which may capture non-linear interaction effects in the configuration space better than polynomials.
- **Implications for the KWS Field**: This paper demonstrates the important idea that "lightweight surrogate models (regression) can replace expensive actual training for network configuration selection." This idea can be generalized to other design decisions in KWS (such as feature extraction method selection: MFCC vs. Filter Bank vs. Learnable Frontend, data augmentation strategy selection, training hyperparameter selection, etc.), significantly accelerating the overall design process of KWS systems.
