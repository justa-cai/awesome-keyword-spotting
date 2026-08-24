# MicroNets: Neural Network Architectures for Deploying TinyML Applications on Commodity Microcontrollers

**Authors/Affiliations**: Colby Banbury, Chuteng Zhou, Igor Fedorov, Ramon Matas Navarro, Urmish Thakker, Dibakar Gope, Vijay Janapa Reddi, Matthew Mattina, Paul N. Whatmough (Arm ML Research, Harvard University)

**Date**: October 2020 (arXiv:2010.11267), MLSys 2021

**Link**: https://arxiv.org/abs/2010.11267

**Keywords**: Neural Architecture Search, TinyML, Microcontrollers, Keyword Spotting, Efficient Inference, Latency Modeling

## Problem Statement

Deploying machine learning workloads on IoT edge devices faces severe technical challenges. Commodity microcontrollers (MCUs) have extremely limited resources:
- **SRAM**: Typically around 256KB, used for storing activation values and working memory
- **Flash**: Typically around 1MB, used for storing model parameters
- **Computing Power**: Lacks dedicated AI accelerators, relying solely on low-power CPUs
- **Power Constraints**: Battery-powered devices require extremely low energy consumption

Deep neural network inference requires substantial computational and memory budgets. The core challenge in the TinyML field is how to achieve the highest possible model accuracy while satisfying the strict constraints of MCUs (memory, latency, and energy consumption).

Traditional Neural Architecture Search (NAS) methods are extremely inefficient when evaluating the latency of candidate architectures, as they typically require actual execution on the target hardware. A fast and accurate latency estimation method is needed to accelerate the NAS search process.

## Methodology

### Key Finding: Linear Relationship Between Operation Count and Latency

The most important finding of this paper:
- In the NAS search space, model latency is statistically approximately linearly related to the operation count (Op Count)
- This linear relationship holds under a uniform prior distribution of models in the search space
- Reason: NAS search spaces are usually composed of structured, repeating units, where different operations have similar computational densities

### Operation Count-Based NAS

Leveraging the aforementioned linear relationship, an efficient NAS method was designed:
- **Latency Proxy Model**: Uses operation count as a proxy metric for latency, eliminating the need for actual execution on the target hardware
- **Differentiable NAS**: Adopts a continuous relaxation approach similar to DARTS
- **Search Space**: Includes various convolutional operations (standard convolution, depthwise separable convolution, dilated convolution, etc.)
- **Search Efficiency**: Only requires training a single Supernet during the search process, without evaluating candidate architectures individually

### MicroNet Architecture Family

The optimized architecture family discovered through NAS has the following characteristics:
- Extensive use of depthwise separable convolutions and inverted bottleneck blocks
- Carefully designed channel expansion ratios and convolutional kernel configurations
- Automatically finds the optimal accuracy-efficiency trade-off under given constraints
- Provides different scales of MicroNet variants tailored to different MCU constraints (memory/latency levels)

### Target Constraints

The design goals for MicroNet:
- **SRAM Constraint**: Activation value storage does not exceed 256KB
- **Flash Constraint**: Model parameters do not exceed 1MB
- **Latency Constraint**: Inference latency meets real-time requirements

## Main Contributions

1. **Discovery of the Linear Relationship Between Operation Count and Latency**: This finding has significant theoretical and practical implications. It simplifies hardware-aware search in NAS by avoiding expensive hardware-measured latency. This discovery offers general reference value for the NAS community.

2. **Efficient Hardware-Aware NAS Method**: By using operation count as a latency proxy, a simple and efficient NAS algorithm was designed, significantly reducing search costs.

3. **MicroNet Architecture Family**: Produced neural network architectures that achieve SOTA accuracy under MCU constraints, covering multiple TinyML benchmark tasks such as KWS and Visual Wake Words.

4. **Multi-Benchmark Validation**: Validated the effectiveness of MicroNet on multiple TinyML benchmark tasks, including Keyword Spotting (Google Speech Commands) and Visual Wake Words.

5. **Industrial Practicality**: All model designs satisfy the hardware constraints of real MCUs and can be directly deployed in actual products.

## Experimental Results

### Latency Proxy Validation
- The correlation coefficient between operation count and actual latency is as high as above 0.95
- The prediction error of the linear proxy model is within the 5-10% range
- Significantly reduces the number of times actual hardware execution measurements are needed during the NAS search process

### Google Speech Commands Results
- MicroNet achieves higher accuracy than previous manually designed architectures under the same SRAM and latency constraints
- Shows significant advantages over general lightweight models like MobileNetV3-tiny under MCU constraints
- Different scales of MicroNet variants meet different latency/accuracy requirements

### Visual Wake Words Results
- MicroNet also performs excellently on the Visual Wake Words task
- Validates the generalizability of the method, which is not limited to specific tasks

### Hardware Constraint Satisfaction
- The activation value storage requirements of all MicroNet models are within the 256KB SRAM range
- Model parameter sizes are within the 1MB Flash range
- Inference latency meets real-time processing requirements

## Limitations and Future Work

### Method Limitations
- **Linear Relationship Assumption**: The linear relationship between operation count and latency may not hold on certain heterogeneous hardware (with dedicated accelerators)
- **Search Space Dependency**: The validity of the linear relationship relies on the uniformity assumption of the NAS search space; caution is needed for non-uniform search spaces
- **Limited Energy Consumption Assessment**: Energy consumption assessment was only performed on specific hardware configurations; energy characteristics may vary significantly across different MCUs
- **NAS Computational Cost**: Although search efficiency is improved, training the Supernet still requires significant computational resources (typically requiring GPU clusters)

### Future Directions
- Extend the latency model to support heterogeneous hardware and dedicated AI accelerators
- Investigate more fine-grained energy consumption modeling methods, using energy consumption as a direct optimization target in NAS
- Explore Zero-Cost NAS methods to further reduce search overhead
- Extend the MicroNet design philosophy to more edge AI tasks (anomaly detection, time series prediction, etc.)
- Research quantization-aware NAS, considering the impact of quantization on accuracy during the search phase
