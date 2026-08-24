# AB/BA Analysis: A Framework for Estimating Keyword Spotting Recall Improvement While Maintaining Audio Privacy

- **Authors/Affiliations**: Raphael Petegrosso, Vasistakrishna Baderdinni, Thibaud Senechal, Benjamin L. Bullough (Amazon, USA)
- **Date**: April 2022 (NAACL 2022 Industry Track)
- **Link**: https://arxiv.org/abs/2204.08474
- **Keywords**: keyword spotting evaluation, privacy, recall estimation, false positive rate, AB testing, semi-supervised analysis

## Problem Statement

### Problem Background and Domain Pain Points
Keyword Spotting (KWS) systems are core entry components for intelligent voice assistants such as Amazon Alexa, Google Assistant, and Apple Siri. These systems continuously listen on billions of devices, waiting for specific wake words (e.g., "Alexa", "Hey Siri"). The two core performance metrics for KWS systems are the False Positive Rate (FPR, the ratio of non-keywords incorrectly identified as keywords) and Recall (the ratio of true keywords correctly identified). In practical product iterations, engineers need to evaluate the improvement of a new model relative to the online baseline model, which serves as the basis for product release decisions.

However, modern privacy regulations (such as GDPR, CCPA, etc.) fundamentally limit the evaluation capabilities of KWS systems. Due to privacy-preserving designs, KWS systems are configured to briefly collect and upload audio data only when a potential keyword is detected—i.e., the system retains only "accept" samples, while all rejected audio (containing a large number of true negatives and false negatives) is discarded on the device. This means researchers cannot obtain a complete set of negative samples to directly calculate the denominator of the FPR, nor can they obtain a complete set of positive samples to directly calculate the denominator of Recall. This constraint of "only seeing detected samples" makes it impossible to construct a complete confusion matrix using traditional methods.

### Specific Shortcomings of Existing Methods
- **Recall Blind Spot in Standard A/B Testing**: In A/B testing, user traffic is randomly allocated to baseline model A or candidate model B. Since only audio "accepted" by the system is uploaded to the server, researchers can count the number of false positives (accepted non-keywords) in both groups, but cannot count the number of missed detections (rejected true keywords) because the rejected audio is inaccessible. Therefore, standard A/B testing can only measure the relative change in FPR between two models and cannot evaluate improvements in Recall—despite Recall being a key metric for user experience.
- **Distribution Shift in Offline Datasets**: Using open-source datasets (such as Google Speech Commands) or recorded supplementary datasets to estimate Recall offline. However, there are significant differences between the acoustic conditions of laboratory-recorded data (quiet environments, near-field microphones, standard pronunciation) and real-world production environments (far-field noise, multi-speaker interference, diverse accents, children's and elderly voices). Empirical evidence suggests that the correlation between Recall estimates on offline datasets and true online Recall can be as low as 0.3-0.5, limiting the reference value of offline evaluation for product decisions.
- **Privacy Conflicts and Scalability Issues with Manual Annotation**: Theoretically, manually annotating audio collected from the production environment could yield precise Recall estimates, but this faces three difficulties: (1) The principle of data minimization requires exposing as little user audio data as possible; large-scale manual listening directly violates this principle; (2) Annotation costs are extremely high—in Amazon's experiments, each annotated sample required approximately 30 seconds of manual listening time; (3) Annotation quality is difficult to guarantee—under far-field noise conditions, the inter-annotator agreement of human annotators is also limited.
- **Lack of Systematic Statistical Frameworks in Prior Work**: Although the industry employs various ad-hoc evaluation methods (such as using proxy metrics or heuristic rules), there is a lack of a theoretically guaranteed, statistically rigorous framework to estimate the relative performance ratio between models under privacy constraints.

### Key Challenges Addressed by This Paper
Design a complete framework that, under the condition that only "accept" data (i.e., candidate keyword data detected by the system) is available, and without making parametric assumptions about the underlying data distribution, can statistically rigorously estimate the relative Recall ratio ($rRecall = Recall_B / Recall_A$) and relative FPR ratio ($rFPR = FPR_B / FPR_A$) between two KWS models. The framework needs to be deployable and scalable in real production environments, with confidence intervals compact enough to support product decisions.

## Methodology

### Overall Architecture Design and Design Motivation
The inspiration for the AB/BA analysis framework comes from the crossover design in medical clinical trials. In crossover trials, each patient receives two treatments in sequence, and individual differences are eliminated by comparing the different responses of the same patient to the two treatments. AB/BA migrates this idea to KWS evaluation: the two models "cross-decode" each other's accept data, inferring relative performance by comparing the different judgments of the same audio under the two models.

The key assumption of this design is that, through randomization, the user traffic in Group A and Group B is statistically homogeneous—i.e., both groups face the same true keyword occurrence rate, the same speaker distribution, and the same acoustic environment distribution. This assumption is guaranteed by the Law of Large Numbers when the sample size is large enough (millions of queries).

### Mathematical Principles of the Core Algorithm

**Step 1: Online Data Collection**
User traffic is evenly and randomly allocated into two groups:
- Group A (50% traffic): Uses baseline model A to process all audio, uploading only audio "accepted" by A to the server.
- Group B (50% traffic): Uses candidate model B to process all audio, uploading only audio "accepted" by B to the server.

Definitions:
- $N_A$: Total number of queries in Group A (including keywords and non-keywords)
- $N_B$: Total number of queries in Group B
- $P_A$: Number of times true keywords appear in Group A (equal to total traffic multiplied by the true keyword occurrence probability)
- $P_B$: Number of times true keywords appear in Group B

Under the randomization assumption: $P_A / N_A \approx P_B / N_B$ (the keyword occurrence rate is the same for both groups)

**Step 2: Cross-Dataset Offline Decoding** (Core Step)
This is the key innovation of the AB/BA framework:
- Decode the accept data collected online by model A using model B offline: obtain $N_{A \to B}^{++}$ (accepted by both A and B), $N_{A \to B}^{+-}$ (accepted by A but rejected by B)
- Decode the accept data collected online by model B using model A offline: obtain $N_{B \to A}^{++}$ (accepted by both B and A), $N_{B \to A}^{+-}$ (accepted by B but rejected by A)

**Step 3: Estimation of Relative Recall Ratio**
Through Bayes' theorem and the randomization assumption, the estimator for rRecall is derived:

$$\widehat{rRecall} = \frac{N_{A \to B}^{++} / (N_{A \to B}^{++} + N_{A \to B}^{+-})}{N_{B \to A}^{++} / (N_{B \to A}^{++} + N_{B \to A}^{+-})}$$

Intuitive interpretation of this estimator: The numerator is the "proportion of samples accepted by model A that are also accepted by model B," and the denominator is the "proportion of samples accepted by model B that are also accepted by model A." Under the randomization assumption, the numerator approximates a function of $P(\text{B accepts} | \text{A accepts}, \text{keyword})$, while the denominator approximates a function of $P(\text{A accepts} | \text{B accepts}, \text{keyword})$. The ratio of the two exactly cancels out the unknown total number of true keywords, depending only on observable cross-decoding counts.

Mathematically, it can be proven that this estimator is a consistent estimator, meaning that as the sample size approaches infinity, the estimate converges to the true value.

**Step 4: Estimation of Relative FPR Ratio**
Define rFPR = FPR_B / FPR_A. In KWS systems, FPR is typically extremely low (e.g., at the 0.1% level), meaning false positives are rare relative to the total traffic. Utilizing this sparsity:

$$\widehat{rFPR} = \frac{N_B^{accept} / N_B}{N_A^{accept} / N_A} \times \frac{1}{\widehat{rRecall}}$$

Where $N_A^{accept}$ and $N_B^{accept}$ are the total number of samples accepted online by model A and model B, respectively. The core idea of this formula is: the online acceptance rate is influenced by both Recall and FPR. When the relative ratio of Recall is known (estimated in Step 3), the relative ratio of FPR can be solved from the ratio of acceptance rates. The sparsity assumption (very few false positives) ensures the high accuracy of this approximation.

### Semi-Supervised AB/BA Analysis
The data after cross-decoding still needs to determine the "ground truth"—i.e., whether each sample is truly a keyword. The paper proposes two alternative solutions:

**Solution 1: Stratified Sampling Annotation**
- Instead of annotating all cross-decoded samples, prioritize selecting samples where the two models produce inconsistent judgments (i.e., samples accepted by A but rejected by B, or rejected by A but accepted by B).
- These "high-information" samples are located in the intersection area of the decision boundaries of the two models and contribute most to the variance of the estimator.
- By annotating only these high-information samples (typically only 5-10% of the total samples), one can achieve estimation accuracy comparable to full annotation with approximately 80% less annotation effort.

**Solution 2: Machine Soft Labels**
- Use a well-trained independent model C (neither A nor B) to generate soft labels for all cross-decoded samples.
- Model C is trained on a large amount of annotated data, and its output probability $p_C(y|x)$ serves as a proxy for ground truth.
- The continuity of soft labels (probability values between 0 and 1) retains more confidence information than hard labels (0 or 1).
- Eliminates the need for manual annotation, improving analysis speed by an order of magnitude—from days to hours.
- The paper validates the high consistency between soft labels and human labels through an annotation experiment with 117 participants.

### Technical Differences from Existing Methods
- Compared to traditional A/B testing: AB/BA transforms the "unobservable Recall" into "observable cross-counts" through cross-decoding, breaking through the Recall blind spot.
- Compared to pure offline evaluation: AB/BA uses real production data, with a data distribution completely consistent with the actual deployment environment, eliminating distribution shift issues.
- Compared to full data annotation: AB/BA requires annotation only for a subset of cross-decoded data (or uses machine labels), significantly reducing the privacy exposure surface—only exposing audio where the two models disagree, rather than all audio.
- Compared to parametric methods: AB/BA does not assume any parametric form for the underlying data distribution (such as normality, Poisson distribution, etc.), relying only on statistical homogeneity guaranteed by randomization, thus having a wider scope of application.

### Specific Network Structure and Training Strategy
This paper does not involve model training but proposes an evaluation framework. Model A and Model B are already trained KWS models (in Amazon's case, end-to-end deep learning-based KWS models). The AB/BA framework only plays a role during the model deployment and evaluation stages.

## Main Contributions

1. **First Recall Estimation Framework under Privacy Constraints**: AB/BA analysis is the first systematic method capable of estimating the relative Recall ratio when only "accept" data is available, without making parametric assumptions about the underlying data distribution. Theoretical proofs show that this estimator is consistent, providing mathematical guarantees for the reliability of the results. This has significant practical value for the evaluation of all speech AI systems facing privacy constraints (not limited to KWS, but also including speech recognition, speaker identification, etc.).

2. **Theoretical Derivation of Low-Variance FPR Estimation**: In practical KWS systems (scenarios with extremely low FPR), the paper derives an rFPR estimator with low variance characteristics. The paper provides a formulaic calculation method and proves that under the assumption of sparse false positives, the variance of this estimator decreases inversely proportional to the sample size—meaning that only a moderate sample size is needed to obtain sufficiently compact confidence intervals to support product decisions.

3. **Engineering Innovation in Semi-Supervised AB/BA**: The introduction of machine-generated soft labels to replace manual annotation improves analysis speed by an order of magnitude while protecting privacy. The stratified sampling strategy further optimizes annotation efficiency. Overall, semi-supervised AB/BA reduces labor costs by approximately 80-90% while maintaining estimation accuracy comparable to fully supervised methods.

4. **Production-Level Validation**: In Amazon Alexa's real production environment (involving hundreds of millions of devices and tens of millions of daily active users), the AB/BA framework was successfully applied to the iterative evaluation of KWS models. This is an important step in moving KWS evaluation under privacy constraints from ad-hoc practices to systematic methodology.

## Experimental Results

### Datasets Used and Their Scale
- **Simulation Experiments**: Used synthetic KWS system outputs to simulate AB/BA scenarios. Controllable parameters include: true Recall, true FPR, sample size, and keyword occurrence rate. The advantage of simulation experiments is that the "true values are known," allowing for precise verification of the accuracy and variance characteristics of the estimators. Simulation sample sizes ranged from 10,000 to 10,000,000.
- **Real Production Data**: Deployed in Amazon Alexa's production environment, using real user interaction data. Approximately 300K independent clients participated, involving millions of queries. Group A and Group B each received approximately 50% of the randomly allocated traffic.

### Definition and Rationale for Evaluation Metrics
- **Relative Recall Ratio (rRecall)**: Defined as Recall_B / Recall_A. The reason for choosing relative metrics over absolute metrics is: (1) Absolute Recall cannot be directly calculated under privacy constraints; (2) Product decisions care about "how much better the new model is than the old one," and the relative ratio exactly answers this question.
- **Relative FPR Ratio (rFPR)**: Defined as FPR_B / FPR_A. The rationale is the same as above.
- **95% Confidence Interval**: Used to quantify estimation uncertainty and support statistical significance judgments.
- **Inter-annotator Agreement**: Used to validate the quality of manual annotation and machine soft labels.

### Detailed Comparison with Baseline Methods and SOTA
- **AB/BA vs. Standard A/B Testing**: Standard A/B testing can only estimate rFPR and cannot estimate rRecall. AB/BA provides both metrics simultaneously.
- **AB/BA Estimated rRecall vs. True rRecall (Simulation)**: In simulation experiments, the rRecall estimated by AB/BA is highly consistent with the true value calculated based on complete ground truth, with errors within the range of statistical noise (RMSE < 0.02). As the sample size increases, the width of the confidence interval decreases at a rate of $O(1/\sqrt{N})$, consistent with theoretical expectations.
- **Semi-Supervised AB/BA vs. Fully Supervised AB/BA**: Semi-supervised AB/BA using machine soft labels achieves estimation quality close to that of fully manually annotated AB/BA (relative error increase not exceeding 5%), while reducing the annotation volume by approximately 80%. This means that under the same annotation budget, the semi-supervised method can handle larger-scale data, obtaining more compact confidence intervals.
- **Stratified Sampling vs. Random Sampling**: Under the same annotation budget (e.g., annotating 1,000 samples), stratified sampling reduces the standard error of the estimate by approximately 50%, as it focuses annotation on the "inconsistent" samples with the highest information content.

### Findings from Ablation Studies
- **Impact of Cross-Decoding Direction**: The decoding results in both directions (A->B and B->A) are highly consistent (correlation coefficient > 0.95), validating the effectiveness of the randomization assumption.
- **Impact of Soft Label Model Selection**: The paper tested models C of different complexities as soft label generators. It was found that as long as the accuracy of model C is higher than 85%, the estimation quality of soft label AB/BA is insensitive to the specific choice of model C. This lowers the implementation threshold.
- **Impact of Keyword Occurrence Rate**: When the keyword occurrence rate is extremely low (<0.01%), the variance of the estimator increases (because positive samples are very rare). The paper suggests extending the data collection period in this scenario to accumulate sufficient positive samples.

### Key Performance Numbers
- Width of the 95% confidence interval for rRecall estimation: Approximately 0.05 under 1 million queries (i.e., the estimation accuracy of rRecall is approximately ±2.5%).
- Analysis speed of the semi-supervised method: Reduced from days to hours.
- In the label accuracy study: 117 participants annotated 11,908 samples, with an inter-annotator agreement Cohen's Kappa > 0.85.

## Limitations and Future Work

### Technical Limitations of the Method
- **Engineering Complexity of Dual-Model Parallel Deployment**: The AB/BA framework requires deploying two KWS models simultaneously to different user groups (50% traffic each) and performing cross-decoding on the server side. This requires: (1) Loading parameters for both models on the server side; (2) An offline batch processing pipeline for cross-decoding; (3) Traffic segmentation and routing infrastructure. For teams without large-scale A/B testing infrastructure, the implementation threshold is high.
- **Provision of Relative Metrics Only**: AB/BA can only estimate the relative performance ratio between two models (rRecall, rFPR) and cannot provide the absolute Recall or absolute FPR for either model. When the question is "what is the current system's Recall" (rather than "how much better is the new system than the old one"), other methods are still needed.
- **Sensitivity to the Randomization Assumption**: The framework relies on random allocation of user traffic to ensure statistical homogeneity between the two groups. If the randomization mechanism has systematic biases (e.g., uneven distribution of device types, regions, or time periods), the estimator may be biased. The paper does not sufficiently discuss how to detect and correct such biases.
- **Latency of Cross-Decoding**: Audio accepted online by model A needs to wait for the offline decoding results of model B, resulting in analysis latency. In rapidly iterating scenarios, this latency may affect decision-making speed.

### Shortcomings in Experimental Design
- **Robustness Not Discussed for Large Differences in Group Traffic Scale**: What happens to the variance of the estimator if the traffic ratio between Group A and Group B is not 50:50 (e.g., 90:10)?
- **Model C Selection Criteria in Semi-Supervised Methods Are Not Systematic**: The paper only validates the rough criterion that "Model C's accuracy needs to be higher than 85%" and does not systematically explore the impact of Model C's architecture, training data, and calibration strategies on the quality of soft labels.
- **Confidence Threshold Differences Between Model A and Model B Not Considered**: Different decision thresholds for the two models may lead to different definitions of "accept," thereby affecting the interpretation of cross-decoding results.

### Possible Directions for Future Improvement
- **Extension to Multi-Model Comparison**: The current framework only supports pairwise comparison (A vs. B). In scenarios where multiple candidate models need to be evaluated simultaneously (e.g., A vs. B vs. C vs. D), it can be extended to an N-way crossover design, constructing a complete performance ranking through multiple pairwise comparisons.
- **Online AB/BA**: The current cross-decoding is offline batch processing. In the future, streaming online cross-decoding can be explored to reduce analysis latency from the hour level to the minute level.
- **Causal Inference Perspective**: Provide a more rigorous theoretical foundation for the AB/BA framework from the perspective of causal inference. Specifically, use DAGs (Directed Acyclic Graphs) to explicitly model the relationships between randomization, confounding variables, and causal effects, handling non-completely randomized scenarios (such as stratified randomization by device type).
- **Differential Privacy Enhancement**: Introduce differential privacy mechanisms into the AB/BA framework to provide formal privacy guarantees for the estimation results, rather than relying solely on the principle of data minimization.
- **Implications for the KWS Field**: The core idea of the AB/BA framework—"bypassing unobservability through cross-decoding"—can be generalized to other fields that need to evaluate detection performance under privacy or data access constraints, such as spam detection (cannot access emails rejected by the filter), fraud detection (cannot access transactions rejected by rules), and content moderation (cannot access content deleted by the system). This approach has inspirational value for the entire AI safety field.
