# Utilizing TTS Synthesized Data for Efficient Development of Keyword Spotting Model

- **Authors/Affiliations**: Hyun Jin Park, Dhruuv Agarwal, Neng Chen, Rentao Sun, Kurt Partridge, Justin Chen, Harry Zhang, Pai Zhu, Jacob Bartel, Kyle Kastner, Gary Wang, Andrew Rosenberg, Quan Wang (Google LLC, Mountain View, CA, USA)
- **Date**: July 26, 2024 (arXiv:2407.18879v1 [cs.SD])
- **Link**: https://arxiv.org/abs/2407.18879
- **Keywords**: keyword spotting, TTS synthesized data, prosody-controlled text generation, speaker diversity, data-efficient training, real and synthetic data mixing, streaming edge-side model

## Problem Statement

### Problem Background and Domain Pain Points

Keyword Spotting (KWS) is the task of detecting specific spoken keywords within a continuous audio stream while rejecting background speech and noise. It serves as the activation mechanism for conversational human-computer interfaces such as Alexa, Siri, and Google Assistant (Section 1). Unlike cloud-based large models invoked on demand, production-grade KWS is an "always-on" component. The paper explicitly lists two product-level constraints: ideally, it must be strictly causal to achieve low latency, and its computational footprint must be small enough to limit energy consumption (Section 1). These constraints dictate that KWS models follow a lightweight streaming network architecture. The baseline model in this paper has approximately 320,000 parameters (Section 3.2), consisting of a two-stage structure with an encoder and decoder in series, comprising 7 layers of SVDF factorized convolutional layers and 3 layers of bottleneck projection layers, specifically optimized for streaming inference.

The true pain point lies not in the model side but in the data side. Production-grade KWS must cover the vast diversity of demographics, speaking styles, accents, and acoustic environments (Section 1), thus typically requiring massive amounts of training data. Collecting audio for specific target keywords requires real recordings from human contributors, which is costly (Section 1). The production-scale data provided in this paper (Table 3) includes 3.8 million real positive samples and 14.1 million real negative samples. In other words, for every new wake word, added language, or new product category, millions of target word recordings must be re-collected. This is a direct bottleneck for product iteration speed and development budgets. Meanwhile, state-of-the-art TTS models can now generate large batches of natural-sounding audio at low cost (Abstract). This leads to the core proposition of this paper: Can we train a KWS model with accuracy close to the full real-data baseline by combining massive TTS synthesized data with a very small amount of real data, thereby reducing both development costs and time?

### Specific Shortcomings of Existing Methods

- **Real data collection mode is not scalable**. Recordings for target keywords must be done by humans. The collection of positive samples (speech containing the keyword) is constrained by the keyword itself and cannot reuse general corpora. Although the paper does not provide specific unit collection costs, the scale of 3.8 million positive samples in Table 3 illustrates the industrial investment involved.
- **TTS enhancement conclusions from the ASR field cannot be directly transferred to KWS**. The success of TTS-generated data in ASR (cited in [13–19]) relies on a key premise: ASR can leverage pure text data that is far richer than annotated audio (Section 1). The training data structure for KWS is completely different—positive samples are phrase structures of "keyword plus subsequent user query," while negative samples are any speech not containing the keyword. The two types of samples are highly imbalanced (14.1M real negative samples vs. 3.8M positive samples in Table 3). More importantly, KWS is extremely sensitive to speaker diversity, accents, and prosodic variations. The same phrase "Hey Google" exhibits numerous systematic variations in real users, such as elongation, pauses, rising intonation, and shouting. These variations are not explicitly modeled in general TTS enhancement pipelines.
- **Existing TTS work in the KWS field is not systematic**. Section 1 points out two prior lines of work: Lin et al. [20] proposed pre-training an embedding model on real data, then fine-tuning a classification head attached to it using a small amount of TTS or real data. This approach essentially does not摆脱 (get rid of) the dependence on large-scale real data; it merely shifts the dependence to a one-time pre-training step. Werchniak et al. [21] conducted a preliminary exploration of TTS data usage for single-keyword detection, arriving at the qualitative conclusion that "mixing real and TTS data yields the best results." However, neither line addresses the core engineering questions.
- **Inherent defects of TTS data are not addressed head-on**. The abstract explicitly states: the distribution of TTS-generated data may not match the distribution of real data. Section 4.3 elaborates further—synthesized data may contain artifacts not present in real recordings, or it may fail to generate all variations present in real speech. Previous work has not provided answers regarding the quantified consequences of this mismatch or the methods to compensate for it.
- **Key engineering gaps**: Under the realistic constraint of "current TTS capabilities," how much real data is still missing? In what structure should real data be invested—more speakers or more sentences per speaker? How much of the role of expensive dedicated real positive samples can be replaced by cheap general real negative samples? These questions had not been systematically experimented on prior to this paper.

### Key Challenges Addressed by This Paper

Accepting the existence of distribution mismatch (not pursuing pure synthetic training) and without changing the model structure, the paper designs a data recipe that achieves extremes in two directions: minimizing the amount of real data used and maximizing the diversity of TTS outputs. The goal is to reach a clearly defined engineering target: controlling the error rate within 3 times that of the full real-data baseline (Abstract). Notably, the goal setting of "3 times error rate" itself carries an engineering judgment flavor: instead of matching the baseline, it accepts a certain range of precision loss in exchange for reducing real data usage by approximately three orders of magnitude, leaving operational space for the rapid cold-start of new categories and languages.

## Methodology

### Overall Architecture Design and Design Motivation

The overall scheme (Fig. 2) can be summarized as "model unchanged, data reconstructed": the baseline KWS model structure is kept completely intact. Training data can come from two sources: real speech or TTS synthesis. The mixing ratio between the two is explored systematically as a hyperparameter in sweep experiments (Section 4). The entire scheme consists of three key components:

1. **Text Generator**: Generates text phrases customized for KWS training, with the design goal of maximizing the diversity of TTS synthesis outputs (Section 4.1).
2. **Dual TTS Engines**: Utilizes the best TTS modules capable of synthesizing speech from large numbers of voices—Virtuoso provides a large number of pre-trained voices, while AudioLM variants support personalized voice generation based on input audio (Section 4.2).
3. **Mixing Strategy**: Evaluates various mixing options of synthetic and real data, focusing on minimizing data costs while maximizing quality (Section 4.3).

Why put all innovations on the data side? First, the 320K parameter streaming SVDF encoder-decoder structure is a configuration already validated by large-scale production (following [3,6]); the risk-reward ratio of changing the model is poor. Second, conclusions on the data side are orthogonal to model architecture; regardless of what streaming network is used at the bottom, the recipe of "how much real data and what structure" holds true, offering strong transferability. Third, the bottleneck this paper aims to solve is explicitly in data collection costs and development cycles, not model design.

Baseline model details (Section 3, all following previous work [3,6]):

- **Input Features**: 40-dimensional filterbank energies, 25 ms analysis window, 10 ms frame shift. Every 20 ms, 3 consecutive frames are stacked and strided to obtain a 120-dimensional input vector $X_t$ (Section 3.1). Data augmentation such as simulated reverberation and noise mixing [26] is applied before feature extraction to improve model robustness and adaptability.
- **Structure**: The encoder maps the stacked filterbank energies to an N-dimensional output $Y_E$, with the design intent of encapsulating N phoneme-class sound units crucial for keyword recognition. The decoder then generates a 2-dimensional output $Y_D$ from $Y_E$, predicting the presence or absence of the keyword in the audio stream. The final prediction logit is the concatenation $Y = [Y_E, Y_D]$ (Section 3.2). The intermediate representation of the encoder is explicitly exposed in the concatenated logit, allowing phoneme-level evidence and whole-word decisions to be supervised simultaneously.
- **Training Objective**: See next section.

### Mathematical Principles of Core Algorithms

The supervised training objective (Eq. 1) is a weighted sum of two loss terms:

$$L_{sup} = \sum_{t=1..n} [ (1-\alpha) \cdot L_{CE}(Y(X_t, \theta), c_t) + \alpha \cdot L_{MP}(Y(X_t, \theta), \omega_{end}) ]$$

Where $Y(X_t, \theta)$ is the joint output of the encoder and decoder given input $X_t$ and parameters $\theta$; $L_{CE}$ is the end-to-end loss proposed by Alvarez [3] (implemented as defined in Eq. 2 of Park et al. [6]), where $c_t$ is the per-frame target label, i.e., cross-entropy is calculated directly between the model logits and labels; $L_{MP}$ is the max-pool loss proposed by Park et al. [6] (Eq. 12 in their paper), where $\omega_{end}$ is the label for the keyword end position, i.e., cross-entropy is calculated on the pooled logits at the end of the word; $\alpha$ is an empirically determined loss weight hyperparameter, the specific value of which is not reported in the paper. Both loss terms have independent components for the encoder and decoder modules, which are weighted and combined to form the final loss (Section 3.3). The reason given by the paper for this combination is that it helps prevent overfitting, improves model performance on unseen data, and ultimately enhances the robustness and effectiveness of the keyword detection task (Section 3.3). From a decision semantics perspective: per-frame CE provides dense frame-level supervision but does not directly correspond to the event of "hearing a complete keyword"; the max-pool loss applies supervision only on the pooled logits at the keyword end position, which aligns perfectly with the whole-word decision semantics of KWS. The two are complementary.

Formalization of text generation (Section 4.1): Define a keyword as a tuple `keyword := (prefix, key_name)`, for example, prefix is "Hey" and key_name is "Google". Given a keyword and random query text (query, i.e., what the user says after the keyword, from random text corpora), positive sample phrases are constructed by concatenation; negative sample phrases come from arbitrary text corpora with keywords filtered out. The text template set (Table 2) defines the combination of variable slots and prosody control symbols (Table 1), generating TTS input text through random sampling of templates.

### Key Technical Innovation 1: Prosody-Controlled KWS Custom Text Generator

This is the third contribution explicitly listed in the paper. The mechanism is built on an experimental feature of the Virtuoso TTS: punctuation marks in the text input can control the prosody of the synthesized speech (Section 2.1). The paper defines 5 control methods (Table 1):

| Control Text | Effect |
|---|---|
| text | Default pronunciation of the text |
| (text) | Speak this text slowly |
| text: | Insert a pause after the text |
| text? | Raise the pitch at the end of the text |
| text! | Speak this text loudly |

Based on this, Table 2 provides text generation templates: 5 positive sample templates—"{prefix} {key_name} {query}" (baseline concatenation), "{prefix} ({key_name}) {query}" (slow keyword), "({prefix}): ({key_name}) {query}" (slow prefix with pause, slow keyword), "{prefix}: ({key_name})? {query}" (prefix pause, slow keyword with rising intonation), "{prefix}: {key_name}! {query}" (prefix pause, loud keyword)—and 1 negative sample template "{query}" (pure query text, naturally不含 keyword).

Why is this move particularly valuable in the KWS scenario? Real pronunciations of wake words have many systematic variations: users may elongate sounds when calling, pause between the prefix and keyword, use a questioning rising intonation to confirm if the device is listening, or shout loudly from across the room. If these variations are not in the training distribution, they will manifest as false rejects upon deployment. Traditional TTS data augmentation only transforms text content and speaker, leaving the prosodic space unexplored. This paper applies prosody symbols independently to the prefix and key_name segments, turning "multiple ways of saying the same sentence" into operations that are enumerable and randomly sampleable at the text level, directly expanding the coverage of the synthetic distribution over the real pronunciation space. Note that the paper explicitly states these symbols are experimental features of Virtuoso (Section 4.1), so the portability of the method is engine-bound.

### Key Technical Innovation 2: Dual TTS Engines and Speaker Personalization

The two engines complement each other in terms of "sources of diversity" (Section 2, Section 4.2):

**Virtuoso** ([22,23]): A multilingual speech-text joint training model that can perform semi-supervised learning using three data sources: untranscribed speech, unlabeled text, and paired speech-text. It supports speech generation in 139 languages and provides 726 predefined speaker profiles (Section 2.1). This paper uses its simple text-to-speech mode: given transcribed text, generate target language speech from a specified speaker, with randomized prosody (Section 2.1). Additionally, the paper leverages its multilingual capability for a clever operation: using fixed English phrases combined with different target languages to generate output that sounds like accented English (Section 4.2), injecting accent diversity into the synthetic data—one of the most critical coverage dimensions for KWS serving global users. This effectively converts the prior of 139 languages' speakers into an English accent space for free.

**AudioLM Variant** ([24,25]): An audio generative language model with long-term coherence and high quality (Section 2.2). The variant used in this paper supports dual conditioning on text and audio. Its key feature is that it preserves the speaker characteristics and prosody of the input audio during synthesis (Section 2.2)—i.e., any real speaker's audio sample can be used as a condition to generate keyword speech in that voice. The paper emphasizes that the diversity of the resulting dataset can match the richness of real human audio prompts (Section 2.2).

Why two engines instead of one? Virtuoso provides breadth: a combination space of 726 preset voices multiplied by 139 target languages. AudioLM provides depth: the ability to clone any voice outside the preset profiles, allowing the speaker distribution of synthetic data to follow the real population's speaker distribution closely, rather than being limited to the hundreds of profiles selected by the engine vendor. The final synthetic data scale (Table 3): 7.5 million synthetic positive samples and 5.1 million synthetic negative samples, combined from both engines. The proportion occupied by each engine is not reported.

### Key Technical Innovation 3: Mixing Strategy for Real and Synthetic Data

Section 4.3 faces the distribution mismatch problem head-on: even the best TTS can generate realistic human speech, but the distribution of generated data may mismatch with real speech data. Synthetic data may contain artifacts not present in real speech, or it may fail to generate all variations present in real speech. The paper's countermeasure is not to fix the TTS itself, but to mix in real data for compensation. The mixing method is treated as an experimental object, designing four groups of sweeps (Section 5.3):

1. **Baseline performance of different dataset combinations**: Including an improved baseline with added real negative data (referred to as base real negative data in the paper). A key cost insight here: positive sample collection is constrained by the keyword (must contain the target word), while negative samples can be obtained from almost any data source not containing the keyword (Section 5.3). Therefore, real negative samples are much cheaper than real positive samples.
2. **TTS plus incremental real positive samples**: Starting from pure TTS training, randomly sample and gradually increase real positive samples to 100k, while testing to what extent base real negative samples can offset the need for real data.
3. **TTS plus incremental number of speakers**: Fix the TTS configuration, uniformly sample by speaker with a fixed 10 sentences per person, and gradually increase the number of speakers—uniformly sampling real sentences should provide data diversity, ideally helping the model train better and faster (Section 5.3).
4. **TTS plus incremental sentences per person**: Fix the number of speakers at 100, increase the number of sentences per person, forming a contrast with the previous group along the "diversity axis vs. quantity axis."

### Technical Differences with Existing Methods

- **Difference from Lin et al. [20]**: Lin's scheme is two-stage—first pre-train an embedding model on large-scale real data, then fine-tune the classification head with a small amount of TTS or real data. Essentially, it shifts the dependence on large-scale real data to a one-time pre-training, without eliminating the dependence, and each new task still requires attaching a fine-tuning head. This paper uses end-to-end single-stage mixed training, directly compressing the demand for real positive samples for each new keyword to the order of thousands (Table 5), without relying on any pre-trained embedding model.
- **Difference from Werchniak et al. [21]**: Werchniak is a preliminary exploration on the single-keyword detection problem, with conclusions remaining at the qualitative level of "mixing real and TTS is best." This paper quantifies and systematizes this conclusion: sweeping along three axes of total real data, number of speakers, and sentences per person, providing reusable data recipes and clear thresholds (e.g., 100 speakers entering the baseline 3x error rate interval).
- **Difference from TTS enhancement work in the ASR field [13–19]**: Different task structures dictate different data construction methods. ASR is a full-vocabulary transcription task, where the value of TTS lies in turning abundant pure text into audio. KWS is an open-set detection of fixed keywords, with highly imbalanced positive and negative samples (Table 3: 14.1M real negative vs. 3.8M positive). The text generator must specifically construct the "keyword plus query" positive sample phrase structure and "keyword-filtered" negative sample corpus. Moreover, prosodic variations are more critical than text content diversity (the existence of Table 1 itself is for this purpose).
- **Relationship with model-side compute-saving work**: The baseline in this paper follows the streaming SVDF structure of [3,6], which is completely orthogonal to compression routes such as quantization and distillation, and can be stacked.

## Experimental Results

### Datasets Used and Their Scales

The training task is "Hey/OK Google" detection (two prefixes, two keywords). Real speech data consists of anonymized utterances collected according to Google's privacy and AI principles [27,28] (Section 5.1). TTS data is generated by two engines, Virtuoso and AudioLM variant, with multi-style data augmentation [29] (i.e., room simulator implementation) applied during training (Section 5.1).

The data scale summarized in Table 3:

| Data Type | Number of Utterances |
|---|---|
| Real Positive Samples | 3.8M |
| Real Negative Samples | 14.1M |
| Synthetic Positive Samples | 7.5M (Combined Virtuoso and AudioLM) |
| Synthetic Negative Samples | 5.1M (Combined Virtuoso and AudioLM) |

Evaluation is performed on the real Hey/OK Google dataset (Section 5.2); the scale and composition of the evaluation set itself are not reported in the paper. Another numerical detail to note: the main text in Section 6.1 states that the real negative samples added in the sweep experiments are approximately 11M ("∼11 M"), which does not match the 14.1M in Table 3. The paper does not explain the difference (speculation: reserved for evaluation or subset, only speculation possible).

### Definition and Rationale for Evaluation Metrics

The primary metric is the False Reject Rate (FRR). The model threshold is optimized under the constraint of a fixed maximum allowed false accept per hour of 0.133 (Section 5.2), which the paper calls a typical operational condition. The rationale for this choice is straightforward: KWS is a product-level resident system, and the frequency of false wake-ups is a hard constraint perceptible to users (the experience disaster of a device randomly waking up is far greater than occasionally needing to say the wake word again). Therefore, FRR must be compared while locking the false wake-up budget; reporting a single accuracy or error rate is meaningless. This is also the standard evaluation protocol for production-grade KWS work. Regarding model complexity, only the parameter count of approximately 320,000 (Section 3.2) is reported; inference latency, FLOPS, training cost, and TTS generation cost are not reported.

### Detailed Comparison with Baseline Methods and SOTA

**Table 4: Baseline Comparison under Simple Mixing Options (FRR under Fixed FA)**

| Training Data | FRR |
|---|---|
| Virtuoso Data Only | 53.10% |
| AudioLM Data Only | 46.50% |
| TTS Total (Virtuoso + AudioLM) | 46.47% |
| Real Data Only | 3.17% |
| Virtuoso + Base Real Negative | 17.75% |
| AudioLM + Base Real Negative | 16.59% |
| TTS + Base Real Negative | 17.94% |
| TTS + Full Real Data | 2.46% |

This table has four key readings. First, pure synthetic training is infeasible: the best pure TTS configuration has an FRR of 46.47%, nearly 15 times that of the pure real baseline of 3.17%, making the cost of distribution mismatch directly visible. Second, dual-engine mixing (46.47%) is slightly better than either single engine (53.10% / 46.50%); but interestingly, after adding real negative samples, the AudioLM single engine with negatives (16.59%) is slightly better than the dual-engine mix with negatives (17.94%), a counter-intuitive result that the paper does not analyze. Third, real negative samples are the single real data investment with the largest leverage: approximately 11M real negative samples pull the pure TTS baseline from 46.47% down to 17.94% (Section 6.1), a relative improvement of about 60%. Mechanistically, this can be understood as: the high false reject rate of pure TTS models comes largely from misclassifying daily speech that "sounds like the keyword but isn't" as positive. Real negative samples exactly fill this decision boundary, and since they do not need to contain the keyword, their collection cost is far lower than positive samples. Fourth, TTS plus full real data achieves 2.46%, which is actually better than pure real at 3.17%—synthetic data still brings an absolute gain of 0.71 points when real data is sufficient, indicating that the value of TTS is not just to replace real collection, but can also be used as traditional data augmentation.

**Fig. 3: TTS plus Incremental Real Positive Samples (0 to 100k)**

Blue bars are configurations without base real negative samples, red bars are with base negative samples. Both groups show similar improvements as real positive samples increase (Section 6.2). Key points: FRR decreases monotonically with the amount of real positive samples; adding real negative samples pulls the pure TTS baseline from about 46.7% to 17.94%; adding full real positive samples on top reduces it from 17.94% to 2.46% (Section 6.2); approximately 100k randomly sampled real positive samples plus base negatives plus TTS yield an FRR of 9.94%, which is about 3 times the pure real baseline (3.17%) (Section 6.2). Special attention should be paid to the figure caption: Fig. 3 uses a "medium sized model" (original caption text), and it is not specified whether this model is of the same specification as in Table 4 and Table 5, so cross-table number comparisons should be cautious.

**Table 5 Upper Half: Sweep Speaker Count (Fixed 10 Sentences per Person, All Configurations Include Full TTS and Base Real Negative)**

| Configuration | Total Real Positive Samples | FRR |
|---|---|---|
| TTS + 1 Speaker | 10 sentences | 15.28% |
| TTS + 10 Speakers | 100 sentences | 14.94% |
| TTS + 100 Speakers | 1k sentences | 9.78% |
| TTS + 200 Speakers | 2k sentences | 9.90% |
| TTS + 500 Speakers | 5k sentences | 7.63% |

The improvement from 1 to 100 speakers is steep (15.28% down to 9.78%), from 100 to 200 it is almost a plateau (9.78% to 9.90%, even a slight rebound, within experimental noise), and from 200 to 500 it drops again (7.63%). The paper concludes from this: when the number of speakers reaches 100 or more, the FRR is similar to or lower than 3 times the baseline (3.17%) (Section 6.3); and it specifically emphasizes that the 100-speaker model used only 1k sentences of real positive samples, while the baseline used 3.8M sentences (Section 6.3).

**Table 5 Lower Half: Sweep Sentences per Person (Fixed 100 Speakers)**

| Configuration | Total Real Positive Samples | FRR |
|---|---|---|
| TTS + 2 Sentences/Speaker | 200 sentences | 10.99% |
| TTS + 6 Sentences/Speaker | 600 sentences | 10.95% |
| TTS + 12 Sentences/Speaker | 1.2k sentences | 10.71% |
| TTS + 20 Sentences/Speaker | 2k sentences | 9.47% |
| TTS + 200 Sentences/Speaker | 20k sentences | 7.99% |

Increasing sentences per person from 2 to 12 yields extremely slow improvement (10.99% to 10.71%), reaching 9.47% at 20 sentences and 7.99% at 200 sentences. Comparing the improvement slope with the upper half, the paper's conclusion is clear: increasing the number of sentences under a fixed number of speakers yields relatively slow improvement in FRR; increasing speaker diversity has a greater impact than increasing sentences per person (Section 6.4). The "100 speakers, 2k sentences" configuration claimed in the abstract and conclusion strictly corresponds to the 20 sentences/speaker row here (FRR 9.47%, within 3 times the baseline of 3.17%). The real positive sample usage of 2k sentences compared to 3.8M sentences represents a compression of about three orders of magnitude (approximately 1900 times).

There is another comparison not explicitly pointed out by the paper but visible by placing Section 6.2 and Table 5 together: randomly sampling 100k real positive samples yields an FRR of 9.94% (Section 6.2), whereas balanced sampling by speaker requires only 2k sentences (100 speakers * 20 sentences) to achieve 9.47% (Table 5 lower half)—under similar precision, balanced sampling uses about 50 times less data. This strongly supports the paper's mechanistic explanation: the value of real data lies mainly in speaker diversity rather than sentence quantity; large corpora sampled randomly are likely concentrated on a few speakers, offering limited diversity gains.

### Findings from Ablation Experiments

This paper does not have traditional structural ablations (model structure remains unchanged); the ablation targets are the data recipes themselves. The findings can be summarized into five points:

1. **Real negative samples as a baseline are a necessary configuration**: Approximately 11M real negative samples pull the pure TTS FRR from 46.47% to 17.94% (Table 4, Section 6.1). All subsequent sweep experiments fixed the use of base real negative samples (decision at the end of Section 6.1).
2. **Positive and negative real data have asymmetric but significant roles**: Real negative samples reduce the error rate from about 46.7% to 17.94%, and full real positive samples further reduce it from 17.94% to 2.46% (Section 6.2). Both types of real data make large independent contributions to the pure TTS baseline.
3. **Speaker diversity takes precedence over sentences per person**: This is the most important actionable conclusion in the paper (comparison of upper and lower halves of Table 5, Section 6.4). The paper's explanation is that uniformly sampled real sentences by speaker provide data diversity, helping the model train better and faster (Section 5.3).
4. **100 speakers is a practical threshold**: Reaching 100 speakers enters the baseline 3x error rate interval (Section 6.3), after which marginal returns slow down (200 speakers 9.90% is even slightly worse than 100 speakers' 9.78%, suggesting noise).
5. **Individual contributions of diversity methods are not disentangled**: Prosody-controlled text (Table 1), cross-language accent synthesis (Section 4.2), and AudioLM voice cloning (Section 4.2) are presented only as a combined solution. The paper does not report the contribution of any single method in isolation, which is a gap in the experimental design.

## Main Contributions

1. **Training Recipe**: Achieving accuracy comparable to the baseline using large-scale synthetic data plus a minimal amount of real data—100 speakers, 2k sentences of real positive samples (Table 5) compared to the baseline's 3.8M sentences, with error rates within 3 times the baseline, compressing real data requirements by about three orders of magnitude (Contribution 1, Section 1).
2. **Trade-off Curves**: Reporting the trade-off relationship between real data usage and model accuracy under various sweep conditions (Contribution 2, Section 1)—covering three dimensions: total real data axis (Fig. 3), number of speakers axis (Table 5 upper), and sentences per person axis (Table 5 lower), providing the basis for the "expand speakers first, then supplement sentences" collection strategy.
3. **Text Generator**: Proposing a KWS-customized text generation method that leverages the experimental prosody control features of Virtuoso TTS (punctuation control symbols in Table 1 and templates in Table 2) with the goal of maximizing TTS output diversity (Contribution 3, Section 1, Section 4.1).
4. **(Implicit Contribution) Two Transferable Engineering Judgments**: First, real negative samples (which need not contain the keyword and have almost no source restrictions) are the most cost-effective real data investment, with approximately 11M negative samples bringing the largest single-item FRR improvement. Second, TTS data can not only replace real collection but also reverse-enhance full real data (2.46% vs. 3.17%, Table 4).

## Limitations and Future Work

### Technical Limitations of the Method

- **Distribution mismatch is not fundamentally solved, only diluted by the mixing strategy**: Pure TTS training yields an FRR as high as 46.47% (Table 4), indicating that synthetic data alone cannot support training. All usable configurations must mix in real data. There is a ceiling to "TTS replacing real collection." The paper does not attempt any explicit means to narrow the synthetic-real domain gap (such as domain adversarial training, consistency loss, or synthetic artifact filtering).
- **Marginal returns of synthetic data narrow when real data is sufficient**: The difference between full real plus TTS (2.46%) and pure real (3.17%) is only 0.71 absolute points (Table 4). For products already possessing large-scale real data, the incremental value of TTS enhancement is limited; its main battlefield should be the cold-start of new words and languages, which is precisely the scenario not covered by the experiments.
- **Prosody control is bound to Virtuoso's experimental features**: The mapping from punctuation to prosody in Table 1 is an experimental feature of that engine (original words in Section 4.1). Changing the TTS engine requires rebuilding the control channel, limiting the portability of the method. There are only 5 prosody symbols, and their application relies on random template sampling, with no search or weighting done on "which prosodic variations contribute most to FRR reduction."
- **Diversity methods are not disentangled**: The individual contributions of cross-language accent synthesis (Section 4.2) and AudioLM voice cloning (Section 4.2) are not ablated, making it impossible to evaluate the cost-effectiveness of each method or determine which subsets of the 726 preset voices and 139 target languages are most valuable.
- **Mixing method is the simplest concatenation**: More fine-grained fusion strategies such as curriculum learning (synthetic first, then real), domain-weighted sampling, and TTS data quality filtering are not explored. The mixing ratios are only presented in a few coarse-grained levels.

### Shortcomings in Experimental Design

- **Single Keyword Group**: All experiments are only针对 (targeted at) Hey/OK Google. The greatest imaginative space for TTS data—zero-collection or low-collection development for any custom wake word—is not experimentally verified. The abstract's claim of "minimizing development cost and time" lacks evidence from the most relevant scenario.
- **Missing Evaluation Details**: The scale, composition, and whether the speakers in the evaluation set overlap with those in the training real data are not reported. The variance of FRR or the number of repeated experiments is not reported. The non-monotonic results in Table 5 (100 speakers 9.78% vs. 200 speakers 9.90%) suggest that experimental noise cannot be ignored, yet the conclusions are built precisely on these fine-grained numbers.
- **Lack of Data for Efficiency Argument**: The keyword in the paper title is "efficient development," but the computational cost of generating 12.6M synthetic sentences via TTS, the cost model for collecting real data, and the comparison of training costs between the two routes are all unreported. The efficiency claim remains at the level of "fewer real data sentences."
- **Questionable Product Usability of 3x Error Rate**: An FRR of 9.47% means about one in ten wake words is rejected. Whether this is usable for a resident product, and for which product lines (prototype verification or mass production), is not discussed in terms of applicability boundaries.
- **Numerical Consistency Issues**: The 14.1M real negative samples in Table 3 do not match the approximately 11M in the main text of Section 6.1, with no explanation. The abstract's "100 speakers, 2k sentences" strictly corresponds to the 20 sentences/speaker configuration in the lower half of Table 5, but the 100-speaker threshold was actually first met at 1k sentences (9.78%). The relationship between the "medium sized model" in Fig. 3 and the model specifications in Table 4 and Table 5 is not explained.
- **Unanalyzed Counter-Intuitive Results**: After adding base negatives, the AudioLM single engine (16.59%) outperforms the dual-engine mix (17.94%, Table 4). This result is unfavorable to the "engine complementarity" narrative, yet the paper does not discuss it.

### Possible Directions for Future Improvement

- **Domain Adaptation and Synthetic Data Cleaning**: Introduce consistency or contrastive pre-training objectives (the paper itself cites TTS4pretrain 2.0 [19] as a ready path) to narrow the distribution gap between synthetic and real data, striving to push the error rates of pure TTS and low-real-data configurations further down from the 46% and 16% intervals, approaching or even surpassing the "3x baseline" line.
- **Automatic Search of Prosody Space**: Change the combination of prosody symbols in Table 1 from random sampling to optimizable variables, selecting which variations to generate based on validation set FRR, upgrading the text generator from "maximizing diversity" to "maximizing information gain."
- **Verification of Cold-Start Keyword Development Process**: Apply the recipe in this paper to completely new custom wake words, end-to-end measuring the relationship between development cycle, real data budget, and accuracy, supplementing the evidence for the most valuable application scenario.
- **Active Learning-Based Real Data Collection**: Since speaker diversity has been proven to be the dominant factor, select the next speaker and sentences to record based on model uncertainty or embedding coverage, further compressing the 2k sentence budget.
- **Orthogonal Combination with Edge-Side Compute-Saving Technologies**: Stack with quantization, distillation, and trimmable architectures to form a complete low-cost development stack of "data recipe plus model compression." The baseline scale of 320K parameters in this paper also provides a clean starting point for such combinations.
