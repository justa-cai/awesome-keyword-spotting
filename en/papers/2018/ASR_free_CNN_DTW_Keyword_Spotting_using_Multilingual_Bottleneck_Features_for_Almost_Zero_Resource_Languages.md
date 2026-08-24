# ASR-free CNN-DTW Keyword Spotting using Multilingual Bottleneck Features for Almost Zero-Resource Languages

- **Authors/Affiliations**: Raghav Menon, Herman Kamper, Emre Yilmaz, John Quinn, Thomas Niesler (Department of Electrical and Electronic Engineering, Stellenbosch University; Centre for Language Studies, Radboud University; National University of Singapore; United Nations Global Pulse lab)
- **Date**: July 2018 (arXiv:1807.08666)
- **Link**: https://arxiv.org/abs/1807.08666
- **Keywords**: CNN-DTW, multilingual bottleneck features, zero-resource, keyword spotting, low-resource languages, humanitarian applications

## Problem Statement

United Nations humanitarian relief projects (such as the Global Pulse lab) need to detect keywords quickly in speech data from crisis regions in order to analyze people's needs (such as "food", "water", "medical care"). However, the languages spoken in these regions are often severely under-resourced.

**Domain Pain Points and Specific Background**
- The African continent has more than 2,000 languages, the vast majority of which have no annotated speech corpus or ASR system
- The time frame of crisis response (days to weeks) is far shorter than the development cycle of a traditional ASR system (months to years)
- The UN needs a rapidly deployable solution that works with only a very small amount of annotated data
- DTW (dynamic time warping) template matching does not require ASR, but its performance is limited when applied directly to MFCC features

**Specific Shortcomings of Existing Methods**
- Traditional ASR approaches require large amounts of annotated data, a pronunciation lexicon, and a language model, which is infeasible under zero-resource conditions
- Although DTW template matching is simple, MFCC features have poor robustness to cross-speaker and noise variation
- Monolingual bottleneck features provide only limited benefit when the target language is extremely under-resourced

**Key Challenges This Paper Aims to Solve**
- How to build a practical keyword spotting system under "almost zero-resource" conditions (only a small number of isolated keyword examples available)
- How to effectively transfer multilingual speech knowledge to a low-resource target language
- How to combine the template-matching flexibility of DTW with the efficient search capability of a CNN

## Methodology

### Overall Architecture Design

The paper proposes a CNN-DTW hybrid method that combines the weakly supervised learning capability of DTW with the efficient search capability of a CNN.

**Stage 1: Multilingual Bottleneck Feature (BNF) Extraction**

Core idea: jointly train a DNN acoustic model on 10 well-resourced languages, and use the bottleneck layer to extract speech features that generalize across languages.

Concrete implementation:
1. **Training languages**: 10 languages (including English, Mandarin, Portuguese, Russian, Turkish, Vietnamese, Hawaiian, French, German, and Spanish)
2. **DNN architecture**: a multi-layer fully connected network whose penultimate layer is the bottleneck layer (42-dimensional)
3. **Training objective**: phoneme classification — mapping MFCC inputs to phoneme posterior probabilities for each language
4. **Multilingual joint training**: the training data of all languages is pooled, the DNN parameters are shared, and only the output layer is split per language
5. **Feature extraction**: after training, the BNF extractor (with the output layer removed) is applied directly to the target language

**Why do multilingual BNFs work?**
- Different languages share low-level acoustic characteristics (such as vocal tract formant patterns and manner of articulation)
- Multilingual training lets the BNF extractor learn a speech representation that generalizes beyond any single language
- The dimensional constraint of the bottleneck layer forces the network to retain only the most discriminative features

**Stage 2: DTW Supervision Generation**

Under the condition that the target language has only a small number (1920) of isolated keyword examples, DTW is used to generate "soft labels":
1. Collect a small number of reference templates for each keyword
2. Compute DTW distances between a large amount of untranscribed audio and these templates
3. The DTW distance acts as a "pseudo-label" — the smaller the distance, the more likely the audio segment contains the target keyword
4. Although these pseudo-labels are imperfect (they contain noise), they provide sufficient supervisory signal

**Stage 3: CNN Keyword Detector Training**

The CNN detector is trained using the DTW pseudo-labels:
1. **Input**: multilingual BNF features
2. **CNN architecture**: a standard convolutional neural network with several convolutional layers and fully connected layers
3. **Training objective**: regression training with the DTW distance as the target value (or conversion into classification labels)
4. **Inference**: the trained CNN can rapidly search large amounts of audio for keywords, without computing DTW distances one by one

**Why use a CNN instead of DTW directly?**
- DTW has time complexity O(N*M), making it very slow for searching large-scale audio archives
- CNN forward inference has a fixed time complexity, so its search speed is far faster than DTW
- A CNN can learn higher-level acoustic patterns that the DTW distance cannot capture

## Main Contributions

1. **Significant gains from multilingual BNFs**: the multilingual BNF trained on 10 languages achieves an absolute improvement of 10.9% in ROC AUC over MFCC. This result quantitatively demonstrates the transferability of cross-lingual speech knowledge and the advantage of multilingual joint training.

2. **An ingenious combination of CNN and DTW**: DTW-based weak supervision (suited to limited annotated data) is combined with CNN-based fast search (suited to large-scale deployment). DTW provides "coarse but valuable supervision", while the CNN provides "fast and accurate search" — the two are complementary.

3. **10-language vs 2-language comparison**: the 10-language BNF significantly outperforms the 2-language BNF, showing that more source languages provide better cross-lingual coverage and generality. This offers a practical guideline for language selection in multilingual training.

4. **A battle-oriented zero-resource method**: the paper presents a complete methodology that goes from a small number of isolated keyword examples to a deployable CNN detector, meeting the rapid-deployment requirements of humanitarian relief scenarios.

## Experimental Results

### Datasets
- South African English speech corpus (NCHLT Speech Corpus)
- 1920 isolated keywords used as DTW templates
- A large amount of untranscribed continuous speech as the search target
- Evaluation metric: ROC AUC (area under the receiver operating characteristic curve)

### Core Results

**ROC AUC comparison**
| Features | ROC AUC | Absolute improvement |
|----------|---------|---------|
| MFCC baseline | ~0.79 | baseline |
| 2-language BNF | ~0.83 | +4% |
| 10-language BNF | ~0.90 | +10.9% |

- The 10-language BNF provides the most significant improvement
- More source languages consistently improve BNF quality
- An absolute improvement of 10.9% in ROC AUC translates into a significant reduction of the miss rate in practical detection

**CNN vs DTW search speed**
- CNN search is far faster than direct DTW template matching
- The CNN's search time over large-scale audio archives scales linearly with the amount of audio
- DTW's search time is proportional to the product of the audio length and the template length

**Generalization of multilingual BNFs**
- Even when the target language (English) is not among the training languages, the BNF still provides a significant improvement
- The degree of phonetic overlap between languages (the proportion of shared phonemes) affects the transfer performance

### Ablation Experiments
- 10-language BNF > 2-language BNF > monolingual BNF > MFCC
- A bottleneck dimensionality of 42 is a good balance point (too small loses information, too large reduces generalization)

## Limitations and Future Work

### Technical Limitations of the Method
- **English-only evaluation**: although the method targets zero-resource languages, the main evaluation was conducted on English (in fact a well-resourced language). Performance on truly low-resource languages (such as indigenous African languages) may differ.
- **Training cost of the BNF extractor**: a DNN must be jointly trained on data from 10 languages; although these data are publicly available, the training process still requires a certain amount of computational resources and engineering effort.
- **Noise in the DTW supervision**: DTW distances used as pseudo-labels contain noise — DTW may mistake non-target words for the keyword, or miss target words due to speaker variability. This noise propagates into CNN training.
- **Dependence on phonetic overlap between source and target languages**: when the target language's phonetic system differs greatly from all source languages (such as African languages containing click consonants), the transfer effect may weaken.

### Shortcomings of the Experimental Design
- No end-to-end evaluation was performed on a truly low-resource African language
- No systematic comparison with other zero-resource KWS methods (such as the correspondence autoencoder (CAE) and unsupervised subword discovery)
- The effect of different numbers of DTW templates on CNN training quality was not evaluated

### Future Improvement Directions
- Combine multilingual BNFs with a correspondence autoencoder (CAE), as in the subsequent work of Menon et al. (2018)
- Explore replacing BNFs with features from modern self-supervised pre-trained models such as wav2vec/XLSR
- Extend the method to keyword search in continuous speech (rather than isolated-word detection)
- Develop a zero-shot version that requires no annotated data at all (based on unsupervised phoneme discovery)

### Insights for the KWS Field
- Multilingual training is an important strategy for solving low-resource speech technology; this idea was later validated and popularized by multilingual pre-trained models such as wav2vec 2.0
- The "weak supervision + deep learning" paradigm provides a practical technical route for resource-constrained scenarios
- The CNN-DTW combination idea can be generalized to other scenarios where a detector must be learned from a small amount of annotated data
- Humanitarian applications (such as the UN Global Pulse project) give KWS research an important social-value dimension
