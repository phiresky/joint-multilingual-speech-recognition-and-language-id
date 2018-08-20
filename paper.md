---
title: Summary of "Language Independent End-to-End Architecture For Joint Language and Speech Recognition (2017)" by Watanabe, S.; Hori, T.; Hershey, J.R. 
# author: Me
classoption: twocolumn
papersize: a4paper
csl: ieee.csl
header-includes:
- \usepackage{siunitx}
- \usepackage{xcolor}
- \usepackage{CJKutf8}
---

# Motivation / Goal

Recognize multiple languages at the same time

- Two tasks: identify language AND recognize speech (simultaneously)
- Use a single model for 10 languages (EN, JP, CH, DE, ES, FR, IT, NL, PT, RU)
- Find out if transfer learning between languages work
- End to end: Directly train sequence to sequence, no lexicon, phoneme pronounciation maps, or manual alignment


# Related Work

<!-- (e.g. only attention) -->

- _Multilingual Speech Recognition With A Single End-To-End Model_ (Shubham Toshniwal, Google) [@to17]
    - separate output for language id
    - only on 9 indian languages, hard to compare

- _Hybrid CTC/Attention Architecture for End-to-End Speech Recognition_ (Watanabe et al. 2017) [@wa17]
    - Same as this paper except only one language and more detailed


# Model description

## Input and output format decisions

### Input {#sec:input}

Apart from the model architecture, there are two main decisions to be made: How the audio should be input into the system, and what the output should look like.

For the input format, the authors chose the common method of extracting spectral features from the audio file, and chunking it into frames of e.g. 10ms. There is no description of what the exact features are. Because the architecture is very similar to the one the same authors mention in [@wa17], we can assume the input structure to be similar. [@wa17] mentions 40 features from the output of a filter bank, and two dimensions from simple pitch features. These are concatenated into a 42-dimensional input feature vector, which causes a potential issue described in [@sec:futurework]

### Output

There are multiple formats that are viable for text output in a end-to-end neural network. Mainly we can either output words as a whole (using one-hot or word embeddings like word2vec [@mi13]) or single characters. For speech recognition it makes sense to output single characters, since that way no fixed dictionary is needed and the network can learn to output words not seen before in the training data. To allow multilingual output, the authors propose to simply unify all the character sets (latin, cyrillic, CJK) to get a total of 5500 characters.

For the language identification output, the authors propose adding a special "language-tag" character that is prepended to the output, for example "`[EN]Hello`" or "\begin{CJK}{UTF8}{gbsn}[CH]你好\end{CJK}". An alternative would be to add a separate one-hot encoded language id output as seen in [@to17].

![Input and output structure overview (from the paper)](img/20180626154145a.png)

<!-- ![Model overview (from the paper)](img/20180626154145b.png) -->


## Simple Model overview

1. Input: Basically a spectrogram as a 2D image
2. Encoder (CNN + LSTM)
3. Decoder
    #. Soft Attention for each input frame to each output character
    #. LSTM Layer
4. Output
    - N characters from union of all languages (one-hot / softmax)


## Input Encoder

The authors use a image processing pipeline for the encoder. The input audio is thus formatted like an RGB image, with y being the feature index and x being time. The first channel is the spectral features described in [@sec:input], and the second and third channel are delta and deltadelta features of the first channel.

The encoder consists of two parts, a convolution network and a recurrent network. The convolutional network is taken directly from the first six layers of the VGG Net architecture, as seen in [@fig:vgg]. Due to the two pooling layers, the dimensionality of the spectral input image is reduced 4 times in both the time and feature dimension. For simplicity, we will refer to $t$ as 1/4th of the input time dimension, so the encoded state can be indexed by $t$.

![VGG Net for image processing - first 6 layers [@vgg]](img/vgg16-cutoff.png){#fig:vgg}

The convolved input is then fed into a bidirectional LSTM with 320 cells in each direction, which results in a encoded vector dimension ($h_t$) of 640 scalars per time step $t$.

This hidden state is then decoded with two separate decoders in parallel, one based on attention and one based on CTC.

## Decoder A (Attention-based)

First the input sequence $\vec{x}_1,\dots,\vec{x}_{4t}$ is encoded to $\vec{h}_1,\dots,\vec{h}_t$ with the above described VGG+BLSTM. The goal is to get $l \leq t$ output characters $c_1,\dots,c_l$. The soft attention weights $a_{lt}$ are calculated based on three values: The attention on the same input for the previous output ($a_{(l-1)t}$), the current encoded state $\vec{h}_t$, and the previous hidden decoder state $\vec{q}_{l-1}$.

The encoded state is then combined weighted with the soft alignment to get the input of the decoder network $\vec{r}_l = \sum_t{a_{lt}\vec{h}_t}$. The decoder is another (unidirectional) LSTM layer, followed by a softmax layer:

$$c_l = \text{Softmax}(\text{FC}(\text{LSTM}(\vec{r}_l, \vec{q}_{l-1}, c_{l-1})))$$

Using this decoder without any additions is possible, but the authors argue that pure temporal attention is too flexible since it allows nonsensical alignments: Compared to machine translation, the word order can not change from input to output in speech recognition, it is strictly monotonic. Thus, the authors add a second decoder based on CTC described in the next section.

## Decoder B (CTC-based)

The input and encoder are the same as before. After the encoder, a simple softmax layer per time step is added that directly converts the 640 outputs ($\vec{h}_t$ from the encoder into one of the N output characters. This results in one output character per timestep $t$, which is then reduced to a flexible number of output characters using the CTC loss function [@gr06].

As a simple explanation, CTC works by adding a blank character `"="` to the output character set. For example, if we only allow the output HELLO our output set would be $\{H, E, L, O, -\}$. For inference, first all duplicate characters are removed and then all blank symbols. For example: `HHHH-EEEEEEEE-LL-LLL-\-\-\-OOOOOO → H-E-L-L-O → HELLO`. Note that the double l sequence is retained since there is a blank symbol between two blocks of `l` symbols. For training, we simply define all combinations of character duplications that result in the ground truth at inference time to be correct. The loss function is then the negative logarithm of the ground truth probability the network predicts. This probability can be efficiently computed using the Viterbi / forward-backward algorithm.

As opposed to the attention-based decoder, this method enforces the monotonic alignment of the output to the input.


![Hybrid CTC/attention-based end-to-end architecture](img/20180629140340.png){width=50%}


## Language Model

With the model described so far, the neural network needs to implicitly learn a language model for all the output languages and store it together with the decoding task in the weights. The authors propose adding a separate, explicit language model called RNN-LM, that is trained only to model the distribution and sequences of the output characters, while ignoring the input speech. It is also based on a single LSTM layer that predicts the next character based on the previous one (together with its hidden state). It is trained separately, though it would be possible to jointly train it.

## Final loss function

The final loss function is a simple linear combination of the loss functions of the attention decoder, the CTC decoder, and the language model. The two decoders are weighted equally ($\lambda=0.5$), and the language model is weighted at 1/10 of the decoders together ($\gamma = 0.1$).


\begin{align}
\mathcal{L}_{\text{MTL}} & = \lambda \log p_{\text{ctc}} (C|X) \\
 & + (1 - \lambda) \log p_{\text{att}}(C|X) \\
 & + \gamma \log p_{\text{rnn-lm}}(C) 
\end{align}


The authors use the AdaDelta optimizer and train for 15 epochs. The inference is done via beam search on the attention output weighted by the above loss function.

# Results

[@Fig:cer] shows the average character error rate for all languages for different experiments. Comparing the second and third columns shows that adding the convolutional network in front of the LSTM encoder improves performance by 7%. The fourth column shows that the RNN-LM improves performance, though not by much (3%). Adding data from three more languages (NL, RU and PT) increases the performance for the _other_ 7 languages by 9%, which shows that transfer learning between the languages works. Note that the authors do not provide a significance analysis, but the data set is of all languages together is very large. The authors also do not provide a baseline or results with only one of both decoders.

![Character Error Rates (abbrev.)](img/20180701122411-ov.png){#fig:cer}

[@Fig:liderror] shows that the language identification task has very good results. The only strong confusion is that Spanish is often (30%) mistaken for Italian, but not the other way around. The authors do not provide an interpretation for this.

![Language identification (LID) accuracies/error rates (%). The diagonal elements correspond to the LID accuracies while the offdiagonal
elements correspond to the LID error rates](img/20180629120038.png){#fig:liderror}

# Potential problems / future work? {#sec:futurework}

- Only fed with a single language utterance at a time
    - maybe we want to allow switching? (append utterances from different languages)
- Uniform random parameter initialization with $[-0.1, 0.1]$ seems statistically unsound? (use Xavier / Hu)
- Input feature convolution is weird
    - _\[...\] we used 40-dimensional filterbank features with 3-dimensional pitch features_
    - redundancy (delta, deltadelta)
- Unbalanced language sets (500h CH, 2.9h PR)
- Same latin characters are used for multiple languages, while others (RU, CH, JP) get their own character set
    - Try transliterating them to Latin?

- Does not work online (without complete input utterance)
    - Bidirectional LSTM in encoder
        - Could try one directional, but Language ID would completely break
        - aggregate limited number of future frames (e.g. add 500ms latency between input and output)
    - Attention does not work in realtime
    - CTC should work online



<!-- 
- adding a pure language model (RNN-LM) improves performance a bit
- [On single language ASR] "Surprisingly, the method achieved performance comparable to, and in some cases superior to, several state-ofthe-art HMM/DNN ASR systems [...] when both multiobjective learning and joint decoding are used."
-->
