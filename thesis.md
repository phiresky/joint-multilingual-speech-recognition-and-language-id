---
# this document is pandoc-1.19-flavored markdown
author:
- 'Robin'
bibliography:
- 'http://localhost:23119/better-bibtex/collection?/0/SMJR24NP.biblatex'
title: |
    Backchannel Prediction for Conversational Speech Using Recurrent Neural
    Networks
---

# Introduction

Motivation, Goals

# Related Work

# Backchannel Prediction {#sec:extraction}

A backchannel is generally defined as any kind of feedback a listener
gives a speaker as an acknowlegment in a primarily one-way conversation.
This can be 
In this paper, we concentrate on short phrasal backchannels consisting
of a maximum of three words. We try to predict these for a given speaker
audio channel in a causal way.

## BC Utterance selection {#sec:extractio:subsec:bc-utterance-selection}

There are different kinds of phrasal backchannels, they can be
non-commital, positive, negative, questioning, et cetera. To simplify
the problem, we initially only try to predict the trigger times for any
type of backchannel, ignoring different kinds of positive or negative
responses. Later we try to distinguish between a limited set of
categories.

## Training area selection

We need to choose which areas of audio we use to train the network. We
want to predict the backchannel without future information (causally /
online), so we need to train the network to detect segments of audio
from the speaker track that probably cause a backchannel in the listener
track. We choose the beginning of the backchannel utterance as an anchor
and use a fixed range before that as the positive prediction area. This
approach is easy, though it may not be optimal because the delay between
the last utterance of the speaker and the backchannel can vary.

bisher nicht probiert
Another interesting anchor is the last few words before a backchannel.
We could for example choose \([t-0.5s, t+0.5s]\), where t is the center
of the last speaker utterance before the backchannel, but then we need
to be careful not to use any future data.

We also need to choose areas to predict zero i.e. “no backchannel”, so
the network is not biased to always predict a backchannel. For this we
can choose the range a few seconds before each backchannel, because in
that area the listener explicitly decided not to give a backchannel
response yet.

## Feature selection

The most commonly used audio features in related research are fast and
slow pitch slopes and pauses of varying lengths.
[@ward2000prosodic; @eemcs18627; @Morency2010]. Because our network does
automatic feature detection, we simply feed it the absolute pitch and
power (volume) values for a given time context, from which it is able to
calculate the pitch slopes and pause triggers on its own.

Additionally, we try to use other tonal features such as the fundamental
frequency variation (FFV) [@laskowski2008fundamental] , the
Mel-frequency cepstral coefficients (MFCC) \[ref\] and a set of
bottleneck features trained on phoneme recognition \[ref\]. Because our
training data set is limited, we easily run into overfitting problems
with a large input feature dimension.

## Training and Neural Network Design {#training}

We begin with a simple feed forward network. The input layer consists of
all the chosen features over a fixed time context. One or more hidden
layers with varying numbers of neurons follow. The output layer is
(n+1)-dimensional, where n is the number of backchannel categories we
want to predict. In the simplest case we train the network on the
outputs \[1, 0\] for backchannels and \[0, 1\] for non-backchannels.


The placement of backchannels is dependent on previous backchannels. If
the previous utterance was a long time ago, the probability of a
backchannel is higher and vice versa. To accommodate for this, we want
the network to also take its previous internal state or outputs into
account. We do this by modifying the above architecture to use
Long-short term memory layers (LSTM) instead of feed forward layers.

## Postprocessing

The neural neural network gives us a noisy value between 0 and 1, which
we interpret as the probability of a backchannel happening at the given
time. To convert this into discrete trigger timestamps, we run a
low-pass filter over the network output and then use the maximum value
of every output segment whose value is over a given threshold.


\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{img/postprocessing.png}
    \caption{Postprocessing example}
    \label{fig:my_label}
\end{figure}

## Evaluation

To evaluate the performance of our predictions, we take a set of
monologuing segments from our training data and compare the prediction
with the ground truth, calculating the precision, recall, and F1 score
with varying margins of error.

low-pass filter, trigger BC at maxima

use random BC sample from training data

# Experimental Setup {#experiments}

## Dataset

We used the switchboard dataset [@swb] which consists of 2400 telephone
conversations of five to ten minutes, 260 hours in total. These
telephone conversations have complete transcriptions and word alignments
[@swbalign].

## Extraction {#extraction-1}

We used annotations from The Switchboard Dialog Act Corpus [@swda] to
decide which utterances to classify as backchannels. The SwDA contains
categorical annotations for utterances for about half of the data of the
Switchboard corpus. We extracted all utterances containing the
“backchannel” label, and counted their frequency.

We chose to use the top 150 unique utterances marked as backchannels
from this set. Because the SwDA is incomplete, we had to identify
utterances as backchannels just by their text. We manually removed some
requests for repetition like “excuse me” from the list, and added some
other utterances that were missing from the SwDA transcriptions but
present in the original transcriptions, by going through the most common
utterances and manually selecting those that seemed relevant such as
‘um-hum yeah’, ‘absolutely’, ‘right uh-huh’.

The most common backchannels in the data set are “yeah”, “um-hum”,
“uh-huh” and “right”, adding up to 41860 instances or 68% of all
extracted backchannel phrases.

To select which utterances should be categorized as backchannels and
used for training, we first filter noise and other markers from the
transcriptions, for example `[laughter-yeah] okay_1 [noise]` becomes
`yeah okay`.

Some utterances such as “uh” can be both backchannels and speech
disfluencies, so we only choose those that have either silence or
another backchannel before them. This method gives us a total of 61645
backchannels out of 391593 utterances (15.7%) or 71207 out of 3228128
words (2.21%). We used the Janus Recognition Toolkit [@janus] for parts
of the feature extraction (power, pitch tracking, FFV, MFCC). Features
are extracted for 32ms windows, with a shift of 10ms. We also added a
word2vec vocabulary trained on the word alignements.

This gives us the following selection algorithm:
```python
def is_backchannel(utterance):
    text = noise_filter(utterance)
    if index(utterance) == 0: return False
    previous_text = noise_filter(previous(utterance))
    return (text in valid_backchannels and
            (is_silent(previous_text) or is_backchannel(previous(utterance)))
```


    aggregated  self    count   category    text
    31.92%  31.92%  14319   b   uh-huh
    61.08%  29.15%  13075   b   yeah
    68.95%  7.87%   3532    b   right
    71.73%  2.78%   1249    b   oh
    73.69%  1.96%   877 b   [silence]
    75.09%  1.40%   629 b   oh yeah
    76.49%  1.40%   627 b   yes
    77.84%  1.35%   607 b   okay
    78.86%  1.02%   458 bk  okay
    79.75%  0.89%   399 b   huh
    80.57%  0.81%   364 b   sure
    81.29%  0.72%   325 bk  oh okay
    81.93%  0.64%   288 b   huh-uh
    82.56%  0.63%   282 bh  oh really
    83.15%  0.59%   264 ba  wow
    83.73%  0.58%   259 b   um
    84.16%  0.43%   193 bh  really
    84.57%  0.41%   186 b   really
    84.97%  0.39%   177 bk  oh


## Training {#training-1}

We used Theano [@theano] with Lasagne [@lasagne] for rapid prototyping
and testing of different parameters. We trained a total of over 200
network configuration with various context lengths (500ms to 2000ms),
context strides (1 to 4 frames), network depths ranging from one to four
hidden layers, layer sizes ranging from 15 to 100 neurons, activation
functions (tanh and relu), gradient descent methods (SGD, Adadelta and
Adam), dropout layers (0 to 50%) and layer types (feed forward and
LSTM).

The LSTM networks we tested were prone to overfitting quickly, but still
provided better results than normal feed forward networks.
L2-Regularization reduced this problem and slightly improved the
results.


- softmax with categorical crossentropy for categorical output (1=BC, 0=NBC)
- sigmoid with mean squared error for bell curve output


validate error functions etc.

## Evaluation {#eval-1}

The training data contains two-sided conversations. Because the output
of our predictors is only relevant for segments with only one person
talking, we only run our evaluation on monologuing segments.

For this we define the predicate `is_listening` (is only emitting
utterances that are backchannels or silence) and `is_talking`
(`= not is_listening`). A monologuing segment is the maximum possible
time range in which one person is consistently talking and the other
only listening. Additionally, we only consider such segments of a
minimum length of 5 seconds to reduce problems arising from predictions
at the edges of the segments, though this only improved the resulting
F1-scores by 2%.

# Results

All the results in \autoref{fig:survey} use the following configuration
if not otherwise stated: LSTM, Hidden layers: 70 $\rightarrow$ 35
neurons, Input features: Power, pitch, FFV, Context frame stride: 2,
Margin of Error: 0ms to +1000ms. Precision, Recall and F1-Score are
given for the validation data set. In \autoref{fig:final}, our final
results are given for the completely independent evaluation data set.

# Conclusion

\bibliographystyle{styles/spmpsci}


# Bibliography