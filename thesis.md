---
# this document is pandoc-1.19-flavored markdown
author:
- 'Robin'
bibliography:
- 'bib.bib'
title: |
    Backchannel Prediction for Conversational Speech Using Recurrent Neural
    Networks
---

# Introduction

Motivation, Goals

# Related Work

# Backchannel Prediction {#sec:extraction}

A listener backchannel is generally defined as any kind of feedback a listener
gives a speaker as an acknowlegment in a primarily one-way conversation.
They include but are not limited to nodding [@watanabe_voice_1989], a shift in the gaze direction and short phrases. Backchannels are said to help build rapport , which is the feeling of comfortableness or being "in sync" with conversation partners [@huang_virtual_2011].

(-> motivation)
This thesis concentrates on short phrasal backchannels consisting
of a maximum of three words. We try to predict these for a given speaker
audio channel in a causal way, using only past information. This allows the predictor to be used in an online environment, for example to make a conversation with an artificial assistant more natural.

## BC Utterance selection {#sec:extractio:subsec:bc-utterance-selection}

The definition of backchannels varies in literature. There are many different kinds of phrasal backchannels, they can be non-commital ("uh huh", "yeah"), positive/confirming ("oh how neat", "great"), negative/surprised ("you're kidding", "oh my god"), questioning ("oh are you", "is that right"), et cetera.
To simplify the problem, we initially only try to predict the trigger times for any
type of backchannel, ignoring different kinds of positive or negative
responses. Later we also try to distinguish between a limited set of
categories.

## Training area selection

We generally assume to have two seperate audio tracks, one for the speaker and one for the listener each with the corresponding transcriptions.
We need to choose which areas of audio we use to train the network. We
want to predict the backchannel without future information (causally /
online), so we need to train the network to detect segments of audio
from the speaker track that probably cause a backchannel in the listener
track. The easiest method is to choose the beginning of the utterance in the transcript of the listener channel as an anchor $t$, and then use a fixed context range of width $w$ before that as the audio range to train the network to predict a backchannel $[t-w, t]$. The width can range from a few hundred milliseconds to multiple seconds. We feed all the extracted features for this time range into the network at once, from which it will predict if a backchannel at time $t$ is appropriate. This approach is easy because it only requires searching for all backchannel utterance timestamps and then extracting the calculated range of audio. It may not be optimal though, because the delay between the last utterance of the speaker and the backchannel can vary significantly in the training data. This means it is not guaranteed that the training range will contain the actual trigger for the backchannel, which is assumed to be the last few words said by the speaker, and even if it does the last word will not be aligned within the context. This causes the need for the network to first learn to align it's input, making training harder and slower.

Another interesting anchor is the last few words before a backchannel.
We could for example choose $t$ as the center of the last speaker utterance before the backchannel, and then use $[t-0.5s, t+0.5s]$ as the training range. This proved to be hard because without manual alignment it isn't clear what the last relevant utterance even is, and in many cases the relevant utterance ends after the backchannel happens, so we would need to be careful not to use any future data. The first approach seemed to work reasonably well, so we did not do any further experiments with the second approach.


We also need to choose areas to predict zero i.e. “no backchannel” (NBC). This should be in about the same amount as backchannel samples, so the network is not biased towards one or the other.
To create this balanced data set, we can choose the range a few seconds before each backchannel as a negative sample. This gives us an exactly balanced data set, and the negative samples are intuitively meaningful, because in that area the listener explicitly decided not to give a backchannel response yet, so it is sensible to assume whatever the speaker is saying is not a trigger for backchannels.

## Feature selection

The most commonly used audio features in related research are fast and
slow voice pitch slopes and pauses of varying lengths.
[@ward2000prosodic; @eemcs18627; @Morency2010]. Because our network does
automatic feature detection, we simply feed it the absolute pitch and
power (volume) values for a given time context, from which it is able to
calculate the pitch slopes and pause triggers on its own by substracting the neighboring values in the time context for each feature.

We also try to use other tonal features used for speech recognition in addition and instead of pitch and power.
The first feature is the fundamental frequency variation spectrum (FFV) [@laskowski2008fundamental], which is a representation of changes in the fundamental frequency over time, giving a more accurate view of the pitch progression than the single-dimensional pitch value which can be very noisy. This feature has seven dimensions in the default configuration given by the Janus Recognition Toolkit.

Other features we tried include the Mel-frequency cepstral coefficients (MFCC) with 20 dimensions \[ref\] and a set of
bottleneck features trained on phoneme recognition using a feed forward network\[ref\].

Because our training data set is limited, we easily run into overfitting problems
with a large input feature dimension.

All of the features are extracted with a window size of 32 milliseconds, overlapping each other with a stride of 10 milliseconds. This gives us 100 frames per second.

## Training and Neural Network Design {#training}

We begin with a simple feed forward network. The input layer consists of
all the chosen features over a fixed time context. With a time context of $c\,\si{ms}$ and a feature dimension of $f$, this gives us a input dimension of $f \times \floor{c \over \SI{10}{ms}}$.
One or more hidden
layers with varying numbers of neurons follow. After every layer we apply an activation or nonlinear function like tanh or ReLU. The output layer is
$(n+1)$--dimensional, where n is the number of backchannel categories we
want to predict. This layer has softmax as an activation function, which maps a $K$-dimensional vector of arbitrary values to values that are in the range $(0, 1]$ and that add up to 1, which allows us to interpret them as class probabilities.

We then calculate categorical cross-entropy of the output values of the network and the ground truth from the training data set. This gives us the loss function as the function mapping from the network inputs to the cross-entropy output. We can now train the parameters of the network by deriving it individually for each of the neurons and descending the resulting gradient using the back-propagation algorithm [@rumelhart_learning_1986].

In the simplest case (ignoring different kinds of backchannels) we train the network on the outputs $[1, 0]$ for backchannels and $[0, 1]$ for non-backchannels. A visualization of this architecture can be seen in @fig:nn_2h. In the following sections we will concentrate on this architecture.

\include{net1}

The placement of backchannels is dependent on previous backchannels: If
the previous backchannel utterance was a long time ago, the probability of a
backchannel happening shortly is higher and vice versa. To accommodate for this, we want
the network to also take its previous internal state or outputs into
account. We do this by modifying the above architecture to use
Long-short term memory layers (LSTM) instead of feed forward layers. LSTM neurons are recurrent, meaning they are connected to themselves in a time-delayed fashion, and they have an internal state cell which is transmitted through time and which has set and clear functions which are triggered by any combination of their inputs. LSTM networks are trained in similar fashion as feed forward networks, with the time-stacked layer instances unrolled into individual copies with some parameters shared before applying the backpropagation algorithm.

## Postprocessing

Our goal is to generate an artificial audio track containing utterances such as "uh-huh" or "yeah" at appropriate times. The neural neural network gives us a noisy value between 0 and 1, which
we interpret as the probability of a backchannel happening at the given
time. To generate our audio track from this output, we need to convert the noisy floating value into discrete trigger timestamps. We first run a low-pass filter over the network output, which gives us
a less noisy and more continous output function. Then we use a fixed trigger threshold to extract the ranges in the output where the predictor is fairly confident that a backchannel should happen.

Within these ranges we have multiple possibilities to choose the trigger anchor, which is a fixed time before the actual trigger.

The easiest method is to use the time of the maximum peak of every range where the value is larger than the threshold, but this requires us to wait until the value is lower than the threshold again before we can decide where the maximum is, which introduces another delay and is thus bad for live detection.

Another possibility is to use the start of the range, but this can give us worse results because it might force the trigger to happen earlier than the time the network would give the highest rating.

A compromise between the best quality and immediate decision is to use the first local maximum within the thresholded range. Because of the low-pass filter we mostly have few local maxima which differ from the global maximum within the given range, and it's easy to decide when the local maximum was reached by simply triggering as soon as the first derivate is $<0$.

An example of this postprocessing procedure can be seen in @fig:postproc.


\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{img/postprocessing.png}
    \caption{Postprocessing example}
    \label{fig:postproc}
\end{figure}

Another problem that arises is which low-pass filter to use. A simple gaussian blur uses future information which we do not want. One approach is to cut off the gaussian filter at some multiple of the standard deviation and have it offset to the left so the newest frame it uses is the present frame. Of course this causes the prediction to be delayed further. Another method is to use a predictive filter such as a Kalman Filter [@kalman_new_1960-1].

## Evaluation

To evaluate the performance of our predictions, we take a set of
monologuing segments from our training data and compare the prediction
with the ground truth, calculating the precision, recall, and F1 score
with varying margins of error.


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