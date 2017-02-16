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


Collection:

- @watanabe_voice_1989-1
    - not found
- @okato_insertion_1996
    - Languages: Japanese
    - Truth type: utterances
    - Features: Pause
    - Method: HMM for pitch contour
    - 
    - Evaluation Method: 
    - Margin of Error: (-100ms, +300ms) from target utterance end (?)
- @ward_using_1996
    - Japanese
- @noguchi_prosody-based_1998
    - Languages: Japanese
    - Features: Prosodic (pause, frequency F0)
- @ward_prosodic_2000
    - English, Japanese
    - Features: Low Pitch Cue, Pause
    - Margin of error: (-500, 500)

- @cathcart_shallow_2003
    - English
    - Features: trigrams, pauses
    - Corpus: HCRC Map Task Corpus
    - Eval Method: Precision, recall, F1

- @fujie_conversation_2004
    - Japanese
    - Features: utterances, prosodic
- @takeuchi_timing_2004
    - Japanese
    - features: porosodic
    - Method: decision tree, C4.5 learning algorithm
    - Corpus: SIG of Corpus-Based Research for Discourse and Dialogue, JSAI, 1999. ”Constructing a spoken dialogue dorpus as sharable research resource”
    - Eval method: recall, precision
- @kitaoka_response_2005
    - Japanese
    - Pitch, pause,
    - Eval: precision, recall, F1

- @nishimura_spoken_2007
    - Japanese
    - Features: Speech recog, 
- @morency_predicting_2008
    - English
    - Features: Eye gaze, low pitch, pause
    - HMM, CRF
    - Margin of error: happens during actual BC utterance
- @de_kok_multimodal_2009
    - English
    - Features: dialog, attention, head gestures, prosody (pitch, pause, etc)
    - Margin: peak in our probabilities (see Section 3) occurs during an actual end-of-speaker-turn.
- @morency_probabilistic_2010
    - Dutch
    - Corpus: MultiLis corpus
    - Special: building consensus Fconsensus
- @de_kok_learning_2010
- @huang_learning_2010
    - Subjective, on live corpus
    - Fconsensus
- @ozkan_concensus_2010
    - RAPPORT dataset
- @ozkan_latent_2010
- @poppe_backchannel_2010
- @de_kok_speaker-adaptive_2013
- @ozkan_latent_2013
- @de_kok_iterative_2014
- @mueller_using_2015-2

# Backchannel Prediction {#sec:extraction}

A listener backchannel is generally defined as any kind of feedback a listener
gives a speaker as an acknowledgment in a segment of conversation that is primarily one-way.
They include but are not limited to nodding [@watanabe_voice_1989-1], a shift in the gaze direction and short phrases. Backchannels are said to help build rapport, which is the feeling of comfortableness or being "in sync" with conversation partners [@huang_virtual_2011].

(-> motivation)
This thesis concentrates on short phrasal backchannels consisting
of a maximum of three words. We try to predict these for a given speaker
audio channel in a causal way, using only past information.

This would allow the predictor to be used in an online environment, for example to make a conversation with an artificial assistant more natural.

## BC Utterance selection {#sec:extractio:subsec:bc-utterance-selection}

The definition of backchannels varies in literature. There are many different kinds of phrasal backchannels, they can be non-committal ("uh huh", "yeah"), positive/confirming ("oh how neat", "great"), negative/surprised ("you're kidding", "oh my god"), questioning ("oh are you", "is that right"), et cetera.
To simplify the problem, we initially only try to predict the trigger times for any
type of backchannel, ignoring the distinction between different kinds of positive or negative
responses. Later we also try to distinguish between a limited set of
categories.

## Feature selection

The most commonly used audio features in related research are fast and
slow voice pitch slopes and pauses of varying lengths.
[@ward_prosodic_2000; @truong_rule-based_2010-1; @morency_probabilistic_2010]. A neural network is able to learn advantageous feature representations on its own, so we simply feed it the absolute pitch and
power (signal energy) values for a given time context, from which it is able to
calculate the pitch slopes and pause triggers on its own by subtracting the neighboring values in the time context for each feature.
The power value used is the logarithm of the raw `adc2pow` value output by the Janus Recognition Toolkit (JRTk).

We also try to use other tonal features used for speech recognition in addition to and instead of pitch and power.
The first feature is the fundamental frequency variation spectrum (FFV) [@laskowski2008fundamental], which is a representation of changes in the fundamental frequency over time, giving a more accurate view of the pitch progression than the single-dimensional pitch value which can be very noisy. This feature has seven dimensions in the default configuration given by Janus.

Other features we tried include the Mel-frequency cepstral coefficients (MFCC) with 20 dimensions \[ref\] and a set of
bottleneck features trained on phoneme recognition using a feed forward neural network, which is used for speech recognition at the Interactive Systems Lab.

## Training area selection

We generally assume to have two separate but synchronized audio tracks, one for the speaker and one for the listener, each with the corresponding transcriptions.
We need to choose which areas of audio we use to train the network. We
want to predict the backchannel without future information (also called _causally_ or
_online_), so we need to train the network to detect segments of audio
from the speaker track that probably cause a backchannel in the listener
track. The easiest method is to choose the beginning of the utterance in the transcript of the listener channel as an anchor $t$, and then use a fixed context range of width $w$ before that as the audio range to train the network to predict a backchannel ($[t-w, t]$). The width can range from a few hundred milliseconds to multiple seconds. We feed all the extracted features for this time range into the network at once, from which it will predict if a backchannel at time $t$ is appropriate. This approach is easy because it only requires searching for all backchannel utterance timestamps and then extracting the calculated range of audio. It may not be optimal though, because the delay between the last utterance of the speaker and the backchannel can vary significantly in the training data. This means it is not guaranteed that the training range will contain the actual trigger for the backchannel, which is assumed to be the last few words said by the speaker, and even if it does the last word will not be aligned within the context. This causes the need for the network to first learn to align it's input, making training harder and slower.

Another interesting anchor is the last few words before a backchannel.
We could for example choose $t$ as the center of the last speaker utterance before the backchannel, and then use $[t-0.5s, t+0.5s]$ as the training range. While experimenting this proved to be hard because without manual alignment it isn't clear what the last relevant utterance even is, and in many cases the relevant utterance ends after the backchannel happens, so we would need to be careful not to use any future data. The first approach seemed to work reasonably well, so we did not do any further experiments with the second approach.


We also need to choose areas to predict zero i.e. “no backchannel” (NBC). The number of training samples of this kind should be about the same amount as backchannel samples so the network is not biased towards one or the other.
To create this balanced data set, we can choose the range a few seconds before each backchannel as a negative sample. This gives us an exactly balanced data set, and the negative samples are intuitively meaningful, because in that area the listener explicitly decided not to give a backchannel response yet, so it is sensible to assume whatever the speaker is saying is not a trigger for backchannels.

The previous method trains the network on binary values, only 1 = 100% = "Backchannel happens here" and 0 = 0% = "Definitely no backchannel happens here.
Another approach for choosing a training area would be to not choose two separate areas to predict binary 1 and 0 values, but to instead use a area around every actual backchannel trigger and train the network on a bell curve with the center being at the ground truth with the maximum value of 1, and lower values between 0 and 1 for later and earlier values. For example, if there is a backchannel at t=5\,s and we identify t=4.5\,s as the actual trigger time, we could train the network on output=1 for a context centered at 4.5\,s, 0.5 for 200ms earlier and later and on $v \ll 1$ for more distant times. This has the same problem as described above in that we would need to know the exact time of the event that triggered the backchannel. Testing this approach by simply using a fixed offset before the onset of the backchannel utterance gave far worse results than the binary training approach, so we discarded the idea in favor of concentrating on finding the optimal context ranges for the binary approach.

## Training and Neural Network Design {#sec:training}

We begin with a simple feed forward network architecture. The input layer consists of
all the chosen features over a fixed time context. With a time context of $c\,\si{ms}$ and a feature dimension of $f$, this gives us a input dimension of $f \times \floor{c \over \SI{10}{ms}}$.
One or more hidden
layers with varying numbers of neurons follow. After every layer we apply an activation function (also called a nonlinearity) like tanh or ReLU. The output layer is
$(n+1)$--dimensional, where n is the number of backchannel categories we
want to predict. This layer has softmax as an activation function, which maps a $K$-dimensional vector of arbitrary values to values that are in the range $(0, 1]$ and that add up to 1, which allows us to interpret them as class probabilities.

We then calculate categorical cross-entropy of the output values of the network and the ground truth from the training data set. This gives us the loss function as the function mapping from the network inputs to the cross-entropy output. We can now train the parameters of the network by deriving it individually for each of the neurons and descending the resulting gradient using the back-propagation algorithm [@rumelhart_learning_1986].

In the simplest case (ignoring different kinds of backchannels) we train the network on the outputs $[1, 0]$ for backchannels and $[0, 1]$ for non-backchannels. When evaluating we can ignore the second dimension of this output, simply interpreting the first dimension as a probability. A visualization of this architecture can be seen in @fig:nn_2h. In the following sections we will concentrate on this architecture.

\include{net1}

The placement of backchannels is dependent on previous backchannels: If
the previous backchannel utterance was a long time ago, the probability of a
backchannel happening shortly is higher and vice versa. To accommodate for this, we want
the network to also take its previous internal state or outputs into
account. We do this by modifying the above architecture to use
Long-short term memory layers (LSTM) instead of dense feed forward layers. LSTM neurons are recurrent, meaning they are connected to themselves in a time-delayed fashion, and they have an internal state cell which is transmitted through time and which has set and clear functions which are triggered by any combination of their inputs. LSTM networks are trained in similar fashion as feed forward networks, with the time-stacked layer instances unrolled into individual copies with shared parameters before applying the backpropagation algorithm.

## Postprocessing

Our goal is to generate an artificial audio track containing utterances such as "uh-huh" or "yeah" at appropriate times. The neural neural network gives us a noisy value between 0 and 1, which
we interpret as the probability of a backchannel happening at the given
time. To generate our audio track from this output, we need to convert the noisy floating value into discrete trigger timestamps. We first run a low-pass filter over the network output, which removes all the higher frequencies and gives us
a less noisy and more continuous output function. 

To ensure our predictor does not use any future information, the low-pass filter must be causal. A common low-pass filter is the gaussian blur, which folds the input function with a bell curve. This filter is symmetric, which in our case means it uses future information as well as past information. To prevent this, we cut the filter of asymmetrically for the right side (that would range in the future), with a cutoff at some multiple $c$ of the standard deviation $\sigma$. Then we shift the filter to the left so the last frame it uses is $\pm\SI{0}{ms}$ from the prediction target time. This means the latency of our prediction increases by $c\cdot\sigma\,ms$. If we choose $c=0$, we cut off the complete right half of the bell curve, meaning we do not need to shift the filter, which keeps the latency at 0 at the cost of accuracy of the low-pass filter.

Another possible filter to use would be the Kalman Filter [@kalman_new_1960-1], which tries to predict the value of a function based on a noisy function masking the actual function. This filter has many parameters requiring tuning, so we used the described gaussian filter method.

After this filter we use a fixed trigger threshold to extract the ranges in the output where the predictor is fairly confident that a backchannel should happen. We trigger exactly once for each of these ranges.
We have multiple possibilities to choose the trigger anchor with in each range.

The easiest method is to use the time of the maximum peak of every range where the value is larger than the threshold, but this requires us to wait until the value is lower than the threshold again before we can decide where the maximum is, which introduces another delay and is thus bad for live detection.

Another possibility is to use the start of the range, but this can give us worse results because it might force the trigger to happen earlier than the time the network would give the highest probability rating.

A compromise between the best quality and immediate decision is to use the first local maximum within the thresholded range. Because of the low-pass filter we mostly have no or few local maxima which differ from the global maximum within the given range, and it's easy to decide when the local maximum was reached by simply triggering as soon as the first derivate is $<0$.

When we have the trigger anchor, we can either immediately trigger, or delay the actual trigger by a fixed amount of time, which can be useful if we notice the prediction would happen to early otherwise.

An example of this postprocessing procedure can be seen in @fig:postproc.


\begin{figure}
    \centering
    \includegraphics[width=\textwidth]{img/postprocessing.pdf}
    \caption{Postprocessing example}
    \label{fig:postproc}
\end{figure}




## Evaluation

The switchboard dataset contains alternating conversation. Because our predictor is only capable of handling situations where one speaker is consistently speaking and the other consistently listening, we need to evaluate it on only those segments. We call these segments "monologuing segments".

For a simple subjective evaluation, we take a some random audio tracks from the training data and extract the monologuing segments. For each segment, we remove the original listener channel and replace it with the artificial one. This audio data is generated by inserting a random backchannel audio sample at every predicted timestamp. We get these audio samples from a random speaker from the training data, keeping the speaker the same over the whole segment so it sounds like a specific person is listening to the speaker.

To get an objective evaluation of the performance of our predictions, we again take a set of
monologuing segments from the evaluation data set and compare the prediction
with the ground truth, i.e. all the timestamps where a backchannel happens in the original data.

We interpret a prediction as correct if it is within a specific margin of the nearest real backchannel in the dataset. For example, with a margin of error of $[\SI{-100}{ms}, \SI{+300}{ms}]$, if the real data has a backchannel at 5.5 seconds, we say the predictor is correct if it also produces a backchannel within $[\SI{5.5}{s} - \SI{100}{ms}, \SI{5.5}{s} + \SI{300}{ms}] = [\SI{5.4}{s}, \SI{5.8}{s}]$.

In other research, varying margins of error have been used. We use a margin of 0ms to +1000ms for our initial tests, and later also do our evaluation with other margins for comparison with related research.

After aligning the prediction and the ground truth using the margin of error, we get two overlapping sets of timestamps. The set of predictions is called "selected elements", the set of true elements is called "relevant elements". The number of true positives is defined as $\mathit{TP} = |\mathit{selected} \cap \mathit{relevant}|$, which is all the backchannels the predictor correctly identified as such. The number of false positives is defined as $\mathit{FP} = |\mathit{selected} \setminus \mathit{relevant}|$, which is all the backchannels the predictor output that are not contained in the dataset. The number of false negatives is defined as $\mathit{FN} = |\mathit{relevant} \setminus \mathit{selected}|$, which is all the backchannels that the predictor should have found but didn't.

With these, we can now calculate the measures _Precision_ and _Recall_ commonly used in information retrieval and binary classification:

$$
\begin{split}
\mathit{Precision} & = {\mathit{TP} \over \mathit{TP} + \mathit{FP}} \\
\mathit{Recall} & = {\mathit{TP} \over \mathit{TP} + \mathit{FN}}
\end{split}
$$

Both of these are values between 0 and 1. The _Precision_ value is the fraction of returned values that were correct, and can thus be seen as a measure of the _quality_ of a algorithm. The _Recall_ value, also known as _sensitivity_ is the fraction of relevant values that the algorithm output, thus it can be interpreted as a measure of _quantity_. Precision and Recall are in a inverse relationship, and it is usually possible to increase one of them while reducing the other by tweaking parameters of the algorithm. Recall can be easily maximized to 100% by simply returning true for every given timestamp. Precision can be maximized by never outputting anything, causing every predicted value to be correct. To solve this problem, we use the normalized harmonic mean of precision and recall, also known as the F1-Score or F-Measure:

$$
\mathit{F1\,Score} = 2 \cdot \frac{1}{\frac{1}{\mathit{Recall}} + \frac{1}{\mathit{Precision}}}
    = 2 \cdot \frac{\mathit{Precision}\cdot\mathit{Recall}}{\mathit{Precision}+\mathit{Recall}}
$$

We use the F1-Score to objectively measure the performance of our prediction systems.

# Experimental Setup {#experiments}

## Dataset

We used the switchboard dataset [@swb] which consists of 2438
telephone conversations of five to ten minutes, 260 hours in total.
Pairs of participants from across the United States were encouraged to talk about a specific topic selected from 70 possibilities. Conversation partners and topics were selected so two people would only talk once with each other and every person would only discuss a specific topic once.
These telephone conversations are annotated with transcriptions and word alignments \cite{swbalign} with a total of 390k utterances or 3.3 million words. The audio data is given as stereo files, where the first speaker is on the left channel and the second speaker on the right channel. We split the dataset randomly into 2000 conversations for training, 200 for validation and 238 for evaluation. As opposed to many other datasets, the transcriptions also contain backchannel utterances like _uh-huh_ and _yeah_, making it ideal for this task.

The transcriptions are split into _utterances_, which are multiple words grouped by speech structure, for example (slashes indicate utterance boundaries): "did you want me to go ahead / okay well one thing i- i- i guess both of us have very much aware of the equality / uh it seems like women are uh just starting to really get some kind of equality not only in uh jobs but in the home where husbands are starting to help out a lot more than they ever did um". The length of these utterances varies in length from one word to whole sentences. Each of these utterance has a start time and stop time attached, where the stop time of one utterance is always the same as the start time of the next utterance. For longer periods of silence, an utterance containing the text  "[silence]" is between them.

The word alignments have the same format, except they are split into single words, each with start and stop time. Here the start and stop times are mostly exactly aligned with the actual word audio start and end. _[silence]_ utterances are between all words that have even a slight pause between them.

To better understand and visualize the dataset, we first wrote a complete visualization GUI for viewing and listening to audio data, together with transcriptions, markers and other data. This proved to be very helpful. A screenshot of the UI inspecting a short portion of one of the phone conversations can be seen in @fig:amazing.

![From top to bottom: Speaker A audio data, transcription, and word alignment, then the same for speaker B.](img/20170208185355.png){#fig:amazing}

## Extraction {#extraction-1}

### Backchannel utterance selection

We used annotations from The Switchboard Dialog Act Corpus [@swda] to
decide which utterances to classify as backchannels. The SwDA contains
categorical annotations for utterances for about half of the data of the
Switchboard corpus. An excerpt of the most common categories can be seen in @tbl:swda.

\begin{longtable}{lp{3cm}lp{4cm}ll}
\caption{Most common categories from the SwDA Corpus}\label{tbl:swda}\tabularnewline
\toprule
~ & name & act\_tag & example & train\_count &
full\_count\tabularnewline
\midrule
\endhead
1 & Statement-non-opinion & sd & Me, I'm in the legal department. &
72824 & 75145\tabularnewline
2 & Acknowledge (Backchannel) & b & Uh-huh. & 37096 &
38298\tabularnewline
3 & Statement-opinion & sv & I think it's great & 25197 &
26428\tabularnewline
4 & Agree/Accept & aa & That's exactly it. & 10820 &
11133\tabularnewline
5 & Abandoned or Turn-Exit & \% & So, - & 10569 & 15550\tabularnewline
6 & Appreciation & ba & I can imagine. & 4633 & 4765\tabularnewline
7 & Yes-No-Question & qy & Do you have to have any special training? &
4624 & 4727\tabularnewline
\bottomrule
\end{longtable}

We extracted all utterances containing one of the tags beginning with "b" (which stands for backchannels or backchannel-like utterances), and counted their frequency. Because the dataset also marks some longer utterances as backchannels, we only use those that are at most three words long to exclude those that transmit a lot of additional information to the speaker.

We chose to use the top 150 unique utterances from this set. For the most common ones, see @tbl:bcs.


 | aggregated | self | count | category | text
 |------------|------|-------|----------|------
 | 31.92% | 31.92% | 14319 | b | uh-huh
 | 61.08% | 29.15% | 13075 | b | yeah
 | 68.95% | 7.87% | 3532 | b | right
 | 71.73% | 2.78% | 1249 | b | oh
 | 73.69% | 1.96% | 877 | b | [silence]
 | 75.09% | 1.40% | 629 | b | oh yeah
 | 76.49% | 1.40% | 627 | b | yes
 | 77.84% | 1.35% | 607 | b | okay
 | 78.86% | 1.02% | 458 | bk | okay
 | 79.75% | 0.89% | 399 | b | huh
 | 80.57% | 0.81% | 364 | b | sure
 | 81.29% | 0.72% | 325 | bk | oh okay
 | 81.93% | 0.64% | 288 | b | huh-uh
 | 82.56% | 0.63% | 282 | bh | oh really
 | 83.15% | 0.59% | 264 | ba | wow
 | 83.73% | 0.58% | 259 | b | um
 | 84.16% | 0.43% | 193 | bh | really
 | 84.57% | 0.41% | 186 | b | really
 | 84.97% | 0.39% | 177 | bk | oh

: Most common backchannel utterances in the SwDA dataset {#tbl:bcs}


The SwDA is incomplete, it only contains labels for about half of the Switchboard dataset. Because we wanted to use as much training data as possible, we had to identify
utterances as backchannels just by their text. As can be seen in @tbl:bcs, the SwDA also has some silence utterances marked as backchannels, as well as only laughter and noise, which we can't distinguish from normal silence, so we ignore them altogether. We manually removed some
requests for repetition like “excuse me” from the SwDA list, and added some
other utterances that were missing from the SwDA transcriptions but
present in the original transcriptions, by going through the most common
utterances and manually selecting those that seemed relevant, including but not limited to ‘um-hum yeah’, ‘absolutely’, ‘right uh-huh’.

In total we now had a list of 161 distinct backchannel utterances. The most common backchannels in the data set are “yeah”, “um-hum”,
“uh-huh” and “right”, adding up to 41860 instances or 68% of all
extracted backchannel phrases.

The transcriptions also contain markers indicating laughter while talking (e.g. "i didn't think that well we wouldn't still be [laughter-living] [laughter-here] so ..."),
laughter on its own (`[laughter]`), noise markers (`[noise]`) for microphone crackling or similar and markers for different pronunciations (for example "mhh-kay" is transcribed as `okay_1`).
To select which utterances should be categorized as backchannels and
used for training, we first filter noise and other markers from the
transcriptions, for example `[laughter-yeah] okay_1 [noise]` becomes
`yeah okay` and then compare the resulting text to our list of backchannel phrases.

Some utterances such as “uh” can be both backchannels and speech
disfluencies. For example: "... pressed a couple of the buttons up in the / uh / the air conditioning panel i think and uh and it reads out codes that way". Note that the first _uh_ is it's own utterance and would thus be seen by our extractor as a backchannel. The second occurrence of "uh" has normal speech around in the same utterance it so we would already ignore it. We only want those utterances that are actual backchannels, so after filtering by utterance text we only choose those that have either silence or another backchannel before them.

This gives us the following selection algorithm:
```python
def is_backchannel(utterance):
    text = noise_filter(utterance)
    if index(utterance) == 0:
        # no other utterance before this, can't be a backchannel
        return False
    previous_text = noise_filter(previous(utterance))
    return (text in valid_backchannels and
            (is_silent(previous_text) or is_backchannel(previous(utterance)))
```

This method gives us a total of 61645
backchannels out of 391593 utterances (15.7%) or 71207 out of 3228128
words (2.21%). Note that the percentage of words is much lower because backchannel utterance are on average much shorter than other utterances.

### Feature extraction


#### Acoustic features

We used the Janus Recognition Toolkit [@janus] for the acoustic feature extraction (power, pitch tracking, FFV, MFCC).

These features are extracted for 32ms frame windows, with a frame shift of 10ms. This gives us 100 frames per feature per second.

A sample of the pitch and power features can be seen in @fig:pitchpow.

![From top to bottom: Audio samples, transcription, pitch and power for a single audio channel. Note that the pitch value is only meaningful when the person is speaking.](img/20170208184917.png){#fig:pitchpow}


#### Linguistic features

In addition to these prosodic features, we also tried training Word2Vec [@mikolov_efficient_2013] on the Switchboard dataset.
Word2Vec is an "Efficient Estimation of Word Representations in Vector Space". After training it on a lot of text, it will learn the meaning of the words from the contexts they appear in, and then give a mapping from each word in the vocabulary to an n-dimensional vector, where n is configurable. Similar words will appear close to each other in this vector space, and it's even possible to run semantic calculations on the result. For example calculating $\mathit{king} - \mathit{man} + \mathit{woman}$ gives the result $=\mathit{queen}$ with the highest confidence. Because our dataset is fairly small, we used relatively small word vectors (5 - 20 dimensions). We train it only on utterances that are not backchannels or silence.

For simplicity, we extract these features parallel to those output by Janus, with a 10 millisecond frame shift. To ensure we don't use any future information, we extract the word vector for the last word that ended _before_ the current frame timestamp. This way the predictor is in theory still online, though this assumes the availability of a speech recognizer with instant output. An example of this can be seen in @fig:w2v.

![five-dimensional Word2Vec feature for some text. The encoding is offset by one word, for example the encoding for "twenty" is seen below the word "four", because we encode the word that _ended_ before the current time. Note that with this method we indirectly encode the length of the words / utterances.](img/20170216222513.png){#fig:w2v}

#### Context and stride

We extract the features for a fixed time context. Then we use a subset of that range as the area we feed into the network. 
As an example, we can extract the range [-2000ms, 0ms] for every feature, giving us 200 frames. We train the network on 1500ms of context, so we treat every offset like [-2000, -500ms], [-1990ms, -490ms], ..., [-1500ms, 0ms] as individual training samples. This gives us 50 training samples per backchannel utterance, greatly increasing the amount of training data, but introducing smear as the network needs to learn to handle a larger variance in when the backchannel cue appears in its inputs, and thus reducing the confidence of its output.

This turned out to not work very well, so in the end we settled on only extracting the features for the range [-w - 10ms, 0] where w is the context width, and training the network on [-w - 10ms, 10ms] and [-w, 0ms]. This gives us two training samples per utterance, reduces the smearing problem and at the same time force the network to learn to correctly handle when its inputs are the same or similar but offset by one.

We can also choose to only use every n-th timestep, which we call the "context stride". This works under the assumption that the input features don't change with a high frequency, which greatly reduces the input dimension and speeds up training. In practice a stride of 2 worked well, meaning one frame every 20 milliseconds. This works great in combination with the above. For example, with a stride of 2 and a context size of 100ms, we would now get these two training samples (described as frame indices relative to the onset of the backchannel utterance):

1. [-10, -8, -6, -4, -2, 0]
2. [-11, -9, -7, -5, -3, -1]

## Training {#training-1}

We used Theano [@theano] with Lasagne v1.0-dev [@lasagne] for rapid prototyping
and testing of different parameters. We trained a total of over 200
network configuration with various context lengths (500ms to 2000ms),
context strides (1 to 4 frames), network depths ranging from one to four
hidden layers, layer sizes ranging from 15 to 100 neurons, activation
functions (tanh and relu), gradient descent methods (SGD, Adadelta and
Adam), dropout layers (0 to 50%) and layer types (feed forward and
LSTM).

In general, we used three variables to monitor training: The first is training loss, which is what the network optimizes, as defined in @sec:training. We expect this variable to decrease more or less monotonically while training, because the network is descending it's gradient. The second variable is validation loss, which is the same function a training loss, but on the separate validation data set. This allows us to tell whether the network is still learning useful information or starting to overfit. In general, we expect this to be about the same as the training loss, possibly a bit higher. If the training loss is still falling but validation loss is starting to increase again, the network is overfitting on the training data.

 The third variable is the validation error, which we define as $$1 - {\sum_{s \in S}\{1 \text{ if prediction}(s) = \text{truth}(s)\text{ else }0\} \over |S|},$$ where $S$ is all the frames for all the samples in the validation data set, and $\text{prediction}(s) = \text{argmax}{\text{output}}$. This means we take the network output, convert it into the category the network is currently most confident in, and compare it with the ground truth. For example, say we are training with two outputs, one for "Non BC" and one for "BC". The network will always give us two values between 0 and 1 that add up to one because of the softmax function as described in @sec:training. If the network output is [0.7, 0.3] for a specific sample we interpret it as "Non BC", because the confidence in the "Non BC" category is higher. If the ground truth for this sample is also "Non BC", the validation error is 0, otherwise 1.
 This gives us a statistical rating of the current state of the predictor that has less resolution than the loss function (because it throws away the "confidence" of the network), but is closer to our interpretation of the network output when evaluating. We use this value as an initial comparison between predictors and to decide which epoch of the network gives will probably give the best results.

We started with a simple model with a configuration of pitch and power as input and 800 ms of context, giving us $80\cdot 2 = 160$ input dimensions, hidden layers of 100 $\rightarrow$ 50 feed forward neurons. We trained this using many different gradient descent methods such as stochastic gradient descent (SGD), SGD with momentum, Nesterov momentum, Adadelta and Adam, each with fixed learning rates to start. The momentum methods add a speed variable to the descent. This can be interpreted similar to its physical name giver. Imagine a ball rolling down a mountain slope. For each time period, it keeps it's previous momentum and is thus able to jump over small dents in the ground (local minima). In our case, momentum worsened the results, so we stayed with SGD and Adadelta. 

We tried different weight initialization methods. Initializing all weights with zero gave significantly worse results than random initialization, so we stayed with the Lasagne default of Glorot uniform initialization, which uses uniformly distributed random values, with the maximum value scaled so it makes statistical sense in the layer dimensions [@glorot_understanding_2010].

We compared online and offline prediction, where offline prediction got 400 ms of past audio and 400 ms of future audio from the onset of the backchannel utterance. Offline prediction gave 18% better results, but of course we are more interested in online prediction.

The first simple LSTM we tested by simply replacing the feed forward layers with LSTM layers immediately improved the results by 16% without any further adjustments. But this showed the issues with a fixed learning rate, as the gradient regularily exploded after every 10 - 15 epochs, as can be seen in @fig:exploding. One epoch is defined as one whole backward pass of all the training data through the network. When adding FFV, increasing the input dimension per time frame from 2 to 9, SGD stopped working at all without manually tuning the learning rate.

![Exploding gradient while training a LSTM network. Shown is the training loss over epochs](img/20170208233441.png){#fig:exploding}

These problem was solved by using Adam [@kingma_adam:_2014] instead of SGD, which is a gradient descent method related to Adadelta and Adagrad which also incorporates momentum in some way, see @fig:adam. No one really understands how these work, but Adam with a fixed learning rate of 0.001 worked great for us, so we did all further testing using Adam.

![The Adam method for stochastic optimization. I have no idea what any of those letters mean, but it's been cited a bazillion times and works great.](img/adam.png){#fig:adam}


The LSTM networks we tested were prone to overfitting very quickly, but they still
provided better results after two to three epochs than normal feed forward networks after 100 epochs. Overfitting happens when the results still improve on the training data set, but plateau or get worse on the validation data set. This means the network is starting to learn specific quirks in the training data set by heart, which it then can't apply on other data.

We tried adding dropout layers to the networks to try and avoid overfitting and to generally improve the results. Dropout layers randomly disconnect a specified fraction of neurons in a layer, different for every training batch. This should in theory help the network interpret it's inputs even when it is partially "blind". For validation the dropout is deactivated, so the network is able to take advantage of every single feature when actually using it as a predictor. In this case, we tried adding different dropout settings such as "input (20% dropout) $\rightarrow$ 125 neurons (50% dropout) $\rightarrow$ 80 neurons (50% dropout) $\rightarrow$ output" but this only increased the noise in the training loss and did not improve the results over a simple "input $\rightarrow$ 70 neurons $\rightarrow$ 45 neurons" configuration, both for feed forward and for LSTM networks.

![The same LSTM network trained without (left) and with (right) L2-Regularization. Note that without regularization the network starts overfitting after two epochs. With regularization training and validation loss mostly stay the same with regularization, and the validation error continues to improve. Training loss is blue, validation loss is red and validation error is orange.](img/20170209000001.png){#fig:l2reg}


L2-Regularization reduced this problem and slightly improved the
results, as can be seen in the example in @fig:l2reg.

## Evaluation {#eval-1}

The training data contains two-sided conversations. Because the output
of our predictors is only relevant for segments with only one person
talking, we only run our evaluation on monologuing segments.

For this we define the predicate `is_listening` (is only emitting
utterances that are backchannels or silence) and `is_talking`
(`= not is_listening`). An example of this can be seen in @fig:listening. A monologuing segment is the maximum possible
time range in which one person is consistently talking and the other
only listening. We only consider segments of a minimum length of five seconds to exclude sections of alternating conversation.
The results did not significantly change when adjusting this minimum length between two and ten seconds.

![A short audio segment showing talking and listening areas. Note that a backchannel from Speaker B in the middle ("yeah") does not classify that area as speaking. At the end both people talk at the same time, so speaker B needs to repeat herself.](img/20170211153622.png){#fig:listening}

An interesting aspect is that in our tests the predictors had difficulty distinguishing segments of speech that indicate a backchannel and those that indicate a speaker change. Subjectively, this makes sense because in many cases a backchannel can be seen as a replacement for starting to talk more extensively.

# Results

We use "$70 : 35$" to denote a network layer configuration of $\text{input} \rightarrow 70\text{ neurons} \rightarrow 35\text{ neurons} \rightarrow \text{output}$.
All results in \autoref{fig:survey} use the following setup if not otherwise stated: LSTM, configuration: $(70 : 35)$, input features: power, pitch, FFV, context frame stride: 2, margin of error: 0\,ms to +1000\,ms. Precision, recall and F1-Score are given for the validation data set.


We tested different context widths. A context width of $n\,\si{ms}$ means we use the range $[-n\,\si{ms},\allowbreak 0\,\si{ms}]$ from the beginning of the backchannel utterance. The results improved significantly when increasing the context width from our initial value of 500\,ms. Performance peaked with a context of about 1500\,ms, as can be seen in Figure \ref{varycontext}. Longer contexts tended to cause the predictor to trigger too late.


We tested using only every n-th frame of input data. Even though we initially did this for performance reasons, we noticed that training on every single frame has worse performance than skipping every second frame due to overfitting. Taking every fourth frame seems to miss too much information, so performance peaks at a context stride of 2, as can be seen in Figure \ref{varystrides}.

We tested different combinations of features. Using FFV as the only prosodic feature performs worse than FFV together with the absolute pitch value. Adding MFCCs does not seem to improve performance in a meaningful way when also using pitch. See Figure \ref{varyfeatures} for more information. Note that using \emph{only} word2vec performs reasonably well, because with our method it indirectly encodes the time since the last utterance.

Figure \ref{varylstm} shows a comparison between feed forward and LSTM networks. The parameter count is the number of connection weights the network learns in training. Note that LSTMs have significantly higher performance, even with similar parameter counts.

We compared different layer sizes for our LSTM networks, as shown in Figure \ref{varylayers}. A network depth of two hidden layers worked best, but the results are adequate with a single hidden layer or three hidden layers.

In \autoref{fig:final}, our final results are given for the completely independent evaluation data set. We compared the results by Mueller et al. (2015) \cite{mueller} with our system. They used the same dataset, but focused on offline predictions, meaning their network had future information available, and they evaluated their performance on the whole corpus including segments with silence and with alternating conversation. We adjusted our baseline and evaluation system to match their setup. As can be seen in Figure \ref{fig:mueller}, our predictor performs significantly better.
All other related research used different languages, datasets or evaluation methods, making a direct comparison meaningless.

Figure \ref{fig:ourbest} shows the results with our own evaluation method. We provide values for different margins of error used in other research. Subjectively, missing a BC trigger may be more acceptable than a false positive, so we also provide a result with a balanced precision and recall. 


\begin{figure}
\caption{Results on the Validation Set}\label{fig:survey}
\centering
\subfloat[Results with various context lengths. Performance peaks at 1500\,ms.]{
    \begin{tabular}{cccc}
    \hline\noalign{\smallskip}
    Context & Precision & Recall & F1-Score \\
    \noalign{\smallskip}\svhline\noalign{\smallskip}
    500\,ms & 0.219 & 0.466 & 0.298 \\
    1000\,ms & 0.280 & 0.497 & 0.358 \\
    1500\,ms & 0.305 & 0.488 & \bf{0.375} \\
    2000\,ms & 0.275 & 0.577 & 0.373 \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \label{varycontext}
}

\subfloat[Results with various context frame strides]{
    \begin{tabular}{cccc}
    \hline\noalign{\smallskip}
    Stride & Precision & Recall & F1-Score \\
    \noalign{\smallskip}\svhline\noalign{\smallskip}
    1 & 0.290 & 0.490 & 0.364 \\
    2 & 0.305 & 0.488 & \bf{0.375} \\
    4 & 0.285 & 0.498 & 0.363 \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \label{varystrides}
}

\centering
\subfloat[Results with various input features, separated into only acoustic features and acoustic plus linguistic features.]{
    \begin{tabular}{lccc}
    \hline\noalign{\smallskip}
    Features & Precision & Recall & F1-Score \\
    \noalign{\smallskip}\svhline\noalign{\smallskip}
    power, pitch & 0.307 & 0.435 & 0.360 \\
    power, pitch, mfcc & 0.278 & 0.514 & 0.360 \\
    power, ffv & 0.259 & 0.513 & 0.344 \\
    power, ffv, mfcc & 0.279 & 0.515 & 0.362 \\
    power, pitch, ffv & 0.305 & 0.488 & \bf{0.375} \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    word2vec$_{dim=30}$ & 0.244 & 0.478 & 0.323 \\
    power, pitch, word2vec$_{dim=30}$ & 0.318 & 0.486 & 0.385 \\
    power, pitch, ffv, word2vec$_{dim=15}$ & 0.321 & 0.475 & 0.383 \\
    power, pitch, ffv, word2vec$_{dim=30}$ & 0.322 & 0.497 & \bf{0.390} \\
    power, pitch, ffv, word2vec$_{dim=50}$ & 0.304 & 0.527 & 0.385 \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \label{varyfeatures}
}


\subfloat[Feed forward vs LSTM]{
    \begin{tabular}{ccccc}
    \hline\noalign{\smallskip}
    Layers & Parameters & Precision & Recall & F1-Score \\
    \noalign{\smallskip}\svhline\noalign{\smallskip}
    FF ($56 : 28$) & 40k & 0.230 & 0.549 & 0.325 \\
    FF ($70 : 35$) & 50k & 0.251 & 0.468 & 0.327 \\
    FF ($100 : 50$) & 72k & 0.242 & 0.490 & 0.324 \\
    LSTM ($70 : 35$) & 38k & 0.305 & 0.488 & \bf{0.375} \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \label{varylstm}
}

\subfloat[Comparison of different network configurations. Two LSTM layers give the best results.]{
    \begin{tabular}{cccc}
    \hline\noalign{\smallskip}
    Layers & Precision & Recall & F1-Score \\
    \noalign{\smallskip}\svhline\noalign{\smallskip}
    $100$ & 0.280 & 0.542 & 0.369 \\
    $50 : 20$ & 0.291 & 0.506 & 0.370 \\
    $70 : 35$ & 0.305 & 0.488 & \bf{0.375} \\
    $100 : 50$ & 0.303 & 0.473 & 0.369 \\
    $70 : 50 : 35$ & 0.278 & 0.541 & 0.367 \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \label{varylayers}
}


\end{figure}


\newcommand{\csubfloat}[2][]{%
  \makebox[0pt]{\subfloat[#1]{#2}}%
}
\captionsetup[subfigure]{width=\textwidth}
\begin{figure}
\centering
\caption{Final best results on the evaluation set (chosen by validation set)}\label{fig:final}


\csubfloat[Comparison with previous research. Mueller et al. did their evaluation without the constraints defined in \autoref{eval-1}, so we adjusted our baseline and evaluation to match their setup]{
    \begin{tabular}{lccc}
    \hline\noalign{\smallskip}
        Predictor & Precision & Recall & F1-Score \\
        \noalign{\smallskip}\svhline\noalign{\smallskip}
        Baseline (random) & 0.0417 & 0.0417 & 0.0417 \\
        Müller et al. (offline) \cite{mueller} & -- & -- & 0.109 \\
        Our results (offline, context of \SIrange{-750}{750}{ms}) & 0.114 & 0.300 & \bf{0.165} \\
        Our results (online, context of \SIrange{-1500}{0}{ms}) & 0.100 & 0.318 & 0.153 \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \label{fig:mueller}
}

\csubfloat[Results with our evaluation method with various margins of error used in other research \cite{survey}. Performance improves with a wider margin width and with a later margin center.]{
    \begin{tabular}{clccc}
    \hline\noalign{\smallskip}
        Margin of Error & Constraint & Precision & Recall & F1-Score \\
        \noalign{\smallskip}\svhline\noalign{\smallskip}
        \SIrange{-200}{+200}{ms} && 0.172 & 0.377 & 0.237 \\
        \SIrange{-100}{+500}{ms} &&	0.239 & 0.406 & 0.301 \\
        \SIrange{-500}{+500}{ms} && 0.247 & 0.536 & 0.339 \\
    \hline\noalign{\smallskip}
        \SIrange{0}{+1000}{ms} & Baseline (random) & 0.111 & 0.0521 & 0.0708 \\
         & Balanced Precision and Recall & 0.342 & 0.339 & 0.341 \\
         & Best F1-Score (only acoustic features) & 0.294 & 0.488 & 0.367 \\
         & Best F1-Score (acoustic and linguistic features) & 0.308 & 0.497 & \bf{0.380} \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \label{fig:ourbest}
}

\end{figure}

# Technical Details

We implemented multiple software components to help us understand the data, extract the relevant parts, train the predictors and evaluate the results.

## Web Visualizer

### Description

The web visualizer is an interactive tool to display time-based data. It can display audio files as waveforms like other popular audio editing software. It can also show arbitrary data that has one or more dimensions per time frame of arbitrary length. We use this to show the features we extract from the audio and use for training of the neural networks, as well as the resulting network outputs. It can show these either as a line graph as seen in @fig:postproc or as a grayscale color value, as seen in @fig:grayscale.

![The output of a neural network trying to predict backchannel categories for a audio segment. The first category is "No Backchannel", so it is roughly inverse to the other categories (neutral, question, surprised, affirmative, positive). From top to bottom: Audio, Text, Raw NN output, Smoothed NN output. White means higher probabilities, black means lower probabilities.](img/20170211151523.png){#fig:grayscale}

In addition, it can also show time-aligned textual labels either as seperate features or as overlays over the other data. We use this to display the transcriptions below the audio, and to highlight ranges of audio, for example which areas we use as positive and negative prediction areas, or which areas we interpret as "talking" or "listening" as can be seen in @fig:listening. This allows us to quickly check if the chosen prediction areas make general sense.

The user can choose a specific or random conversation from the training, validation or evaluation data set. Then they can choose to view and combination of the available features from a categorical tree [@fig:categories]. The user can zoom into any section of the data and play the audio, the UI will follow the current playback position.

\begin{wrapfigure}{R}{0.3\textwidth}
\centering
\includegraphics{img/20170211154422.png}
\caption{Selecting a feature to display in the Web Visualizer.\label{fig:categories}}
\end{wrapfigure}

The user can also save the extact current GUI state including zoom and playback position to generate a unique URL that will restore the visualizer to that state when loaded.

Some presets are also available, such as the default view which will show both channels of a conversation, together with the transcriptions, power, pitch and highlights for the training areas. Another meta preset is the NN output view, which includes a single audio channel and the raw output, smoothed output and resulting audio track for a trained network, as seen for example in @fig:postproc. The exact configuration can be chosen in a form [@fig:loadingnn]. Newly trained networks will automatically be available when the training is finished.

![Loading the output of the best epoch of specific network for channel A of the current conversation](img/20170211161913.png){#fig:loadingnn width=70%}


### Implementation

The visualizer is split into two parts, the server side (backend) and the client side (frontend). The server is written in python. It accepts connections via Websockets. It sends a list of available conversation IDs and corresponding features. The client can then select a conversation and dynamically load any combination of the available features, which the server will evaluate on demand while caching the most recently used features. The backend is in a shared code base with the extraction code, so it uses parts of the same code we also use for training and evaluation. The client is written in TypeScript with MobX and React.



## Extraction and Learning

We wrote our own parsing toolkit for the transcriptions and word alignments from the SwDA [@swda].
We used the Python interface of the Janus Recognition Toolkit, numpy, scipy and sklearn for feature extraction and filters. We implemented a small learning toolkit on top of Lasagne that reads the extraction and training configuration from a JSON file and outputs the training history as JSON. We used git to track the changes and git tags so every extraction and training output had the exact state of the code attached. This allowed us to easily reproduce different outputs. We also wrote a meta configuration generator, that takes a set of interesting configuration parameters and outputs all the relevant permutations, which can then be extracted, trained and evaluated in parallel.

Many of the methods for extraction were written as pure functions, which allowed us to create a transparent caching system as a function decorator that automatically caches the result of expensive functions on disk. This way, we do not need to explicitly seperate the extraction from training and write data files, we can simply run the training and the extraction only runs when the extraction parameters (features / context size) changes. We also used joblib, a python library for easy parallelization, to enable the extraction and evaluation processes to use all available CPU cores.

## Evaluation Visualizer

The evaluation visualizer reads all the output JSON files from training and displays the resulting loss graphs as seen in @fig:l2reg. It also shows a table of the evaluation results with various parameters for a every neural network sorted from best validation result to worst [@fig:fullsingle]. There is also a graph for detailed comparison of the postprocessing and evaluation methods, as shown in @fig:varysigma, @fig:varycenter, and @fig:varymonologuing.

![Full overview over a single training result, showing the loss graph for checking for overfitting issues and the evaluation results.](img/20170212134323.png){#fig:fullsingle}

![Precision, Recall and F1-Score depending on the smoothing sigma used. The F1-Score peaks at $\sigma=350\,ms$, Precision peaks at $\sigma=525\,ms$. With lower sigmas, the predictor triggers more often causing the recall to rise](img/20170212003936.png){#fig:varysigma}

![Precision, Recall and F1-Score depending on the margin of error center, showing a very clear normal distribution. The center of this distribution depends on the filter sigma and cutoff, meaning we can optimize it depending on the evaluation method.](img/20170212134705.png){#fig:varycenter}

![Precision, Recall and F1-Score depending on the minimum monologuing segment length. -1 means we evaluate on all data, including silence.](img/20170212132651.png){#fig:varymonologuing}

# Conclusion and Future Work

We have presented...(paper umschreiben)


We only scraped the surface of adding linguistic features via word2vec because it assumes the availability of an instant speech recognizer. Further work is needed to evaluate other methods for adding word vectors to the input features and to analyse problems with our approach.
We only tested feed forward neural networks and LSTMs, other architectures like convolutional neural networks may also give interesting results.
Our approach of choosing the training areas may not be optimal because the delay between the last utterance of the speaker and the backchannel can vary significantly. A possibility would be to align the training area by the last speaker utterance instead of the backchannel start. Our initial tests of predicting multiple separate categories of BCs did not produce any notable results, further work is needed to analyse whether more context or features are needed to facilitate this. 

Because backchannels are a largely subjective phenomenon, a user study would be helpful to subjectively evaluate the performance of our predictor and to compare it with human performance in our chosen evaluation method. Another method would be to find consensus for backchannel triggers by combining the predictions of multiple human subjects for a single speaker channel as described by Huang et al. (2010) as "parasocial consensus sampling" \cite{huang2010learning}.

An interesting extension for use in Virtual Assistants would be to also predict speaker changes in addition to backchannels. Our current model already often predicts a backchannel at a time where the speaker stops talking and expects a longer response. Our current model of predicting [No BC, BC] could be extended to predict [No BC, BC, Speaker Change]. This would allow a virtual assistant to give natural sounding backchannel responses and detect when it is supposed to start answering the query.


# Bibliography
