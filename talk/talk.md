---
title: Backchannel Prediction for Conversational Speech Using Recurrent Neural Networks
header-includes:
- |-
   \makeatletter
   \def\svhline{%
   \noalign{\ifnum0=`}\fi\hrule \@height2\arrayrulewidth \futurelet
   \reserved@a\@xhline}
   \makeatother
   \setbeamertemplate{sidebar right}{}
    \setbeamertemplate{footline}{%
    \hfill\usebeamertemplate***{navigation symbols}
    \hspace{1cm}\insertframenumber{}/\inserttotalframenumber}
- \usepackage{siunitx}
---

# Introduction

## Definition

What are backchannels?

- Nodding / head movement
- Eye gaze shift
- Short phrases like "uh-huh", "yeah", "right"
- Vary from culture to culture (e.g. Japanese)

## Motivation / Goal

- BCs help build _rapport_ (comfortableness between conversation partners)

→ Improve conversation with artificial assistants

<!--## BC categories

- non-committal ("uh huh", "yeah")
- positive / confirming ("oh how neat", "great")
- negative / surprised ("you're kidding", "oh my god")
- questioning ("oh are you", "is that right")
- et cetera.

## Why backchannel prediction?

- Artificial assistants are becoming ubiquitous (Siri, Google Now, Alexa, Cortana, ...)
- Conversation with these is still distinctively unhuman
- BCs can help make conversations with an AI agent feel more natural
-->
How?

- Simplify backchannels to only short phrases
- Predict when to emit backchannels
- (Predict what kind of backchannel to emit)


# Related Work

## Related Work

Common approach: manually tuned rules.

Ward (2000):

> produce backchannel feedback upon detection of:
>
> - a region of pitch less than the 26th-percentile pitch level and
> - continuing for at least 110 milliseconds,
> - coming after at least 700 milliseconds of speech,
> - providing you have not output back-channel feedback within the preceding 800 milliseconds,

Almost always based on pitch and power

## Related Work

Common approach: manually tuned rules.

- error-prone
- a lot of manual work
- hard to generalize

semi-automatic approaches, e.g. Morency (2010)

- hand-picked features such as binary pause regions and different speech slopes
- train Hidden Markov Models to predict BCs from that

# NN-based approach

## Dataset

Switchboard dataset:

- 2400 English telephone conversations 
- 260 hours total
- Randomly selected topics
- Transcriptions and word alignments that include BC utterances
 
## BC Utterance Selection

- Get a list of all backchannel phrases
- Separate those into categories
- BC phrase list from the _Switchboard Dialog Act Corpus_ (SwDA)

\begin{figure}
\centering
\scriptsize
\begin{tabular}{cp{3cm}cp{4cm}r}
    \hline\noalign{\smallskip}

    ~ & name & act\_tag & example & full\_count \\

    \noalign{\smallskip}\svhline\noalign{\smallskip}

    1 & Statement-non-opinion & sd & Me, I'm in the legal department. &  75145\tabularnewline
    2 & \bf{Acknowledge (Backchannel)} & b & Uh-huh. &  38298\tabularnewline
    3 & Statement-opinion & sv & I think it's great &  26428\tabularnewline
    4 & Agree/Accept & aa & That's exactly it. &  11133\tabularnewline
    5 & Abandoned or Turn-Exit & \% & So, - &  15550\tabularnewline
    6 & Appreciation & ba & I can imagine. &  4765\tabularnewline
    7 & Yes-No-Question & qy & Do you have to have any special training? &  4727\tabularnewline

    \noalign{\smallskip}\hline\noalign{\smallskip}
\end{tabular}
\caption{SwDA categories}
\label{tbl:swda}
\end{figure}

## BC Utterance Selection (Practice)

SwDA incomplete

→ Identify utterances only from their text



Something like "uh" can be a disfluency or a BC.

→ only include phrases with silence or BC before them.


## Training Area Selection

![Sample Audio Segment](img/tminus0.svg)

## Training Area Selection

![Sample Audio Segment](img/tminus.svg)

## Training Area Selection

![Positive Training Area (width=1.5s)](img/tminus-b.svg)

## Training Area Selection

Need area to predict non-BC.

→ Area of audio where no BC follows

. . .

Want balanced data set.

→ Choose area 0.5 seconds before BC area

## Training Area Selection

![Pos/Neg Training areas](img/tminus-c.svg)


→ Balanced data

## Feature Selection (Theory)

- Acoustic features like power, pitch

- Linguistic features (from the transcriptions)

## Feature Selection -- Acoustic

![Audio, Power, Pitch](img/20170311192331.png)

## Feature Selection -- Linguistic

![Word2Vec](img/20170311192910.png)

<!-- "what i" has same encoding -->

# Neural network design

## Input layer

![](img/nn-a.svg)

## Input layer

![](img/nn-b.svg)

## Input layer

![](img/nn-c.svg)

## Input layer

![](img/nn-d.svg)

## Hidden layers (Feed forward)

![](img/nn-hidden2.svg)

<!--## Hidden layers (Feed forward)

![](img/nn-hidden3.svg) -->

## Hidden layers (Feed forward)

![](img/nn-hidden4.svg)

## Hidden layers (Feed forward)


![](img/nn-hidden-a.svg)

## Problem with feed forward networks

Feed Forward can not take its previous state into account.

BCs are more probable after a longer period without BCs.

. . .

LSTM is able to take into account it's own past internal state.

![Recurrent Neural Net architecture \tiny (Christopher Olah)](img/RNN-unrolled.png)


## Postprocessing

NN output is

- a value between 0 and 1
- quickly changing
- noisy

## Postprocessing -- Low-pass filter

![Low-pass convolution](img/postproc-convolve.svg)

## Postprocessing -- Low-pass filter

Gauss filter looks into future

→ Cut off filter and shift it


![](img/gauss-cutoff.svg)

## Thresholding / Triggering

- Use areas of output > threshold (0 < t < 1)
- Trigger at local maximum

![Example of the postprocessing process.](../img/postprocessing.pdf){#fig:postproc}

# Evaluation

## Objective Evaluation


- Precision (portion of predictions that were correct)
- Recall (portion of correct BCs that were predicted)
- F1-Score (harmonic mean of Precision and Recall)

![Evaluating the performance of a network while varying the threshold. Note the inverse relationship between _Precision_ and _Recall_.](../img/20170220200731.png){#fig:varythreshold}


## Lots of parameters to tune

- Context width
- Context stride
- Which features
- NN depth
- NN layer sizes
- LSTM vs Feed forward
- Trigger threshold
- Gaussian filter sigma
- Gaussian filter cutoff
- Prediction delay

## Lots of parameters to tune

manually through trial and error:

- Context width
- Context stride
- Which features
- NN depth
- NN layer sizes
- LSTM vs Feed forward

automatically with Bayesian optimization:

- Trigger threshold
- Gaussian filter sigma
- Gaussian filter cutoff
- Prediction delay

# Results

## Context width

\begin{table}
    \centering
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
    \caption{Results with various context lengths. Performance peaks at 1500\,ms.}\label{tbl:varycontext}
\end{table}

## Context stride

\begin{table}
    \centering
    \begin{tabular}{cccc}
    \hline\noalign{\smallskip}
    Stride & Precision & Recall & F1-Score \\
    \noalign{\smallskip}\svhline\noalign{\smallskip}
    1 & 0.290 & 0.490 & 0.364 \\
    2 & 0.305 & 0.488 & \bf{0.375} \\
    4 & 0.285 & 0.498 & 0.363 \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \caption{Results with various context frame strides.}\label{tbl:varystrides}
\end{table}

## LSTM vs FF


\begin{table}
    \centering
    \begin{tabular}{ccccc}
    \hline\noalign{\smallskip}
    Layers & Parameter count & Precision & Recall & F1-Score \\
    \noalign{\smallskip}\svhline\noalign{\smallskip}
    FF ($56 : 28$) & 40k & 0.230 & 0.549 & 0.325 \\
    FF ($70 : 35$) & 50k & 0.251 & 0.468 & 0.327 \\
    FF ($100 : 50$) & 72k & 0.242 & 0.490 & 0.324 \\
    LSTM ($70 : 35$) & 38k & 0.305 & 0.488 & \bf{0.375} \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \caption{LSTM outperforms feed forward architectures.}\label{tbl:varylstm}
\end{table}

## Layer sizes

\begin{table}
    \centering
    \begin{tabular}{cccc}
    \hline\noalign{\smallskip}
    Layer sizes & Precision & Recall & F1-Score \\
    \noalign{\smallskip}\svhline\noalign{\smallskip}
    $100$ & 0.280 & 0.542 & 0.369 \\
    $50 : 20$ & 0.291 & 0.506 & 0.370 \\
    $70 : 35$ & 0.305 & 0.488 & \bf{0.375} \\
    $100 : 50$ & 0.303 & 0.473 & 0.369 \\
    $70 : 50 : 35$ & 0.278 & 0.541 & 0.367 \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \caption{Comparison of different network configurations. Two LSTM layers give the best results.}\label{tbl:varylayers}
\end{table}

## Features

\begin{table}
    \centering
    \begin{tabular}{lccc}
    \hline\noalign{\smallskip}
    Features & Precision & Recall & F1-Score \\
    \noalign{\smallskip}\svhline\noalign{\smallskip}
    power & 0.244 & 0.516 & 0.331 \\
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
    \caption{Results with various input features, separated into only acoustic features and acoustic plus linguistic features.}\label{tbl:varyfeatures}
\end{table}



## Other research

\begin{table}
    \small
    \centering
    %\caption{Final best results on the evaluation set (chosen by validation set)}\label{tbl:final}
    \begin{tabular}{p{6cm}ccc}
    \hline\noalign{\smallskip}
        Predictor & Precision & Recall & F1-Score \\
        \noalign{\smallskip}\svhline\noalign{\smallskip}
        Baseline (random) & 0.042 & 0.042 & 0.042 \\
        Müller et al. (offline) & -- & -- & 0.109 \\
        Our results (offline, context of \SIrange{-750}{750}{ms}) & 0.114 & 0.300 & \bf{0.165} \\
        Our results (online, context of \SIrange{-1500}{0}{ms}) & 0.100 & 0.318 & 0.153 \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \caption{Comparison with previous research.\label{tbl:mueller}}
\end{table}

---

## Varying margin of error

\begin{table}
    \scriptsize
    \centering
    \begin{tabular}{p{2.4cm}p{3.7cm}ccc}
    \hline\noalign{\smallskip}
        Margin of Error & Constraint & Precision & Recall & F1-Score \\
        \noalign{\smallskip}\svhline\noalign{\smallskip}
        \SIrange{-200}{200}{ms} && 0.172 & 0.377 & 0.237 \\
        \SIrange{-100}{500}{ms} &&	0.239 & 0.406 & 0.301 \\
        \SIrange{-500}{500}{ms} && 0.247 & 0.536 & 0.339 \\
    \hline\noalign{\smallskip}
        \SIrange{0}{1000}{ms} %& Baseline (random, correct BC count) & 0.111 & 0.052 & 0.071 \\
         & Baseline (random) & 0.079 & 0.323 & 0.127 \\
         %& Balanced Precision and Recall & 0.342 & 0.339 & 0.341 \\
         & Only acoustic features & 0.294 & 0.488 & 0.367 \\
         & Acoustic and linguistic features & 0.312 & 0.511 & 0.388 \\
    \noalign{\smallskip}\hline\noalign{\smallskip}
    \end{tabular}
    \caption{Results with various margins of error used in other research. Performance improves with a wider margin width and a later margin center.\label{tbl:ourbest}}
\end{table}

## Survey

Randomly show participants 6 samples of the following categories

1. Random predictor
2. NN predictor
3. Ground truth

## Survey

![Screenshot of the survey interface.](../img/survey-bw.png){#fig:bcsurvey}

## Survey Results

\begin{table}
\begin{tabular}{ccccc}
\hline\noalign{\smallskip}
Predictor & Sample & Timing & Naturalness & Sample Size \\
\noalign{\smallskip}\svhline\noalign{\smallskip}
random & average & 2.33 points & 2.63 points & 40 \\
\noalign{\smallskip}\hline\noalign{\smallskip}
nn & average & 3.48 points & 3.08 points & 40 \\
\noalign{\smallskip}\hline\noalign{\smallskip}
truth & average & 4.20 points & 4.08 points & 40 \\
\noalign{\smallskip}\hline\noalign{\smallskip}
\end{tabular}
    \caption{Average user ratings of different BC predictors}
\end{table}

<!--# Implementation


# Conclusion and Future Work-->












