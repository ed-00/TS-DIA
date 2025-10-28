## Problem formulation (ego‑centric diarization)

This document explains the math behind the ego‑centric formulation used in TS‑DIA. It follows the project overview but focuses strictly on notation, objectives, and intuition.

### Notation
- **Acoustic features**: $X = [\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T]$, where $X \in \mathbb{R}^{T\times F}$
  - $T$: number of frames; $F$: feature dimension per frame.
- **Target enrollment**: $\mathbf{e} \in \mathbb{R}^{D}$ is a fixed embedding for the target speaker.
- **Labels**: $Y = [y_1, y_2, \dots, y_T]$, one label per frame.

### Label space (mutually exclusive classes)
We replace multi‑label speaker activity with a single categorical label per frame drawn from:

$$
\mathcal{C} = \{\texttt{ts},\ \texttt{ts\_{ovl}},\ \texttt{others\_{sgl}},\ \texttt{others\_{ovl}},\ \texttt{ns}\}
$$

where:
- $\texttt{ts}$: target speaker only
- $\texttt{ts\_{ovl}}$: target speaker overlapping with others
- $\texttt{others\_{sgl}}$: exactly one non‑target speaker
- $\texttt{others\_{ovl}}$: overlap of non‑target speakers
- $\texttt{ns}$: non‑speech

At each time step $t$: $y_t \in \mathcal{C}$.

### Auto‑regressive modeling
We model the probability of the current label conditioned on past audio, past labels, and the target embedding:

$$
P\!\left(y_t\ \middle|\ X_{1:t},\ Y_{1:t-1},\ \mathbf{e}\right) \;=\; f_{\theta}\!\left(\mathbf{x}_1,\dots,\mathbf{x}_t,\ y_1,\dots,y_{t-1},\ \mathbf{e}\right),
$$

where $f_{\theta}$ is the neural network (e.g., Transformer encoder–decoder). This is a standard sequence model with categorical outputs (softmax over $\mathcal{C}$).

### Training objective
Given a dataset of $N$ utterances, each with ground‑truth label sequence $Y_i^*$ and enrollment $\mathbf{e}_i$, we minimize the negative log‑likelihood (equivalently, sum of cross‑entropies):

$$
\mathcal{L}(\theta) \;=\; - \sum_{i=1}^{N} \log P\!\left( Y_i^*\ \middle|\ X_i,\ \mathbf{e}_i;\ \theta \right).
$$

Using the chain rule over time for a single sample:

$$
\log P\!\left( Y^*\ \middle|\ X,\ \mathbf{e};\ \theta \right)
\;=\; \sum_{t=1}^{T} \log P\!\left( y_t^*\ \middle|\ X_{1:t},\ Y_{1:t-1}^*,\ \mathbf{e};\ \theta \right).
$$

Thus, the optimization target becomes:

$$
\theta^* 
\;=\; \arg\min_{\theta} \left( - \sum_{i=1}^{N} \sum_{t=1}^{T_i} \log P\!\left( y_{i,t}^*\ \middle|\ X_{i,1:t},\ Y_{i,1:t-1}^*,\ \mathbf{e}_i;\ \theta \right) \right).
$$

### Why this formulation helps
- **Permutation‑free**: one categorical variable per frame avoids matching speaker columns (no PIT needed).
- **Speaking‑style awareness**: with history \(Y_{1:t-1}\), the model can learn temporal patterns (turn‑taking cues), not just timbre.
- **Generalizable decoding**: iterate over speakers to construct global diarization (who‑spoke‑when) from ego‑centric predictions.

### Glossary
- $X_{1:t}$: frames 1 through $t$ of the acoustic features.
- $Y_{1:t-1}$: previously predicted (or teacher‑forced) labels up to $t\!-\!1$.
- $\mathbf{e}$: target speaker embedding used to condition predictions.
- $f_{\theta}$: model parameterized by $\theta$.

