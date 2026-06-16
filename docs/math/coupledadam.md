# Coupled Adam

Implements Coupled Adam, an Adam variant that couples the second moment across output embedding vectors.

Standard Adam treats every embedding vector independently, which the authors show drives anisotropy and gives rare tokens disproportionately large updates. Coupled Adam targets only the output embedding matrix: it computes per-vector first and second moments as usual, then replaces each vector's bias-corrected second moment with the average second moment taken over the whole vocabulary. Sharing $\hat{v}$ removes the per-token scaling difference in the denominator, yielding better-conditioned embeddings while leaving the rest of the network on ordinary Adam.

$$
\begin{aligned}
g_{i,t} &= \nabla_{\theta_i} f_t(\theta_{i,t-1}) \\
m_{i,t} &= \beta_1 m_{i,t-1} + (1-\beta_1) g_{i,t} \\
v_{i,t} &= \beta_2 v_{i,t-1} + (1-\beta_2) g_{i,t}^2 \\
\hat{m}_{i,t} &= \frac{m_{i,t}}{1-\beta_1^{t}}, \qquad \hat{v}_{i,t} = \frac{v_{i,t}}{1-\beta_2^{t}} \\
\bar{v}_t &= \frac{1}{V} \sum_{i=1}^{V} \hat{v}_{i,t} \\
\theta_{i,t} &= \theta_{i,t-1} - \eta \, \frac{\hat{m}_{i,t}}{\sqrt{\bar{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta_i$ is the embedding vector for token $i$, $V$ is the vocabulary size, $g_{i,t}$ its gradient, $m_{i,t}/v_{i,t}$ the first and second moments with decays $\beta_1,\beta_2$, $\hat{m}_{i,t}/\hat{v}_{i,t}$ their bias-corrected forms, $\bar{v}_t$ the vocabulary-averaged second moment shared by all tokens, $\eta$ the learning rate, and $\epsilon$ a stability constant.

Reference: Felix Stollenwerk, Tobias Stollenwerk, "Better Embeddings with Coupled Adam", arXiv 2025. https://arxiv.org/abs/2502.08441

---
[Back to the Canon](../index.md)
