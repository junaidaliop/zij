# DPIS

Implements DPIS, differentially private SGD with importance sampling for variance reduction.

DPIS follows the DP-SGD framework but replaces uniform mini-batch sampling with importance sampling, drawing records with probability proportional to their clipped gradient norm. Each sampled gradient is reweighted by the inverse selection probability so the estimate stays unbiased, which lowers the variance of the privatized gradient under the same noise budget. Per-example gradients are clipped to an $\ell_2$ bound, Gaussian noise is added for the differential-privacy guarantee, and parameters take a plain SGD step on the resulting noisy gradient.

$$
\begin{aligned}
\bar{g}_t(x_i) &= \frac{g_t(x_i)}{\max\!\left(1, \frac{\lVert g_t(x_i) \rVert_2}{C}\right)}, \qquad p(x_i) = \frac{\lVert \bar{g}_t(x_i) \rVert}{K}, \quad K = \sum_j \lVert \bar{g}_t(x_j) \rVert \\
\tilde{g}_t &= \frac{1}{b}\left( \sum_{i \in \mathcal{B}} \frac{1}{N\, p(x_i)}\, \bar{g}_t(x_i) + Z \right), \qquad Z \sim \mathcal{N}(0, \sigma_G^2 C^2 I) \\
\theta_{t+1} &= \theta_t - \eta\, \tilde{g}_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t(x_i)$ the per-example gradient, $\bar{g}_t$ its clipped version, $C$ the clipping bound, $\mathcal{B}$ the importance-sampled batch of expected size $b$, $N$ the dataset size, $p(x_i)$ the sampling probability, $K$ the sum of clipped gradient norms, and $Z$ Gaussian noise with multiplier $\sigma_G$.

Reference: Jianxin Wei, Ergute Bao, Xiaokui Xiao, Yin Yang, "DPIS: An Enhanced Mechanism for Differentially Private SGD with Importance Sampling", CCS 2022. https://arxiv.org/abs/2210.09634

---
[Back to the Canon](../index.md)
