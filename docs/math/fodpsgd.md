# FO-DP-SGD

Implements FO-DP-SGD, a fractional-order differentially private SGD that augments the DP-SGD query with a tempered power-law memory of past private releases.

Standard DP-SGD treats each step's clipped, noised gradient sum independently. FO-DP-SGD instead blends the current clipped sum with a confidence-aware tempered fractional memory state: a Grünwald-Letnikov-type power-law weighting $(j+1)^{\alpha-1}$ over prior private releases, exponentially tempered by a baseline rate $\lambda$ plus an inconsistency-aware term that down-weights stale or off-trend history. Because the memory is built only from already-released private sums, recycling it incurs no additional privacy cost. Noise is added at the sum level after mixing, and the released sum is rescaled by the expected batch size to form the descent direction.

$$
\begin{aligned}
\bar{g}_t(x_i) &= g_t(x_i) \,/\, \max\!\left(1,\ \|g_t(x_i)\|_2 / C\right) \\
s_t &= \sum_{i \in S_t} \bar{g}_t(x_i) \\
a_{t,j} &= (j+1)^{\alpha-1}\,\exp\!\left(-(\lambda + \chi_t\,\tau\,\nu_{t,j})\,j\right), \qquad u_{t-1} = \sum_{j=1}^{K_t-1} \frac{a_{t,j}}{\sum_{l} a_{t,l}}\,\tilde{s}_{t-j} \\
\tilde{s}_t &= \beta\,s_t + (1-\beta)\,u_{t-1} + Z_t, \qquad Z_t \sim \mathcal{N}(0,\ \sigma^2 C^2 I) \\
\theta_{t+1} &= \theta_t - \eta\,\tilde{s}_t / L
\end{aligned}
$$

where $g_t(x_i)$ is the per-example gradient, $C$ the clipping threshold, $S_t$ the sampled minibatch, $\alpha \in (0,1]$ the fractional order, $\lambda \ge 0$ the baseline tempering rate, $\tau \ge 0$ the inconsistency-aware tempering coefficient, $\beta \in (0,1]$ the mixing coefficient between current sum and memory, $K_t$ the memory window depth, $\tilde{s}_{t-j}$ past private releases, $\nu_{t,j}$ the normalized inconsistency of a lagged release against an EMA trend, $\chi_t$ a confidence factor, $\sigma$ the noise multiplier, $\eta$ the learning rate, and $L = Nq$ the expected batch size.

Reference: Mohammad Partohaghighi, Roummel Marcia, "Deep Learning under Fractional-Order Differential Privacy", arXiv 2026. https://arxiv.org/abs/2605.09890

---
[Back to the Canon](../index.md)
