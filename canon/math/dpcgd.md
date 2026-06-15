# DP-λCGD

Implements DP-$\lambda$CGD, differentially private SGD with single-step anti-correlated noise.

Each per-example gradient is clipped to $\ell_2$-norm $\zeta$ and summed over the batch. Calibrated Gaussian noise is then added, but instead of being independent across iterations the noise is correlated with only the immediately preceding step: a $\lambda$-fraction of the previous step's noise is cancelled before fresh noise is injected. This lightweight correlation reduces the effective noise variance accumulated by the optimization trajectory while preserving the same $(\varepsilon,\delta)$ guarantee, and the previous noise is regenerated from a stored PRNG state rather than kept in memory.

$$
\begin{aligned}
\tilde g_{t,j} &= \min\!\left(1, \frac{\zeta}{\lVert g_{t,j}\rVert}\right) g_{t,j} \\
x_t &= \sum_{j} \tilde g_{t,j} \\
\hat x_t &= x_t - \zeta\sigma\lambda\, Z_{t-1} + \zeta\sigma\, Z_t,\qquad Z_t \sim \mathcal{N}(0, I) \\
\theta_t &= \theta_{t-1} - \frac{\eta}{B}\,\hat x_t
\end{aligned}
$$

where $g_{t,j}$ is the per-example gradient, $\zeta$ the clipping norm, $\sigma$ the noise multiplier calibrated for $(\varepsilon,\delta)$-differential privacy, $\lambda \in [0,1)$ the correlation coefficient, $Z_t$ standard Gaussian noise, $B$ the batch size, and $\eta$ the learning rate.

Reference: Nikita P. Kalinin, Ryan McKenna, Rasmus Pagh, Christoph Lampert, "DP-$\lambda$CGD: Efficient Noise Correlation for Differentially Private Model Training", ICML 2026. https://arxiv.org/abs/2601.22334

---
[Back to the Canon](../README.md)
