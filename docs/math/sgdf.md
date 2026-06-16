# SGDF

Implements SGDF, SGD whose gradient estimate is denoised by a Wiener filter.

SGDF treats the noisy stochastic gradient $g_t$ as a measurement of an underlying signal and fuses it with the momentum estimate $\hat{m}_t$ through a Wiener (Kalman-style) gain. The gain $K_t$ weighs the two sources by their variances: when the estimated gradient variance $\hat{s}_t$ is small relative to the instantaneous deviation $(g_t - \hat{m}_t)^2$, the filter trusts the smoothed history; when it is large, it trusts the current gradient. The gain is the minimizer of the mean-squared error of the fused estimate. The filtered gradient $\hat{g}_t$ then drives a plain SGD step, so the method adds no per-coordinate preconditioning, only variance-aware smoothing.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
s_t &= \beta_2 s_{t-1} + (1 - \beta_2)(g_t - m_t)^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{t}}, \qquad \hat{s}_t = \frac{s_t}{1 - \beta_2^{t}} \\
K_t &= \frac{\hat{s}_t}{\hat{s}_t + (g_t - \hat{m}_t)^2} \\
\hat{g}_t &= \hat{m}_t + K_t\,(g_t - \hat{m}_t) \\
\theta_t &= \theta_{t-1} - \eta\,\hat{g}_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the first-moment (momentum) estimate, $s_t$ the running estimate of the gradient variance, $\hat{m}_t$/$\hat{s}_t$ their bias-corrected forms, $\beta_1,\beta_2$ the decay rates, $K_t$ the Wiener gain that fuses the smoothed and instantaneous gradients, and $\hat{g}_t$ the resulting filtered gradient.

Reference: Zhipeng Yao, Yu Zhang, Dazhou Li, "Signal Processing Meets SGD: From Momentum to Filter", 2023. https://arxiv.org/abs/2311.02818

---
[Back to the Canon](../index.md)
