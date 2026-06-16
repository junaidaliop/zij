# AClipped-dpSGD

Implements AClipped-dpSGD, differentially private SGD for heavy-tailed data using one-time clipping with output averaging.

For stochastic convex optimization on heavy-tailed data, each noisy stochastic gradient is clipped a single time per step to bound its sensitivity, calibrated Gaussian noise is added for $(\epsilon,\delta)$-differential privacy, and a plain SGD step is taken. The estimator returned is the uniform average of all iterates, which together with the one-time clipping (rather than repeated multi-pass clipping) yields an efficient, privacy-calibrated method with sharp convergence rates.

$$
\begin{aligned}
\hat g_t &= \min\!\left(1,\ \frac{\lambda}{\lVert g_t\rVert_2}\right) g_t \\
\tilde g_t &= \hat g_t + z_t,\qquad z_t \sim \mathcal{N}\!\left(0,\ \hat\sigma^2 I_d\right) \\
\hat\sigma &= c\,\frac{\lambda\, m\, \sqrt{T\,\ln(1/\delta)}}{n\,\epsilon} \\
\theta_{t+1} &= \theta_t - \gamma\, \tilde g_t \\
\bar\theta &= \frac{1}{T}\sum_{t=0}^{T-1}\theta_t
\end{aligned}
$$

where $g_t$ is the stochastic gradient, $\lambda$ the clipping threshold, $\gamma$ the step size, $T$ the number of iterations, $n$ the dataset size, $m$ the mini-batch size, $(\epsilon,\delta)$ the privacy budget, $c$ a constant from the privacy analysis, $I_d$ the $d$-dimensional identity, and $\bar\theta$ the averaged output. In the constrained case the update step is followed by a projection onto the feasible set $\mathcal{X}$.

Reference: Chenhan Jin, Kaiwen Zhou, Bo Han, Ming-Chang Yang, James Cheng, "Efficient Private SCO for Heavy-Tailed Data via Averaged Clipping", Machine Learning 2024. https://arxiv.org/abs/2206.13011

---
[Back to the Canon](../index.md)
