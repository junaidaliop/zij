# Hybrid SignSGD-SGD switching

Implements Hybrid SignSGD-SGD switching, a SWATS-style schedule that begins with momentum SignSGD and transitions to plain SGD using a projection-calibrated learning rate.

SignSGD-M compresses each momentum coordinate to its sign, which saves communication and memory but discards gradient magnitude and leaves a generalization gap. This method runs SignSGD-M for the early phase while continuously estimating the SGD step size that best matches the sign step via a projection of the gradient onto the sign direction. The estimate is tracked with an exponential moving average, and once a switch step is reached the optimizer hands over to SGD using that calibrated rate, recovering magnitude information for the later phase.

$$
\begin{aligned}
m_{t} &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \\
\lambda_t &= \frac{\gamma\, \lvert \langle \mathrm{sign}(m_t),\, g_t \rangle \rvert}{\lVert g_t \rVert_2^2 + \epsilon}, \\
\bar{\lambda}_t &= \beta_2 \bar{\lambda}_{t-1} + (1-\beta_2)\, \lambda_t, \\
\theta_t &=
\begin{cases}
\theta_{t-1} - \gamma\, \mathrm{sign}(m_t), & t < T_{\mathrm{switch}}, \\
\theta_{t-1} - \bar{\lambda}_t\, g_t, & t \ge T_{\mathrm{switch}}.
\end{cases}
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the stochastic gradient, $m_t$ the momentum with decay $\beta_1$, $\gamma$ the fixed sign step size, $\lambda_t$ the per-step projection of the gradient onto the sign direction, $\bar{\lambda}_t$ its EMA with decay $\beta_2$, $\epsilon$ a stability constant, and $T_{\mathrm{switch}}$ the epoch at which the optimizer transitions from SignSGD-M to SGD.

Reference: Haoran Chen, Wentao Wang, "Enhancing SignSGD: Small-Batch Convergence Analysis and a Hybrid Switching Strategy", arXiv 2026. https://arxiv.org/abs/2604.25550

---
[Back to the Canon](../index.md)
