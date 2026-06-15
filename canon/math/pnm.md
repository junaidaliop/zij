# PNM

Implements PNM (Positive-Negative Momentum), a momentum scheme that controls stochastic gradient noise to improve generalization.

PNM maintains two interleaved momentum buffers updated on alternating steps, then combines them with positive and negative weights. The negative momentum term amplifies the stochastic gradient noise in a controlled way while keeping the expected update direction intact, decoupling the effective noise magnitude from the batch size and base learning rate. The step is rescaled by $\sqrt{(1+\beta_0)^2 + \beta_0^2}$ so that the noise variance stays calibrated.

$$
\begin{aligned}
m_t &= \beta_1^2 \, m_{t-2} + (1 - \beta_1^2)\, g_t \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{(1+\beta_0)^2 + \beta_0^2}} \left[ (1+\beta_0)\, m_t - \beta_0\, m_{t-1} \right]
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the momentum buffer (the recursion uses $m_{t-2}$, so even and odd steps form two independent sequences), $\beta_1$ the momentum decay, and $\beta_0$ the positive-negative momentum coefficient that sets the noise magnitude.

Reference: Zeke Xie, Li Yuan, Zhanxing Zhu, Masashi Sugiyama, "Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization", ICML 2021. https://arxiv.org/abs/2103.17182

---
[Back to the Canon](../README.md)
