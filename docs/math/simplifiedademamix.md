# Simplified-AdEMAMix

Implements Simplified-AdEMAMix, an AdEMAMix variant that keeps a single momentum buffer and adds the raw gradient.

AdEMAMix accelerates training by mixing a fast and a slow exponential moving average of the gradient, but this needs a second momentum buffer and a schedule for its decay rate. Simplified-AdEMAMix drops the slow EMA entirely. It keeps one accumulator $m_t$ updated without the usual $(1-\beta_1)$ normalization, so $m_t$ behaves like a slow, large-mass momentum, and recovers the responsiveness of a fast EMA by adding the current gradient $g_t$ directly into the step with a fixed weight $\alpha$. The second moment $v_t$ and its bias correction are inherited from AdamW.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^{\,t}}, \\
\theta_t &= \theta_{t-1} - \eta_t\left(\frac{m_t + \alpha\, g_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\, \theta_{t-1}\right).
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t$ the gradient, $m_t$ the unnormalized momentum accumulator, $v_t$ the second moment with bias-corrected $\hat{v}_t$, $\beta_1,\beta_2$ the decay rates, $\alpha$ the fixed weight on the current gradient, $\lambda$ the decoupled weight decay, and $\epsilon$ a stability constant. An optional warmup schedule may ramp $\beta_1$ from $\beta_{\mathrm{start}}$ over $T_{\beta_1}$ steps.

Reference: Depen Morwani, Nikhil Vyas, Hanlin Zhang, Sham Kakade, "Connections between Schedule-Free Optimizers, AdEMAMix, and Accelerated SGD Variants", 2025. https://arxiv.org/abs/2502.02431

---
[Back to the Canon](../index.md)
