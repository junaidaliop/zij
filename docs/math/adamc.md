# AdamC

Implements AdamC, AdamW with a learning-rate-corrected weight decay for normalized layers.

Defazio observes that under a decaying learning-rate schedule, AdamW's weight decay term shrinks faster than the gradient step, so the weight norm of normalized layers drifts and gradients spike near the end of training. AdamC fixes this by scaling the decay of normalized layers by $\gamma_t^2/\gamma_{\max}$ instead of $\gamma_t$, keeping the gradient-to-weight ratio invariant to the schedule. Non-normalized layers keep the ordinary AdamW decay.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \gamma_t \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} - c_t\, \lambda\, \theta_{t-1}, \qquad
c_t = \begin{cases} \dfrac{\gamma_t^2}{\gamma_{\max}} & \text{normalized layer} \\ \gamma_t & \text{otherwise} \end{cases}
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma_t$ the scheduled learning rate, $\gamma_{\max}$ its peak value over the schedule, $g_t$ the gradient, $m_t,v_t$ the first and second moment estimates with decays $\beta_1,\beta_2$, $\lambda$ the weight decay, and $\epsilon$ the stability constant.

Reference: Aaron Defazio, "Why Gradients Rapidly Increase Near the End of Training", 2025. https://arxiv.org/abs/2506.02285

---
[Back to the Canon](../index.md)
