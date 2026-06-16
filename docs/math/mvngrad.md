# MVN-Grad

Implements MVN-Grad, an adaptive method that applies momentum to variance-normalized gradients.

MVN-Grad normalizes each gradient coordinate by an exponential moving average of its variance (the squared deviation from the gradient momentum), and only then accumulates momentum on the normalized signal. This normalize-then-momentum ordering decouples a stale momentum buffer from the stochastic normalizer, in contrast to Adam-style momentum-then-normalize updates, yielding smaller one-step update variance and bounded sensitivity to single gradient spikes.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)(g_t - m_t)^2 + \epsilon_s \\
z_t &= \frac{g_t}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon} \\
u_t &= \beta_1 u_{t-1} + (1 - \beta_1) z_t \\
\theta_t &= \theta_{t-1} - \eta \, \frac{u_t}{1 - \beta_1^t}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the step size, $g_t$ the gradient, $m_t$ the gradient momentum, $v_t$ the variance estimate, $z_t$ the variance-normalized gradient, $u_t$ the momentum on the normalized gradient, $\beta_1, \beta_2$ the decay rates, $\epsilon$ the stability constant, and $\epsilon_s \ge 0$ a variance floor.

Reference: Francisco Patitucci, Aryan Mokhtari, "Adaptive Optimization via Momentum on Variance-Normalized Gradients", arXiv 2026. https://arxiv.org/abs/2602.10204

---
[Back to the Canon](../index.md)
