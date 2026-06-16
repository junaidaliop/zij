# Ano

Implements Ano, a momentum optimizer that decouples step direction from step magnitude.

The direction of each step is taken from the sign of the momentum $m_t$, which aggregates gradient history for stability, while the magnitude is taken from the instantaneous gradient $|g_t|$, so the optimizer reacts quickly in noisy landscapes instead of being damped by stale momentum. The denominator uses a Yogi-style second moment $v_t$ that decreases additively, and a decoupled weight decay term is applied directly to the parameters. The variant Anolog replaces the fixed $\beta_1$ with the schedule $\beta_{1,t} = 1 - 1/\log(t+2)$.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= v_{t-1} - (1 - \beta_2)\,\mathrm{sign}(v_{t-1} - g_t^2)\,g_t^2 \\
\eta_t &= \frac{\eta}{(t+2)^{3/4}} \\
\theta_t &= \theta_{t-1} - \eta_t \frac{|g_t|}{\sqrt{v_t} + \epsilon}\,\mathrm{sign}(m_t) - \eta_t \lambda\, \theta_{t-1}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the base learning rate with decaying schedule $\eta_t$, $g_t$ is the gradient, $m_t$ and $v_t$ are the first and second moments, $\beta_1, \beta_2$ are the decay rates, $\lambda$ is the weight decay, and $\epsilon$ is a stability constant.

Reference: Adrien Kegreisz, "Ano: Faster is Better in Noisy Landscapes", arXiv 2025. https://arxiv.org/abs/2508.18258

---
[Back to the Canon](../index.md)
