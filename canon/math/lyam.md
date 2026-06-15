# LyAm

Implements LyAm, an Adam variant with a Lyapunov-inspired adaptive learning rate for noisy, non-convex optimization.

LyAm keeps Adam's bias-corrected first and second moments but replaces the usual $\sqrt{\hat{v}_t}+\epsilon$ denominator with a Lyapunov-stability-motivated scaling $\eta_0/(1+\hat{v}_t)$. Scaling each coordinate by the inverse of its bias-corrected second moment damps steps along high-variance (noisy) directions, which the authors derive to yield a monotonically decreasing loss surrogate.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\eta_t &= \frac{\eta_0}{1+\hat{v}_t} \\
\theta_t &= \theta_{t-1} - \eta_t \odot \hat{m}_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t=\nabla L(\theta_{t-1})$ the gradient, $m_t,v_t$ the first and second moment estimates, $\hat{m}_t,\hat{v}_t$ their bias-corrected versions, $\beta_1,\beta_2$ the decay rates, $\eta_0$ the base learning rate, $\eta_t$ the per-coordinate adaptive learning rate, and $\odot$ elementwise multiplication (all moment operations are elementwise).

Reference: Elmira Mirzabeigi, Sepehr Rezaee, Kourosh Parand, "LyAm: Robust Non-Convex Optimization for Stable Learning in Noisy Environments", arXiv 2025. https://arxiv.org/abs/2507.11262

---
[Back to the Canon](../README.md)
