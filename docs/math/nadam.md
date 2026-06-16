# NAdam

Implements Nadam, an Adam variant that folds Nesterov momentum into the
first-moment estimate.

Nadam keeps Adam's running averages of the gradient $m_t$ and the squared
gradient $v_t$, but replaces Adam's bias-corrected first moment with a
Nesterov-style lookahead: the update mixes the freshly decayed moment $m_t$
with the current gradient $g_t$, so the step anticipates where the momentum
is heading rather than relying only on past accumulation. The mixing uses a
per-step momentum coefficient $\mu_t$ that warms up through a schedule
governed by the momentum decay $\psi$, and the corresponding running product
$\prod_i \mu_i$ supplies the bias correction.

$$
\begin{aligned}
\mu_t &= \beta_1 \left(1 - \tfrac{1}{2}\, 0.96^{\, t\psi}\right), \qquad
\mu_{t+1} = \beta_1 \left(1 - \tfrac{1}{2}\, 0.96^{\, (t+1)\psi}\right) \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{\mu_{t+1}\, m_t}{1 - \prod_{i=1}^{t+1} \mu_i}
  + \frac{(1 - \mu_t)\, g_t}{1 - \prod_{i=1}^{t} \mu_i} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^{\, t}} \\
\theta_t &= \theta_{t-1} - \frac{\gamma\, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ is the learning rate, $g_t$ is
the gradient, $m_t$ and $v_t$ are the first and second moments,
$\beta_1, \beta_2$ are the decay rates, $\mu_t$ is the scheduled momentum
coefficient set by the momentum decay $\psi$, and $\epsilon$ is a small
constant for numerical stability.

Reference: Timothy Dozat, "Incorporating Nesterov Momentum into Adam", ICLR Workshop 2016.
https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ

---
[Back to the Canon](../index.md)
