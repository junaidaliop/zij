# Gravity

Implements Gravity, a kinematic optimizer.

Gravity treats each parameter as a point mass rolling down an inclined
plane whose slope is the gradient, and integrates a constant-acceleration
kinematic step. The per-coordinate step is largest for moderate gradients
and saturates as the gradient grows, giving a bounded velocity increment.
The velocity buffer is seeded from a normal distribution and smoothed by a
running average whose decay anneals from $\frac{1}{2}$ toward
$\beta$ as training proceeds.


$$
\begin{aligned}
     V_0 &\sim \mathcal{N}\!\left(0, \sigma^2\right),\ \sigma = \frac{\alpha}{\eta}  \\
     \hat{\beta}_t &= \frac{\beta t + 1}{t + 2}                         \\
     m_t &= \frac{1}{\max_i |g_{t,i}|}                                  \\
     \zeta_t &= \frac{g_t}{1 + (g_t / m_t)^2}                           \\
     V_t &= \hat{\beta}_t V_{t-1} + (1 - \hat{\beta}_t) \zeta_t         \\
     \theta_t &= \theta_{t-1} - \eta V_t
\end{aligned}
$$

where $g_t$ is the gradient, $m_t$ the reciprocal of the largest
gradient magnitude, $\zeta_t$ the saturating gravity step,
$V_t$ the velocity buffer, $\eta$ the learning rate,
$\alpha$ the velocity initialization scale, and $\beta$ the
asymptotic running-average decay.

Reference: Dariush Bahrami, Sadegh Pouriyan Zadeh, "Gravity Optimizer: a
Kinematic Approach on Optimization in Deep Learning", arXiv 2021.
https://arxiv.org/abs/2101.09192

---
[Back to the Canon](../index.md)
