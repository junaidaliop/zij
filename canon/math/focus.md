# FOCUS

Implements FOCUS, a sign-momentum optimizer with attraction to a moving
target.

FOCUS extends Signum with an attraction force toward an exponential moving
average of the parameters. The momentum supplies the descent direction
through its sign, while the attraction term pulls the parameters toward the
smoothed target with strength $\gamma$.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                           \\
     \bar{\theta}_t &= \beta_2 \bar{\theta}_{t-1} + (1 - \beta_2) \theta_t \\
     \hat{\theta}_t &= \frac{\bar{\theta}_t}{1 - \beta_2^t}               \\
     \theta_t &= \theta_t - \eta \, \omega \, \hat{\theta}_t              \\
     \theta_{t+1} &= \theta_t - \eta \left(
         \mathrm{sign}(m_t)
         + \gamma \mathrm{sign}(\theta_t - \hat{\theta}_t)
     \right)
\end{aligned}
$$

where $m_t$ is the gradient moment, $\bar{\theta}_t$ the moving
average of the parameters with bias-corrected form $\hat{\theta}_t$,
$\eta$ the learning rate, $\gamma$ the attraction strength, and
$\omega$ the decoupled weight decay applied toward the moving target.

Reference: Yizhou Liu, Ziming Liu, Jeff Gore, "FOCUS: First Order
Concentrated Updating Scheme", arXiv 2025.
https://arxiv.org/abs/2501.12243

---
[Back to the Canon](../README.md)
