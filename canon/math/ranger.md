# Ranger

Implements Ranger, RAdam with a Lookahead wrapper and gradient
centralization.

Each step takes a rectified Adam (RAdam) update on the fast weights and,
every $k$ steps, interpolates a set of slow weights toward them
(Lookahead). Gradients are optionally centralized by subtracting their mean
before the moment updates.


$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\rho_\infty &= \frac{2}{1 - \beta_2} - 1 \\
\rho_t &= \rho_\infty - \frac{2 t\, \beta_2^t}{1 - \beta_2^t}
\end{aligned}
$$

When the length of the approximated simple moving average satisfies
$\rho_t \geq$ `n_sma_threshold` (default 5), the variance is
tractable and the step is rectified:


$$
\begin{aligned}
r_t &= \sqrt{\frac{(1 - \beta_2^t)(\rho_t - 4)(\rho_t - 2)\rho_\infty}
                  {(\rho_\infty - 4)(\rho_\infty - 2)\rho_t}} \\
\theta_t &= \theta_{t-1} - \frac{\eta\, r_t}{1 - \beta_1^t}
             \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

Otherwise, with `degenerated_to_sgd=True`, the update falls back to the
unscaled first moment,
$\theta_t = \theta_{t-1} - \frac{\eta}{1 - \beta_1^t} m_t$; with the
default `degenerated_to_sgd=False` the rectified branch is simply skipped
until the moving average becomes tractable. Every
$k$ steps the slow weights $\phi$ track the fast weights,
$\phi_t = \phi_{t-1} + \alpha (\theta_t - \phi_{t-1})$, and the fast
weights are reset to $\phi_t$.

Here $\theta$ are the parameters, $\eta$ is the learning rate,
$g_t$ is the gradient, $m_t$ and $v_t$ are the first and
second moments, $\beta_1, \beta_2$ are their decay rates, $k$ is
the Lookahead synchronization period, and $\alpha$ is the Lookahead
interpolation factor.

Reference: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong
Liu, Jianfeng Gao, Jiawei Han, "On the Variance of the Adaptive Learning
Rate and Beyond", ICLR 2020. https://arxiv.org/abs/1908.03265
Reference: Michael R. Zhang, James Lucas, Geoffrey Hinton, Jimmy Ba,
"Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019.
https://arxiv.org/abs/1907.08610
Reference: Hongwei Yong, Jianqiang Huang, Xiansheng Hua, Lei Zhang,
"Gradient Centralization: A New Optimization Technique for Deep Neural
Networks", ECCV 2020. https://arxiv.org/abs/2004.01461

---
[Back to the Canon](../README.md)
