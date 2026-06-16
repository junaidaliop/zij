# Adam++

Implements Adam++, a parameter-free variant of Adam whose step size is set automatically from the optimization trajectory.

Adam++ removes the learning-rate hyperparameter by tracking the distance the iterate has traveled from its starting point. The per-step scale $\eta_t$ is the running maximum of the normalized displacement $\lVert \theta_t - \theta_0 \rVert_2 / \sqrt{d}$, so it grows as the optimizer moves away from the initialization and never needs manual tuning. The first moment uses a time-decayed momentum coefficient $\beta_{1,t} = \beta_1 \lambda^{t-1}$, and the diagonal preconditioner is built from accumulated squared gradients.

$$
\begin{aligned}
r_t &= \frac{\lVert \theta_t - \theta_0 \rVert_2}{\sqrt{d}}, \\
\eta_t &= \max(\eta_{t-1},\, r_t), \\
\beta_{1,t} &= \beta_1 \lambda^{t-1}, \\
m_t &= \beta_{1,t}\, m_{t-1} + (1 - \beta_{1,t})\, g_t, \\
v_t &= \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2, \quad s_t = \sqrt{(t+1)\, \max_{t' \le t} v_{t'}}, \\
H_t &= \delta + \mathrm{diag}(s_t), \\
\theta_{t+1} &= \theta_t - \eta_t\, H_t^{-1} m_t,
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the gradient, $m_t$ the first moment, $v_t$ the second moment, $s_t$ the second-moment scale, $\eta_t$ the trajectory-derived step size (initialized $\eta_0 = \epsilon$), $\beta_1,\beta_2$ the decay rates, $\lambda$ the momentum decay factor, $\delta$ a regularization constant, and $d$ the parameter dimension. A simpler variant replaces the recursion with $s_t = \big(\sum_{i=0}^{t} g_i^2\big)^{1/2}$.

Reference: Yuanzhe Tao, Huizhuo Yuan, Xun Zhou, Yuan Cao, Quanquan Gu, "Towards Simple and Provable Parameter-Free Adaptive Gradient Methods", arXiv 2024. https://arxiv.org/abs/2412.19444

---
[Back to the Canon](../index.md)
