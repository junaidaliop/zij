# PowerStep

Implements PowerStep, a memory-efficient optimizer that applies a signed power transform to a heavy-ball momentum buffer.

PowerStep avoids the per-coordinate second-moment buffer of Adam by deriving coordinate-wise adaptivity from an $\ell_p$-norm steepest-descent view. It first accumulates gradients into a heavy-ball momentum buffer for temporal smoothing, then applies a signed power transform to that buffer, which compresses large coordinates and amplifies small ones. This yields adaptive-style behavior with a single state buffer, halving the optimizer memory relative to Adam, while decoupled weight decay is added directly in the update.

$$
\begin{aligned}
m_t &= \gamma\, m_{t-1} + g_t \\
u_t &= \mathrm{sign}(m_t) \odot |m_t|^{\beta} \\
\theta_t &= \theta_{t-1} - \eta_t \left( u_t + \lambda\, \theta_{t-1} \right)
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the gradient, $m_t$ the heavy-ball momentum buffer, $\gamma \in [0,1)$ the momentum coefficient, $\beta \in [0,1]$ the power exponent applied elementwise, $\eta_t$ the learning-rate schedule, $\lambda \ge 0$ the decoupled weight decay, and $\odot$ elementwise multiplication.

Reference: Yao Lu, Dengdong Fan, Shixun Zhang, Yonghong Tian, "PowerStep: Memory-Efficient Adaptive Optimization via $\ell_p$-Norm Steepest Descent", arXiv 2026. https://arxiv.org/abs/2605.10335

---
[Back to the Canon](../index.md)
