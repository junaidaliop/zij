# AdaBoundW

Implements AdaBound with decoupled weight decay (AdamW-style).

The adaptive update matches `AdaBound`; weight decay is applied
directly to the parameters rather than added to the gradient.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
     \hat{\eta}_t &= \mathrm{Clip}\!\left(
         \frac{\eta \sqrt{1 - \beta_2^t}}{(1 - \beta_1^t)
             (\sqrt{v_t} + \epsilon)},
         \eta_l(t), \eta_u(t)\right)                                       \\
     \theta_t &= \theta_{t-1} - \hat{\eta}_t \odot m_t - \lambda \theta_{t-1}
\end{aligned}
$$

where $\lambda$ is the weight decay and $\eta_l(t)$,
$\eta_u(t)$ are the AdaBound bounds.

Reference: Liangchen Luo, Yuanhao Xiong, Yan Liu, Xu Sun, "Adaptive Gradient
Methods with Dynamic Bound of Learning Rate", ICLR 2019.
https://arxiv.org/abs/1902.09843

---
[Back to the Canon](../README.md)
