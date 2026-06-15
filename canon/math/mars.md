# MARS

Implements MARS, a variance-reduced preconditioned optimizer (MARS-AdamW variant).


$$
\begin{aligned}
c_t &= g_t + \gamma\, \frac{\beta_1}{1 - \beta_1}\,(g_t - g_{t-1}) \\
\tilde{c}_t &= \begin{cases}
    c_t / \lVert c_t \rVert_2 & \text{if } \lVert c_t \rVert_2 > 1 \\
    c_t & \text{otherwise}
\end{cases} \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \tilde{c}_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \tilde{c}_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1}
    - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
    + \lambda\, \theta_{t-1}\right)
\end{aligned}
$$

where $c_t$ is the scaled stochastic recursive momentum correction,
$\gamma$ the gradient-correction scaling factor, and $\lambda$
the decoupled weight decay. By default the correction uses the approximate
(`is_approx`) form that reuses the previous step's gradient as
$g_{t-1}$. One-dimensional parameters fall back to AdamW unless
`optimize_1d` is set. `mars_type` selects the preconditioner among the
`mars-adamw`, `mars-lion`, and `mars-shampoo` instantiations.

Reference: Huizhuo Yuan, Yifeng Liu, Shuang Wu, Xun Zhou, Quanquan Gu,
"MARS: Unleashing the Power of Variance Reduction for Training Large Models",
ICML 2025.
https://arxiv.org/abs/2411.10438

---
[Back to the Canon](../README.md)
