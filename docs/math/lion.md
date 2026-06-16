# Lion

Implements Lion, a sign-momentum optimizer discovered by symbolic search.


$$
\begin{aligned}
c_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
\theta_t &= \theta_{t-1} - \gamma \left( \mathrm{sign}(c_t)
            + \lambda \theta_{t-1} \right) \\
m_t &= \beta_2 m_{t-1} + (1 - \beta_2)\, g_t
\end{aligned}
$$

where $m_t$ is the single momentum buffer, $\lambda$ is the
decoupled weight decay, and the update direction is the element-wise sign of
the interpolated momentum $c_t$. The interpolation rate
$\beta_1$ and the momentum decay $\beta_2$ are passed as
`betas`.

Reference: Xiangning Chen et al., "Symbolic Discovery of Optimization
Algorithms", NeurIPS 2023.
https://arxiv.org/abs/2302.06675

---
[Back to the Canon](../index.md)
