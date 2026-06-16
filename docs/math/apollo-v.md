# Apollo

Implements Apollo, an adaptive parameter-wise diagonal quasi-Newton method for nonconvex stochastic optimization.

Apollo approximates the Hessian by a diagonal matrix $B_t$ that is updated to satisfy a parameter-wise weak secant condition using only first-order gradient information, giving linear time and memory cost. To guarantee a descent direction in the nonconvex setting, the diagonal preconditioner is rectified by taking element-wise absolute values bounded below by a convexity threshold $\sigma$.

$$
\begin{aligned}
m_t &= \frac{\beta(1-\beta^{t-1})}{1-\beta^{t}}\, m_{t-1} + \frac{1-\beta}{1-\beta^{t}}\, g_t \\
\alpha_t &= \frac{d_{t-1}^{\top}(m_t - m_{t-1}) + d_{t-1}^{\top} B_{t-1} d_{t-1}}{(\lVert d_{t-1} \rVert_4 + \epsilon)^4} \\
B_t &= B_{t-1} - \alpha_t\, \mathrm{Diag}(d_{t-1}^2) \\
D_t &= \max(\lvert B_t \rvert,\, \sigma) \\
d_t &= D_t^{-1} m_t \\
\theta_t &= \theta_{t-1} - \eta\, d_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the bias-corrected exponential moving average of gradients with decay $\beta$, $B_t$ the diagonal Hessian approximation, $d_t$ the update direction, $\mathrm{Diag}(d_{t-1}^2)$ the diagonal matrix of squared direction entries, $D_t = \max(\lvert B_t \rvert, \sigma)$ the rectified preconditioner with convexity threshold $\sigma$, and $\epsilon$ a stability constant.

Reference: Xuezhe Ma, "Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization", arXiv 2020. https://arxiv.org/abs/2009.13586

---
[Back to the Canon](../index.md)
