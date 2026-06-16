# MARS-M

Implements MARS-M, a matrix-aware variance-reduced optimizer that brings MARS-style gradient correction to the Muon update.

MARS-M forms a corrected gradient $c_t$ by adding a scaled difference between the current gradient and the previous gradient evaluated at the same minibatch, which reduces stochastic variance. The corrected gradient is clipped to unit norm, accumulated into a heavy-ball momentum matrix, and then orthogonalized via a Newton-Schulz iteration before the decoupled-weight-decay parameter step, so the matrix structure of the layer is preserved exactly as in Muon.

$$
\begin{aligned}
c_t &= g_t + \gamma_t \frac{\beta}{1-\beta}\left(g_t - g_{t-1}\right) \\
\hat{c}_t &= \frac{c_t}{\max(1, \lVert c_t \rVert_2)} \\
m_t &= \beta\, m_{t-1} + (1-\beta)\, \hat{c}_t \\
o_t &= \mathrm{NewtonSchulz}(m_t) \\
\theta_{t+1} &= \theta_t - \eta_t\left(0.2\, \sqrt{\max(m,n)}\; o_t + \lambda\, \theta_t\right)
\end{aligned}
$$

where $\theta$ are the (matrix) parameters with dimensions $m \times n$, $g_t = \nabla f(\theta_t, \xi_t)$ and $g_{t-1} = \nabla f(\theta_{t-1}, \xi_t)$ are gradients on the same minibatch $\xi_t$, $\gamma_t$ is the variance-reduction scaling, $\beta$ is the momentum coefficient, $\mathrm{NewtonSchulz}(\cdot)$ approximates the orthogonalization $U V^\top$ of $m_t = U \Sigma V^\top$, $\eta_t$ is the learning rate, and $\lambda$ is the decoupled weight decay.

Reference: Yifeng Liu, Angela Yuan, Quanquan Gu, "MARS-M: When Variance Reduction Meets Matrices", 2025. https://arxiv.org/abs/2510.21800

---
[Back to the Canon](../index.md)
