# NIGT

Implements NIGT (Normalized SGD with Implicit Gradient Transport), normalized SGD with momentum that corrects the moment estimate for the iterate shift.

NIGT combines normalized SGD with a momentum buffer, but evaluates the gradient at an extrapolated auxiliary point rather than the current iterate. This implicit gradient transport extrapolates along the recent step direction so the stale gradients held in the momentum buffer better match the current parameters, which removes the bias normally introduced by momentum and lets the method match the convergence guarantees of normalized SGD.

$$
\begin{aligned}
x_t &= \theta_t + \frac{\beta}{1-\beta}\,(\theta_t - \theta_{t-1}) \\
m_t &= \beta\, m_{t-1} + (1-\beta)\, \nabla f(x_t, \xi_t) \\
\theta_{t+1} &= \theta_t - \eta\, \frac{m_t}{\lVert m_t \rVert}
\end{aligned}
$$

where $\theta_t$ are the parameters, $x_t$ the extrapolated auxiliary point at which the stochastic gradient $\nabla f(x_t,\xi_t)$ is sampled, $m_t$ the momentum buffer, $\eta$ the learning rate, and $\beta$ the momentum coefficient; the buffer is initialized as $m_1 = \nabla f(\theta_1,\xi_1)$ with $\theta_2 = \theta_1 - \eta\, m_1 / \lVert m_1 \rVert$.

Reference: Ashok Cutkosky, Harsh Mehta, "Momentum Improves Normalized SGD", ICML 2020. https://arxiv.org/abs/2002.03305

---
[Back to the Canon](../index.md)
