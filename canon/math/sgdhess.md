# SGDHess

Implements SGDHess, SGD with a second-order momentum correction built from Hessian-vector products.

SGDHess augments variance-reduced momentum (STORM-style) with a curvature term: instead of carrying the raw stale momentum forward, it corrects it using the Hessian-vector product along the most recent step, which estimates how the gradient should have changed between iterates. The momentum buffer is clipped before each parameter step to keep its norm bounded.

$$
\begin{aligned}
\hat{g}_t &= (1-\alpha_{t-1})\left(\hat{g}^{\,\mathrm{clip}}_{t-1} + \nabla^2 f(\theta_t, z_t)\,(\theta_t - \theta_{t-1})\right) + \alpha_{t-1}\, g_t \\
\hat{g}^{\,\mathrm{clip}}_t &= \hat{g}_t \cdot \min\!\left(1, \frac{G}{\lVert \hat{g}_t \rVert}\right) \\
\theta_{t+1} &= \theta_t - \eta_t\, \hat{g}^{\,\mathrm{clip}}_t
\end{aligned}
$$

where $\theta_t$ are the parameters, $g_t = \nabla f(\theta_t, z_t)$ the stochastic gradient at the current sample $z_t$, $\nabla^2 f(\theta_t, z_t)\,(\theta_t - \theta_{t-1})$ the Hessian-vector product along the last step, $\hat{g}_t$ the second-order momentum buffer, $\hat{g}^{\,\mathrm{clip}}_t$ its clipping to norm $G$, $\eta_t$ the learning rate, and $\alpha_{t-1}$ the momentum coefficient. Initialization uses $\hat{g}_1 = g_1$.

Reference: Hoang Tran, Ashok Cutkosky, "Better SGD using Second-order Momentum", arXiv 2021. https://arxiv.org/abs/2103.03265

---
[Back to the Canon](../README.md)
