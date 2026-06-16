# BCOS

Implements BCOS, stochastic approximation with block-coordinate optimal stepsizes.

BCOS chooses each coordinate's stepsize to minimize the expected squared distance to the optimum, yielding a closed-form optimal stepsize that scales the search direction by the ratio of its conditional mean to its conditional second moment. With momentum as the search direction and a conditional EMA estimator of the second moment, this gives an Adam-like coordinate-wise update that requires fewer hyperparameters. The conditional estimator $v_t$ blends the squared previous momentum and the squared current gradient, which removes the need for a separate $\beta_2$.

$$
\begin{aligned}
g_t &= \nabla f(\theta_{t-1}, \xi_t) \\
m_t &= \beta\, m_{t-1} + (1-\beta)\, g_t \\
v_t &= \bigl(1-(1-\beta)^2\bigr)\, m_{t-1}^2 + (1-\beta)^2\, g_t^2 \\
\theta_t &= (1 - \alpha_t \lambda)\, \theta_{t-1} - \alpha_t\, \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\alpha_t$ is the (block) stepsize, $g_t$ the stochastic gradient, $m_t$ the momentum search direction, $v_t$ the conditional second-moment estimate, $\beta$ the momentum decay, $\lambda$ the decoupled weight decay, and $\epsilon$ a stability constant; all products are element-wise.

Reference: Tao Jiang, Lin Xiao, "Stochastic Approximation with Block Coordinate Optimal Stepsizes", arXiv 2025. https://arxiv.org/abs/2507.08963

---
[Back to the Canon](../index.md)
