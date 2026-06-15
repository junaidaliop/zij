# VRAdam

Implements VRAdam, a physics-inspired Adam variant that regularizes the learning rate by the squared norm of the momentum (velocity).

VRAdam treats the first-moment buffer as a velocity and borrows the idea of a quartic kinetic-energy term: when the parameters move fast (large velocity norm) the effective learning rate is throttled, giving a self-braking mechanism that damps overshoot. The braking term is clipped so the step size cannot collapse arbitrarily. Apart from the dynamic learning rate, the moment estimates, bias correction, and decoupled weight decay follow AdamW.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\eta_t &= \frac{\alpha_0}{1 + \min(\beta_3 \lVert m_t \rVert^2,\ \alpha_1)} \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1}(1 - \eta_t \lambda) - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the gradient, $m_t$ the first-moment (velocity) buffer with decay $\beta_1$, $v_t$ the second-moment buffer with decay $\beta_2$, $\alpha_0$ the maximal learning rate, $\alpha_1$ the learning-rate cutoff that caps the braking term, $\beta_3$ the velocity-penalty coefficient, $\lVert m_t \rVert$ the Euclidean norm of the velocity, $\lambda$ the decoupled weight decay, and $\epsilon$ a stability constant.

Reference: Pranav Vaidhyanathan, Lucas Schorling, Natalia Ares, Michael A. Osborne, "A Physics-Inspired Optimizer: Velocity Regularized Adam", arXiv 2025. https://arxiv.org/abs/2505.13196

---
[Back to the Canon](../README.md)
