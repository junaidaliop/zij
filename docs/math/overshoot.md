# Overshoot

Implements Overshoot, momentum SGD that evaluates gradients at weights shifted ahead in the momentum direction.

Standard momentum updates the base weights $\theta_t$, but Overshoot keeps a second set of "overshoot" weights $\theta'_t = \theta_t + \gamma\eta\, m_t$ that are pushed further along the accumulated momentum. Gradients are computed at these shifted weights, so each step anticipates the trajectory the optimizer is about to take, taking advantage of future gradients. The base weights are recovered at the end of training; in practice the shift is folded into a single closed-form update so no extra forward pass is needed.

$$
\begin{aligned}
m_{t+1} &= \mu\, m_t + g_t \\
\theta'_{t+1} &= \theta'_t - \eta\big((\gamma - \gamma\mu^{-1} + 1)\, m_{t+1} + \gamma\mu^{-1}\, g_t\big)
\end{aligned}
$$

where $g_t = \nabla f(\theta'_t)$ is the gradient at the overshoot weights, $\mu \in (0,1]$ is the momentum coefficient, $\gamma \ge 0$ is the overshoot factor, $\eta$ is the learning rate, and the base weights are obtained as $\theta_t = \theta'_t - \gamma\eta\, m_t$.

Reference: Jakub Kopal, Michal Gregor, Santiago de Leon-Martinez, Jakub Simko, "Overshoot: Taking advantage of future gradients in momentum-based stochastic optimization", 2025. https://arxiv.org/abs/2501.09556

---
[Back to the Canon](../index.md)
