# Accelerated Distance-adaptive Method (DoG-lineage)

Implements the Accelerated Distance-adaptive Method (AGDA), a learning-rate-free accelerated dual-averaging scheme in the DoG lineage.

This method extends the distance-over-gradients idea to Nesterov-style acceleration without any learning rate. It maintains three coupled sequences: a dual-averaging iterate $v_t$, an extrapolated query point $\theta_t$, and an averaged output point $y_t$. The effective step is set entirely from an adaptively estimated distance to the optimum, $\bar r_t$, taken as the running maximum of $\|\theta_0 - v_t\|$, so no learning rate is tuned.

Acceleration weights $a_t$ grow with the accumulated square-root distance, giving the classical $\mathcal{O}(1/t^2)$ schedule when the distance estimate stabilizes. The regularization coefficient $\beta_t$ (the inverse of the dual-averaging step) is chosen by a line search / closed-form balance so the accelerated descent inequality holds; the gradients are then accumulated into a single dual-averaging update.

$$
\begin{aligned}
r_t &= \|\theta_0 - v_t\|, \qquad \bar r_t = \max\{\bar r_{t-1},\, r_t\} \\
A_{t+1} &= \Big(\textstyle\sum_{i=0}^{t} \bar r_i^{1/2}\Big)^2, \qquad a_{t+1} = A_{t+1} - A_t, \qquad \tau_t = \frac{a_{t+1}}{A_{t+1}} \\
\theta_{t+1} &= \tau_t\, v_t + (1-\tau_t)\, y_t \\
v_{t+1} &= \theta_0 - \frac{1}{\beta_{t+1}} \sum_{i=1}^{t+1} a_i\, \nabla f(\theta_i) \\
y_{t+1} &= \tau_t\, v_{t+1} + (1-\tau_t)\, y_t
\end{aligned}
$$

where $\theta_0$ is the initial point, $v_t$ is the dual-averaging iterate (with $v_0 = y_0 = \theta_0$), $\theta_t$ is the query point at which the gradient $\nabla f(\theta_t)$ is taken, $y_t$ is the returned averaged iterate, $\bar r_t$ is the adaptive distance estimate, $A_t$ and $a_t$ are the cumulative and incremental acceleration weights, $\tau_t$ is the momentum interpolation factor, and $\beta_{t+1}$ is the dual-averaging regularization coefficient set by line search to satisfy the accelerated descent condition. The displayed $v$ update is the unconstrained ($g \equiv 0$) closed form of $v_{t+1} = \arg\min_\theta \{ \tfrac{\beta_{t+1}}{2}\|\theta_0 - \theta\|^2 + \sum_{i=1}^{t+1} a_i(\langle \nabla f(\theta_i), \theta - \theta_i\rangle + g(\theta))\}$.

Reference: Yijin Ren, Haifeng Xu, Qi Deng, "Accelerated Distance-adaptive Methods for Hölder Smooth and Convex Optimization", arXiv 2025. https://arxiv.org/abs/2510.22135

---
[Back to the Canon](../README.md)
