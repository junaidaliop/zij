# OptiQ

Implements OptiQ, a second-order optimizer that discretizes gradient flow with adaptive step sizes set by the dominant time constant.

OptiQ views optimization as integrating the gradient-flow ODE $\dot\theta = -\nabla f(\theta)$ toward a critical point. Instead of forcing monotonic descent, it estimates a first-order time constant for each parameter from the ratio of its velocity to its acceleration, and advances by the smallest such constant. The parameter with the shortest rise time is driven into "quiescence" ($\ddot\theta = 0$) and thereafter moved along the quasi-steady-state implied by the remaining variables, so each step requires inverting only the small Hessian block of the quiescent set rather than the full Hessian.

At each iteration the state is partitioned as $\theta = [\theta_q; \theta_{nq}]$ (quiescent and non-quiescent). With $g = \nabla f$ and Hessian blocks $H_{nq} = \partial^2 f/\partial\theta_{nq}^2$, $H_q = \partial^2 f/\partial\theta_q^2$, $H_{q,nq} = \partial^2 f/\partial\theta_q\partial\theta_{nq}$:

$$
\begin{aligned}
\dot\theta_{nq} &= -\,g_{nq}, \qquad \ddot\theta_{nq} = H_{nq}\,\dot\theta_{nq}, \\
\tilde\tau_i &= -\,\frac{\dot\theta_{nq,i}}{\ddot\theta_{nq,i}}, \qquad \Delta t = \min_i \tilde\tau_i, \\
\dot\theta_q &= -\,H_q^{-1}\,H_{q,nq}\,\dot\theta_{nq}, \\
\theta_{nq} &\leftarrow \theta_{nq} + \Delta t\,\dot\theta_{nq}, \\
\theta_q &\leftarrow \theta_q + \Delta t\,\dot\theta_q,
\end{aligned}
$$

where $\tilde\tau_i$ is the estimated time constant of non-quiescent variable $i$, $\Delta t$ is the chosen step (the smallest dominant time constant), and the variable attaining the minimum is appended to the quiescent set $\theta_q$. A variable is returned to the non-quiescent set when its quiescence residual $|\dot\theta_q + g_q|$ exceeds a tolerance, and iteration continues until $\|g\|^2$ falls below the threshold.

Reference: Aayushya Agarwal, Larry Pileggi, Ronald Rohrer, "Second-Order Optimization via Quiescence", arXiv 2024. https://arxiv.org/abs/2410.08033

---
[Back to the Canon](../index.md)
