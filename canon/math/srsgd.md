# SRSGD

Implements SRSGD, Nesterov accelerated gradient with a scheduled momentum restart.

Plain Nesterov acceleration uses a momentum coefficient that grows toward $1$ as iterations proceed. With stochastic (inexact) gradients this lets error accumulate and hurts convergence. SRSGD keeps the same gradient step but replaces the iteration counter inside the momentum coefficient with $k \bmod F_i$, so the momentum is reset to zero every $F_i$ steps. The restart frequency $F_i$ is itself scheduled across training, either linearly or exponentially, giving acceleration early in each cycle while periodically discarding accumulated error.

$$
\begin{aligned}
v_{t+1} &= \theta_t - \eta\, g_t \\
\theta_{t+1} &= v_{t+1} + \frac{t \bmod F_i}{(t \bmod F_i) + 3}\,(v_{t+1} - v_t)
\end{aligned}
$$

where $\theta_t$ are the parameters, $v_t$ the post-gradient-step iterate, $g_t$ the (mini-batch) gradient, $\eta$ the step size, $F_i$ the restart frequency in the current schedule interval, and $t \bmod F_i$ the steps since the last restart. The frequency is updated between intervals by a linear schedule $F_{i+1} = F_1\,(1 + (r-1)\,i)$ or an exponential schedule $F_{i+1} = F_1\, r^{\,i}$.

Reference: Bao Wang, Tan M. Nguyen, Tao Sun, Andrea L. Bertozzi, Richard G. Baraniuk, Stanley J. Osher, "Scheduled Restart Momentum for Accelerated Stochastic Gradient Descent", arXiv 2020. https://arxiv.org/abs/2002.10583

---
[Back to the Canon](../README.md)
