# ADAGB2

Implements ADAGB2, a stochastic second-order Adagrad for nonconvex bound-constrained optimization.

ADAGB2 generalizes Adagrad to second-order steps over a feasible set. At each iteration it forms the projected first-order direction $d_t$, updates a per-coordinate accumulator $w_t$ from the running sum of squared projected steps, and uses it to set a per-coordinate trust radius $\Delta_t$ that bounds the step. A diagonal (or symmetric) Hessian approximation $B_t$ supplies curvature through a scalar Cauchy-style scaling $\gamma_t$, so the method interpolates between adaptive first-order and second-order behavior. In the unconstrained case with $B_t = 0$ it collapses to the familiar Adagrad rule.

$$
\begin{aligned}
d_t &= P_{\mathcal{F}}(\theta_t - g_t) - \theta_t, \\
w_{t,i} &= \sqrt{w_{t-1,i}^2 + d_{t,i}^2}, \\
\Delta_{t,i} &= \frac{|d_{t,i}|}{w_{t,i}}, \\
\gamma_t &= \min\!\left(1,\; \frac{-g_t^\top s_t}{s_t^\top B_t s_t}\right), \\
\theta_{t+1} &= \theta_t + \gamma_t\, s_t, \quad |s_{t,i}| \le \kappa_s\, \Delta_{t,i}.
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the stochastic gradient, $P_{\mathcal{F}}$ the projection onto the feasible set $\mathcal{F}$, $w_t$ the per-coordinate Adagrad accumulator (initialized $w_{-1,i} = \varsigma \in (0,1]$), $\Delta_t$ the per-coordinate trust radius, $B_t$ a symmetric Hessian approximation, $\gamma_t$ the Cauchy scaling (set to $1$ when $s_t^\top B_t s_t \le 0$), and $\kappa_s \ge 1$ the step-bound constant; with $B_t = 0$ and $\mathcal{F} = \mathbb{R}^n$ this reduces to $\theta_{t+1,i} = \theta_{t,i} - g_{t,i}/w_{t,i}$.

Reference: Stefania Bellavia, Serge Gratton, Benedetta Morini, Philippe L. Toint, "Fast Stochastic Second-Order Adagrad for Nonconvex Bound-Constrained Optimization", arXiv 2025. https://arxiv.org/abs/2505.06374

---
[Back to the Canon](../README.md)
