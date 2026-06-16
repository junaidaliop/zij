# S-BFGS

Implements S-BFGS, a stochastic BFGS quasi-Newton method that regularizes the inverse-Hessian update with Bayesian-derived terms.

Standard BFGS breaks down under noisy gradients because the curvature pairs are corrupted. S-BFGS treats the curvature information as observed under a likelihood and prior, which adds two regularizing quantities to the secant denominators: a likelihood parameter $\rho$ and a precision $p_t$ (the inverse trace of the covariance of $y_t$). The result is a damped inverse-Hessian recursion that stays stable with modest batch sizes; the limited-memory variant L-S-BFGS recovers the $O(d)$ per-step cost.

$$
\begin{aligned}
g_t &= \frac{1}{N}\sum_{n=1}^{N}\nabla_\theta f(\theta_t,\xi_{t,n}), \\
s_t &= \theta_t-\theta_{t-1}, \qquad
y_t = \frac{1}{N}\sum_{n=1}^{N}\bigl[\nabla_\theta f(\theta_t,\xi_{t,n})-\nabla_\theta f(\theta_{t-1},\xi_{t,n})\bigr], \\
H_{t+1} &= H_t
+\frac{1+\dfrac{y_t^\top H_t y_t}{s_t^\top y_t+\rho/p_t}}{\,s_t^\top y_t+\rho/(2p_t)\,}\,s_t s_t^\top
-\frac{H_t y_t s_t^\top+s_t y_t^\top H_t}{s_t^\top y_t+\rho/p_t}, \\
\theta_{t+1} &= \theta_t-\eta\,H_t\,g_t.
\end{aligned}
$$

where $H_t$ is the inverse-Hessian approximation, $g_t$ the mini-batch gradient over $N$ samples $\xi_{t,n}$, $(s_t,y_t)$ the curvature pair, $\rho>0$ the likelihood parameter, $p_t>0$ the precision of $y_t$, and $\eta$ the step size. The pair is accepted only when $y_t^\top s_t\ge m\,\lVert s_t\rVert^2$ for a tuning constant $m>0$.

Reference: André Carlon, Luis Espath, Raúl Tempone, "Efficient Stochastic BFGS methods Inspired by Bayesian Principles", arXiv 2025. https://arxiv.org/abs/2507.07729

---
[Back to the Canon](../index.md)
