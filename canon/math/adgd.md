# AdGD

Implements AdGD (Adaptive Gradient Descent without Descent), a learning-rate-free step size that adapts to the local curvature using only gradients.

The method removes the need for a fixed learning rate, line search, or function evaluations. At each step it estimates a local Lipschitz constant from the most recent change in iterates and gradients, and sets the step size to the smaller of two quantities: one that prevents the step from growing too quickly, and one that prevents overshooting the local curvature.

$$
\begin{aligned}
\gamma_t &= \min\left\{ \sqrt{1 + r_{t-1}}\,\gamma_{t-1},\ \frac{\lVert \theta_t - \theta_{t-1} \rVert}{2\,\lVert g_t - g_{t-1} \rVert} \right\} \\
r_t &= \frac{\gamma_t}{\gamma_{t-1}} \\
\theta_{t+1} &= \theta_t - \gamma_t\, g_t
\end{aligned}
$$

where $\theta_t$ are the parameters at step $t$, $g_t = \nabla f(\theta_t)$ is the gradient, $\gamma_t$ is the adaptive step size, and $r_t$ is the ratio of consecutive step sizes (initialized $r_0 = +\infty$, with arbitrary $\gamma_0 > 0$).

Reference: Yura Malitsky, Konstantin Mishchenko, "Adaptive Gradient Descent without Descent", ICML 2020. https://arxiv.org/abs/1910.09529

---
[Back to the Canon](../README.md)
