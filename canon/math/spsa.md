# SPSA

Implements SPSA, a gradient-free stochastic approximation method using simultaneous perturbation.

SPSA minimizes a noisy loss $L(\theta)$ without analytic gradients. Instead of perturbing each coordinate separately (which costs $2p$ loss evaluations for a $p$-dimensional parameter), it perturbs all coordinates at once along a single random direction $\Delta_t$, so the full gradient estimate needs only two loss measurements per step regardless of dimension. The estimate is biased per step but is an asymptotically unbiased descent direction in expectation, and the iterate converges under decaying gains.

$$
\begin{aligned}
(\hat g_t)_i &= \frac{L(\theta_t + c_t \Delta_t) - L(\theta_t - c_t \Delta_t)}{2 c_t (\Delta_t)_i} \\
\theta_{t+1} &= \theta_t - a_t\, \hat g_t \\
a_t &= \frac{a}{(t + 1 + A)^{\alpha}}, \qquad c_t = \frac{c}{(t + 1)^{\gamma}}
\end{aligned}
$$

where $L$ is the (noisy) objective, $\hat g_t$ is the simultaneous-perturbation gradient estimate with $i$-th component formed from a single symmetric two-point difference divided by the perturbed coordinate $(\Delta_t)_i$, $\Delta_t$ is a mean-zero perturbation vector with independent, symmetric components having finite inverse moments (typically Rademacher $\pm 1$), $a_t$ and $c_t$ are decaying gain sequences with constants $a, c, A > 0$ and exponents $\alpha, \gamma$ (standard asymptotic choices $\alpha = 0.602$, $\gamma = 0.101$).

Reference: J. C. Spall, "Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation", IEEE Transactions on Automatic Control 1992. https://doi.org/10.1109/9.119632

---
[Back to the Canon](../README.md)
