# SPSA

Implements SPSA, a zeroth-order method that estimates the gradient from a single random perturbation, regardless of dimension.

Simultaneous Perturbation Stochastic Approximation perturbs all parameters at once along a random direction $\Delta_t$ and forms a finite-difference estimate of the gradient using only two loss evaluations. This makes the per-step cost independent of the problem dimension, in contrast to coordinate-wise finite differencing which needs $2p$ evaluations. The estimate is biased but consistent, and the recursion converges to a stationary point under decaying gain sequences.

$$
\begin{aligned}
(\hat{g}_t)_i &= \frac{f(\theta_t + c_t \Delta_t) - f(\theta_t - c_t \Delta_t)}{2 c_t (\Delta_t)_i} \\
\theta_{t+1} &= \theta_t - a_t \, \hat{g}_t \\
a_t &= \frac{a}{(t + 1 + A)^\alpha}, \qquad c_t = \frac{c}{(t + 1)^\gamma}
\end{aligned}
$$

where $f$ is the (noisy) objective, $\theta$ are the parameters, $\hat{g}_t$ is the simultaneous-perturbation gradient estimate with components indexed by $i$, $\Delta_t$ is a random perturbation vector whose entries are independent and symmetric about zero with finite inverse moments (typically Rademacher $\pm 1$), $a_t$ and $c_t$ are the decaying gain and perturbation sequences, and $a, c, A, \alpha, \gamma$ are positive tuning constants (with the standard asymptotically optimal choice $\alpha = 1$, $\gamma = 1/6$).

Reference: J. C. Spall, "Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation", IEEE Transactions on Automatic Control 1992. https://doi.org/10.1109/9.119632

---
[Back to the Canon](../README.md)
