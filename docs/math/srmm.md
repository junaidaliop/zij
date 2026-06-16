# SRMM

Implements SRMM (Stochastic Regularized Majorization-Minimization), a majorization-minimization scheme for nonconvex stochastic optimization with weakly convex or block multi-convex surrogates.

At each step a new data point defines the instantaneous loss $f_n(\theta)=\ell(x_n,\theta)$, for which a surrogate $g_n$ majorizing $f_n$ near $\theta_{n-1}$ is chosen. SRMM maintains a running average $\bar g_n$ of these surrogates and minimizes it under a regularizer that keeps the iterate close to the previous one. The regularizer either adds a proximal term $\tfrac{\hat\rho}{2}\lVert\theta-\theta_{n-1}\rVert^2$ (to convexify weakly convex surrogates with $\hat\rho>-\rho$) or imposes a diminishing trust-region radius $\lVert\theta-\theta_{n-1}\rVert\le c\,w_n$ for block multi-convex surrogates.

$$
\begin{aligned}
g_n &\in \mathrm{Srg}_{L,\rho_n}(f_n,\theta_{n-1},\varepsilon_n), \\
\bar g_n &= (1-w_n)\,\bar g_{n-1} + w_n\, g_n, \\
\theta_n &\approx \arg\min_{\theta\in\Theta}\ \Big[\, \bar g_n(\theta) + \Psi_n\big(\lVert\theta-\theta_{n-1}\rVert\big) \,\Big], \\
\Psi_n(\lVert\theta-\theta_{n-1}\rVert) &= \tfrac{\hat\rho}{2}\,\lVert\theta-\theta_{n-1}\rVert^2 \quad\text{or}\quad \Psi_n = \begin{cases} 0, & \lVert\theta-\theta_{n-1}\rVert\le c\,w_n,\\ +\infty, & \text{otherwise.} \end{cases}
\end{aligned}
$$

where $g_n$ is a surrogate majorizing the loss $f_n$ at $\theta_{n-1}$ within tolerance $\varepsilon_n$, $w_n\in(0,1]$ is a non-increasing averaging weight, $\bar g_n$ is the weighted-average surrogate, $\hat\rho$ is the proximal regularization strength ($\hat\rho>-\rho$ for weakly convex surrogates), $\Psi_n$ is the regularizer (proximal term or diminishing-radius indicator), $c>0$ sets the trust-region radius, and $\Theta\subseteq\mathbb{R}^p$ is the constraint set.

Reference: Hanbaek Lyu, "Stochastic regularized majorization-minimization with weakly convex and multi-convex surrogates", arXiv 2022. https://arxiv.org/abs/2201.01652

---
[Back to the Canon](../index.md)
