# ZoPro

Implements ZoPro, a zeroth-order proximal (regularized Newton) method for decentralized consensus optimization.

ZoPro solves a network consensus problem in which each agent $i$ holds a local objective $f_i$ accessible only through function values. At every iteration each agent forms Gaussian-smoothed zeroth-order estimates of its local gradient and Hessian, then takes a regularized Newton step on the penalized consensus objective. A symmetric weight matrix $W$ couples neighbors, the penalty parameter $\rho$ enforces agreement, and a dual variable $q$ accumulates the consensus residual; an Armijo backtracking line search picks the per-agent step size.

The two-point estimates over $b$ random directions $u_j \sim \mathcal{N}(0, I_d)$ and smoothing radius $\mu$ are

$$
\begin{aligned}
\tilde g_{\mu,i}(x_i) &= \frac{1}{b}\sum_{j=1}^{b}\frac{f_i(x_i+\mu u_j)-f_i(x_i)}{\mu}\,u_j, \\
\tilde H_{\mu,i}(x_i) &= \frac{1}{b}\sum_{j=1}^{b}\frac{f_i(x_i+\mu u_j)+f_i(x_i-\mu u_j)-2f_i(x_i)}{2\mu^2}\,u_j u_j^{\top},
\end{aligned}
$$

and the per-agent update at iteration $k$ is

$$
\begin{aligned}
d_i^{k} &= -\bigl(\tilde H_{\mu,i}(x_i^{k}) + D_i\bigr)^{-1}\bigl(\tilde g_{\mu,i}(x_i^{k}) + \rho\, y_i^{k} + q_i^{k}\bigr), \\
x_i^{k+1} &= x_i^{k} + \alpha_i^{k}\, d_i^{k}, \\
q_i^{k+1} &= q_i^{k} + \rho\, y_i^{k+1},
\end{aligned}
$$

where $y_i^{k} = \sum_{j} W_{ij}\, x_j^{k}$ is the weighted consensus residual at agent $i$, $D_i$ is a regularization matrix chosen so $\tilde H_{\mu,i}(x_i)+D_i \succ 0$, $q_i$ is the dual variable, $\rho$ is the penalty parameter, $\mu$ is the smoothing radius, $b$ is the number of sampling directions, and $\alpha_i^{k}$ is the step size returned by Armijo backtracking line search.

Reference: Chengan Wang, Zichong Ou, Jie Lu, "A Zeroth-Order Proximal Algorithm for Consensus Optimization", arXiv 2024. https://arxiv.org/abs/2406.09816

---
[Back to the Canon](../index.md)
