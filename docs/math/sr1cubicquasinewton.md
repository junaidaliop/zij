# SR1 Cubic Quasi-Newton

Implements SR1 Cubic Quasi-Newton, a limited-memory symmetric rank-one method with adaptive cubic regularization.

A curvature estimate $B_k$ is built from the SR1 (symmetric rank-one) update, which—unlike BFGS—admits indefinite approximations and so can capture negative curvature. Each step solves a cubic-regularized model of the loss in which the regularization replaces a trust region: the cubic term penalizes large steps and guarantees a well-defined minimizer even when $B_k$ is indefinite. The compact eigenbasis of the low-rank $B_k$ yields a closed-form step, and the regularization weight $\mu_k$ is adapted from the ratio of actual to predicted decrease.

$$
\begin{aligned}
B_{k+1} &= B_k + \frac{(y_k - B_k s_k)(y_k - B_k s_k)^\top}{s_k^\top (y_k - B_k s_k)}, \qquad s_k = \theta_k - \theta_{k-1}, \quad y_k = g_k - g_{k-1} \\
s^\ast &= \arg\min_{s}\; m_k(s) \equiv g_k^\top s + \tfrac{1}{2} s^\top B_k s + \frac{\mu_k}{3}\,\big(\Phi_k(s)\big)^3 \\
\theta_{k+1} &= \begin{cases} \theta_k + s^\ast & \text{if } \rho_k \ge \eta_1 \\ \theta_k & \text{otherwise} \end{cases}, \qquad \rho_k = \frac{f(\theta_k) - f(\theta_k + s^\ast)}{-m_k(s^\ast)} \\
\mu_{k+1} &= \begin{cases} \tfrac{1}{2}\mu_k & \text{if } \rho_k > \eta_2 \\ \tfrac{1}{2}\mu_k(1+\gamma_1) & \text{if } \eta_1 \le \rho_k \le \eta_2 \\ \tfrac{1}{2}\mu_k(\gamma_1+\gamma_2) & \text{otherwise} \end{cases}
\end{aligned}
$$

where $g_k = \nabla f(\theta_k)$, $B_k$ is the limited-memory SR1 Hessian approximation, $\mu_k > 0$ is the cubic regularization weight, $\Phi_k(s) = \lVert U_k^\top s \rVert_3$ is a shape-changing norm in the eigenbasis $U_k$ of $B_k$, $\rho_k$ is the ratio of actual to predicted reduction, and $\eta_1 \le \eta_2$, $\gamma_1, \gamma_2 > 1$ control step acceptance and regularization adjustment.

Reference: Aditya Ranganath, Mukesh Singhal, Roummel Marcia, "Symmetric Rank-One Quasi-Newton Methods for Deep Learning Using Cubic Regularization", arXiv 2025. https://arxiv.org/abs/2502.12298

---
[Back to the Canon](../index.md)
