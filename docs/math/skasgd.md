# SKA-SGD

Implements SKA-SGD, stochastic gradient descent accelerated by projection onto a low-dimensional Krylov subspace.

To handle poorly-conditioned problems, SKA-SGD keeps a sliding window of the last $s$ stochastic gradients and projects the current gradient onto the subspace they span. The projection coefficients solve a small regularized Gram system, which is computed by streaming Gauss-Seidel (equivalent to modified Gram-Schmidt) at $O(s^2)$ cost rather than forming the full system explicitly. The resulting direction replaces the raw gradient in an ordinary (optionally Nesterov-momentum) SGD step.

$$
\begin{aligned}
P_t &= [\, g_t,\ g_{t-1},\ \ldots,\ g_{t-s+1} \,] \\
\alpha_t &= (P_t^\top P_t + \lambda I)^{-1} P_t^\top g_t \\
d_t &= P_t\, \alpha_t \\
v_t &= \beta\, v_{t-1} - \eta\, d_t \\
\theta_t &= \theta_{t-1} + v_t
\end{aligned}
$$

where $g_t$ is the stochastic gradient, $P_t$ stacks the most recent $s$ gradients as a Krylov basis, $\alpha_t$ are the projection coefficients obtained by streaming Gauss-Seidel on the regularized Gram system, $\lambda$ is the regularization parameter, $d_t$ is the projected search direction, $\eta$ is the learning rate, $\beta$ is the momentum coefficient, and $v_t$ is the velocity.

Reference: Stephen Thomas, "Streaming Krylov-Accelerated Stochastic Gradient Descent", arXiv 2025. https://arxiv.org/abs/2505.07046

---
[Back to the Canon](../index.md)
