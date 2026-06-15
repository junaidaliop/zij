# Optimal Projection-Free Adaptive SGD

Implements Optimal Projection-Free Adaptive SGD, an accelerated adaptive matrix optimizer that replaces projections with a dual (FTRL) step.

The method ("Generalized Leon with Nesterov Acceleration") accumulates raw stochastic gradients into a dual variable $m_k$ and recovers the primal iterate through the gradient of a conjugate regularizer built from a trace norm. Adaptivity comes from a matrix preconditioner $S_k$ that accumulates the symmetric rank-one outer products of successive gradient differences, so the geometry tracks the curvature seen along the trajectory. Constraints are handled implicitly by the dual map rather than by an explicit projection, and a Nesterov-style averaging produces the accelerated reference point $\bar{x}_k$.

$$
\begin{aligned}
m_k &= m_{k-1} + g_k, \qquad g_k = \nabla_{\xi_k} f_k(x_k) \\
x_{k+1} &= -\nabla \Psi_k^*(m_k), \qquad \Psi_k^*(m) = \eta \,\big\| \sqrt{\,\mathrm{proj}_{\mathcal{H}}(\mathrm{out}(m)) + S_k}\,\big\|_{\mathrm{tr}} \\
\bar{x}_{k+1} &= \tfrac{1}{\alpha_k}\, x_{k+1} + \big(1 - \tfrac{1}{\alpha_k}\big)\, \bar{x}_k, \qquad \alpha_k = 1 + \tfrac{k}{2} \\
\tilde{g}_{k+1} &= \nabla_{\xi'_{k+1}} f_k(x_{k+1}) \\
S_{k+1} &= S_k + \mathrm{proj}_{\mathcal{H}}\big(\mathrm{out}(\tilde{g}_{k+1} - g_k)\big)
\end{aligned}
$$

where $\mathrm{out}(z) = z\langle z, \cdot\rangle$ is the symmetric rank-one operator, $\mathrm{proj}_{\mathcal{H}}$ is the orthogonal projection onto the admissible subspace $\mathcal{H}$ of self-adjoint operators, $\|A\|_{\mathrm{tr}} = \mathrm{tr}(\sqrt{A A^*})$ is the trace norm, $\eta$ is the dual scale (set to the constraint radius $\mathcal{R}$), $S_0 = \delta^2 I$ is the preconditioner with stabilization $\delta$, and $f_k(x) = \alpha_k^2 f\big(x/\alpha_k + (1 - 1/\alpha_k)\bar{x}_k\big)$ is the rescaled objective driving the accelerated update.

Reference: Dmitry Kovalev, "Optimal Projection-Free Adaptive SGD for Matrix Optimization", arXiv 2026. https://arxiv.org/abs/2604.02505

---
[Back to the Canon](../README.md)
