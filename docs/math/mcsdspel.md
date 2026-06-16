# MCSD / SPEL

Implements MCSD / SPEL, manifold-constrained steepest descent via a norm-induced linear minimization oracle and projection back to the manifold.

Norm-constrained LMO-based optimizers such as spectral gradient descent and Muon extend awkwardly to manifold-constrained problems, usually requiring nested loops that solve tangent-space subproblems iteratively. MCSD is a single-loop alternative: at each step it picks a steepest-descent direction induced by a chosen norm by applying the LMO to the Riemannian gradient, takes a step, and returns to the manifold by projection. The stochastic variant smooths the gradient with momentum before the LMO. SPEL is the spectral-norm specialization on the Stiefel manifold, where the LMO and the projection both reduce to the matrix sign (polar) factor.

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + (1-\beta)\, g_t \\
\theta_{t+1} &= P_{\mathcal{M}}\!\left(\theta_t + \eta_t\, \mathrm{LMO}_{\|\cdot\|}\!\left(P_{T_{\theta_t}\mathcal{M}}(m_t)\right)\right) \\
\theta_{t+1}^{\,\mathrm{SPEL}} &= \mathrm{msign}\!\left(\theta_t - \eta_t\, \mathrm{msign}\!\left(\nabla_{\mathcal{M}} f(\theta_t)\right)\right)
\end{aligned}
$$

where $\theta_t$ are the parameters constrained to the manifold $\mathcal{M}$, $g_t$ is the stochastic gradient, $m_t$ its momentum estimate with decay $\beta \in [0,1)$, $\eta_t$ is the step size, $P_{T_{\theta_t}\mathcal{M}}$ projects onto the tangent space at $\theta_t$, $\nabla_{\mathcal{M}} f$ is the Riemannian gradient, $P_{\mathcal{M}}$ projects back onto the manifold, $\mathrm{LMO}_{\|\cdot\|}$ is the linear minimization oracle for the chosen norm, and $\mathrm{msign}(Y) = Y(Y^\top Y)^{-1/2}$ is the matrix sign (polar factor) onto the Stiefel manifold.

Reference: Kaiwei Yang, Lexiao Lai, "Manifold constrained steepest descent", ICML 2026. https://arxiv.org/abs/2601.21487

---
[Back to the Canon](../index.md)
