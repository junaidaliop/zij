# OptMuon

Implements OptMuon, a Muon-style orthogonalized momentum method with a closed-loop, self-normalized step-size schedule.

OptMuon keeps the Muon update structure -- the search direction is the polar factor $\mathrm{Orth}(M_t)$ of a momentum matrix $M_t$ -- but replaces the fixed or open-loop magnitude rule with a trajectory-dependent, AdaGrad-Norm-style coefficient. A lagged self-normalized coefficient $\alpha_t = \rho_{t-1}$ is built from the running gradient-norm history; its numerator carries a running maximum that compensates for occasional gradient spikes, so the step does not collapse after a single large gradient. Direction and magnitude are thus cleanly separated: the polar factor sets the direction, while the scalar $\theta\gamma_t\|M_t\|_F$ sets the magnitude.

The framework has two variants sharing the same orthogonalized template. Option A (average smoothness, $q=1/2$) accumulates a single stochastic gradient per step; Option I (individual smoothness, $q=2/3$) uses a STORM-type recursive momentum estimator with two gradients on the same mini-batch. The polar factor is computed exactly via SVD in the analysis and approximated by a few Newton-Schulz iterations in practice.

$$
\begin{aligned}
G_t &= \nabla f(X_t; \xi_t), \qquad g_t = \|G_t\|_F \\
\alpha_t &= \rho_{t-1}, \qquad \rho_t = \min_{1 \le k \le t} \left( \frac{1 + \max_{1 \le i \le k} g_i^2}{1 + \sum_{i=1}^{k} g_i^2} \right)^{q}, \qquad \rho_0 = 1 \\
\text{Option A:}\quad M_t &= G_t + (1 - \alpha_t) M_{t-1}, \qquad \gamma_t = \min\!\left\{ \alpha_t^2,\ \alpha_t \Big( 1 + \sum_{j=1}^{t} \alpha_j^2 \|M_j\|_F^2 \Big)^{-1/2} \right\} \\
\text{Option I:}\quad M_t &= (1 - \alpha_t)\big( M_{t-1} - \nabla f(X_{t-1}; \xi_t) \big) + G_t, \qquad \gamma_t = \min\!\left\{ \sqrt{\alpha_t},\ \Big( 1 + \sum_{j=1}^{t} \alpha_j^{-1/2} \|M_j\|_F^2 \Big)^{-1/2} \right\} \\
X_{t+1} &= X_t - \theta \gamma_t \|M_t\|_F\, \mathrm{Orth}(M_t)
\end{aligned}
$$

where $X$ are the matrix parameters, $\theta$ the learning rate, $G_t$ the stochastic gradient with norm $g_t$, $M_t$ the momentum matrix, $\alpha_t$ the lagged self-normalized coefficient, $\gamma_t$ the closed-loop scalar step factor, $q$ the smoothness-regime exponent ($1/2$ for average, $2/3$ for individual smoothness), and $\mathrm{Orth}(M) = U W^\top$ the polar factor from the thin SVD $M = U \Sigma W^\top$ (with $\mathrm{Orth}(0) = 0$).

Reference: Ganzhao Yuan, "OptMuon: Closed-Loop Orthogonalized Momentum Methods for Stochastic Optimization with Zero-Noise Optimality", arXiv 2026. https://arxiv.org/abs/2606.08783

---
[Back to the Canon](../index.md)
