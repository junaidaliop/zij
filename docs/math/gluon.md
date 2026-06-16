# Gluon

Implements Gluon, a layer-wise LMO-based optimizer that performs norm-constrained steepest descent with momentum.

Gluon casts each layer's update as a linear minimization oracle (LMO) over a norm ball centered at the current iterate. The gradient is first smoothed into a momentum buffer $M_t$, and the new parameters are obtained by minimizing the inner product with $M_t$ over a ball of radius $t$ in the layer's chosen norm $\|\cdot\|$. This recovers Muon when the spectral norm $\|\cdot\|_{2\to2}$ is used: the LMO returns the orthogonal factor $UV^\top$ from the SVD of the (momentum) gradient, and the parameters move along that direction. The framework unifies Muon and Scion as special cases and supplies layer-wise adaptive step sizes derived from generalized smoothness.

$$
\begin{aligned}
M_t &= \beta\, M_{t-1} + (1-\beta)\, g_t, \\
\theta_t &= \mathrm{LMO}_{\mathcal{B}_t}(M_t) = \arg\min_{\theta \in \mathcal{B}_t} \langle M_t, \theta\rangle, \qquad \mathcal{B}_t = \{\theta : \|\theta - \theta_{t-1}\| \le t_k\}, \\
\theta_t &= \theta_{t-1} - t_k\, U_t V_t^\top, \qquad M_t = U_t \Sigma_t V_t^\top \quad (\text{spectral norm}).
\end{aligned}
$$

where $g_t$ is the (stochastic) gradient for the layer, $M_t$ is the momentum buffer, $\beta \in [0,1)$ the momentum coefficient, $t_k > 0$ the adaptive trust-region radius (step size), $\|\cdot\|$ the layer-specific norm, $\mathcal{B}_t$ the norm ball around the current iterate, and $U_t \Sigma_t V_t^\top$ the SVD of $M_t$ for matrix layers under the spectral norm.

Reference: Artem Riabinin, Egor Shulgin, Kaja Gruntkowska, Peter Richtárik, "Gluon: Making Muon & Scion Great Again! (Bridging Theory and Practice of LMO-based Optimizers for LLMs)", 2025. https://arxiv.org/abs/2505.13416

---
[Back to the Canon](../index.md)
