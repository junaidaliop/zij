# SNSM

Implements SNSM, a memory-efficient optimizer pairing a subset-norm adaptive step size with subspace momentum.

SNSM combines two complementary state-reduction techniques. The subset-norm step size partitions the coordinates into subsets and shares a single AdaGrad-style accumulated norm across each subset, shrinking the second-moment state from $O(d)$ to $O(\sqrt{d})$. Subspace momentum keeps a momentum buffer only inside a low-dimensional subspace defined by a projection $P$ (the top-$k$ left singular vectors of the gradient, refreshed every few steps), while taking plain SGD steps in the orthogonal complement, so the momentum state shrinks to the subspace rank.

The two pieces are defined by the following updates. Subset-norm accumulates a per-subset squared norm and divides each coordinate by its subset's denominator; subspace momentum filters the projected gradient and adds back the orthogonal residual.

$$
\begin{aligned}
b_{t,i}^2 &= b_{t-1,i}^2 + \lVert g_{t,\Psi_i} \rVert^2, &\quad \theta_{t+1,j} &= \theta_{t,j} - \frac{\eta}{b_{t,\psi(j)}}\, g_{t,j} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, P g_t, &\quad r_t &= g_t - P^{*} P g_t \\
\theta_{t+1} &= \theta_t - \eta\,(P^{*} m_t + r_t)
\end{aligned}
$$

where $g_t$ is the gradient, $\Psi_i$ is the $i$-th coordinate subset with $\psi(j)$ mapping coordinate $j$ to its subset, $b_{t,i}$ is that subset's accumulated gradient norm, $m_t$ is the subspace momentum with decay $\beta_1$, $P$ projects onto the top-$k$ gradient singular subspace with adjoint $P^{*}$, $r_t$ is the orthogonal-complement residual, $\eta$ is the step size, and $\theta$ the parameters.

Reference: Thien Hang Nguyen, Huy Le Nguyen, "Lean and Mean Adaptive Optimization via Subset-Norm and Subspace-Momentum with Convergence Guarantees", ICML 2025 (PMLR v267). https://arxiv.org/abs/2411.07120

---
[Back to the Canon](../index.md)
