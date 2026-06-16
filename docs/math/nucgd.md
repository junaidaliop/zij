# NucGD

Implements NucGD, normalized steepest descent under the nuclear norm.

NucGD treats the gradient as a matrix and takes the steepest-descent step whose nuclear norm is bounded by the step size. By nuclear–spectral duality, that optimal step is the rank-one matrix built from the top singular vector pair of the momentum matrix, so each update concentrates on the single dominant singular direction.

Because every step is rank one, accumulating updates places mass on a few dominant directions, which biases the iterates toward low-rank solutions without explicit regularization. The momentum matrix $m_t$ is decomposed by SVD (or an SVD-free power iteration) and only its leading factors are used.

$$
\begin{aligned}
m_t &= \mu\, m_{t-1} + (1 - \mu)\, g_t \\
U_t \Sigma_t V_t^\top &= \mathrm{SVD}(m_t) \\
\theta_{t+1} &= \theta_t - \gamma\, u_1 v_1^\top
\end{aligned}
$$

where $\theta$ are the (matrix-shaped) parameters, $g_t$ the gradient, $m_t$ the momentum, $\mu \in [0,1)$ the momentum weight, $\gamma$ the step size, and $u_1, v_1$ the top left and right singular vectors of $m_t$, so that $\gamma\, u_1 v_1^\top \in \arg\max_{\lVert \Delta \rVert_* \le \gamma} \langle m_t, \Delta \rangle$ with $\lVert \cdot \rVert_*$ the nuclear norm.

Reference: Shengping Xie, Zekun Wu, Quan Chen, Kaixu Tang, "Towards The Implicit Bias on Multiclass Separable Data under Norm Constraints", arXiv 2026. https://arxiv.org/abs/2603.22824

---
[Back to the Canon](../index.md)
