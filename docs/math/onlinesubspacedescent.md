# Online Subspace Descent

Implements Online Subspace Descent, a memory-efficient training method that projects gradients into a low-rank subspace whose projection matrix is itself learned online.

Like GaLore, the weight matrix is updated through a low-rank projection $P_t$, so the optimizer state (e.g. Adam moments) lives in the reduced $k$-dimensional subspace rather than the full parameter space. Unlike GaLore, the projection matrix is not recomputed by periodic SVD; instead $P_t$ is updated every step by one optimizer step on an online PCA objective that tracks the current gradient. This continuous, dynamics-based update admits a Hamiltonian descent interpretation that guarantees convergence to stationary points for arbitrary smooth choices of the projection dynamics.

$$
\begin{aligned}
\hat{g}_t &= P_t^\top g_t \\
(\hat{\Delta}_t, \hat{S}_t) &= \mathrm{Optimizer}_W(\hat{g}_t, \hat{S}_{t-1}) \\
\theta_{t+1} &= \theta_t + \eta^W_t \left( P_t \hat{\Delta}_t - \lambda^W \theta_t \right) \\
\tilde{g}_t &= g_t / \lVert g_t \rVert \\
L_t(P) &= \lVert P P^\top \tilde{g}_t - \tilde{g}_t \rVert^2 + \lambda \lVert P^\top P - I_{k\times k} \rVert^2 \\
(\Delta^P_t, S^P_t) &= \mathrm{Optimizer}_P\!\left( \nabla_P L_t(P_t), S^P_{t-1} \right) \\
P_{t+1} &= P_t + \eta^P_t \left( \Delta^P_t - \lambda^P P_t \right)
\end{aligned}
$$

where $\theta$ is a weight matrix in $\mathbb{R}^{n\times m}$, $g_t = \nabla_\theta L(\theta_t)$ its gradient, $P_t \in \mathbb{R}^{n\times k}$ ($k \ll n$) the learned projection, $\hat{g}_t$ the projected gradient, $\hat{\Delta}_t$ the subspace optimizer update with state $\hat{S}_t$, $\Delta^P_t$ the optimizer update for $P$ with state $S^P_t$, $\eta^W_t,\eta^P_t$ the learning rates, and $\lambda^W,\lambda^P,\lambda$ weight-decay / orthogonality coefficients. The authors recommend Adam for both optimizers with $\eta^P_t = \alpha\,\eta^W_t$ (e.g. $\alpha=5$) and $\lambda^W=\lambda^P$.

Reference: Kaizhao Liang, Bo Liu, Lizhang Chen, Qiang Liu, "Memory-Efficient LLM Training with Online Subspace Descent", NeurIPS 2024. https://arxiv.org/abs/2408.12857

---
[Back to the Canon](../index.md)
