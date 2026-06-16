# Lotus

Implements Lotus, a memory-efficient low-rank gradient-projection trainer that adapts when to switch subspaces using the displacement of the unit gradient.

Lotus extends GaLore-style training, where the gradient $G_t$ of a weight matrix is projected into a low-rank subspace by an orthonormal matrix $P_t$, an Adam regularizer $\rho(\cdot)$ acts on the projected gradient, and the result is mapped back to full rank for the weight update. GaLore refreshes $P_t$ on a fixed period, paying a periodic SVD cost. Lotus replaces the fixed schedule and the exact SVD with two changes: $P_t$ is computed by randomized SVD of $G_t$, and the subspace is switched only when it stops aligning with the recent gradient flow.

The switching trigger is a path efficiency ratio $\rho_t$: the norm of the current subspace's projection of the accumulated unit gradients, divided by the norm of those accumulated unit gradients. When $\rho_t$ drops below a threshold $\gamma$ (the projected path has become inefficient) and a minimum dwell time has elapsed, a fresh subspace is drawn.

$$
\begin{aligned}
R_t &= P_t^{\top} G_t \\
\tilde{R}_t &= \rho(R_t) \\
\theta_t &= \theta_{t-1} - \eta\, P_t \tilde{R}_t \\
\hat{g}_t &= \frac{g_t}{\lVert g_t \rVert_2} \\
\rho_t &= \frac{\bigl\lVert \sum_{i=0}^{k-1} P_t\, \hat{g}_{t-i} \bigr\rVert_2}{\bigl\lVert \sum_{i=0}^{k-1} \hat{g}_{t-i} \bigr\rVert_2} \in [0,1] \\
P_t &= \begin{cases} \mathrm{randSVD}(G_t) & \rho_t < \gamma \ \text{and}\ t - t_{\mathrm{last}} \ge T_{\min} \\ P_{t-1} & \text{otherwise} \end{cases}
\end{aligned}
$$

where $\theta$ are the parameters reshaped into the weight matrix, $\eta$ is the learning rate, $G_t$ is the full gradient with unit-normalized form $\hat{g}_t$, $P_t$ is the orthonormal projection matrix, $R_t$ the projected low-rank gradient and $\tilde{R}_t$ its Adam-regularized form via $\rho(\cdot)$, $\rho_t$ the path efficiency ratio over the last $k$ unit gradients, $\gamma$ the switching threshold, $T_{\min}$ the minimum steps between switches since $t_{\mathrm{last}}$, and $\mathrm{randSVD}$ a randomized truncated SVD.

Reference: Tianhao Miao, Zhongyuan Bao, Lejun Zhang, "Lotus: Efficient LLM Training by Randomized Low-Rank Gradient Projection with Adaptive Subspace Switching", ICASSP 2026. https://arxiv.org/abs/2602.01233

---
[Back to the Canon](../index.md)
