# DP-Muon

Implements DP-Muon, a differentially private variant of Muon that clips and noises per-sample matrix gradients before orthogonalizing the momentum.

DP-Muon adapts the Gaussian mechanism to Muon's matrix-aware update. For each weight matrix $W$ it clips every per-sample gradient in Frobenius norm to a threshold $C_W$, averages the batch, and adds Gaussian noise calibrated to $C_W$, giving a differentially private stochastic gradient. This noisy gradient feeds a heavy-ball momentum buffer, and the momentum matrix is then orthogonalized by a Newton–Schulz iteration (after orienting it to have no more rows than columns) so that the applied step has near-orthogonal singular structure.

$$
\begin{aligned}
H_t(z) &= g_t(z)\,\big/\,\max\!\Big(1,\ \tfrac{\lVert g_t(z)\rVert_F}{C_W}\Big), \quad g_t(z) = \nabla_W \ell(\theta_{t-1};z) \\
\tilde g_t &= \frac{1}{B}\sum_{z\in\mathcal{B}_t} H_t(z) + Z_t, \qquad Z_t \sim \mathcal{N}\!\Big(0,\ \tfrac{\sigma^2 C_W^2}{B^2}\, I\Big) \\
m_t &= \beta\, m_{t-1} + \tilde g_t \\
Y_0 &= \frac{\mathcal{T}(m_t)}{\max\!\big(1,\ \lVert \mathcal{T}(m_t)\rVert_F\big)}, \qquad Y_j = p_\kappa\!\big(Y_{j-1} Y_{j-1}^\top\big)\, Y_{j-1}, \quad j=1,\dots,q \\
O_t &= \mathcal{T}^{-1}(Y_q), \qquad \theta_t = \theta_{t-1} - \eta\, O_t
\end{aligned}
$$

where $g_t(z)$ is the per-sample gradient of the loss $\ell$, $C_W$ the Frobenius clipping threshold, $B$ the batch size, $\sigma$ the noise multiplier, $\beta$ the momentum decay, $\mathcal{T}$ the operator that transposes $W$ iff it has more rows than columns (so $\mathcal{T}(W)$ has fewer rows than columns) with inverse $\mathcal{T}^{-1}$, $p_\kappa(\lambda)=\sum_{s=0}^{\kappa}\frac{(2s)!}{4^s (s!)^2}(1-\lambda)^s$ the degree-$\kappa$ Newton–Schulz polynomial applied for $q$ iterations, $O_t$ the orthogonalized update, and $\eta$ the learning rate.

Reference: Jihwan Kim, Chenglin Fan, "DP-Muon: Differentially Private Optimization via Matrix-Orthogonalized Momentum", arXiv 2026. https://arxiv.org/abs/2605.12994

---
[Back to the Canon](../index.md)
