# FFT-based Subspace Selection

Implements FFT-based Subspace Selection, an SVD-free low-rank AdamW that projects gradients onto adaptively chosen columns of a fixed DCT basis.

Like GaLore, this method runs Adam in a low-rank subspace, but it replaces the costly per-layer SVD projection with a single predefined orthogonal Discrete Cosine Transform (DCT) matrix $Q$ (computable via FFT in $O(n^2 \log n)$). At each refresh it picks the $r$ DCT columns whose alignment with the current gradient is largest, so only $r$ integer indices are stored per layer rather than a dense projector. Because the subspace changes between refreshes, the first and second moments are rotated through $R = Q_{\mathrm{prev}}^{\top} Q_{\mathrm{crt}}$ to stay in the new coordinates, and an error-feedback buffer $\Xi_t$ accumulates the part of the gradient discarded by the projection.

$$
\begin{aligned}
G_t &= \nabla_\theta f(\theta_t) + \Xi_t, \qquad \mathcal{I}_{\mathrm{crt}} = \mathrm{RankCols}(G_t Q,\, r) \\
g_t &= G_t Q_{\mathrm{crt}}, \qquad \Xi_t = G_t - g_t Q_{\mathrm{crt}}^{\top} \\
R &= Q_{\mathrm{prev}}^{\top} Q_{\mathrm{crt}} \\
m_t &= \beta_1\, m_{t-1} R + (1-\beta_1)\, g_t \\
v_t &= \beta_2\, |v_{t-1} R| + (1-\beta_2)\, g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \eta_t\, \frac{\hat m_t}{\epsilon + \sqrt{\hat v_t}}\, Q_{\mathrm{crt}}^{\top}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $Q$ the fixed DCT matrix, $Q_{\mathrm{crt}}$ its $r$ columns selected at the current step by $\mathrm{RankCols}$ (top-$r$ by $L_1$ alignment with $G_t Q$), $g_t$ the projected gradient, $\Xi_t$ the error-feedback buffer holding the discarded component, $R$ the rotation carrying the moments from the previous subspace to the current one (identity except on refresh steps $t \bmod T_u = 0$), $m_t,v_t$ the first and second moments in the subspace, $\beta_1,\beta_2$ their decay rates, and $\epsilon$ the stability constant. The projection is applied on the right; a symmetric left-projection variant projects $Q_{\mathrm{crt}}^{\top} G_t$ instead.

Reference: Ionut-Vlad Modoranu, Mher Safaryan, Erik Schultheis, Max Ryabinin, Artem Chumachenko, Dan Alistarh, "FFT-based Dynamic Subspace Selection for Low-Rank Adaptive Optimization of Large Language Models", arXiv 2025. https://arxiv.org/abs/2505.17967

---
[Back to the Canon](../index.md)
