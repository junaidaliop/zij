# MoFaSGD

Implements MoFaSGD, memory-efficient training via low-rank factorization of the momentum.

Instead of storing a full-size momentum buffer, MoFaSGD keeps a rank-$r$ SVD of the exponentially averaged first moment, $\hat{M}_t = U_{t+1}\,\mathrm{Diag}(\sigma_{t+1})\,V_{t+1}^\top \approx \beta\hat{M}_{t-1} + (1-\beta)G_t$, and updates each parameter matrix using only the orthogonal factors $U_{t+1}V_{t+1}^\top$. This spectrally normalizes the step (it uses the singular vectors, not the singular-value magnitudes), so the cost is dominated by the rank-$r$ momentum buffer rather than a dense one.

The factors are refreshed each step by the UMF (update momentum factor) routine, which projects the new gradient onto the current tangent space, augments the bases via QR, assembles a small $2r\times 2r$ matrix $S_t$, and takes its rank-$r$ SVD:

$$
\begin{aligned}
(U_t', R_{U_t}) &= \mathrm{QR}\!\left(\left[\,U_t \;\; G_t V_t\,\right]\right), \\
(V_t', R_{V_t}) &= \mathrm{QR}\!\left(\left[\,V_t \;\; G_t^\top U_t\,\right]\right), \\
S_t &= R_{U_t}\begin{bmatrix} \beta\Sigma_t - U_t^\top G_t V_t & I_r \\ I_r & 0_r \end{bmatrix} R_{V_t}^\top, \\
U_t''\,\Sigma_t''\,(V_t'')^\top &= \mathrm{SVD}_r(S_t), \\
U_{t+1} &= U_t' U_t'', \quad V_{t+1} = V_t' V_t'', \quad \Sigma_{t+1} = \Sigma_t'', \\
W_{t+1} &= W_t - \eta\, U_{t+1} V_{t+1}^\top.
\end{aligned}
$$

where $W_t$ is the parameter matrix, $\eta$ the learning rate, $\beta$ the momentum decay, $G_t$ the gradient, $r$ the rank, $U_t,V_t$ the orthogonal momentum factors with singular values $\Sigma_t = \mathrm{Diag}(\sigma_t)$, $\mathrm{QR}$ and $\mathrm{SVD}_r$ the QR and rank-$r$ truncated SVD, and $I_r, 0_r$ the $r\times r$ identity and zero blocks.

Reference: Pouria Mahdavinia, Mehrdad Mahdavi, "Low-rank Momentum Factorization for Memory Efficient Training", arXiv 2025. https://arxiv.org/abs/2507.08091

---
[Back to the Canon](../index.md)
