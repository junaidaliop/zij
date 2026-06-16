# Spectral Compact Training (SCT)

Implements Spectral Compact Training (SCT), a memory-efficient scheme that stores and trains weights directly in truncated SVD form.

Each linear weight $W \in \mathbb{R}^{m \times n}$ is replaced by a permanent rank-$k$ factorization $W = U \, \mathrm{diag}(s) \, V^\top$, and the dense matrix is never materialized. AdamW updates are applied to the compact factors $U$, $s$, $V$ using their own gradients, then $U$ and $V$ are retracted back onto the Stiefel manifold by a QR step with a sign correction $\mathrm{sign}(\mathrm{diag}(R))$ that keeps the columns orthonormal and the parameterization continuous.

$$
\begin{aligned}
W &= U \, \mathrm{diag}(s) \, V^\top, \\
(U, s, V) &\leftarrow \mathrm{AdamW}\big(U, s, V; \, \nabla_U \mathcal{L}, \nabla_s \mathcal{L}, \nabla_V \mathcal{L}, \, \eta\big), \\
Q_U, R_U &= \mathrm{QR}(U), \quad U \leftarrow Q_U \, \mathrm{sign}(\mathrm{diag}(R_U)), \\
Q_V, R_V &= \mathrm{QR}(V), \quad V \leftarrow Q_V \, \mathrm{sign}(\mathrm{diag}(R_V)).
\end{aligned}
$$

where $U \in \mathbb{R}^{m \times k}$ and $V \in \mathbb{R}^{n \times k}$ have orthonormal columns, $s \in \mathbb{R}^{k}$ holds the singular values, $\eta$ is the learning rate, $\mathcal{L}$ is the loss, $\mathrm{QR}(\cdot)$ is the QR decomposition, and $\mathrm{sign}(\mathrm{diag}(R))$ flips column signs to make the retraction unique.

Reference: Björn R. Kohlberger, "Spectral Compact Training: Pre-Training Large Language Models via Permanent Truncated SVD and Stiefel QR Retraction", arXiv preprint 2026. https://arxiv.org/abs/2604.00733

---
[Back to the Canon](../index.md)
