# Dion

Implements Dion, communication-efficient distributed training via amortized low-rank orthogonalization of the momentum matrix.

Dion targets the same orthogonalized-momentum update as Muon but makes it scalable across many devices. Each parameter matrix keeps an error-feedback buffer $M_t$ that accumulates the gradient. A single step of power iteration, warm-started from the right factor $Q_{t-1}$ carried over between steps, extracts a rank-$r$ approximation $P_t R_t^\top$ of the buffer; the orthonormalized left factor $P_t$ together with the column-normalized right factor gives a cheap stand-in for the full orthogonalization. The portion of the buffer that the rank-$r$ update consumes is removed via error feedback (scaled by $1-\mu$), so the residual energy is retained for later steps. Only the low-rank factors need to be communicated, making the per-step cost independent of the full matrix dimensions.

$$
\begin{aligned}
B_t &= M_{t-1} + G_t \\
P_t &= \mathrm{Orthogonalize}(B_t\, Q_{t-1}) \\
R_t &= B_t^\top P_t \\
M_t &= B_t - (1-\mu)\, P_t R_t^\top \\
Q_t &= \mathrm{ColumnNormalize}(R_t) \\
X_t &= X_{t-1} - \eta\, P_t Q_t^\top
\end{aligned}
$$

where $X_t$ is the parameter matrix, $G_t$ its gradient, $M_t$ the error-feedback momentum buffer, $B_t$ the buffer plus current gradient, $P_t \in \mathbb{R}^{m\times r}$ and $R_t \in \mathbb{R}^{n\times r}$ the low-rank factors from one power-iteration step, $Q_t$ the column-normalized right factor reused as the next warm start, $r$ the rank, $\mu$ the momentum decay, and $\eta$ the learning rate. $\mathrm{Orthogonalize}$ returns an orthonormal column basis (QR) and $\mathrm{ColumnNormalize}$ rescales each column to unit norm. The reported experiments additionally multiply the update by a $\sqrt{m/n}$ scale factor for transfer across model sizes; this scalar sits outside the core Algorithm 1 update above.

Reference: Kwangjun Ahn, Byron Xu, "Dion: A Communication-Efficient Optimizer for Large Models", arXiv 2025. https://arxiv.org/abs/2504.05295

---
[Back to the Canon](../README.md)
