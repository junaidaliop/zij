# SUMO

Implements SUMO, subspace-aware moment orthogonalization for memory-efficient LLM training.

SUMO trains in a low-rank subspace: every $K$ steps it computes an orthonormal basis $Q_t$ for the top-$r$ subspace of the gradient (via truncated randomized SVD), projects the gradient and carries the momentum across subspace changes, and accumulates a first-order moment inside that subspace. The projected moment is orthogonalized exactly through its SVD, $m_t = U \Sigma V^\top \Rightarrow O_t = U V^\top$, replacing Muon's Newton–Schulz approximation and avoiding its accumulated error. The orthogonalized update is mapped back to the full space by $Q_t$ before the weight step, with optional norm-growth clipping and decoupled weight decay.

$$
\begin{aligned}
Q_t &= \mathrm{TruncatedSVD}_r(g_t), \quad m_{t-1} \leftarrow (Q_t^\top Q_{t-1})\, m_{t-1} && \text{(every } K \text{ steps)} \\
\hat{g}_t &= Q_t^\top g_t \\
m_t &= \mu\, m_{t-1} + \hat{g}_t \\
O_t &= U V^\top, \quad \text{where } m_t = U \Sigma V^\top \\
\theta_t &= \theta_{t-1} - \alpha\, \eta\, Q_t O_t - \eta\, \lambda\, \theta_{t-1}
\end{aligned}
$$

where $\theta$ are the weights, $g_t$ the full gradient, $Q_t$ the orthonormal subspace basis of rank $r$, $\hat{g}_t$ the projected gradient, $m_t$ the in-subspace first moment, $\mu$ the momentum coefficient, $O_t$ the SVD-orthogonalized direction, $\eta$ the learning rate, $\alpha$ a scale factor, and $\lambda$ the weight-decay coefficient.

Reference: Yehonathan Refael, Guy Smorodinsky, Tom Tirer, Ofir Lindenbaum, "SUMO: Subspace-Aware Moment-Orthogonalization for Accelerating Memory-Efficient LLM Training", arXiv 2025. https://arxiv.org/abs/2505.24749

---
[Back to the Canon](../index.md)
