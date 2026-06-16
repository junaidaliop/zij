# MUD

Implements MUD (MomentUm Decorrelation), a triangular whitening surrogate for Muon's polar update on matrix-shaped parameters.

MUD targets the 2D weight matrices of a transformer (other parameters such as embeddings and biases use AdamW). It keeps a momentum buffer with a Nesterov-style lookahead, then replaces Muon's orthogonalization with a cheaper Cholesky-like decorrelation: the lookahead matrix is row-normalized, its Gram matrix is formed, the lower triangle is solved against the rows, and the result is renormalized. Repeating this for a small number of passes $p$ (typically one) approximately whitens the update at a fraction of Muon's cost. The whitened direction is then applied with a shape-dependent scale and decoupled weight decay.

$$
\begin{aligned}
V_t &= \beta V_{t-1} + G_t, \\
M_t &= G_t + \beta V_t, \\
Q^{(0)} &= M_t, \\
\text{for } j = 1,\dots,p:\quad
Q &\leftarrow \mathrm{diag}(r + \epsilon)^{-1} Q, \quad r_i = \lVert Q_{i,:} \rVert_2, \\
\mathcal{G} &\leftarrow Q Q^\top, \\
T &\leftarrow \mathrm{tril}(\mathcal{G}), \\
Q &\leftarrow T^{-1} Q, \\
Q &\leftarrow \mathrm{diag}(r + \epsilon)^{-1} Q, \quad r_i = \lVert Q_{i,:} \rVert_2, \\
\theta_{t+1} &= (1 - \eta \lambda)\, \theta_t - \eta\, s(\theta)\, Q_t .
\end{aligned}
$$

where $\theta$ are the matrix parameters, $G_t$ the gradient, $V_t$ the momentum buffer, $M_t$ the Nesterov lookahead direction, $\beta$ the momentum coefficient, $\eta$ the learning rate, $\lambda$ the decoupled weight decay, $\epsilon$ a stability constant, $\mathrm{tril}(\cdot)$ the lower-triangular part, $p$ the number of decorrelation passes, $Q_t$ the whitened update after $p$ passes, and $s(\theta) = 0.2\sqrt{\max(n,m)}$ a shape-dependent scale for an $n \times m$ matrix.

Reference: Ben S. Southworth, Stephen Thomas, "Beyond Muon: MUD (MomentUm Decorrelation) for Faster Transformer Training", arXiv 2026. https://arxiv.org/abs/2603.17970

---
[Back to the Canon](../index.md)
