# AuON

Implements AuON, a linear-time alternative to semi-orthogonal momentum updates.

AuON replaces the Newton-Schulz orthogonalization of Muon with an $O(n)$ elementwise transform: the momentum matrix is first normalized by its Frobenius norm, then rescaled by the root-mean-square of its hyperbolic cosine. The $\cosh$ map amplifies large entries, so heavy-tailed updates produce a larger RMS and thus stronger global shrinkage, yielding a spectrally contractive update ($\lVert U \rVert_2 < 1$) without any iterative matrix factorization. A per-matrix factor $\sqrt{\max(1, m/n)}$ decouples the step scale from the aspect ratio, as in Muon.

$$
\begin{aligned}
m_t &= (1-\beta)\, m_{t-1} + \beta\, g_t \\
\hat{g}_t &= \beta\, m_t + (1-\beta)\, g_t \quad \text{(Nesterov, else } \hat{g}_t = m_t) \\
G_t &= \frac{\hat{g}_t}{\lVert \hat{g}_t \rVert_F + 10^{-7}} \\
U_t &= \frac{G_t}{\sqrt{\tfrac{1}{N}\sum_i \cosh(G_t)_i^2} + 10^{-8}} \\
\theta_t &= \theta_{t-1} - \eta\, \sqrt{\max\!\big(1, \tfrac{m}{n}\big)}\; U_t
\end{aligned}
$$

where $\theta$ are the (matrix) parameters, $g_t$ the gradient, $m_t$ the momentum buffer with decay $\beta$, $\lVert \cdot \rVert_F$ the Frobenius norm, $\cosh(z) = (e^z + e^{-z})/2$ applied elementwise, $N = m \cdot n$ the number of entries with $m, n$ the matrix dimensions, $\eta$ the learning rate, and $\epsilon = 10^{-7}, 10^{-8}$ stability constants. A hybrid variant prepends a single Newton-Schulz iteration before the $\cosh$-RMS scaling.

Reference: Dipan Maity, "AuON: A Linear-time Alternative to Semi-Orthogonal Momentum Updates", arXiv 2025. https://arxiv.org/abs/2509.24320

---
[Back to the Canon](../index.md)
