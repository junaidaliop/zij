# Turbo-Muon

Implements Turbo-Muon, Muon with an almost-orthogonal-layer preconditioner that accelerates the Newton-Schulz orthogonalization.

Turbo-Muon keeps the Muon outer loop—a momentum buffer over the gradient whose orthogonalized form drives the weight update—but replaces the standard quintic Newton-Schulz polar-factor computation with a preconditioned variant. Before iterating, an AOL (almost-orthogonal-layer) step rescales the columns of the momentum matrix by the inverse square root of their absolute Gram-row sums, which guarantees $\lVert X_1 \rVert_2 \le 1$ and improves conditioning. The better-conditioned start lets the quintic iteration converge in four steps instead of the usual five, cutting the orthogonalization cost while preserving approximation quality.

$$
\begin{aligned}
M_t &= \beta M_{t-1} + g_t, \\
X_0 &= M_t, \quad A_0 = X_0^\top X_0, \\
s_i &= \Big( \textstyle\sum_j |A_0|_{ij} \Big)^{-1/2}, \\
X_1 &= X_0\, \mathrm{diag}(s), \quad A_1 = \mathrm{diag}(s)\, A_0\, \mathrm{diag}(s), \\
A_k &= X_k^\top X_k, \\
B_k &= b_k A_k + c_k A_k A_k, \\
X_{k+1} &= a_k X_k + X_k B_k, \quad k = 1,\dots,4, \\
O_t &= X_5, \\
\theta_t &= \theta_{t-1} - \eta\, O_t.
\end{aligned}
$$

where $M_t$ is the momentum buffer with decay $\beta$, $g_t$ the gradient, $A_0$ the Gram matrix of the momentum, $s$ the AOL column-scaling vector, $(a_k, b_k, c_k)$ the per-iteration quintic coefficients inherited from Muon, $O_t$ the orthogonalized update after the four preconditioned iterations, $\eta$ the learning rate, and $\theta$ the parameters.

Reference: Thibaut Boissin, Thomas Massena, Franck Mamalet, Mathieu Serrurier, "Turbo-Muon: Accelerating Orthogonality-Based Optimization with Pre-Conditioning", arXiv 2025. https://arxiv.org/abs/2512.04632

---
[Back to the Canon](../index.md)
