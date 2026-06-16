# Muon+

Implements Muon+, Muon augmented with one additional row/column normalization step after orthogonalization.

Muon updates matrix-shaped parameters by momentum followed by an approximate orthogonalization $\mathrm{Ortho}(\cdot)$, which Newton-Schulz iterations use to approach the $UV^\top$ factor of the SVD of the momentum buffer. Muon+ observes that the orthogonalized update can still have ill-balanced rows or columns, so it inserts an $\ell_2$ normalization $\mathrm{Norm}_{(d)}$ before applying the step. The normalization rescales each column (dividing by its column norm), each row (dividing by its row norm), or a composition of both, leaving the orthogonal structure intact while equalizing per-row or per-column scale.

$$
\begin{aligned}
m_t &= \mu\, m_{t-1} + (1-\mu)\, g_t \\
O_t &= \mathrm{Norm}_{(d)}\big(\mathrm{Ortho}(m_t)\big) \\
\theta_t &= \theta_{t-1} - \eta \sqrt{\tfrac{m}{n}}\; O_t
\end{aligned}
$$

where $\mathrm{Norm}_{(\mathrm{col})}(X) = X D_{\mathrm{col}}^{-1}$ with $D_{\mathrm{col}} = \mathrm{diag}\big(\|X_{:,1}\|_2, \dots, \|X_{:,n}\|_2\big)$, $\mathrm{Norm}_{(\mathrm{row})}(X) = D_{\mathrm{row}}^{-1} X$ with $D_{\mathrm{row}} = \mathrm{diag}\big(\|X_{1,:}\|_2, \dots, \|X_{m,:}\|_2\big)$, $\mathrm{Ortho}(\cdot)$ is the Newton-Schulz orthogonalization (5 iterations), $g_t$ is the gradient, $m_t$ the momentum buffer, $\mu$ the momentum coefficient, $\eta$ the learning rate, and $m \times n$ the shape of the parameter matrix.

Reference: Ruijie Zhang, Yequan Zhao, Ziyue Liu, Zhengyang Wang, Zheng Zhang, "Muon+: Towards Better Muon via One Additional Normalization Step", arXiv 2026. https://arxiv.org/abs/2602.21545

---
[Back to the Canon](../index.md)
