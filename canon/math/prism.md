# PRISM

Implements PRISM, a matrix-aware optimizer that orthogonalizes an anisotropically shaped momentum matrix.

PRISM extends the Muon idea of orthogonalizing the momentum before stepping. It tracks an EMA momentum $M_t$ and an innovation term $D_t = G_t - M_t$ (the part of the gradient the momentum failed to predict). The two are stacked into an augmented matrix $\tilde M_t = [\,M_t;\ \gamma D_t\,]$, whose polar factor blends the smoothed direction with a damped correction. Orthogonalizing this stacked matrix and keeping the rows corresponding to the original parameter block yields the update direction, which is equivalent to preconditioning by $(M_t^\top M_t + \gamma^2 D_t^\top D_t)^{-1/2}$.

$$
\begin{aligned}
G_t &= \nabla_\theta \mathcal{L}(\theta_{t-1}) \\
M_t &= \beta\, M_{t-1} + (1-\beta)\, G_t \\
D_t &= G_t - M_t \\
\tilde M_t &= [\,M_t;\ \gamma D_t\,] \in \mathbb{R}^{2m \times n} \\
\tilde O_t &= \mathrm{polar}(\tilde M_t), \qquad O_t = \tilde O_t[{:}m,\, :] \\
\theta_t &= \theta_{t-1} - \eta\, O_t
\end{aligned}
$$

where $\theta$ are the (matrix-shaped) parameters, $\eta$ the learning rate, $G_t$ the gradient, $M_t$ the momentum, $D_t$ the innovation, $\beta$ the momentum coefficient, $\gamma$ the damping coefficient that scales the innovation block, $\tilde M_t$ the vertically stacked augmented matrix, and $\mathrm{polar}(\cdot)$ the orthogonal polar factor $UV^\top$ of $\tilde M_t = U S V^\top$ (computed via Newton-Schulz). $O_t = \tilde O_t[{:}m,:]$ keeps the top $m$ rows matching the original momentum block.

Reference: Yujie Yang, "PRISM: Structured Optimization via Anisotropic Spectral Shaping", arXiv 2026. https://arxiv.org/abs/2602.03096

---
[Back to the Canon](../README.md)
