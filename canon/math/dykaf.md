# DyKAF

Implements DyKAF, an Adam-style update preconditioned by a dynamical Kronecker approximation of the Fisher information matrix.

DyKAF treats a matrix parameter $W \in \mathbb{R}^{m \times n}$ and approximates the empirical Fisher $F_t = \sum_{i \le t} \mathrm{vec}(G_i)\,\mathrm{vec}(G_i)^\top$ by a Kronecker product $F_t \approx L_t \otimes R_t$, with the factors $L_t \in \mathbb{R}^{m \times m}$ and $R_t \in \mathbb{R}^{n \times n}$ tracked over time by a low-rank projector-splitting integrator. Diagonalizing the factors as $L_t = Q_L \Lambda_L Q_L^\top$ and $R_t = Q_R \Lambda_R Q_R^\top$ rotates the gradient into the Fisher eigenbasis, where preconditioning reduces to a diagonal (Adam-like) rescaling before rotating back.

Each step rotates the momentum, divides it by a square-rooted second moment accumulated in the rotated basis, rotates back, and applies the step; the Kronecker factors and their eigenvectors $Q_L, Q_R$ are refreshed periodically.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \\
m_t' &= Q_L^\top\, m_t\, Q_R, \qquad g_t' = Q_L^\top\, g_t\, Q_R, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, (g_t' \odot g_t'), \\
n_t' &= m_t' \oslash \big(v_t^{\circ 1/2} + \epsilon\big), \\
n_t &= Q_L\, n_t'\, Q_R^\top, \\
\theta_t &= \theta_{t-1} - \eta\, n_t.
\end{aligned}
$$

where $\theta = W$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the momentum, $v_t$ the second moment formed in the rotated basis, $\beta_1, \beta_2$ the decay rates, $\epsilon$ the stability constant, $Q_L, Q_R$ the eigenvector matrices of the Kronecker factors $L_t, R_t$, and $\odot, \oslash, (\cdot)^{\circ 1/2}$ elementwise product, division, and square root.

Reference: Nikolay Yudin, Ekaterina Grishina, Andrey Veprikov, Alexandr Beznosikov, Maxim Rakhuba, "DyKAF: Dynamical Kronecker Approximation of the Fisher Information Matrix for Gradient Preconditioning", arXiv 2025. https://arxiv.org/abs/2511.06477

---
[Back to the Canon](../README.md)
