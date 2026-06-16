# ASGO

Implements ASGO (Adaptive Structured Gradient Optimization), a one-sided full-matrix adaptive method for matrix-shaped parameters.

ASGO treats the gradient of a weight matrix as a structured object rather than a flat vector. Instead of a diagonal preconditioner over individual entries (as in Adam), it accumulates the full outer product $G_t G_t^\top$ to capture row-space curvature, then preconditions the gradient on the left by the inverse square root of this matrix. This is the single-sided analogue of Shampoo: one preconditioner of the parameter's row dimension rather than two, which keeps the curvature estimate full-matrix along one axis while staying cheaper than a two-sided method.

$$
\begin{aligned}
G_t &= \nabla_W f(W_t) \\
V_t &= V_{t-1} + G_t G_t^\top \\
\Lambda_t &= V_t^{1/2} + \epsilon I \\
W_{t+1} &= W_t - \eta_t\, \Lambda_t^{-1} G_t
\end{aligned}
$$

where $W_t \in \mathbb{R}^{m \times n}$ is the parameter matrix, $G_t$ its gradient, $V_t \in \mathbb{R}^{m \times m}$ the accumulated left preconditioner from the gradient outer products, $V_t^{1/2}$ the matrix square root, $\Lambda_t^{-1}$ the inverse preconditioner, $\eta_t$ the learning rate, $\epsilon$ a stability constant, and $I$ the identity. The practical variant replaces the plain accumulations with exponential moving averages (momentum $M_t = \beta_1 M_{t-1} + (1-\beta_1) G_t$ and $V_t = \beta_2 V_{t-1} + (1-\beta_2) G_t^\top G_t$), recomputes the inverse square root every $\tau$ steps for efficiency, and applies the preconditioner on the right.

Reference: Kang An, Yuxing Liu, Rui Pan, Shiqian Ma, Donald Goldfarb, Tong Zhang, "ASGO: Adaptive Structured Gradient Optimization", arXiv 2025. https://arxiv.org/abs/2503.20762

---
[Back to the Canon](../index.md)
