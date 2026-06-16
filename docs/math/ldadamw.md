# LDAdamW

Implements LDAdam, an Adam variant that keeps its moment estimates in an
adaptively tracked low-dimensional subspace.

With $P_t$ the rank-$r$ orthogonal projection tracked by one
round of block power iteration per step (initialized from a truncated SVD
of the first gradient), $Q_t = P_t^\top P_{t-1}$ the change-of-basis
matrix, and hats denoting Adam bias correction:


$$
\begin{aligned}
a_t &= g_t + e_t \\
b_t &= \rho\, P_{t-1} \hat{m}_{t-1} + (1 - \rho)\, a_t \\
P_t &= \mathrm{QR}\big(b_t b_t^\top P_{t-1}\big), \qquad
\tilde{g}_t = P_t^\top a_t \\
\tilde{m}_{t-1} &= Q_t m_{t-1} \\
\tilde{v}_{t-1} &= (1 - \beta_2^{t-1})\,
    \big| (Q_t \circ Q_t)\big(\hat{v}_{t-1} - \hat{m}_{t-1}^{\,2}\big)
    + (Q_t \hat{m}_{t-1})^2 \big| \\
m_t &= \beta_1 \tilde{m}_{t-1} + (1 - \beta_1)\, \tilde{g}_t \\
v_t &= \beta_2 \tilde{v}_{t-1} + (1 - \beta_2)\, \tilde{g}_t^{\,2} \\
\theta_{t+1} &= (1 - \eta\lambda)\, \theta_t
    - \eta\, P_t\, \hat{m}_t / \big(\sqrt{\hat{v}_t} + \epsilon\big) \\
e_{t+1} &= a_t - P_t \tilde{g}_t
    + \tfrac{\beta_1}{1 - \beta_1}\big(P_{t-1} m_{t-1} - P_t \tilde{m}_{t-1}\big)
\end{aligned}
$$

where $\rho$ interpolates between the past momentum and the fresh
gradient when adapting the subspace, $\lambda$ is the decoupled
weight decay, and $e_t$ is the generalized error feedback buffer
that captures both gradient and optimizer-state compression error. The
equations are written for left projection; tall matrices (more rows than
columns) are projected from the right, wide matrices from the left,
mirroring the update.


**Note:** Low-rank compression applies to 2D parameters only. Group parameters and set `enable_lowrank=False` on groups holding biases, norms, and embeddings, which then take plain AdamW steps; the upstream experiments enable low-rank only for linear-layer weights. With `error_feedback=True`, gradients of low-rank parameters double as the error buffer: `zero_grad()` leaves them in place, and gradient accumulation adds onto the buffer as the algorithm expects.

Reference: Thomas Robert, Mher Safaryan, Ionut-Vlad Modoranu, Dan Alistarh,
"LDAdam: Adaptive Optimization from Low-Dimensional Gradient Statistics",
ICLR 2025.
https://arxiv.org/abs/2410.16103

---
[Back to the Canon](../index.md)
