# AdaDiag

Implements AdaDiag, an adaptive method that diagonalizes the preconditioner by rotating gradients into the singular-vector basis.

The diagonal second moment of Adam implicitly assumes the gradient's covariance is diagonal, which fails for the structured (matrix-shaped) gradients of neural layers. AdaDiag instead periodically computes a singular value decomposition of the gradient matrix $G_t = P_t \Sigma_t Q_t^{\top}$ and projects the gradient into the rotated basis $P_t^{\top} G_t$ before forming the moments. In that basis the gradient covariance is closer to diagonal, so the per-coordinate Adam statistics $M_t,V_t$ approximate a full preconditioner far better; the resulting update is rotated back into the original coordinates. A one-sided variant rotates only by $P_t$, while a two-sided variant rotates by both $P_t$ and $Q_t$. The SVD is recomputed every $T$ steps and the rotation matrices are reused in between.

$$
\begin{aligned}
P_t,\,\Sigma_t,\,Q_t^{\top} &= \mathrm{SVD}(G_t) \quad \text{every } T \text{ steps, else } P_t,Q_t \leftarrow P_{t-1},Q_{t-1} \\
\tilde G_t &= P_t^{\top} G_t \quad (\text{two-sided: } \tilde G_t = P_t^{\top} G_t\, Q_t) \\
M_t &= \beta_1 M_{t-1} + (1-\beta_1)\,\tilde G_t \\
V_t &= \beta_2 V_{t-1} + (1-\beta_2)\,\tilde G_t^{2} \\
W_{t+1} &= W_t - \eta_t\!\left( P_t\,\frac{M_t}{\sqrt{V_t}+\epsilon} + \lambda W_t \right) \quad (\text{two-sided: } P_t\,\tfrac{M_t}{\sqrt{V_t}+\epsilon}\,Q_t^{\top})
\end{aligned}
$$

where $W$ (i.e. $\theta$) are the layer's weight matrix, $G_t$ its gradient, $P_t,Q_t$ the left/right singular-vector rotation matrices, $\tilde G_t$ the projected gradient, $M_t,V_t$ the first- and second-moment estimates of the projected gradient (with bias-corrected decays $\beta_1,\beta_2$), $\eta_t$ the learning rate, $\lambda$ the decoupled weight decay, $\epsilon$ the stability constant, and $T$ the SVD recomputation period.

Reference: Son Nguyen The, Bo Liu, Lizhang Chen, Qiang Liu, "Improving Adaptive Moment Optimization via Preconditioner Diagonalization", 2025. https://arxiv.org/abs/2502.07488

---
[Back to the Canon](../index.md)
