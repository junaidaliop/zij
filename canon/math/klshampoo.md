# KL-Shampoo

Implements KL-Shampoo, a Shampoo variant whose Kronecker factors are fit by Kullback-Leibler minimization.

Shampoo and SOAP can be derived as estimating a structured (Kronecker-factored) preconditioner. KL-Shampoo recasts that estimation as a coupled KL-minimization problem, which couples the two factors: each factor is accumulated from the gradient outer product *whitened by the inverse of the other factor*, rather than from the raw outer product as in standard Shampoo. The preconditioned parameter update itself is unchanged, applying the inverse square roots of the two factors on either side of the matrix gradient.

$$
\begin{aligned}
S_a &\leftarrow (1-\beta_2)\, S_a + \frac{\beta_2}{d_b}\, G_t\, S_b^{-1} G_t^{\top} \\
S_b &\leftarrow (1-\beta_2)\, S_b + \frac{\beta_2}{d_a}\, G_t^{\top} S_a^{-1} G_t \\
\theta_t &\leftarrow \theta_{t-1} - \gamma\, S_a^{-1/2}\, G_t\, S_b^{-1/2}
\end{aligned}
$$

where $G_t$ is the matrix-valued gradient of shape $d_a \times d_b$, $S_a$ and $S_b$ are the left and right Kronecker factors (preconditioner covariances), $\beta_2$ is the factor decay rate, $\gamma$ is the learning rate, and the inverse opposite-factor terms $S_b^{-1}$ and $S_a^{-1}$ inside the accumulations are the KL-minimization coupling that distinguishes it from Shampoo's raw $G_t G_t^{\top}$ and $G_t^{\top} G_t$ updates.

Reference: Wu Lin, Scott C. Lowe, Felix Dangel, Runa Eschenhagen, Zikun Xu, Roger B. Grosse, "Understanding and Improving Shampoo and SOAP via Kullback-Leibler Minimization", arXiv 2025. https://arxiv.org/abs/2509.03378

---
[Back to the Canon](../README.md)
