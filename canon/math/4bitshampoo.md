# 4-bit Shampoo

Implements 4-bit Shampoo, Shampoo with its preconditioners and their inverse roots stored at 4-bit precision.

The base method is Shampoo: per layer it keeps left and right second-order statistics $L_t, R_t$ and preconditions the gradient by their inverse fourth roots. To save memory, 4-bit Shampoo does not quantize the matrices $L_t, R_t$ directly. It eigendecomposes a symmetric matrix $A=U\Lambda U^{\top}$ and quantizes only the eigenvector matrix $U$ to 4-bit while keeping the eigenvalues $\Lambda$ in full precision; this keeps the quantized factor close to orthogonal, which the linear (preconditioner) state is far from. On dequantization the eigenvectors are re-orthogonalized by a few Björck iterations, and the inverse fourth root is rebuilt from the rectified eigenvectors and the eigenvalues. The preconditioned gradient is rescaled to match the original gradient norm and then consumed by a first-order optimizer (SGDM or AdamW).

$$
\begin{aligned}
L_t &= \beta L_{t-1} + (1-\beta)\, G_t G_t^{\top}, \qquad R_t = \beta R_{t-1} + (1-\beta)\, G_t^{\top} G_t \\
A &= U \Lambda U^{\top}, \qquad (\lambda, \bar U) = (\mathrm{diag}(\Lambda),\, Q(U)) \\
V^{(0)} &= D(\bar U), \qquad V^{(k)} = \tfrac{3}{2} V^{(k-1)} - \tfrac{1}{2} V^{(k-1)} (V^{(k-1)})^{\top} V^{(k-1)} \\
\hat A &= V\big(\Lambda + \max(\lambda)\,\epsilon\, I\big)^{-1/4} V^{\top}, \qquad (a, \bar A) = (\mathrm{diag}(\hat A),\, Q(\hat A - \mathrm{diag}(a))) \\
\hat G_t &= \hat L_t\, G_t\, \hat R_t, \qquad \tilde G_t = \hat G_t \cdot \frac{\lVert G_t \rVert_F}{\lVert \hat G_t \rVert_F} \\
\theta_t,\, s_t &= F(\theta_{t-1},\, s_{t-1},\, \tilde G_t)
\end{aligned}
$$

where $\theta$ are the parameters, $G_t$ the gradient (reshaped to a matrix), $\beta$ the statistics decay rate, $\epsilon$ a damping constant, $Q$ the 4-bit quantizer and $D$ its dequantizer, $V^{(k)}$ the Björck orthonormalization restoring the dequantized eigenvectors over $k$ iterations, $\hat L_t,\hat R_t$ the quantized inverse-fourth-root preconditioners obtained by applying the $\hat A$ construction to $L_t$ and $R_t$, $\tilde G_t$ the norm-rescaled preconditioned gradient, $s_t$ the inner first-order state, and $F$ the wrapped first-order optimizer (SGDM or AdamW) carrying the learning rate $\eta$ and weight decay $\lambda$.

Reference: Sike Wang, Pan Zhou, Jia Li, Hua Huang, "4-bit Shampoo for Memory-Efficient Network Training", NeurIPS 2024. https://arxiv.org/abs/2405.18144

---
[Back to the Canon](../README.md)
