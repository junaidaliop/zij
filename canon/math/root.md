# ROOT

Implements ROOT, a robust orthogonalized optimizer that orthogonalizes the outlier-suppressed momentum.

ROOT builds on the matrix-orthogonalization idea of Muon but addresses two robustness gaps. First, instead of a fixed Newton–Schulz iteration, it uses an adaptive quintic iteration whose coefficients $a^{(m,n)}, b^{(m,n)}, c^{(m,n)}$ are tuned to the parameter shape $m \times n$, giving a more accurate orthogonalization across layers of differing dimensions. Second, before orthogonalizing, it applies element-wise soft-thresholding to the momentum to separate heavy-tailed outliers $O_t$ from a clipped robust component $B_t$; only $B_t$ is orthogonalized and used to update the weights, so a few large entries cannot dominate the resulting search direction.

$$
\begin{aligned}
M_t &= \mu\, M_{t-1} + g_t \\
O_t &= \mathcal{T}_\epsilon[M_t], \qquad \big(\mathcal{T}_\epsilon[x]\big)_i = \mathrm{sign}(x_i)\,\max(|x_i|-\epsilon,\,0) \\
B_t &= M_t - O_t \\
X_0 &= B_t / \lVert B_t \rVert_F \\
X_k &= a^{(m,n)} X_{k-1} + b^{(m,n)} X_{k-1}\big(X_{k-1}^\top X_{k-1}\big) + c^{(m,n)} X_{k-1}\big(X_{k-1}^\top X_{k-1}\big)^2 \\
\theta_t &= \theta_{t-1} - \eta\, X_K
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $M_t$ the momentum buffer, $\mu$ its decay, $\epsilon$ the soft-threshold, $\mathcal{T}_\epsilon$ the element-wise soft-thresholding (outlier) operator, $O_t$ the suppressed outliers, $B_t$ the robust clipped component, $\lVert\cdot\rVert_F$ the Frobenius norm, and $X_K$ the adaptive Newton iterate after $K$ steps (the orthogonalized direction $B_t^{\mathrm{orth}}$), with shape-dependent coefficients $a^{(m,n)}, b^{(m,n)}, c^{(m,n)}$ for a parameter matrix of size $m \times n$.

Reference: Wei He, Kai Han, Hang Zhou, Hanting Chen, Zhicheng Liu, Xinghao Chen, Yunhe Wang, "ROOT: Robust Orthogonalized Optimizer for Neural Network Training", 2025. https://arxiv.org/abs/2511.20626

---
[Back to the Canon](../README.md)
