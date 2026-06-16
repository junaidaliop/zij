# Mousse

Implements Mousse, a curvature-aware variant of Muon that rectifies its geometry with Kronecker-factored preconditioning.

Muon orthogonalizes the momentum matrix with a matrix sign (Newton-Schulz iteration), implicitly assuming an isotropic geometry. Mousse instead estimates the curvature through Kronecker factors of the gradient covariance and whitens the momentum into that geometry before orthogonalizing, then unwhitens. The factors are trace-normalized for scale stability, tempered with a milder exponent than Shampoo, and the resulting direction is rescaled (grafted) to the Frobenius norm of the orthogonalized matrix.

For a weight matrix $\theta_t$ with gradient $g_t$:

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + g_t, \\
L_t &= \beta_2 L_{t-1} + (1-\beta_2)\, g_t g_t^\top, \qquad R_t = \beta_2 R_{t-1} + (1-\beta_2)\, g_t^\top g_t, \\
\tilde{L}_t &= \frac{\dim(L_t)}{\mathrm{Tr}(L_t)+\epsilon}\, L_t, \qquad \tilde{R}_t = \frac{\dim(R_t)}{\mathrm{Tr}(R_t)+\epsilon}\, R_t, \\
\bar{m}_t &= \mathrm{msign}\!\left( \tilde{L}_t^{-\alpha}\, m_t\, \tilde{R}_t^{-\alpha} \right), \\
u_t &= \tilde{L}_t^{\alpha}\, \bar{m}_t\, \tilde{R}_t^{\alpha}, \qquad \hat{u}_t = \frac{\lVert \bar{m}_t \rVert_F}{\lVert u_t \rVert_F}\, u_t, \\
\theta_{t+1} &= \theta_t - \eta\, \hat{u}_t.
\end{aligned}
$$

where $m_t$ is the momentum buffer with decay $\beta$, $L_t$ and $R_t$ are the left and right Kronecker factors of the gradient covariance with EMA decay $\beta_2$ (bias-corrected and refreshed every $T$ steps), $\mathrm{msign}(\cdot)$ is the matrix sign approximated by Newton-Schulz iteration, $\alpha=0.125$ is the spectral-tempering (curvature) exponent, $\epsilon$ stabilizes the trace normalization, and $\eta$ is the learning rate.

Reference: Yechen Zhang, Shuhao Xing, Junhao Huang, Kai Lv, Yunhua Zhou, Xipeng Qiu, Kai Chen, Qipeng Guo, "Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning", arXiv 2026. https://arxiv.org/abs/2603.09697

---
[Back to the Canon](../index.md)
