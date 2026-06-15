# Spectral Sphere Optimizer (SSO)

Implements Spectral Sphere Optimizer (SSO), a constrained matrix optimizer that keeps each 2D weight on a spectral-norm sphere.

SSO maintains the spectral norm of every weight matrix at a fixed radius $R=\sqrt{d_\mathrm{out}/d_\mathrm{in}}$ (the spectral $\mu$P scale) so that activation magnitudes stay invariant across width. The momentum is Frobenius-normalized and orthogonalized via the matrix sign function $\mathrm{msign}$, while a Lagrange multiplier $\lambda_t$ removes the component of the step that would leave the sphere along the top singular direction $\Theta_t=u_t v_t^\top$. After a retraction back onto the sphere, the orthogonalized direction is applied with the $\mu$P scaling.

$$
\begin{aligned}
M_t &= \beta M_{t-1} + (1-\beta)\, G_t, \\
\widehat{M}_t &= M_t / \lVert M_t \rVert_F, \\
(\sigma_t, u_t, v_t) &= \mathrm{PowerIteration}(W_t), \qquad \Theta_t = u_t v_t^\top, \\
W_t &\leftarrow W_t \cdot R / \sigma_t, \\
\lambda_t &= \text{root of } \; \langle \Theta_t,\; \mathrm{msign}(\widehat{M}_t + \lambda\,\Theta_t) \rangle = 0, \\
\Phi_t &= \mathrm{msign}(\widehat{M}_t + \lambda_t\,\Theta_t), \\
W_{t+1} &= W_t - \eta\, R\, \Phi_t,
\end{aligned}
$$

where $G_t=\nabla_W \mathcal{L}(W_t)$ is the gradient, $M_t$ the momentum with decay $\beta$, $\lVert\cdot\rVert_F$ the Frobenius norm, $\mathrm{PowerIteration}$ returns the leading singular value $\sigma_t$ and singular vectors $u_t,v_t$, $\mathrm{msign}(\cdot)=UV^\top$ from the SVD (the orthogonal polar factor), $\Theta_t=u_t v_t^\top$ is the tangent-space projector of the sphere constraint, $\lambda_t$ is the Lagrange multiplier found by bisection, $\eta$ is the learning rate, and $R=\sqrt{d_\mathrm{out}/d_\mathrm{in}}$ is the spectral $\mu$P radius. Weights are initialized as $W_0 \leftarrow R\, W_0 / \lVert W_0 \rVert_2$.

Reference: Tian Xie, Haoming Luo, Haoyu Tang, Yiwen Hu, Jason Klein Liu, Qingnan Ren, Yang Wang, Wayne Xin Zhao, Rui Yan, Bing Su, Chong Luo, Baining Guo, "Controlled LLM Training on Spectral Sphere", arXiv 2026. https://arxiv.org/abs/2601.08393

---
[Back to the Canon](../README.md)
