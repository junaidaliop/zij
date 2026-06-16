# Compositional Muon (CM)

Implements Compositional Muon (CM), a Muon variant that orthogonalizes composed weight products rather than individual matrices.

Standard Muon applies the matrix sign $\mathrm{msign}(G) = UV^\top$ (from the thin SVD $G = U\Sigma V^\top$) to each matrix gradient separately. CM instead controls the perturbation of a *composed* map such as $M = W_Q W_K^\top$, whose differential is $\Delta M = \Delta W_Q W_K^\top + W_Q \Delta W_K^\top$. To take steepest descent in the composed map's spectral norm, each factor's gradient is whitened by its partner's Gram-matrix square root before the spectral sign, then rescaled by the same inverse, so the resulting steps are balanced across the two factors. Momentum accumulates the raw gradients, and the partner whitening uses the current geometry at each step.

For an attention QK pair with partner Gram roots $C_Q = (W_Q^\top W_Q)^{1/2}$ and $C_K = (W_K^\top W_K)^{1/2}$, the per-head update is

$$
\begin{aligned}
m_{Q,t} &= \beta\, m_{Q,t-1} + g_{Q,t}, \\
m_{K,t} &= \beta\, m_{K,t-1} + g_{K,t}, \\
\Delta W_Q &= -\tfrac{\eta}{2}\, \mathrm{msign}\!\left(m_{Q,t}\, C_K^{-1}\right) C_K^{-1}, \\
\Delta W_K &= -\tfrac{\eta}{2}\, \mathrm{msign}\!\left(m_{K,t}\, C_Q^{-1}\right) C_Q^{-1},
\end{aligned}
$$

where $\theta = \{W_Q, W_K\}$ are the paired weight matrices, $g_{Q,t} = G_M W_K$ and $g_{K,t} = G_M^\top W_Q$ are the factor gradients induced by the composed-map gradient $G_M$, $m_t$ are momentum buffers with decay $\beta$, $\eta$ is the learning rate, $\mathrm{msign}(A) = UV^\top$ is the matrix sign, and $C_Q^{-1}, C_K^{-1}$ are the partner whitening factors. The OV pair uses the analogous hybrid form, whitening per head on the V side and applying a single concatenated spectral sign on the O side.

Reference: Ben Keigwin, Li Yang, Dhruv Pai, Yunzhe Zhang, Alec DeWulf, "Towards Compositional Steepest Descent", Tilde Research 2026. https://blog.tilderesearch.com/blog/compositional-muon

---
[Back to the Canon](../index.md)
