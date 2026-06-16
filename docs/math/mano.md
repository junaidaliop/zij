# Mano

Implements Mano, a manifold optimizer that projects momentum onto the tangent space of a rotating row/column-normalized weight matrix.

Mano treats each 2D weight as a point on an Oblique manifold and constrains the update to that manifold's tangent space, retaining curvature structure that global spectral normalization (Muon) discards. At step $t$ it builds heavy-ball momentum, picks the manifold orientation $k = t \bmod 2$ (alternating between row-wise, $k=0$, and column-wise, $k=1$, normalization), normalizes the weight along that axis, subtracts the component of the momentum that lies along the normalized weight to obtain a tangent direction, normalizes that direction the same way, and steps with a fixed rescaling $0.2\sqrt{n_k}$ plus decoupled weight decay.

$$
\begin{aligned}
m_t &= \mu\, m_{t-1} + g_t \\
k &= t \bmod 2 \\
\hat{\theta}_t &= \theta_{t-1} \oslash \lVert \theta_{t-1} \rVert_{2,k} \\
v_t &= m_t - \hat{\theta}_t \odot \langle m_t, \hat{\theta}_t \rangle_k \\
\hat{v}_t &= v_t \oslash \lVert v_t \rVert_{2,k} \\
\theta_t &= \theta_{t-1} - \eta_t \left( 0.2\sqrt{n_k}\; \hat{v}_t + \lambda\, \theta_{t-1} \right)
\end{aligned}
$$

where $g_t = \nabla f(\theta_{t-1})$ is the gradient, $m_t$ the momentum, $\mu$ its decay, $\eta_t$ the learning rate, $\lambda$ the weight decay, $\odot$ and $\oslash$ elementwise product and division (the vector of per-slice norms broadcasts over its axis), $\lVert \cdot \rVert_{2,k}$ the L2 norm taken over each row when $k=0$ and each column when $k=1$, $\langle \cdot, \cdot \rangle_k$ the matching per-row/per-column inner product, and $n_k \in \{m, n\}$ with $n_0 = m$, $n_1 = n$ the size of the active dimension of the $m \times n$ weight, so $0.2\sqrt{n_k}$ rescales the step to AdamW-comparable magnitude.

Reference: Yufei Gu, Zeke Xie, "Mano: Restriking Manifold Optimization for LLM Training", arXiv 2026. https://arxiv.org/abs/2601.23000

---
[Back to the Canon](../index.md)
