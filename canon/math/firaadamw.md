# FiraAdamW

Implements FiraAdamW, AdamW with full-rank updates under a GaLore-style
low-rank optimizer-memory budget.


$$
\begin{aligned}
&P_t = U[:, {:}r] \quad \text{where} \quad
    U S V^\top = \mathrm{SVD}(g_t)
    \quad \text{(recomputed every $T$ steps)}                       \\
&r_t = P_t^\top g_t                                                 \\
&m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, r_t                        \\
&v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, r_t^2                      \\
&\eta_t = \eta\, \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)             \\
&\psi_t = m_t / (\sqrt{v_t} + \epsilon)                             \\
&(\phi_t)_i = \lVert (\psi_t)_{:,i} \rVert_2
    \, / \, \lVert (r_t)_{:,i} \rVert_2                             \\
&S_t = \phi_t \odot (g_t - \alpha P_t r_t)                          \\
&S_t \leftarrow \gamma\, S_t\, \lVert S_{t-1} \rVert
    / \lVert S_t \rVert
    \quad \text{if } \lVert S_t \rVert > \gamma \lVert S_{t-1} \rVert \\
&\theta_{t+1} = (1 - \eta\lambda)\,
    (\theta_t - \eta_t\, (\alpha P_t \psi_t + S_t))
\end{aligned}
$$

where $r$ is the projection rank, $T$ the subspace change
frequency (`update_proj_gap`), $\alpha$ the `alpha` scale
factor, $\gamma = 1.01$ the norm-growth limit, and $\lambda$
the decoupled weight decay, applied after the gradient step as upstream
does. Bias correction is folded into the step size $\eta_t$. The
Adam statistics $m_t, v_t$ live in the rank-$r$ subspace, as
in GaLore; the residual gradient $g_t - \alpha P_t r_t$ outside the
subspace is applied with the norm-based scaling $\phi_t$, one
factor per column of $r_t$ (per row when the gradient is projected
from the right), so the full-rank direction is trained without full-rank
optimizer state. The norm-growth limiter caps the step-to-step growth of
$\lVert S_t \rVert$ at $\gamma$ to suppress loss spikes.


**Note:** Projection is enabled per parameter group: groups carrying `rank`, `update_proj_gap`, `alpha`, and `proj_type` keys are projected (2D parameters only), all other groups get plain AdamW.

Reference: Xi Chen, Kaituo Feng, Changsheng Li, Xunhao Lai, Xiangyu Yue,
Ye Yuan, Guoren Wang,
"Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank
Constraint?", NeurIPS 2025.
https://arxiv.org/abs/2410.01623

---
[Back to the Canon](../README.md)
