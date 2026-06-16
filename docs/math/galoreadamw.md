# GaLoreAdamW

Implements GaLoreAdamW, AdamW with gradient low-rank projection.


$$
\begin{aligned}
&P_t = U[:, {:}r] \quad \text{where} \quad
    U S V^\top = \mathrm{SVD}(g_t)
    \quad \text{(recomputed every $T$ steps)}                       \\
&r_t = P_t^\top g_t                                                 \\
&m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, r_t                        \\
&v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, r_t^2                      \\
&\eta_t = \eta\, \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)            \\
&\tilde{g}_t = \alpha\, P_t\, m_t / (\sqrt{v_t} + \epsilon)         \\
&\theta_{t+1} = (1 - \eta\lambda)\,(\theta_t - \eta_t\, \tilde{g}_t)
\end{aligned}
$$

where $r$ is the projection rank, $T$ the subspace change
frequency (`update_proj_gap`), $\alpha$ the `scale` factor, and
$\lambda$ the decoupled weight decay, applied after the gradient
step as upstream does. Bias correction is folded into the step size
$\eta_t$, the formulation the official implementation inherits
from the transformers AdamW. The Adam statistics
$m_t, v_t$ live in the rank-$r$ subspace, which is what saves
the optimizer memory. The paper states the update for a matrix with
$m \le n$ and a left projector; this implementation picks the
projector side from the gradient shape so the smaller factor is kept.


**Note:** Projection is enabled per parameter group: groups carrying `rank`, `update_proj_gap`, `scale`, and `proj_type` keys are projected (2D parameters only), all other groups get plain AdamW. The upstream tensor projector for dim > 2 parameters needs tensorly and is not vendored.

Reference: Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang,
Anima Anandkumar, Yuandong Tian,
"GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection",
ICML 2024.
https://arxiv.org/abs/2403.03507

---
[Back to the Canon](../index.md)
