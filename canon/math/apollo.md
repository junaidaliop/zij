# APOLLO

Implements APOLLO, a memory-efficient AdamW variant that replaces
element-wise gradient scaling with channel-wise or tensor-wise factors
estimated in a low-rank space under random projection.


$$
\begin{aligned}
     R_t &= P G_t, \qquad P_{ij} \sim \mathcal{N}(0, 1/r)               \\
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) R_t                         \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) R_t^2                       \\
     \tilde{R}_t &= \frac{m_t / (1 - \beta_1^t)}
         {\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}                      \\
     s_j &= \frac{\|\tilde{R}_t[:, j]\|_2}{\|R_t[:, j]\|_2}
         \;\text{(channel-wise)}, \qquad
     s = \frac{\|\tilde{R}_t\|_2}{\|R_t\|_2}
         \;\text{(tensor-wise)}                                         \\
     \theta_t &= \theta_{t-1} - \eta \, \alpha \, G_t \,
         \mathrm{diag}(s) - \eta \lambda \theta_{t-1}
\end{aligned}
$$

where the projection $P \in \mathbb{R}^{r \times m}$ is resampled
every `update_proj_gap` steps and $\alpha$ is `scale`. A
norm-growth limiter caps the ratio of consecutive scaled-gradient norms
at $\gamma = 1.01$. `scale_type='channel'` gives APOLLO;
`scale_type='tensor'` with a small rank (1 in the paper) gives
APOLLO-Mini. With `rank=None` no projection is applied and the update
reduces to AdamW.


**Note:** Following the upstream reimplementation, the scaling factors are applied to the projected gradient $R_t$ and the update is mapped back through $P^\top$ scaled by $\alpha^{3/2}$, rather than scaling the full-rank gradient $G_t$ directly as in Algorithm 1 of the paper.

Reference: Hanqing Zhu, Zhenyu Zhang, Wenyan Cong, Xi Liu, Sem Park,
Vikas Chandra, Bo Long, David Z. Pan, Zhangyang Wang, Jinwon Lee,
"APOLLO: SGD-like Memory, AdamW-level Performance", MLSys 2025.
https://arxiv.org/abs/2412.05270

---
[Back to the Canon](../README.md)
