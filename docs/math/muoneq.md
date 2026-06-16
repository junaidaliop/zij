# MuonEq

Implements MuonEq, Muon with lightweight diagonal equilibration applied before orthogonalization.

MuonEq targets matrix-valued parameters. Like Muon, it keeps a momentum buffer and orthogonalizes the update via a fixed number of Newton-Schulz iterations. The addition is a cheap pre-orthogonalization step: the momentum matrix is rescaled by row and/or column squared-norm statistics so that it is better conditioned before entering the Newton-Schulz map. Three forms are available: two-sided (RC), row-only (R, the default), and column-only (C); all use only $O(m+n)$ extra statistics.

$$
\begin{aligned}
m_t &= \beta_t\, m_{t-1} + (1-\beta_t)\, g_t,\\
D_{r,t} &= \mathrm{diag}\big(\mathrm{rowsum}(m_t \odot m_t) + \epsilon\big),\\
D_{c,t} &= \mathrm{diag}\big(\mathrm{colsum}(m_t \odot m_t) + \epsilon\big),\\
\hat{m}_t &= D_{r,t}^{-1/2}\, m_t\, D_{c,t}^{-1/2} \quad (\text{R: drop } D_{c,t};\ \text{C: drop } D_{r,t}),\\
o_t &= \mathrm{NS}_5(\hat{m}_t),\\
\theta_{t+1} &= (1 - \lambda\, \eta_t)\, \theta_t - a\, \eta_t\, o_t,
\end{aligned}
$$

where $\theta$ is a parameter matrix of shape $m \times n$, $g_t$ the gradient, $m_t$ the momentum, $\beta_t = 1 - t^{-1/2}$ the momentum decay, $\odot$ elementwise product, $\mathrm{rowsum}$ and $\mathrm{colsum}$ the per-row and per-column sums, $\epsilon$ a stability constant, $\mathrm{NS}_5$ five Newton-Schulz orthogonalization steps, $a = 0.2\sqrt{\max(m,n)}$ the update scale, $\eta_t = t^{-3/4}$ the learning rate, and $\lambda$ the decoupled weight decay.

Reference: Da Chang, Qiankun Shi, Lvgang Zhang, Yu Li, Ruijie Zhang, Yao Lu, Yongxiang Liu, Ganzhao Yuan, "MuonEq: Balancing Before Orthogonalization with Lightweight Equilibration", arXiv 2026. https://arxiv.org/abs/2603.28254

---
[Back to the Canon](../index.md)
