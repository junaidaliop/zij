# Muon^2

Implements Muon$^2$, Muon with Adam-style second-moment preconditioning applied before orthogonalization.

Muon orthogonalizes the momentum matrix via Newton-Schulz iterations, but its convergence is limited by the ill-conditioned spectrum of that momentum. Muon$^2$ rescales the momentum element-wise by a running second moment of the gradient, sharpening the spectrum so that fewer Newton-Schulz iterations are needed to reach a sufficiently orthogonal update.

For a parameter matrix $\theta \in \mathbb{R}^{n \times m}$ with gradient $G_t$:

$$
\begin{aligned}
M_t &= \beta_1 M_{t-1} + (1 - \beta_1) G_t \\
V_t &= \beta_2 V_{t-1} + (1 - \beta_2) (G_t \odot G_t) \\
\tilde{M}_t &= M_t \oslash \sqrt{V_t + \epsilon \mathbf{1}} \\
O_t &= \mathrm{NewtonSchulz}(\tilde{M}_t, K) \\
\theta_{t+1} &= \theta_t - \eta \sqrt{m/n}\, O_t
\end{aligned}
$$

where $M_t$ is the momentum, $V_t$ the second-moment accumulator, $\odot$ and $\oslash$ are element-wise product and division, $\sqrt{\cdot}$ acts element-wise, $\mathrm{NewtonSchulz}(\cdot, K)$ applies $K$ orthogonalization iterations, $\eta$ is the learning rate, $\sqrt{m/n}$ is a dimension-aware scaling factor, $\beta_1, \beta_2$ are decay rates, and $\epsilon$ is a stability constant.

Reference: Ziyue Liu, Ruijie Zhang, Zhengyang Wang, Yequan Zhao, Yupeng Su, Zi Yang, Zheng Zhang, "Muon$^2$: Boosting Muon via Adaptive Second-Moment Preconditioning", arXiv 2026. https://arxiv.org/abs/2604.09967

---
[Back to the Canon](../index.md)
