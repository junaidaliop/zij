# NAMO

Implements NAMO, a Muon variant that gives the orthogonalized momentum step an Adam-style adaptive scale.

Muon orthogonalizes the momentum to take a structure-aware matrix step but uses a fixed learning rate. NAMO keeps the orthogonalized direction yet rescales it adaptively: alongside the matrix momentum $M_t$ it tracks a scalar second-moment estimate $v_t$ of the squared Frobenius norm of the gradient. The step size is then modulated by $\lVert M_t\rVert_F / \sqrt{v_t}$ with bias correction, so the effective scale behaves like Adam applied at the level of the whole matrix rather than coordinate-wise. Decoupled weight decay is folded into the scaled step.

$$
\begin{aligned}
M_t &= \mu_1 M_{t-1} + (1-\mu_1)\, G_t \\
v_t &= \mu_2 v_{t-1} + (1-\mu_2)\, \lVert G_t\rVert_F^2 \\
O_t &= \mathrm{Orth}(M_t) \\
\alpha_t &= \frac{\sqrt{1-\mu_2^{\,t}}}{1-\mu_1^{\,t}}\cdot\frac{\lVert M_t\rVert_F}{\sqrt{v_t}+\epsilon} \\
\theta_t &= \theta_{t-1} - \eta\,\alpha_t\bigl(O_t + \lambda\,\theta_{t-1}\bigr)
\end{aligned}
$$

where $\theta$ are the matrix-shaped parameters, $G_t$ the stochastic gradient, $M_t$ the momentum matrix and $v_t$ the scalar second-moment estimate with decays $\mu_1,\mu_2$, $\mathrm{Orth}(M_t)=UV^\top$ the nearest orthogonal matrix to $M_t$ from its polar decomposition (computed in practice by Newton-Schulz iterations as in Muon), $\alpha_t$ the bias-corrected adaptive scale, $\eta$ the learning rate, $\lambda$ the decoupled weight decay, and $\epsilon$ a small stability constant.

Reference: Minxin Zhang, Yuxuan Liu, Hayden Schaeffer, "Adam Improves Muon: Adaptive Moment Estimation with Orthogonalized Momentum", arXiv 2026. https://arxiv.org/abs/2602.17080

---
[Back to the Canon](../index.md)
