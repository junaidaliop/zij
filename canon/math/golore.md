# GoLore

Implements GoLore, gradient low-rank training with a randomly sampled projection subspace.

GoLore follows the GaLore framework: each weight matrix's gradient $g_t$ is projected into a rank-$r$ subspace, an Adam update is run on the small projected gradient (storing low-rank moments), and the resulting update is projected back to full size before being applied. Unlike GaLore, which builds the projection from the top singular vectors of the gradient, GoLore draws the projection matrix uniformly from the Stiefel manifold, i.e. a random orthonormal basis. This avoids the bias that SVD-based projection introduces under small-batch stochastic gradients and is what yields the convergence guarantee; in practice the authors run GaLore in the early phases and switch to GoLore for the final phase. The projection $P_t$ is held fixed for $\tau$ steps and resampled every $\tau$ iterations.

$$
\begin{aligned}
P_t &\sim \mathcal{U}(\mathrm{St}_{m,r}) \quad\text{(resampled every } \tau \text{ steps)} \\
R_t &= P_t^{\top} g_t \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) R_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, R_t \odot R_t \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^{t}}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}} \\
N_t &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \\
\theta_t &= \theta_{t-1} - \eta\, P_t N_t
\end{aligned}
$$

where $\theta$ is the weight matrix ($m \times n$, shown for $m \le n$; for $m > n$ project on the right with $Q_t \sim \mathcal{U}(\mathrm{St}_{n,r})$ so $R_t = g_t Q_t$ and the update is $-\eta\, N_t Q_t^{\top}$), $g_t$ is its gradient, $P_t$ the random orthonormal down-projection, $R_t$ the projected gradient, $m_t/v_t$ the low-rank Adam moments with decays $\beta_1,\beta_2$, $N_t$ the in-subspace update, $\eta$ the learning rate, $r$ the rank, $\tau$ the subspace-change interval, $\odot$ elementwise product, and $\epsilon$ a stability constant.

Reference: Yutong He, Pengrui Li, Yipeng Hu, Chuyan Chen, Kun Yuan, "Subspace Optimization for Large Language Models with Convergence Guarantees", arXiv 2024. https://arxiv.org/abs/2410.11289

---
[Back to the Canon](../README.md)
