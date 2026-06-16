# RaCO-DP

Implements RaCO-DP, a differentially private gradient descent-ascent solver for rate-constrained optimization.

Rate constraints (e.g., fairness rates such as demographic parity) are functions of the model's prediction distribution, which RaCO-DP estimates through a differentially private histogram $\hat{H}_t$ built with Laplace noise. Training solves the Lagrangian min-max problem over primal parameters $\theta$ and dual multipliers $\lambda$ via stochastic gradient descent-ascent (SGDA). Privacy is enforced on the primal step with per-sample gradient clipping and Gaussian noise (DP-SGD style); the dual gradient reuses the already-private histogram, so it incurs no extra privacy cost.

$$
\begin{aligned}
\hat{H}_t &= H(\theta_t) + \mathrm{Lap}(1/b),\\
g_\theta^t &= \sum_{x \in B_t} \mathrm{clip}\!\left(g_{x,\theta}^t,\ \tfrac{C}{r|D|}\right) + Z_t, \qquad Z_t \sim \mathcal{N}(0,\ \sigma^2 I_d),\\
\theta_{t+1} &= \theta_t - \eta_\theta\, g_\theta^t,\\
[g_\lambda^t]_j &= \Gamma_j^{\mathrm{post}}(\hat{H}_t) - \gamma_j,\\
\lambda_{t+1} &= \Pi_\Lambda\!\left(\lambda_t + \eta_\lambda\, g_\lambda^t\right).
\end{aligned}
$$

where $B_t$ is a Poisson subsample of dataset $D$ at rate $r$, $g_{x,\theta}^t$ is the per-sample gradient of the Lagrangian, $C$ the clipping norm, $\sigma$ the Gaussian noise scale, $b$ the Laplace parameter, $\eta_\theta,\eta_\lambda$ the primal and dual learning rates, $\Gamma_j^{\mathrm{post}}$ the $j$-th constraint rate evaluated on the private histogram, $\gamma_j$ its slack, and $\Pi_\Lambda$ the projection onto the dual feasible set $\Lambda$.

Reference: Mohammad Yaghini, Tudor Cebere, Michael Menart, Aurélien Bellet, Nicolas Papernot, "Private Rate-Constrained Optimization with Applications to Fair Learning", arXiv 2025. https://arxiv.org/abs/2505.22703

---
[Back to the Canon](../index.md)
