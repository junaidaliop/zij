# AMUSE

Implements AMUSE, a learning-rate-free optimizer that fuses Muon with Schedule-Free iterate averaging.

AMUSE views training through a river-valley loss landscape: progress accumulates along a flat, low-curvature bulk subspace (the river), while high-curvature directions form steep valley walls that drive oscillations. Muon's orthogonalization accelerates river progress but also amplifies dominant-direction noise. AMUSE evaluates the gradient at a time-varying interpolation between the fast base iterate $Z_t$ and the stabilized average $X_t$, then orthogonalizes the resulting momentum. A coefficient $\beta_t$ shifts the evaluation point from the average toward the base iterate over training, balancing rapid adaptation against suppression of oscillations and removing any need for a learning rate schedule.

$$
\begin{aligned}
Y_t &= (1 - \beta_t) Z_t + \beta_t X_t, \\
M_t &= \mu M_{t-1} + \nabla \mathcal{L}(Y_t), \\
Z_{t+1} &= Z_t - \eta \, \mathcal{O}(M_t), \\
X_{t+1} &= (1 - c_{t+1}) X_t + c_{t+1} Z_{t+1}, \\
c_{t+1} &= \frac{1}{t+1}, \qquad
\beta_t = 1 - \left( \frac{T_0 - 1}{t - 1} \right)^{\rho} (1 - \beta_1) \;\; \text{for } t \ge T_0 .
\end{aligned}
$$

where $Z_t$ is the fast base iterate, $X_t$ the Schedule-Free average, $Y_t$ the gradient evaluation point, $M_t$ the momentum with decay $\mu$, $\eta$ the learning rate, $\mathcal{O}(\cdot)$ the orthogonalization operator (approximated by a Newton-Schulz iteration), $c_{t+1}$ the averaging weight, and $\beta_t$ the time-varying interpolation coefficient with warmup horizon $T_0$, exponent $\rho$, and base value $\beta_1$. Non-matrix parameters are updated with Schedule-Free AdamW or SGD.

Reference: Jueun Kim, Baekrok Shin, Jihun Yun, Beomhan Baek, Minhak Song, Chulhee Yun, "AMUSE: Anytime Muon with Stable Gradient Evaluation", arXiv 2026. https://arxiv.org/abs/2605.22432

---
[Back to the Canon](../index.md)
