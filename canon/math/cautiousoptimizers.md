# Cautious Optimizers

Implements Cautious Optimizers, a one-line modification that masks any base momentum optimizer's update to components agreeing in sign with the current gradient.

Given a base optimizer (e.g. AdamW or Lion) that proposes an update direction $u_t$, the cautious variant zeros out the coordinates of $u_t$ whose sign disagrees with the gradient $g_t$, applying the step only where the proposed direction and the gradient point the same way. The effective learning rate is rescaled by the fraction of retained coordinates so the average step magnitude is preserved. This keeps the modified update a descent direction and can be shown to leave the base optimizer's Hamiltonian and convergence guarantees intact.

$$
\begin{aligned}
\phi_t &= \mathbb{1}(u_t \odot g_t > 0) \\
\bar{\eta}_t &= \eta_t \cdot \frac{\dim(\phi_t)}{\sum_i \phi_{t,i} + \xi} \\
\theta_{t+1} &= \theta_t - \bar{\eta}_t \, (u_t \odot \phi_t)
\end{aligned}
$$

where $u_t$ is the update proposed by the base optimizer, $g_t$ the current gradient, $\odot$ the elementwise product, $\mathbb{1}(\cdot)$ the elementwise indicator giving a binary mask $\phi_t$, $\dim(\phi_t)$ the number of parameters, $\sum_i \phi_{t,i}$ the count of aligned (retained) coordinates, $\eta_t$ the base learning rate, $\bar{\eta}_t$ the rescaled learning rate, and $\xi$ a small constant ($\xi=1$ by default) for numerical stability.

Reference: Kaizhao Liang, Lizhang Chen, Bo Liu, Qiang Liu, "Cautious Optimizers: Improving Training with One Line of Code", arXiv preprint 2024. https://arxiv.org/abs/2411.16085

---
[Back to the Canon](../README.md)
