# Local MixVR

Implements Local MixVR, a distributed optimizer that combines local update steps with STORM-style variance reduction to break the dependence of communication complexity on the total sample size.

Each of the $M$ machines runs several local $\mu^2$-SGD steps before synchronizing. A local step couples a STORM corrected-momentum estimator $d_t$ with anytime averaging: gradients are evaluated at the running average $\bar{\theta}_t$, and the correction term $d_{t-1}-\tilde{g}_{t-1}$ reuses the previous machine sample at the previous average to cancel variance. Machines periodically average their iterates and apply a drift correction to the momentum, so the communication rounds needed scale with the target accuracy rather than with $N$.

$$
\begin{aligned}
g_t &= \frac{1}{K_{\mathrm{avg}}}\sum_{k} \nabla f(\bar{\theta}_t; z_{t,k}), \qquad
\tilde{g}_{t-1} = \frac{1}{K_{\mathrm{avg}}}\sum_{k} \nabla f(\bar{\theta}_{t-1}; z_{t,k}), \\
d_t &= g_t + (1-\beta_t)\,(d_{t-1} - \tilde{g}_{t-1}), \\
\theta_{t+1} &= \theta_t - \eta_t\, d_t, \\
\bar{\theta}_{t+1} &= \gamma_t\, \theta_{t+1} + (1-\gamma_t)\, \bar{\theta}_t, \\
\bar{\theta}_t &\leftarrow \frac{1}{M}\sum_{i=1}^{M} \bar{\theta}_t^{(i)} \quad \text{(at synchronization)},
\end{aligned}
$$

where $\theta_t$ are the parameters, $\bar{\theta}_t$ the anytime average at which gradients are taken, $d_t$ the corrected-momentum direction, $g_t$ and $\tilde{g}_{t-1}$ the averaged gradients at the current and previous iterates over the same samples $z_{t,k}$, $M$ the number of machines, and the schedules are $\beta_t = 1/t$, $\gamma_t = 2/(t+2)$, $\eta_t = t\,\eta$, with $K_{\mathrm{avg}} = \lceil \alpha K \rceil$ averaging samples and $K_{\mathrm{loc}} = \lfloor (1-\alpha) K \rfloor$ local steps for a budget split $\alpha \in (0,1)$.

Reference: Tehila Dahan, Bassel Hamoud, Roie Reshef, Martin Jaggi, Kfir Y. Levy, "Local MixVR: Breaking the Communication-Sample Dependence in Distributed Learning", arXiv 2026. https://arxiv.org/abs/2606.01128

---
[Back to the Canon](../index.md)
