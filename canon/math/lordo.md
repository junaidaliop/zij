# LoRDO

Implements LoRDO, distributed low-rank optimization with infrequent communication.

Each worker keeps Adam-style moments in a low-rank subspace defined by a projection $Q_t$, so per-worker optimizer memory drops from $O(pq)$ to $O(r(p+q))$. Gradients are clipped and error-fed before projection, and workers communicate only every $K_x$ steps, at which point the averaged update direction is used to recompute the shared projection. To stop the iterates from stalling inside a fixed low-rank subspace, a full-rank quasi-hyperbolic term mixes the raw (full-rank) gradient with the low-rank momentum, keeping the aggregated pseudo-gradient full-rank.

For worker $m$ at step $t$, the local update (full-rank quasi-hyperbolic variant) is

$$
\begin{aligned}
\hat{G}^m_t &= \mathrm{clip}\!\left(\nabla F(\theta^m_t;\xi^m_t),\,\rho\right) \\
\hat{g}^m_t &= Q_t^{\top}\!\left(\hat{G}^m_t + E^m_{t-1}\right) \\
E^m_t &= \hat{G}^m_t + E^m_{t-1} - Q_t\,\hat{g}^m_t \\
u^m_t &= \beta_1 u^m_{t-1} + (1-\beta_1)\,\hat{g}^m_t \\
v^m_t &= \beta_2 v^m_{t-1} + (1-\beta_2)\,(\hat{g}^m_t)^2 \\
\theta^m_t &\leftarrow \theta^m_t - \eta_t\left[(1-\omega)\,\frac{\hat{G}^m_t}{\mu\,(\sqrt{v^m_t}+\epsilon)} + \omega\, Q_t\!\left(\frac{u^m_t}{\sqrt{v^m_t}+\epsilon}\right)\right]
\end{aligned}
$$

and every $K_x$ steps the workers synchronize and refresh the projection:

$$
\begin{aligned}
\Delta^m_t &= \theta^m_t - \theta^m_{t-K_x} \\
\Delta_t &= \mathbb{E}_m\!\left[\Delta^m_t\right] \\
Q_{t+1} &= \mathrm{ComputeProjection}(\Delta_t)
\end{aligned}
$$

where $\theta$ are parameters, $\eta_t$ the learning rate, $\hat{G}^m_t$ the clipped full-rank gradient and $\hat{g}^m_t$ its low-rank projection, $E^m_t$ the projection error feedback, $u^m_t,v^m_t$ the low-rank first/second moments with decays $\beta_1,\beta_2$, $\epsilon$ a stability constant, $\rho$ the clipping threshold, $\omega$ the quasi-hyperbolic mixing weight, $\mu$ a scaling factor, $Q_t$ the rank-$r$ projection, and $K_x$ the communication interval.

Reference: Andrej Jovanović, Alex Iacob, Mher Safaryan, Ionut-Vlad Modoranu, Lorenzo Sani, William F. Shen, Xinchi Qiu, Dan Alistarh, Nicholas D. Lane, "LoRDO: Distributed Low-Rank Optimization with Infrequent Communication", ICML 2026. https://arxiv.org/abs/2602.04396

---
[Back to the Canon](../README.md)
