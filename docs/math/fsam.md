# F-SAM

Implements F-SAM, a Sharpness-Aware Minimization variant that perturbs along the stochastic gradient noise rather than the raw mini-batch gradient.

SAM builds its adversarial perturbation from the full mini-batch gradient $g_t = \nabla\mathcal{L}_{\mathcal{B}}(\theta_t)$. The authors argue that the deterministic full-batch component of this gradient is the "unfriendly" part: it degrades generalization while inflating gradient norms, whereas the stochastic noise component carries the beneficial sharpness signal. F-SAM estimates the full-batch component with an exponential moving average $m_t$ across iterations and subtracts it (scaled by $\sigma$) from the current gradient, so the perturbation points along the recovered stochastic noise $d_t$. The update step is otherwise identical to SAM: a single gradient evaluated at the perturbed point.

$$
\begin{aligned}
g_t &= \nabla\mathcal{L}_{\mathcal{B}}(\theta_t), \\
m_t &= \lambda\,m_{t-1} + (1-\lambda)\,g_t, \\
d_t &= g_t - \sigma\,m_t, \\
\epsilon_t &= \rho\,\frac{d_t}{\|d_t\|}, \\
\theta_{t+1} &= \theta_t - \gamma\,\nabla\mathcal{L}_{\mathcal{B}}(\theta_t + \epsilon_t),
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the mini-batch gradient at the clean iterate, $\gamma$ the learning rate, $\rho$ the SAM neighborhood radius, $m_t$ the EMA estimate of the full-batch gradient component with decay $\lambda \in (0,1)$, $\sigma$ the projection strength removing that component (set to $1$ in experiments), $d_t$ the recovered stochastic gradient noise, and $\epsilon_t$ the friendly perturbation.

Reference: Tao Li, Pan Zhou, Zhengbao He, Xinwen Cheng, Xiaolin Huang, "Friendly Sharpness-Aware Minimization", CVPR 2024. https://arxiv.org/abs/2403.12350

---
[Back to the Canon](../index.md)
