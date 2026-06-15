# DES-LOC

Implements DES-LOC, a desynchronized low-communication adaptive optimizer for distributed training.

DES-LOC runs Adam locally on each of $M$ workers and synchronizes each quantity on its own schedule, rather than averaging everything every step. Because the second moment decays more slowly than the first moment, and the first moment more slowly than the parameters, each is averaged across workers at a coarser interval ($K_v > K_u > K_x$), cutting communication while tracking the natural half-life $\tau_{0.5}(\beta) = \ln(0.5)/\ln(\beta)$ of each state.

Per worker $m$, the gradient is clipped and the moments and parameters are updated as in Adam; at the relevant interval the local state is replaced by the worker average $\mathbb{E}_m[\cdot] = \tfrac{1}{M}\sum_{m=1}^{M}(\cdot)^m$ before the step:

$$
\begin{aligned}
\hat{g}_t^m &= \mathrm{clip}(g_t^m, \rho), \\
\bar m_{t-1}^m &= \begin{cases} \mathbb{E}_m[m_{t-1}^m] & t \bmod K_u = 0 \\ m_{t-1}^m & \text{otherwise} \end{cases}, \qquad
m_t^m = \beta_1 \bar m_{t-1}^m + (1-\beta_1)\hat{g}_t^m, \\
\bar v_{t-1}^m &= \begin{cases} \mathbb{E}_m[v_{t-1}^m] & t \bmod K_v = 0 \\ v_{t-1}^m & \text{otherwise} \end{cases}, \qquad
v_t^m = \beta_2 \bar v_{t-1}^m + (1-\beta_2)\,\hat{g}_t^m \odot \hat{g}_t^m, \\
\bar\theta_{t}^m &= \begin{cases} \mathbb{E}_m[\theta_{t}^m] & t \bmod K_x = 0 \\ \theta_{t}^m & \text{otherwise} \end{cases}, \qquad
\theta_{t+1}^m = \bar\theta_{t}^m - \eta_t \, \frac{m_t^m}{\sqrt{v_t^m} + \epsilon}.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t^m$ the local gradient, $\rho$ the clipping radius, $m_t^m, v_t^m$ the local first and second moments, $\beta_1, \beta_2$ the decay rates, $\epsilon$ the stability constant, $M$ the number of workers, and $K_x, K_u, K_v$ the synchronization periods for parameters, first moment, and second moment (recommended $K_u = 3K_x$, $K_v = 6K_x$).

Reference: Iacob, Sani, Safaryan, Giampouras, Horváth, Jovanović, Kurmanji, et al., "DES-LOC: Desynced Low Communication Adaptive Optimizers for Training Foundation Models", arXiv 2025. https://arxiv.org/abs/2505.22549

---
[Back to the Canon](../README.md)
