# TeZO

Implements TeZO (Tensorized Zeroth-Order), a zeroth-order fine-tuning method that factorizes the perturbation tensor across both the model and the temporal dimension.

Memory-efficient zeroth-order (ZO) fine-tuning estimates gradients from forward passes by perturbing each weight matrix with random noise, but storing or regenerating a fresh full-size direction at every step is wasteful. TeZO views the stacked perturbations over all $T$ iterations as a single $m\times n\times T$ tensor and approximates it with a rank-$r$ canonical polyadic (CP) decomposition. For each weight matrix $\theta$ of size $m\times n$, the spatial factors $U\in\mathbb{R}^{m\times r}$ and $V\in\mathbb{R}^{n\times r}$ are sampled once and held fixed, while only a small temporal vector $\tau_t\in\mathbb{R}^{r}$ is resampled each step. The per-step perturbation is then $Z_t = U\,\mathrm{diag}(\tau_t)\,V^\top$, and the gradient is estimated by symmetric finite differences along $Z_t$.

This collapses the cost of generating perturbations from $\mathcal{O}(d\,T)$ to $\mathcal{O}(d+T)$ random numbers per matrix, since the heavy $m\times r$ and $n\times r$ factors are reused across all iterations. The momentum and Adam variants (TeZO-m, TeZO-Adam) accumulate statistics only on the cheap temporal factors.

$$
\begin{aligned}
U &\sim \mathcal{N}(0, I)_{m\times r}, \qquad V \sim \mathcal{N}(0, I)_{n\times r} \quad (\text{sampled once}), \\
\tau_t &\sim \mathcal{N}(0, I_r), \qquad Z_t = U\,\mathrm{diag}(\tau_t)\,V^\top = \sum_{s=1}^{r} \tau_{t,s}\, u_s v_s^\top, \\
\kappa_t &= \frac{f(\theta_t + \rho\, Z_t) - f(\theta_t - \rho\, Z_t)}{2\rho}, \\
\theta_{t+1} &= \theta_t - \eta\, \kappa_t\, Z_t,
\end{aligned}
$$

where $\theta$ is a weight matrix of size $m\times n$, $\eta$ is the learning rate, $\rho$ is the perturbation (smoothing) radius, $r\ll\min\{m,n\}$ is the CP rank, $U,V$ are the fixed spatial factors with columns $u_s,v_s$, $\tau_t$ is the per-step temporal factor, $Z_t$ is the rank-$r$ perturbation direction, $\kappa_t$ is the scalar finite-difference coefficient from two forward passes, and $f$ is the loss on the current minibatch.

Reference: Yan Sun, Tiansheng Huang, Liang Ding, Li Shen, Dacheng Tao, "TeZO: Empowering the Low-Rankness on the Temporal Dimension in the Zeroth-Order Optimization for Fine-tuning LLMs", arXiv 2025. https://arxiv.org/abs/2501.19057

---
[Back to the Canon](../README.md)
