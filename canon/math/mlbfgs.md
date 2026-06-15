# mL-BFGS

Implements mL-BFGS, a momentum-based L-BFGS that smooths the curvature pairs to stabilize quasi-Newton updates in stochastic training.

Vanilla L-BFGS approximates the inverse Hessian from history of parameter and gradient differences, but in stochastic settings those differences are noisy and convergence becomes unstable. mL-BFGS keeps running momentum buffers of the parameters and gradients, and forms the curvature pairs $(s_k, y_k)$ from the smoothed quantities every $T$ steps. A damping factor $\tau$ rescales $y_k$ to keep the curvature within eigenvalue bounds $[\sigma_L, \sigma_H]$ so the inverse-Hessian estimate stays positive definite. The step is the usual two-loop recursion applied to the current gradient.

$$
\begin{aligned}
\mathcal{M}^\theta_t &= \beta\,\mathcal{M}^\theta_{t-1} + (1-\beta)\,\theta_t \\
\mathcal{M}^g_t &= \beta\,\mathcal{M}^g_{t-1} + (1-\beta)\,g_t \\
s_k &= \mathcal{M}^\theta_{(k+1)T} - \mathcal{M}^\theta_{kT}, \qquad y_k = \mathcal{M}^g_{(k+1)T} - \mathcal{M}^g_{kT} \\
\mu &= \frac{s_k^\top y_k}{s_k^\top s_k}, \qquad
\tau = \begin{cases}
\min\!\big(\tfrac{1-\sigma_L}{1-\mu},\, \tau_0\big) & \mu \le \sigma_L < 1 \\
\min\!\big(\tfrac{\sigma_H-1}{\mu-1},\, \tau_0\big) & \mu \ge \sigma_H > 1 \\
\tau_0 & \text{otherwise}
\end{cases} \\
\hat{y}_k &= \tau\,y_k + (1-\tau)\,s_k, \qquad \rho_k = \frac{1}{\hat{y}_k^\top s_k} \\
H_k &= \big(I - \rho_k\,s_k\hat{y}_k^\top\big)\,H_{k-1}\,\big(I - \rho_k\,\hat{y}_k s_k^\top\big) + \rho_k\,s_k s_k^\top \\
\theta_{t+1} &= \theta_t - \eta_t\,H_k\,g_t
\end{aligned}
$$

where $\mathcal{M}^\theta,\mathcal{M}^g$ are the momentum buffers for parameters and gradients, $\beta$ is the momentum coefficient, $T$ is the Hessian update period, $(s_k,y_k)$ are the smoothed curvature pairs, $\tau$ is the adaptive damping factor with bounds $\sigma_L,\sigma_H$ and base value $\tau_0\in(0,1)$, $\hat{y}_k$ is the damped gradient change, $H_k$ is the approximate inverse Hessian (applied via the two-loop recursion), and $\eta_t$ is the learning rate.

Reference: Yue Niu, Zalan Fabian, Sunwoo Lee, Mahdi Soltanolkotabi, Salman Avestimehr, "mL-BFGS: A Momentum-based L-BFGS for Distributed Large-Scale Neural Network Optimization", TMLR 2023. https://arxiv.org/abs/2307.13744

---
[Back to the Canon](../README.md)
