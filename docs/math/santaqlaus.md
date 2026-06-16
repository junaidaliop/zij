# SantaQlaus

Implements SantaQlaus, a shot-adaptive optimizer that tunes the number of quantum measurements so that the intrinsic quantum shot-noise plays the role of the thermal noise in the Santa annealing thermostat.

SantaQlaus builds on the classical Santa optimizer (Stochastic AnNealing Thermostats with Adaptive momentum), which simulates a discretized stochastic differential equation: an adaptive-momentum dynamics with a Nosé–Hoover thermostat $\boldsymbol{\Xi}$ and injected Gaussian noise of variance $\propto 1/\beta_t$, where the inverse temperature $\beta_t$ is annealed from small (exploration) to large (refinement). The central observation is that evaluating a variational-quantum-algorithm gradient with a finite number of shots $N_t$ already produces an asymptotically Gaussian estimator $\hat g_t = g_t + \mathcal{N}(0, \Sigma_t / N_t)$. SantaQlaus therefore does not inject artificial thermal noise; instead it *chooses the shot count* so that the unavoidable quantum shot-noise matches the thermal noise that Santa would have added. Early iterations (small $\beta_t$, high temperature) need only a few shots to explore, while later iterations (large $\beta_t$) demand many shots for precise refinement.

Per iteration, the shot count is set by equating the two noise variances, and the parameters then follow the leading-order Santa step driven by the noisy gradient:

$$
\begin{aligned}
\frac{\Sigma_t}{N_t} &\;=\; \frac{2}{\beta_t \, \eta_t}
\quad\Longrightarrow\quad
N_t \;=\; \left\lceil \frac{\beta_t \, \eta_t \, \Sigma_t}{2} \right\rceil \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, \hat g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, \hat g_t^{\,2}, \qquad G_t = \frac{1}{\sqrt{\,\mathrm{diag}(v_t)\,}+\epsilon} \\
\theta_t &= \theta_{t-1} - \eta_t\, G_t\, m_t + \sqrt{\tfrac{2}{\beta_t}\,\eta_t\, G_t}\;\boldsymbol{\zeta}_t
\end{aligned}
$$

where $\theta$ are the variational parameters, $\eta_t$ the step size, $\hat g_t$ the shot-estimated gradient, $\Sigma_t$ its (per-component) shot-noise variance for one shot, $N_t$ the allocated shot count, $\beta_t$ the annealed inverse temperature, $m_t$/$v_t$ the first- and second-moment estimates forming the adaptive preconditioner $G_t$, $\beta_1,\beta_2$ the decay rates, $\boldsymbol{\zeta}_t$ standard Gaussian noise, and $\epsilon$ a stability constant. The defining shot rule $N_t = \lceil \beta_t \eta_t \Sigma_t / 2 \rceil$ makes the residual quantum shot-noise equal to the Santa thermal-noise term, so the injected-noise contribution can be dropped once $N_t$ is chosen.

Reference: Kosuke Ito, Keisuke Fujii, "SantaQlaus: A resource-efficient method to leverage quantum shot-noise for optimization of variational quantum algorithms", arXiv 2023. https://arxiv.org/abs/2312.15791

---
[Back to the Canon](../index.md)
