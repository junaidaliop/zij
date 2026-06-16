# S-Adam

Implements S-Adam, an Adam variant that brakes the step size near non-smooth singularities.

S-Adam augments Adam with a randomized geometric probe of the local landscape. At each step it samples $k$ unit directions and estimates a Local Geometric Instability (LGI) score $\rho_t$ from the variance of directional finite differences, an empirical proxy for the diameter of the Clarke subdifferential. The standard Adam direction is then scaled by a multiplicative brake $\exp(-\lambda\rho_t)$, which shrinks the effective learning rate in unstable, high-curvature-variance regions while leaving smooth basins essentially untouched.

$$
\begin{aligned}
D_i &= \frac{f(\theta_t + \delta u_i) - f(\theta_t)}{\delta}, \quad u_i \sim \mathcal{U}(\mathbb{S}^{d-1}), \; i = 1,\dots,k \\
\rho_t &= \frac{\mathrm{Var}(\{D_i\})}{\mathbb{E}[D_i^2] + \epsilon} \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{\eta}_t &= \eta \exp(-\lambda \rho_t) \\
\theta_{t+1} &= \theta_t - \hat{\eta}_t \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the base learning rate, $g_t$ the gradient, $m_t$ and $v_t$ the first and second moment estimates with decays $\beta_1,\beta_2$, $u_i$ random unit probe directions on the sphere $\mathbb{S}^{d-1}$, $\delta$ the probe scale, $\rho_t$ the LGI instability score, $\lambda$ the damping coefficient, and $\epsilon$ the stability constant.

Reference: Ruoran Xu, Borong She, Xiaobo Jin, Qiufeng Wang, "Singularity-aware Optimization via Randomized Geometric Probing: Towards Stable Non-smooth Optimization", ICML 2026. https://arxiv.org/abs/2605.29547

---
[Back to the Canon](../index.md)
