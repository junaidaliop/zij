# HeLoCo

Implements HeLoCo, a direction-aware correction for asynchronous low-communication training that realigns stale pseudo-gradients with the outer momentum before the global update.

In DiLoCo-style training each worker $i$ starts from a shared model, runs $H$ local inner steps, and returns a pseudo-gradient $\Delta^{(i)}$ to a global outer optimizer. Asynchrony makes these pseudo-gradients stale: they are computed from outdated states and can point against the current optimization direction, especially under device and data heterogeneity. HeLoCo treats the outer Nesterov momentum $m_t$ as a reference for the trajectory and, per tensor block $b$, measures the cosine alignment $c_b$ between the incoming pseudo-gradient and the momentum. Well-aligned blocks pass through unchanged; anti-aligned blocks are shrunk along the momentum direction; weakly aligned blocks are partially reoriented toward the momentum. A confidence factor scales each correction by the pseudo-gradient's norm relative to the momentum's, and the corrected, delay-weighted update feeds the outer momentum and parameter step.

$$
\begin{aligned}
\theta^{(s_i)}_{i,0} &= \bar\theta_{s_i}, \qquad \theta^{(s_i)}_{i,h+1} = \theta^{(s_i)}_{i,h} - \alpha_h\, g_i\!\big(\theta^{(s_i)}_{i,h}\big), \quad h = 0,\dots,H-1 \\
\Delta^{(i)} &= \bar\theta_{s_i} - \theta^{(s_i)}_{i,H} \\
\hat u_b &= \frac{\Delta_b}{\lVert \Delta_b \rVert}, \quad \hat v_b = \frac{(m_t)_b}{\lVert (m_t)_b \rVert}, \quad c_b = \hat u_b^{\top}\hat v_b \\
\mathrm{conf}_b &= \frac{\lVert \Delta_b \rVert}{\lVert \Delta_b \rVert + \kappa\,\lVert (m_t)_b \rVert + \epsilon} \\
\hat\Delta_b &=
\begin{cases}
\Delta_b, & c_b \ge c_{\mathrm{ok}} \\
\Delta_b - \beta_b\, c_b\, \lVert \Delta_b \rVert\, \hat v_b, \quad \beta_b = \min\{k_s(-c_b)\,\mathrm{conf}_b,\ \beta_{\max}\}, & c_b < 0 \\
\dfrac{\lVert \Delta_b \rVert\, \tilde u_b}{\max\{\lVert \tilde u_b \rVert,\ \epsilon\}}, \quad \tilde u_b = (1-\lambda_b)\hat u_b + \lambda_b \hat v_b, \ \lambda_b = \min\{k_d(1-c_b)\,\mathrm{conf}_b,\ 1\}, & 0 \le c_b < c_{\mathrm{ok}} \\
\end{cases} \\
G_t &= \rho_t\, \hat\Delta \\
m_{t+1} &= \mu\, m_t + (1-\mu)\, G_t \\
\theta_{t+1} &= \theta_t - \eta_t\,\big(G_t + \mu\, m_{t+1}\big)
\end{aligned}
$$

where $\theta$ are the parameters, $\bar\theta_{s_i}$ the shared model that worker $i$ starts from at step $s_i$, $\alpha_h$ the inner learning rate, $g_i$ worker $i$'s gradient, $H$ the number of local steps, $\Delta^{(i)}$ its pseudo-gradient, $b$ a tensor block, $\hat u_b,\hat v_b$ the unit pseudo-gradient and unit outer momentum directions, $c_b$ their cosine alignment, $\mathrm{conf}_b$ a confidence weight, $\hat\Delta$ the block-wise corrected pseudo-gradient, $\rho_t$ a worker/delay weight ($\rho_t=1$ if none), $G_t$ the aggregated corrected update, $m_t$ the outer (Nesterov) momentum with coefficient $\mu$, $\eta_t$ the outer learning rate, and $\epsilon$ a small constant; $c_{\mathrm{ok}}$ is the keep threshold, $k_s,\beta_{\max}$ the shrinkage strength and cap, $k_d$ the reorientation strength, and $\kappa$ the relative-momentum weight.

Reference: Abdullah Al Asif, Patrick Diem, Juan Pablo Muñoz, Felix Wolf, Ali Jannesari, Arya Mazaheri, "HeLoCo: Efficient asynchronous low-communication training under data and device heterogeneity", arXiv 2026. https://arxiv.org/abs/2606.00271

---
[Back to the Canon](../README.md)
