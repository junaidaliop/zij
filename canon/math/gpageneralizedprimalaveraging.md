# GPA (Generalized Primal Averaging)

Implements GPA (Generalized Primal Averaging), a primal-averaging wrapper that smooths any base optimizer.

GPA maintains two coupled sequences: an unsmoothed iterate $z_t$ that accumulates the raw optimizer steps and a smoothed sequence $\theta_t$ returned at the end of training. Gradients (or, more generally, the base optimizer's search direction) are evaluated at an interpolation point $y_t$ that blends the two sequences. The generalized form replaces the plain stochastic gradient with a base-optimizer direction $d_t = -H_t m_t$ (preconditioner $H_t$ times gradient estimator $m_t$, e.g. AdamW) evaluated at $y_t$, and folds in decoupled weight decay.

$$
\begin{aligned}
y_t &= \mu_y\,\theta_t + (1-\mu_y)\,z_t \\
z_{t+1} &= (1-\gamma_t\lambda)\,z_t + \gamma_t\,d_t(y_t) \\
\theta_{t+1} &= \mu_x\,\theta_t + (1-\mu_x)\,z_{t+1}
\end{aligned}
$$

where $\theta_t$ is the smoothed (returned) iterate, $z_t$ the unsmoothed iterate, $y_t$ the point at which the direction is computed, $\gamma_t$ the learning rate schedule, $\lambda$ the weight decay, $d_t(y_t)$ the base optimizer's search direction at $y_t$ (with $d_t = -g_t$ recovering plain SGD averaging), and $\mu_x\in[0,1)$, $\mu_y\in[0,1]$ the two independent interpolation coefficients.

Reference: Aaron Defazio, Konstantin Mishchenko, Parameswaran Raman, Hao-Jun Michael Shi, Lin Xiao, "Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs", arXiv 2026. https://arxiv.org/abs/2512.17131

---
[Back to the Canon](../README.md)
