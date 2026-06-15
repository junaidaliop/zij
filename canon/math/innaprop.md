# INNAprop

Implements INNAprop, a second-order-like optimizer pairing inertial Newton dynamics with RMSprop-style adaptive gradient scaling.

INNAprop discretizes a dissipative inertial system (the INNA dynamics) whose trajectory uses both the parameter $\theta$ and an auxiliary variable $\psi$ to emulate second-order behavior with only first-order cost. The gradient is rescaled per-coordinate by a bias-corrected RMSprop second-moment estimate, and an optional decoupled weight decay is applied before each step.

$$
\begin{aligned}
\theta_t &\leftarrow (1 - \lambda \gamma_t)\,\theta_t \\
v_{t} &= \sigma\, v_{t-1} + (1-\sigma)\, g_t^2 \\
\hat v_{t} &= \frac{v_{t}}{1 - \sigma^{t}} \\
\psi_{t} &= \left(1 - \frac{\gamma_t}{\beta}\right)\psi_{t-1} + \gamma_t\left(\frac{1}{\beta} - \alpha\right)\theta_t \\
\theta_{t+1} &= \left(1 + \frac{\gamma_t(1-\alpha\beta)}{\beta - \gamma_t}\right)\theta_t - \frac{\gamma_t}{\beta - \gamma_t}\,\psi_{t} - \gamma_t\,\beta\,\frac{g_t}{\sqrt{\hat v_{t}} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma_t$ the step size (with $\gamma_t < \beta$), $g_t$ the mini-batch gradient, $v_t$ the RMSprop second moment with decay $\sigma$, $\hat v_t$ its bias-corrected value, $\psi$ the auxiliary inertial variable initialized at $\psi_0 = (1-\alpha\beta)\theta_0$, $\alpha \ge 0$ the friction and $\beta > 0$ the geometric-damping parameter from the INNA dynamics, $\lambda$ the weight decay, and $\epsilon$ a stability constant.

Reference: Jérôme Bolte, Ryan Boustany, Edouard Pauwels, Andrei Purica, "A second-order-like optimizer with adaptive gradient scaling for deep learning", arXiv 2024. https://arxiv.org/abs/2410.05871

---
[Back to the Canon](../README.md)
