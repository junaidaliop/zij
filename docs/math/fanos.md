# FANoS

Implements FANoS, RMS-preconditioned symplectic momentum with a feedback-controlled friction thermostat.

FANoS treats optimization as a damped Hamiltonian system integrated by a semi-implicit (symplectic) Euler scheme. A per-coordinate velocity is driven by the RMS-preconditioned gradient and damped by a scalar friction coefficient, while parameters advance from the freshly updated velocity. The friction itself is not fixed: a Nosé–Hoover-style thermostat adjusts it via feedback on the system's kinetic ("update") energy, pushing the running kinetic energy toward an annealed target temperature so the dynamics start hot for exploration and cool for refinement.

$$
\begin{aligned}
s_t &= \beta\, s_{t-1} + (1-\beta)\, g_t^2, \qquad m_t = \sqrt{s_t} + \epsilon \\
v_t &= (1 - h\,\zeta_{t-1})\, v_{t-1} - h\, \frac{g_t}{m_t} \\
\theta_t &= \theta_{t-1} + h\, v_t \\
\hat{T}_t &= \frac{1}{d}\sum_i m_{t,i}\, v_{t,i}^2, \qquad \bar{T}_t = \rho\, \bar{T}_{t-1} + (1-\rho)\, \hat{T}_t \\
T_0(t) &= T_{\min} + (T_{\max} - T_{\min})\, e^{-t/\tau} \\
\zeta_t &= \mathrm{clip}\!\left(\zeta_{t-1} + \frac{h}{Q}\,\bigl(\bar{T}_t - T_0(t)\bigr),\; -\zeta_{\max},\; \zeta_{\max}\right)
\end{aligned}
$$

where $\theta$ are parameters, $g_t$ the gradient, $h$ the step size, $\beta$ the RMS decay, $\epsilon$ the stability constant, $v_t$ the velocity, $\zeta_t$ the scalar friction, $\hat{T}_t$ the instantaneous kinetic energy over the $d$ coordinates, $\bar{T}_t$ its EMA with rate $\rho$, $T_0(t)$ the annealed target temperature between $T_{\max}$ and $T_{\min}$ with time constant $\tau$, $Q$ the thermostat mass, and $\zeta_{\max}$ the friction clip bound.

Reference: Nalin Dhiman, "FANoS: Friction-Adaptive Nosé–Hoover Symplectic Momentum for Stiff Objectives", arXiv 2026. https://arxiv.org/abs/2601.00889

---
[Back to the Canon](../index.md)
