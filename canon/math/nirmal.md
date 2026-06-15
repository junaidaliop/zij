# NIRMAL

Implements NIRMAL (Novel Integrated Robust Multi-Adaptation Learning), an optimizer that blends five chess-inspired update components into a single weighted step.

NIRMAL maintains an exponential moving average of the gradient ($m_t$) and of the squared gradient ($v_t$), then forms five distinct displacements — plain gradient descent (wazir), momentum (elephant), stochastic perturbation (knight), Adam-style adaptive scaling (camel), and a non-linear momentum transform (horse). The parameter step is a fixed convex combination of these components, intended to combine fast convergence, escape from shallow minima, and adaptive scaling in one rule. Weight decay is folded into the gradient and no bias correction is applied to the moments.

$$
\begin{aligned}
g_t &\leftarrow g_t + \alpha\, \theta_t \\
m_t &= \mu\, m_{t-1} + (1-\mu)\, g_t \\
v_t &= \beta\, v_{t-1} + (1-\beta)\, g_t^2 \\
\Delta_{\mathrm{wazir}} &= -\eta\, g_t \\
\Delta_{\mathrm{elephant}} &= -\eta\, m_t \\
\Delta_{\mathrm{knight}} &= \eta\, \kappa\, \mathcal{N}(0,1) \\
\Delta_{\mathrm{camel}} &= -\eta\, \gamma\, \frac{m_t}{\sqrt{v_t}+\epsilon} \\
\Delta_{\mathrm{horse}} &= -\eta\, \lambda\, \tanh(m_t) \\
\theta_{t+1} &= \theta_t + \sum_{c} w_c\, \Delta_c
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$/$v_t$ the first/second moment estimates, $\mu,\beta$ their decay rates, $\alpha$ the weight decay, $\epsilon$ a stability constant, $\kappa$ the perturbation scale, $\gamma$ the adaptive scaling factor, $\lambda$ the non-linear momentum factor, $\mathcal{N}(0,1)$ standard Gaussian noise, and $w_c$ the fixed component weights $(0.3, 0.25, 0.1, 0.2, 0.15)$ for $(\mathrm{wazir}, \mathrm{elephant}, \mathrm{knight}, \mathrm{camel}, \mathrm{horse})$.

Reference: Nirmal Gaud, Surej Mouli, Preeti Katiyar, Vaduguru Venkata Ramya, "Comparative Analysis of Novel NIRMAL Optimizer Against Adam and SGD with Momentum", arXiv 2025. https://arxiv.org/abs/2508.04293

---
[Back to the Canon](../README.md)
