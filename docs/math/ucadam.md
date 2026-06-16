# UCAdam

Implements UCAdam, an Adam variant whose moments are accumulated through the Unified Conformable Fractional Derivative (UCFD).

Adam advances its moments with fixed decay rates $\beta_1,\beta_2$. UCAdam instead treats the moment update as a fractional difference of order $\alpha$ computed with the UCFD, which turns Adam's constant decay coefficients into iteration-dependent ones. For a fractional order $\alpha \in (0,1)$ this yields a one-step recurrence in which the weight on the previous moment grows with the step count $k$ while the weight on the current gradient shrinks, giving an adaptive blend between momentum and gradient. Setting $\alpha = 1$ recovers standard Adam exactly.

$$
\begin{aligned}
m_t &= \left(1 + \frac{\beta_1 - 1}{t^{\,1-\alpha}}\right) m_{t-1} + \frac{1 - \beta_1}{t^{\,1-\alpha}}\, g_t \\
v_t &= \left(1 + \frac{\beta_2 - 1}{t^{\,1-\alpha}}\right) v_{t-1} + \frac{1 - \beta_2}{t^{\,1-\alpha}}\, g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^{\,t}} \\
\theta_t &= \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient (with $g_t^2$ the elementwise square), $m_t,v_t$ the fractional first and second moments, $\beta_1,\beta_2$ the decay rates, $\alpha \in (0,1]$ the UCFD order, and $\epsilon$ a stability constant; the coefficient $1 + (\beta-1)/t^{1-\alpha}$ on the previous moment and $(1-\beta)/t^{1-\alpha}$ on the current gradient sum to one, so each moment is a step-dependent weighted mean.

Reference: Beibei Hou, Zongxi Song, Baopeng Li, "Improved Adam: Incorporating Unified Conformable Fractional Derivative for fractional-order Momentum", Journal of Electrical Systems 20(10s) 2024. https://journal.esrgroups.org/jes/article/view/5687

---
[Back to the Canon](../index.md)
