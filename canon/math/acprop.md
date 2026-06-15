# ACProp

Implements ACProp, an adaptive method combining momentum centering with an asynchronous update.

ACProp builds the second-moment estimate from the centered gradient $g_t - m_t$ rather than the raw $g_t^2$, so the denominator tracks the variance of the gradient (the "centering" idea shared with AdaBelief). It also makes the update asynchronous: the step at time $t$ divides by the second moment $v_{t-1}$ from the previous iteration, while the numerator $m_t$ already includes the current gradient. This decorrelation of numerator and denominator is what gives the method a convergence guarantee in the stochastic setting, unlike Adam and RMSProp.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)(g_t - m_t)^2 \\
\theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{v_{t-1}} + \epsilon}\, m_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ the first-moment estimate, $v_t$ the centered second-moment estimate, $\beta_1,\beta_2$ the decay rates, and $\epsilon$ a small stability constant. Note the asynchronous denominator $\sqrt{v_{t-1}}$, which uses information up to step $t-1$ while $m_t$ uses the gradient at step $t$.

Reference: Juntang Zhuang, Yifan Ding, Tommy Tang, Nicha Dvornek, Sekhar Tatikonda, James S. Duncan, "Momentum Centering and Asynchronous Update for Adaptive Gradient Methods", NeurIPS 2021. https://arxiv.org/abs/2110.05454

---
[Back to the Canon](../README.md)
