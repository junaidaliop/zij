# ACMo

Implements ACMo, an angle-calibrated moment method for stochastic optimization.

ACMo replaces the fixed momentum coefficient of heavy-ball methods with a calibration that depends on the ratio between the current gradient norm and the running moment norm. This rescaling keeps the momentum term aligned with the gradient direction (acting as an angle bisector when the calibration is saturated), giving Adam-like adaptivity while storing only first-order moments.

$$
\begin{aligned}
\hat\beta_t &= \beta_t \cdot \frac{\lVert g_t \rVert}{\lVert m_{t-1} \rVert + \delta_t} \\
m_t &= g_t + \hat\beta_t \, m_{t-1} \\
\theta_t &= \theta_{t-1} - \gamma_t \, m_t
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma_t$ is the step size, $g_t$ is the stochastic gradient, $m_t$ is the angle-calibrated moment ($m_0 = 0$), $\beta_t$ is the base momentum coefficient, $\hat\beta_t$ is the calibrated coefficient, and $\delta_t$ is a stabilizing constant in the denominator.

Reference: Xunpeng Huang, Runxin Xu, Hao Zhou, Zhe Wang, Zhengyang Liu, Lei Li, "ACMo: Angle-Calibrated Moment Methods for Stochastic Optimization", AAAI 2021. https://arxiv.org/abs/2006.07065

---
[Back to the Canon](../README.md)
