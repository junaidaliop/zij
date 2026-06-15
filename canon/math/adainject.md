# AdaInject

Implements AdaInject, an adaptive optimizer that injects curvature-like second-order information into the first-moment estimate using the recent parameter change.

AdaInject augments the gradient driving the first moment with a term proportional to the squared gradient weighted by the short-term parameter change $\Delta\theta = \theta_{t-2} - \theta_{t-1}$. Intuitively, the sign and magnitude of the recent step modulate how much the squared gradient pushes the moment, letting the optimizer adapt its effective step near minima. The construction is generic; applied to Adam it yields the AdamInject variant shown below, replacing Adam's $m_t$ with the injected moment $s_t$ while leaving the second moment $v_t$ unchanged.

$$
\begin{aligned}
\Delta\theta &= \theta_{t-2} - \theta_{t-1} \\
s_t &= \beta_1 s_{t-1} + (1 - \beta_1)\,\frac{g_t + \Delta\theta \cdot g_t^2}{k} \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\,g_t^2 \\
\hat{s}_t &= \frac{s_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \, \frac{\hat{s}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $g_t^2$ the elementwise squared gradient, $\Delta\theta$ the previous parameter change, $k$ the injection control constant (typically $k=2$), $s_t$ the injected first moment, $v_t$ the second moment, $\hat{s}_t$ and $\hat{v}_t$ their bias-corrected forms, $\beta_1,\beta_2$ the decay rates, and $\epsilon$ a small stability constant.

Reference: Shiv Ram Dubey, S.H. Shabbeer Basha, Satish Kumar Singh, Bidyut Baran Chaudhuri, "AdaInject: Injection Based Adaptive Gradient Descent Optimizers for Convolutional Neural Networks", IEEE Transactions on Artificial Intelligence 2022. https://arxiv.org/abs/2109.12504

---
[Back to the Canon](../README.md)
