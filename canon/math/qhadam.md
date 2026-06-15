# QHAdam

Implements QHAdam, the quasi-hyperbolic counterpart of Adam.


$$
\begin{aligned}
   m_t &= \beta_1\, m_{t-1} + (1 - \beta_1)\, g_t \\
   v_t &= \beta_2\, v_{t-1} + (1 - \beta_2)\, g_t^2 \\
   \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
   \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
   \theta_t &= \theta_{t-1} - \alpha\,
       \frac{(1 - \nu_1)\, g_t + \nu_1\, \hat{m}_t}
            {\sqrt{(1 - \nu_2)\, g_t^2 + \nu_2\, \hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\alpha$ is the learning rate, $\beta_1, \beta_2$ the
moment decay rates, and $\nu_1, \nu_2$ the immediate discount factors
that interpolate each moment estimate toward the current gradient. Setting
$\nu_1 = \nu_2 = 1$ recovers Adam. The NAdam optimizer is recovered
through `from_nadam`.

Reference: Jerry Ma, Denis Yarats, "Quasi-hyperbolic momentum and Adam for
deep learning", ICLR 2019.
https://arxiv.org/abs/1810.06801

---
[Back to the Canon](../README.md)
