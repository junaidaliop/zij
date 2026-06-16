# AdaNorm

Implements AdaNorm, an Adam variant with adaptive gradient norm
correction.

AdaNorm tracks an exponential moving average of the gradient norm and, when
the norm of the current gradient falls below that average, rescales the
gradient up to the running norm before it enters the first moment. This
keeps the first moment driven by a high and representative gradient
magnitude throughout training, while the second moment continues to use the
raw gradient.


$$
\begin{aligned}
s_t &= r\, s_{t-1} + (1 - r)\, \lVert g_t \rVert \\
\tilde{g}_t &=
    \begin{cases}
        \dfrac{s_t}{\lVert g_t \rVert}\, g_t & s_t > \lVert g_t \rVert \\
        g_t & \text{otherwise}
    \end{cases} \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \tilde{g}_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t}\,
    \frac{m_t}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate,
$g_t$ is the gradient, $s_t$ is the running gradient norm with
decay $r$, $m_t$ and $v_t$ are the first and second
moments, and $\beta_1, \beta_2$ are their decay rates.

Reference: Shiv Ram Dubey, Satish Kumar Singh, Bidyut Baran Chaudhuri, "AdaNorm: Adaptive
Gradient Norm Correction based Optimizer for CNNs", WACV 2023.
https://arxiv.org/abs/2210.06364

---
[Back to the Canon](../index.md)
