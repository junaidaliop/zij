# Adam

Implements Adam, an adaptive optimizer that combines momentum with
per-parameter step sizes derived from the second moment of the gradient.

Adam maintains exponential moving averages of the gradient $m_t$ and of the
squared gradient $v_t$. Because both averages start at zero they are biased
toward zero early in training, so Adam applies a bias correction to each
before forming the update. The parameter step rescales the corrected first
moment by the square root of the corrected second moment, giving each
coordinate an effective learning rate that shrinks where gradients are large
and noisy.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \frac{\eta\, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the
gradient, $m_t$ and $v_t$ are the first and second moment estimates,
$\hat{m}_t$ and $\hat{v}_t$ are their bias-corrected counterparts,
$\beta_1, \beta_2$ are the decay rates, and $\epsilon$ is a small constant for
numerical stability.

Reference: Diederik P. Kingma, Jimmy Ba, "Adam: A Method for Stochastic
Optimization", ICLR 2015.
https://arxiv.org/abs/1412.6980

---
[Back to the Canon](../README.md)
