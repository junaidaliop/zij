# RAdam

Implements RAdam, an Adam variant that rectifies the variance of the
adaptive learning rate during early training.

RAdam tracks the length of the exponential moving average that backs the
second moment through $\rho_t$, an estimate of the degrees of freedom of the
adaptive term, and its limiting value $\rho_\infty$. When $\rho_t > 4$ the
variance of the adaptive learning rate is tractable and RAdam applies a
rectification factor $r_t$ to the usual Adam step, scaling it down when the
estimate is still noisy and recovering Adam as $\rho_t \to \rho_\infty$. When
$\rho_t \le 4$ the adaptive term is unreliable and the step falls back to
momentum on the bias-corrected first moment, which removes the need for a
warmup schedule.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\rho_\infty &= \frac{2}{1 - \beta_2} - 1, \qquad
\rho_t = \rho_\infty - \frac{2 t \beta_2^t}{1 - \beta_2^t} \\
r_t &= \sqrt{\frac{(\rho_t - 4)(\rho_t - 2)\,\rho_\infty}
{(\rho_\infty - 4)(\rho_\infty - 2)\,\rho_t}} \\
\theta_t &=
\begin{cases}
\theta_{t-1} - \eta\, r_t \dfrac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
& \rho_t > 4 \\
\theta_{t-1} - \eta\, \hat{m}_t & \rho_t \le 4
\end{cases}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the
gradient, $m_t$ and $v_t$ are the first and second moments, $\beta_1, \beta_2$
are the decay rates, $\rho_t$ is the estimated length of the second-moment
average with limit $\rho_\infty$, $r_t$ is the rectification factor, and
$\epsilon$ is a numerical-stability term.

Reference: Liyuan Liu, Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu,
Jianfeng Gao, Jiawei Han, "On the Variance of the Adaptive Learning Rate and
Beyond", ICLR 2020.
https://arxiv.org/abs/1908.03265

---
[Back to the Canon](../README.md)
