# SWATS

Implements SWATS, switching from Adam to SGD during training.

Each parameter group starts in an Adam phase. After every Adam step the
method estimates the learning rate an equivalent SGD update would use by
projecting the Adam step $p_t$ onto the gradient, and tracks a
bias-corrected running average $\Lambda_t$ of that estimate. When the
estimate stabilizes, the group switches to SGD with momentum using
$\Lambda_t$ as its learning rate.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                          \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                         \\
     p_t &= -\frac{\eta\,\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
         \frac{m_t}{\sqrt{v_t} + \epsilon}                               \\
     \gamma_t &= \frac{p_t^\top p_t}{-\,p_t^\top g_t}                     \\
     \lambda_t &= \beta_2 \lambda_{t-1} + (1 - \beta_2)\gamma_t,
         \qquad \Lambda_t = \lambda_t / (1 - \beta_2^t)
\end{aligned}
$$

When $\Lambda_t \approx \gamma_t$ and $\Lambda_t > 0$, the group
switches to SGD with learning rate $\Lambda_t$.

Reference: Nitish Shirish Keskar, Richard Socher, "Improving Generalization
Performance by Switching from Adam to SGD", 2017.
https://arxiv.org/abs/1712.07628

---
[Back to the Canon](../README.md)
