# AdaMod

Implements AdaMod, an Adam variant that bounds the per-parameter
learning rates by an exponential moving average of their past values.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                               \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                              \\
     \eta_t &= \frac{\alpha \sqrt{1 - \beta_2^t}}{(1 - \beta_1^t)
         (\sqrt{v_t} + \epsilon)}                                             \\
     s_t &= \beta_3 s_{t-1} + (1 - \beta_3) \eta_t                            \\
     \hat{\eta}_t &= \min(\eta_t, s_t)                                        \\
     \theta_t &= \theta_{t-1} - \hat{\eta}_t \odot m_t
\end{aligned}
$$

The adaptive learning rate $\eta_t$ computed by Adam is smoothed by a
third exponential moving average $s_t$ with decay $\beta_3$, and
each element of the update is capped at this momental bound. This restrains
the large learning rates that can appear early in training.

Reference: Jianbang Ding, Xuancheng Ren, Ruixuan Luo, Xu Sun, "An Adaptive
and Momental Bound Method for Stochastic Learning", 2019.
https://arxiv.org/abs/1910.12249

---
[Back to the Canon](../README.md)
