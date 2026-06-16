# AdaPNM

Implements AdaPNM, the adaptive (Adam) form of positive-negative
momentum.


$$
\begin{aligned}
     m_t &= \beta_1^2 m_{t-2} + (1 - \beta_1^2) g_t                        \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
     \hat{v}_t &= \max(\hat{v}_{t-1}, v_t)                                 \\
     \pi_t &= \frac{(1 + \beta_3) m_t - \beta_3 m_{t-1}}
         {\sqrt{(1 + \beta_3)^2 + \beta_3^2}}                              \\
     \theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t}
         \frac{\pi_t}{\sqrt{\hat{v}_t / (1 - \beta_2^t)} + \epsilon}
\end{aligned}
$$

Two momentum buffers are kept and their roles swap every step, so the buffer
that receives the current gradient is decayed by $\beta_1^2$ and is two
steps stale relative to itself. The update direction $\pi_t$ mixes the
fresh positive momentum $m_t$ with the previous (negative) momentum
$m_{t-1}$ and renormalizes by
$\sqrt{(1 + \beta_3)^2 + \beta_3^2}$ so that its variance matches a
plain momentum term. The difference amplifies the stochastic gradient noise,
which the paper links to improved generalization. The denominator is the Adam
second moment, taken with the AMSGrad running maximum when `ams_bound` is
set.

Reference: Zeke Xie, Li Yuan, Zhanxing Zhu, Masashi Sugiyama, "Positive-
Negative Momentum: Manipulating Stochastic Gradient Noise to Improve
Generalization", ICML 2021.
https://arxiv.org/abs/2103.17182

---
[Back to the Canon](../index.md)
