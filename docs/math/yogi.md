# Yogi

Implements Yogi, an adaptive method that controls the increase in the
effective learning rate.

Yogi replaces the multiplicative second-moment update of Adam with an
additive, sign-based one, so that the second moment can decrease as well as
increase and large gradients do not cause it to grow uncontrollably.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                          \\
     v_t &= v_{t-1} - (1 - \beta_2)
         \mathrm{sign}(v_{t-1} - g_t^2) \, g_t^2                   \\
     \hat{v}_t &= v_t / (1 - \beta_2^t)                                  \\
     \theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t}
         \frac{m_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$


**Note:** Following the upstream implementation, both the first moment $m_0$ and second moment $v_0$ are initialized to `initial_accumulator` (default `1e-6`) rather than to zero.

Reference: Manzil Zaheer, Sashank J. Reddi, Devendra Sachan,
Satyen Kale, Sanjiv Kumar, "Adaptive Methods for Nonconvex Optimization",
NeurIPS 2018.
https://papers.nips.cc/paper_files/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html

---
[Back to the Canon](../index.md)
