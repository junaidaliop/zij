# AggMo

Implements AggMo, momentum descent aggregated over several damping rates.

AggMo keeps one velocity buffer per damping coefficient $\beta^{(i)}$
and averages their contributions, so that the small coefficients react
quickly to gradient changes while the large ones supply passive damping:


$$
\begin{aligned}
     v_t^{(i)} &= \beta^{(i)} v_{t-1}^{(i)} - g_t,
         \quad i = 1, \dots, K                                       \\
     \theta_t &= \theta_{t-1} + \frac{\eta}{K}
         \sum_{i=1}^{K} v_t^{(i)}
\end{aligned}
$$

Reference: James Lucas, Shengyang Sun, Richard Zemel, Roger Grosse,
"Aggregated Momentum: Stability Through Passive Damping", ICLR 2019.
https://arxiv.org/abs/1804.00325

---
[Back to the Canon](../index.md)
