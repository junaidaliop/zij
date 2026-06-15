# LaProp

Implements LaProp, which separates momentum from adaptivity in Adam.

LaProp divides the gradient by the second-moment estimate *before*
accumulating momentum, so the momentum buffer holds already-normalized
steps rather than raw gradients.


$$
\begin{aligned}
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1)
         \frac{g_t}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon}               \\
     \theta_t &= \theta_{t-1} - \frac{\eta}{1 - \beta_1^t} \, m_t
\end{aligned}
$$

The bias-correction terms are tracked as exponential moving averages so
that a learning rate that changes across steps is handled correctly.

Reference: Liu Ziyin, Zhikang T. Wang, Masahito Ueda,
"LaProp: Separating Momentum and Adaptivity in Adam", arXiv 2020.
https://arxiv.org/abs/2002.04839

---
[Back to the Canon](../README.md)
