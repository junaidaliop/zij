# Grams

Implements Grams, gradient descent with adaptive momentum scaling.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
     \hat{m}_t &= m_t / (1 - \beta_1^t), \quad
         \hat{v}_t = v_t / (1 - \beta_2^t)                                 \\
     \theta_t &= \theta_{t-1} - \eta \,
         \mathrm{sign}(g_t) \odot
         \frac{\lvert \hat{m}_t \rvert}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

The update direction is taken from the sign of the current gradient, while
the Adam-style first moment supplies only the per-coordinate magnitude, so
direction and magnitude are decoupled. Bias correction is applied through
the step size when `correct_bias` is set, and weight decay is decoupled.

Reference: Yang Cao, Xiaoyu Li, Zhao Song,
"Grams: Gradient Descent with Adaptive Momentum Scaling", arXiv 2024.
https://arxiv.org/abs/2412.17107

---
[Back to the Canon](../index.md)
