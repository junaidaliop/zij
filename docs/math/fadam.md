# FAdam

Implements FAdam (Fisher Adam), recasting Adam as natural gradient descent
with a diagonal empirical Fisher information matrix.

This is Fisher Adam, not a fractional Adam variant: the name FAdam refers to
the Fisher information interpretation, where the second-moment buffer is read
as a diagonal empirical Fisher and the update is a natural gradient step.


$$
\begin{aligned}
     f_t &= \beta_2 f_{t-1} + (1 - \beta_2) g_t^2                             \\
     \bar{g}_t &= \frac{g_t}{f_t^{p} + \epsilon}                              \\
     \hat{g}_t &= \frac{\bar{g}_t}{\max(1, \lVert \bar{g}_t \rVert_{rms} / c)}\\
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \hat{g}_t                         \\
     w_t &= \frac{\theta_{t-1}}{f_t^{p} + \epsilon}                           \\
     \hat{w}_t &= \frac{w_t}{\max(1, \lVert w_t \rVert_{rms} / c)}            \\
     \theta_t &= \theta_{t-1} - \eta \left( m_t + \lambda \hat{w}_t \right)
\end{aligned}
$$

The buffer $f_t$ accumulates the squared gradient as a diagonal
empirical Fisher. The gradient is divided by $f_t^{p}$ to form the
natural gradient $\bar{g}_t$ (with $p = 1/2$ recovering the Adam
denominator), both the natural gradient and the weight-decay term are
root-mean-square clipped to a maximum norm $c$, and momentum is applied
to the clipped natural gradient. The decoupled weight decay $\lambda$
is itself preconditioned by the Fisher.


**Note:** following the official implementation, the Fisher EMA uses a debiased

decay $\hat{\beta}_2 = \beta_2 (1 - \beta_2^{t-1}) / (1 - \beta_2^{t})$
in place of $\beta_2$, and the stability constant is scaled by the
gradient RMS, $\epsilon_t = \min(\mathrm{RMS}(g_t), 1)\,\epsilon$, so
the denominator is $f_t^p + \epsilon_t$.

Reference: Dongseong Hwang, "FAdam: Adam is a natural gradient optimizer using
diagonal empirical Fisher information", 2024.
https://arxiv.org/abs/2405.12807

---
[Back to the Canon](../index.md)
