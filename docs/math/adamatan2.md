# AdamAtan2

Implements Adam-atan2, a scale-invariant epsilon-free variant of Adam.

The standard Adam update divides the bias-corrected first moment by the
square root of the bias-corrected second moment plus a small constant
$\epsilon$ to avoid division by zero. Adam-atan2 replaces that
division with $\mathrm{atan2}$, which removes the
$\epsilon$ hyperparameter and makes the update invariant to the scale
of the gradient.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                          \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                         \\
     \hat{m}_t &= m_t / (1 - \beta_1^t), \quad
         \hat{v}_t = v_t / (1 - \beta_2^t)                               \\
     \theta_t &= \theta_{t-1} - \gamma\, a \,
         \mathrm{atan2}\!\left(\hat{m}_t,\, b \sqrt{\hat{v}_t}\right)
\end{aligned}
$$

where $a$ and $b$ are fixed constants that recover the scale
and shape of the original Adam step.

Reference: Katie Everett et al., "Scaling Exponents Across Parameterizations
and Optimizers", ICML 2024.
https://arxiv.org/abs/2407.05872

---
[Back to the Canon](../index.md)
