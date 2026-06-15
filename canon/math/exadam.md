# EXAdam

Implements EXAdam, an extension of Adam with adaptive cross-moment
debiasing and a gradient-based acceleration term.

Following the official implementation, decoupled (AdamW-style) weight decay
is applied first when `weight_decay` is nonzero; it is not part of the
paper's Algorithm 1.


$$
\begin{aligned}
     \theta_{t-1} &\leftarrow \theta_{t-1} - \eta \lambda \theta_{t-1}
         \quad(\text{decoupled weight decay})                             \\
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
     d_1 &= 1 + \frac{v_t}{v_t + \epsilon} \beta_2^t                       \\
     d_2 &= 1 + \frac{m_t^2}{m_t^2 + \epsilon} \beta_1^t                   \\
     \tilde{m}_t &= \frac{m_t}{1 - \beta_1^t} \, d_1                       \\
     \tilde{v}_t &= \frac{v_t}{1 - \beta_2^t} \, d_2                       \\
     \tilde{g}_t &= \frac{g_t}{1 - \beta_1^t} \, d_1                       \\
     \theta_t &= \theta_{t-1} - \eta \,
         \frac{\tilde{m}_t + \tilde{g}_t}{\sqrt{\tilde{v}_t} + \epsilon}
\end{aligned}
$$

The cross-moment factors $d_1$ and $d_2$ rescale the first and
second moments using information from the other moment, sharpening the bias
correction in early steps. The current gradient enters the numerator through
$\tilde{g}_t$, which accelerates the update along the instantaneous
descent direction.

Reference: Ahmed M. Adly, "EXAdam: The Power of Adaptive Cross-Moments",
arXiv 2024.
https://arxiv.org/abs/2412.20302

---
[Back to the Canon](../README.md)
