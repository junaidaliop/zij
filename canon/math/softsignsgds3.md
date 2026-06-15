# SoftSignSGD (S3)

Implements SoftSignSGD (S3), a sign-style optimizer whose Nesterov momentum is normalized by a $p$-th-power running average of the gradient magnitude.

S3 replaces Adam's second-moment denominator with a $p$-norm aggregate of past and current gradients, so the per-coordinate step $n_t/b_t$ stays bounded in $[-1,1]$ much like a softsign of the gradient. A single decay $\beta$ drives both the momentum $m_t$ and the magnitude track $s_t$, and a Nesterov-style combination $n_t$ feeds the numerator. The bounded, sign-like step damps the loss spikes that destabilize large-model training, and as $p\to\infty$ the rule approaches signSGD with momentum.

$$
\begin{aligned}
m_t &= \beta\, m_{t-1} + (1-\beta)\, g_t \\
s_t &= \beta\, s_{t-1} + (1-\beta)\, |g_t|^p \\
n_t &= \beta\, m_t + (1-\beta)\, g_t \\
b_t &= \left(\beta\, s_t + (1-\beta)\, |g_t|^p\right)^{1/p} \\
\theta_t &= \theta_{t-1} - \gamma_t\, \frac{n_t}{b_t}
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma_t$ the learning rate, $g_t$ the gradient, $m_t$ the first moment, $s_t$ the $p$-th-power magnitude track, $n_t$ the Nesterov momentum, $b_t$ the $p$-norm normalizer, $\beta\in[0,1)$ the single decay rate, and $p\ge 1$ the momentum order (default $p=3$).

Reference: Hanyang Peng, Shuang Qin, Yue Yu, Fangqing Jiang, Hui Wang, Wen Gao, "SoftSignSGD(S3): An Enhanced Optimizer for Practical DNN Training and Loss Spikes Minimization Beyond Adam", 2025. https://arxiv.org/abs/2507.06464

---
[Back to the Canon](../README.md)
