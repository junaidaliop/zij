# Adai

Implements Adai (Adaptive Inertia), which disentangles the adaptive
learning rate of Adam into a parameter-wise adaptive momentum.


$$
\begin{aligned}
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
     \hat{v}_t &= \frac{v_t}{1 - \beta_2^t}                                \\
     \beta_{1,t} &= \left( 1 - \beta_0 \frac{\hat{v}_t}{\bar{v}_t} \right)
         \text{ clamped to } [0, 1 - \epsilon]                            \\
     m_t &= \beta_{1,t} \, m_{t-1} + (1 - \beta_{1,t}) g_t                 \\
     \hat{m}_t &= \frac{m_t}{1 - \prod_{i=1}^{t} \beta_{1,i}}              \\
     \theta_t &= \theta_{t-1} - \eta \, \hat{m}_t
\end{aligned}
$$

Unlike Adam, the adaptive second moment is not used to scale the step size
directly. Instead it modulates a parameter-wise inertia (momentum) factor
$\beta_{1,t}$: parameters whose bias-corrected second moment
$\hat{v}_t$ is large relative to the mean $\bar{v}_t$ over all
parameters receive a smaller momentum, while parameters with small second
moment are driven by heavier inertia. The first moment uses a per-parameter
cumulative product of the inertia factors for bias correction.

The `dampening` argument generalizes the rule: with $d$ the dampening,
the inertia exponent becomes $1 / (3 - 2 d)$, the gradient is scaled by
$(1 - \beta_{1,t})^d$, and the update is rescaled by
$\beta_0^{1 - d}$. The default $d = 1$ recovers the published
Adai update.

Reference: Zeke Xie, Xinrui Wang, Huishuai Zhang, Issei Sato, Masashi
Sugiyama, "Adaptive Inertia: Disentangling the Effects of Adaptive Learning
Rate and Momentum", ICML 2022.
https://arxiv.org/abs/2006.15815

---
[Back to the Canon](../README.md)
