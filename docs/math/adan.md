# Adan

Implements Adan, an adaptive optimizer with Nesterov-style momentum.


$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, (g_t - g_{t-1}) \\
n_t &= \beta_3 n_{t-1} + (1 - \beta_3)\,
       \bigl(g_t + \beta_2 (g_t - g_{t-1})\bigr)^2 \\
\eta_t &= \frac{\eta}{\sqrt{n_t / (1 - \beta_3^t)} + \epsilon} \\
\theta_t &= \frac{1}{1 + \lambda \eta}\left(\theta_{t-1}
            - \eta_t \odot \Bigl(\frac{m_t}{1 - \beta_1^t}
            + \beta_2 \frac{v_t}{1 - \beta_2^t}\Bigr)\right)
\end{aligned}
$$

where $m_t$ is the gradient moment, $v_t$ is the moment of the
gradient difference, $n_t$ is the second moment of the
Nesterov-corrected gradient, and $\lambda$ is the decoupled weight
decay. The decay rates $(\beta_1, \beta_2, \beta_3)$ are passed as
`betas`. When `no_prox` is set, the weight decay multiplies
$\theta_{t-1}$ before the gradient step rather than being applied
proximally.

Reference: Xingyu Xie, Pan Zhou, Huan Li, Zhouchen Lin, Shuicheng Yan,
"Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep
Models", 2022.
https://arxiv.org/abs/2208.06677

---
[Back to the Canon](../index.md)
