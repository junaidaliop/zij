# Stable-SPAM / GradientStabilizer

Implements Stable-SPAM, an Adam variant that stabilizes low-precision training by adaptively clipping and normalizing gradients before the moment update.

Stable-SPAM augments Adam with three mechanisms. AdaClip detects spike entries whose magnitude exceeds an exponentially tracked, bias-corrected threshold and rescales them down to that threshold. AdaGN (the GradientStabilizer) normalizes the whole gradient by its current norm and reweights it by a bias-corrected ratio of first to second moments of past gradient norms, damping bursts that would otherwise destabilize 4-bit and BF16 optimizer states. Periodic momentum reset (every $\Delta T$ steps the moments $m,v$ are zeroed) keeps stale moments from amplifying instability. The cleaned gradient then drives a standard Adam update.

$$
\begin{aligned}
g_{\max} &= \max_i |g_t[i]|, &
T_t &= \gamma_3 T_{t-1} + (1-\gamma_3)\,g_{\max}, &
\hat T_t &= \frac{T_t}{1-\gamma_3^{\,t}}, \\
\tilde g_t[i] &= \begin{cases} \dfrac{g_t[i]}{g_{\max}}\,\hat T_t & |g_t[i]| > \hat T_t \\ g_t[i] & \text{otherwise} \end{cases} \\
n_t &= \|\tilde g_t\|_2, &
\mu_t &= \gamma_1 \mu_{t-1} + (1-\gamma_1)\,n_t, &
\nu_t &= \gamma_2 \nu_{t-1} + (1-\gamma_2)\,n_t^2, \\
\hat\mu_t &= \frac{\mu_t}{1-\gamma_1^{\,t}}, &
\hat\nu_t &= \frac{\nu_t}{1-\gamma_2^{\,t}}, &
\hat g_t &= \frac{\tilde g_t}{n_t}\cdot\frac{\hat\mu_t}{\sqrt{\hat\nu_t}+\epsilon}, \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,\hat g_t, &
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,\hat g_t^{\,2}, \\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, &
\hat v_t &= \frac{v_t}{1-\beta_2^{\,t}}, &
\theta_t &= \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $\tilde g_t$ its spike-clipped version and $\hat g_t$ the further normalized gradient, $T_t$ the tracked spike threshold ($\gamma_3 = 0.999$), $\mu_t,\nu_t$ the first and second moments of the gradient norm $n_t$ ($\gamma_1,\gamma_2$ controlling their decay), $m_t,v_t$ the Adam moments with decays $\beta_1,\beta_2$, $\epsilon$ a stability constant, and the moments $m,v$ are reset to zero every $\Delta T$ steps.

Reference: Tianjin Huang, Haotian Hu, Zhenyu Zhang, Gaojie Jin, Xiang Li, Li Shen, Tianlong Chen, Lu Liu, Qingsong Wen, Zhangyang Wang, Shiwei Liu, "Stable-SPAM: How to Train in 4-Bit More Stably than 16-Bit Adam", ICML 2025. https://arxiv.org/abs/2502.17055

---
[Back to the Canon](../README.md)
