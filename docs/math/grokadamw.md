# GrokAdamW

Implements GrokAdamW, AdamW with Grokfast-style amplification of slow-varying gradients.


$$
\begin{aligned}
\alpha_t &= \alpha_{\text{init}}\, e^{-\kappa s_t} \\
\mu_t &= \alpha_t \mu_{t-1} + (1 - \alpha_t)\, g_t \\
\hat{g}_t &= g_t + \lambda\, \mu_t \\
\beta_1^{(l)} &= \beta_1 (1 - \gamma)^l \\
m_t &= \beta_1^{(l)} m_{t-1} + (1 - \beta_1^{(l)})\, \hat{g}_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, \hat{g}_t^2 \\
\theta_t &= (1 - \eta w)\, \theta_{t-1}
            - \eta\, \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}\,
              \frac{m_t}{\sqrt{v_t} + \epsilon}
\end{aligned}
$$

where $\mu_t$ is the slow-gradient EMA, $s_t$ the grokking
signal averaged over `grokking_signal_fns`, $\kappa$ the signal
decay rate, $\lambda$ the amplification factor `lamb`, $w$
the decoupled weight decay, and $l$ the index of the parameter
within its group, so $\gamma$ decays the momentum of later layers.
Gradients are norm-clipped per parameter before the update when
`gradient_clipping` is positive.


**Note:** When no `grokking_signal_fns` are given, the signal is computed from `train_loss` and `eval_loss` entries set on the parameter group and is zero while those are absent. Optimizer state is kept in CPU memory and moved to the parameter device for each step.

GrokAdamW was written by Eric Hartford and has no dedicated paper; its
slow-gradient amplification follows Grokfast.

Reference: Jaerin Lee, Bong Gyun Kang, Kihoon Kim, Kyoung Mu Lee,
"Grokfast: Accelerated Grokking by Amplifying Slow Gradients", arXiv 2024.
https://arxiv.org/abs/2405.20233

---
[Back to the Canon](../index.md)
