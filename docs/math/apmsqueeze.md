# APMSqueeze

Implements APMSqueeze, a communication-efficient Adam-preconditioned momentum SGD with error-compensated 1-bit gradient compression.

Adam's per-coordinate variance preconditioner is nonlinear, which breaks the error-compensation analysis that makes 1-bit compression work for plain SGD. APMSqueeze sidesteps this in two phases. In a short warmup it runs ordinary Adam for $T_w$ steps to learn a good second-moment estimate, then freezes the variance at $v_{T_w}$. In the second phase it switches to momentum SGD preconditioned by the frozen $v_{T_w}$, so the only quantity that has to be communicated is the momentum, which can now be safely 1-bit compressed with local and global error feedback.

For $t \le T_w$ each worker runs standard Adam (updating both $m_t$ and $v_t$). For $t > T_w$ the variance is held fixed and each worker $i$ performs:

$$
\begin{aligned}
m_t^{(i)} &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t^{(i)}, \\
\hat{m}_t^{(i)} &= C_\omega\!\big[m_t^{(i)} + \delta_{t-1}^{(i)}\big], \\
\delta_t^{(i)} &= m_t^{(i)} + \delta_{t-1}^{(i)} - \hat{m}_t^{(i)}, \\
\bar{m}_t &= C_\omega\!\Big[\tfrac{1}{n}\textstyle\sum_{j=1}^{n}\hat{m}_t^{(j)} + \bar{\delta}_{t-1}\Big], \\
\theta_{t+1} &= \theta_t - \gamma_t\, \bar{m}_t \oslash \sqrt{v_{T_w}},
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma_t$ the learning rate, $g_t^{(i)}$ the local gradient on worker $i$ (of $n$ workers), $m_t$ the momentum with decay $\beta_1$, $v_{T_w}$ the second-moment estimate frozen after $T_w$ Adam warmup steps, $C_\omega[\cdot]$ the 1-bit compression operator, $\delta_t^{(i)}$ and $\bar{\delta}_t$ the local and global accumulated compression-error terms, and $\oslash$ element-wise division.

Reference: Hanlin Tang, Shaoduo Gan, Samyam Rajbhandari, Xiangru Lian, Ji Liu, Yuxiong He, Ce Zhang, "APMSqueeze: A Communication Efficient Adam-Preconditioned Momentum SGD Algorithm", arXiv 2020. https://arxiv.org/abs/2008.11343

---
[Back to the Canon](../index.md)
