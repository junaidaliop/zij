# 1-bit Adam

Implements 1-bit Adam, a communication-efficient Adam that freezes the second moment after a warmup phase and transmits an error-compensated 1-bit compression of the momentum.

The paper observes that Adam's second moment $v_t$ stabilizes early in training, while the nonlinear dependence of the update on $v_t$ blocks the error-feedback compression that works for plain momentum SGD. 1-bit Adam runs vanilla Adam for a warmup of $T_w$ steps, then freezes $v_{T_w}$ and switches to a compression phase: each worker compresses its momentum with a 1-bit (sign-based) operator $C_\omega$, carries the residual forward as local error feedback $\delta_t$, and the averaged compressed momenta are compressed again on the server, so only 1-bit messages cross the network while the frozen variance still preconditions the step.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
\hat{m}_t &= C_\omega\!\left[ m_t + \delta_{t-1} \right] \\
\delta_t &= m_t + \delta_{t-1} - \hat{m}_t \\
\bar{m}_t &= C_\omega\!\left[ \tfrac{1}{n}\sum_{i=1}^{n} \hat{m}_t^{(i)} + \bar{\delta}_{t-1} \right] \\
\bar{\delta}_t &= \tfrac{1}{n}\sum_{i=1}^{n} \hat{m}_t^{(i)} + \bar{\delta}_{t-1} - \bar{m}_t \\
\theta_t &= \theta_{t-1} - \frac{\gamma}{\sqrt{v_{T_w}} + \epsilon}\, \bar{m}_t
\end{aligned}
$$

where $g_t$ is the gradient, $m_t$ is the local first moment, $v_{T_w}$ is the second moment frozen at the end of warmup, $C_\omega[\cdot]$ is the error-compensated 1-bit compression operator, $\delta_t$ and $\bar{\delta}_t$ are the worker and server compression-error (feedback) terms, $\hat{m}_t^{(i)}$ is the compressed momentum from worker $i$, $n$ is the number of workers, $\beta_1, \beta_2 \in [0,1)$ are the decay rates, $\gamma$ is the learning rate, and $\epsilon$ is a small constant for numerical stability. During the warmup phase ($t \le T_w$) the update reduces to standard Adam with $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$.

Reference: Hanlin Tang, Shaoduo Gan, Ammar Ahmad Awan, Samyam Rajbhandari, Conglong Li, Xiangru Lian, Ji Liu, Ce Zhang, Yuxiong He, "1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed", ICML 2021. https://arxiv.org/abs/2102.02888

---
[Back to the Canon](../README.md)
