# WarpAdam

Implements WarpAdam, Adam with a learnable distortion matrix applied to the gradient.

WarpAdam imports the warped-gradient-descent idea from meta-learning into Adam. Instead of feeding the raw gradient into the moment estimates, it first linearly transforms it by a square matrix $P$ learned across tasks. The matrix preconditions the gradient so the optimizer can adapt to the characteristics of a given dataset, while the rest of the Adam machinery (exponential moving averages, bias correction, adaptive step) is unchanged.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\,(P g_t) \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\,(P g_t)^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}} \\
\hat{v}_t &= \frac{v_t}{1 - \beta_2^{\,t}} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\,\hat{m}_t
\end{aligned}
$$

where $g_t = \nabla_\theta f(\theta_t)$ is the gradient, $P$ is an $n \times n$ distortion matrix learned by meta-learning, $m_t, v_t$ are the first and second moment estimates with decay rates $\beta_1, \beta_2$, $\hat{m}_t, \hat{v}_t$ are their bias-corrected forms, $\eta$ is the learning rate, and $\epsilon$ guards the denominator.

Reference: Chengxi Pan, Junshang Chen, Jingrui Ye, "WarpAdam: A new Adam optimizer based on Meta-Learning approach", arXiv 2024. https://arxiv.org/abs/2409.04244

---
[Back to the Canon](../README.md)
