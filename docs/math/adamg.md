# AdamG

Implements AdamG, a parameter-free Adam with the golden step size.


$$
\begin{aligned}
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
     r_t &= \beta_3 r_{t-1} + (1 - \beta_3) s(v_t),
         \quad s(x) = p \, x^q                                             \\
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) \, r_t \odot g_t               \\
     \hat{m}_t &= m_t / (1 - \beta_1^t), \quad
         \hat{v}_t = v_t / (1 - \beta_2^t)                                 \\
     \theta_t &= \theta_{t-1} - \min(\eta, 1/\sqrt{t}) \,
         \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
\end{aligned}
$$

Reference: Yijiang Pang, Shuyang Yu, Bao Hoang, Jiayu Zhou,
"Towards Stability of Parameter-free Optimization", arXiv 2024.
https://arxiv.org/abs/2405.04376

---
[Back to the Canon](../index.md)
