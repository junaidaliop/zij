# CAME

Implements CAME, a confidence-guided variant of Adafactor-style factored optimization.


$$
\begin{aligned}
r_t &= \beta_2 r_{t-1} + (1 - \beta_2)\,
       \bigl(g_t^2 + \epsilon_1 1_n 1_m^\top\bigr) 1_m \\
c_t &= \beta_2 c_{t-1} + (1 - \beta_2)\,
       1_n^\top \bigl(g_t^2 + \epsilon_1 1_n 1_m^\top\bigr) \\
v_t &= r_t c_t / (1_n^\top r_t) \\
u_t &= g_t / \sqrt{v_t} \\
\hat{u}_t &= u_t / \max\bigl(1, \mathrm{RMS}(u_t) / d\bigr) \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \hat{u}_t \\
U_t &= (\hat{u}_t - m_t)^2 \\
R_t &= \beta_3 R_{t-1} + (1 - \beta_3)\,
       \bigl(U_t + \epsilon_2 1_n 1_m^\top\bigr) 1_m \\
C_t &= \beta_3 C_{t-1} + (1 - \beta_3)\,
       1_n^\top \bigl(U_t + \epsilon_2 1_n 1_m^\top\bigr) \\
S_t &= R_t C_t / (1_n^\top R_t) \\
\theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{S_t}}\, m_t
\end{aligned}
$$

where $d$ is the clipping threshold, $\epsilon_1$ and
$\epsilon_2$ are the regularization constants given by `eps`, and
$(\beta_1, \beta_2, \beta_3)$ are the decay rates of the update,
square-gradient, and instability moving averages. Parameters with fewer
than two dimensions are not factored and skip the confidence-guided
correction.

Reference: Yang Luo, Xiaozhe Ren, Zangwei Zheng, Zhuo Jiang, Xin Jiang,
Yang You, "CAME: Confidence-guided Adaptive Memory Efficient Optimization",
ACL 2023.
https://arxiv.org/abs/2307.02047

---
[Back to the Canon](../README.md)
