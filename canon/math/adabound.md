# AdaBound

Implements AdaBound, Adam with a dynamic bound on the learning rate.


$$
\begin{aligned}
     m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t                            \\
     v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2                          \\
     \hat{\eta}_t &= \mathrm{Clip}\!\left(
         \frac{\eta \sqrt{1 - \beta_2^t}}{(1 - \beta_1^t)
             (\sqrt{v_t} + \epsilon)},
         \eta_l(t), \eta_u(t)\right)                                       \\
     \eta_l(t) &= \eta^* \left(1 - \frac{1}{\gamma t + 1}\right),
         \quad \eta_u(t) = \eta^* \left(1 + \frac{1}{\gamma t}\right)      \\
     \theta_t &= \theta_{t-1} - \hat{\eta}_t \odot m_t
\end{aligned}
$$

where $\eta^*$ is the final (SGD) learning rate. The lower and upper
bounds converge to $\eta^*$ as $t \to \infty$, so AdaBound
transitions smoothly from Adam to SGD.

Reference: Liangchen Luo, Yuanhao Xiong, Yan Liu, Xu Sun, "Adaptive Gradient
Methods with Dynamic Bound of Learning Rate", ICLR 2019.
https://arxiv.org/abs/1902.09843

---
[Back to the Canon](../README.md)
