# AdaL

Implements AdaL, an Adam variant that transforms the gradient by its $\ell_1$-norm before accumulating moments.

AdaL keeps Adam's structure but applies an adaptive gradient transformation: each gradient is scaled by its own $\ell_1$-norm prior to the moment updates. This amplifies large gradients early in training to speed convergence and damps them near a minimum to improve generalization. The step size is annealed as $\eta_t = \eta/\sqrt{t}$, and a small $\epsilon$ is added inside the preconditioner rather than to the denominator.

$$
\begin{aligned}
\hat{g}_t &= \lVert g_t \rVert_1\, g_t \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\hat{g}_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\hat{g}_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\eta_t &= \frac{\eta}{\sqrt{t}} \\
\theta_{t+1} &= \theta_t - \eta_t\left(\hat{v}_t^{-1/2} + \epsilon\right)\hat{m}_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the base learning rate, $g_t$ the gradient, $\lVert g_t \rVert_1$ its $\ell_1$-norm, $\hat{g}_t$ the transformed gradient, $m_t, v_t$ the first and second moment estimates with decay rates $\beta_1, \beta_2$, and $\epsilon$ a stability constant.

Reference: Hongwei Zhang, Weidong Zou, Hongbo Zhao, Qi Ming, Tijin Yan, Yuanqing Xia, Weipeng Cao, "AdaL: Adaptive Gradient Transformation Contributes to Convergences and Generalizations", arXiv 2021. https://arxiv.org/abs/2107.01525

---
[Back to the Canon](../index.md)
