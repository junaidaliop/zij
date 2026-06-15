# DualAdam

Implements DualAdam, a convex blend of Adam and an inverse "InvAdam" update that anneals toward Adam during training.

DualAdam keeps Adam's first- and second-moment estimates but defines two competing steps from them: the usual Adam step $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$, which shrinks where the gradient variance is large, and an inverse step $\hat m_t\sqrt{\hat v_t}$, which instead grows there. Early training is dominated by the inverse step to push the iterates toward flatter regions, and a linearly decaying mixing weight $\alpha_t$ hands control back to plain Adam, recovering its convergence behavior.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
u_t &= \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}, \qquad \tilde u_t = \hat m_t \sqrt{\hat v_t} \\
\alpha_t &= \max(0,\ 1 - \xi t) \\
\theta_t &= \theta_{t-1} - \eta\,\big(\alpha_t\, \tilde u_t + (1-\alpha_t)\, u_t\big)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t,v_t$ the first and second moment estimates with bias-corrected forms $\hat m_t,\hat v_t$, $\beta_1,\beta_2$ the moment decays, $\epsilon$ the stability constant, $u_t$ the Adam step, $\tilde u_t$ the inverse (InvAdam) step, and $\xi$ the switching rate setting how fast $\alpha_t$ decays from InvAdam to Adam.

Reference: Tao Shi, Liangming Chen, Long Jin, Mengchu Zhou, "Combining Adam and its Inverse Counterpart to Enhance Generalization of Deep Learning Optimizers", arXiv 2026. https://arxiv.org/abs/2603.07122

---
[Back to the Canon](../README.md)
