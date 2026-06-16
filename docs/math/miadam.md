# MIAdam

Implements MIAdam, Adam with multiple-integral momentum smoothing for flatter minima.

MIAdam (Multiple-Integral Adam) replaces Adam's single momentum accumulation with an $n$-fold nested summation that discretizes repeated integration of the gradient signal. Each integration order applies a decay factor $\kappa$, smoothing the optimization trajectory so the optimizer is steered away from sharp minima and toward flat regions of the loss landscape, which the authors associate with better generalization.

The multiple-integral term is used only in the early phase of training. After a switching step $\zeta$ the optimizer reverts to standard Adam to guarantee convergence.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \\
\bar{m}_t^{(0)} &= m_t, \qquad \bar{m}_t^{(j)} = \kappa\, \bar{m}_{t-1}^{(j)} + \bar{m}_t^{(j-1)} \quad (j = 1,\dots,n), \\
\hat{m}_t &= \frac{\bar{m}_t^{(n)}}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \\
\theta_t &= \theta_{t-1} - \frac{\eta^{n}\, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \qquad (t < \zeta),
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ and $v_t$ the first and second moments with decays $\beta_1,\beta_2$, $\bar{m}_t^{(j)}$ the $j$-th order integral accumulator with integration rate $\kappa$, $n$ the integration order, $\zeta$ the switching step, and $\epsilon$ a stability constant. For $t \ge \zeta$ the update uses the plain Adam step $\theta_t = \theta_{t-1} - \eta\, \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$ with $\hat{m}_t = m_t/(1-\beta_1^t)$.

Reference: Long Jin, Han Nong, Liangming Chen, Zhenming Su, "A Method for Enhancing Generalization of Adam by Multiple Integrations", arXiv 2024. https://arxiv.org/abs/2412.12473

---
[Back to the Canon](../index.md)
