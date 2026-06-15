# AdamMC

Implements AdamMC, Adam with moment centralization.

AdamMC augments Adam with a centralization step on the first-order moment: before bias correction, the mean of the accumulated momentum is subtracted from it. Computed per layer over the momentum tensor, this enforces a zero-mean constraint on the momentum, which the authors find improves generalization for convolutional networks. The rest of the update is identical to Adam.

$$
\begin{aligned}
g_t &= \nabla_\theta f_t(\theta_{t-1}) \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
m_t &= m_t - \mathrm{mean}(m_t) \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ and $v_t$ the first- and second-order moments, $\beta_1,\beta_2$ the decay rates, $\mathrm{mean}(m_t)$ the mean of the momentum tensor (taken per layer), and $\epsilon$ a stability constant.

Reference: Sumanth Sadu, Shiv Ram Dubey, S. R. Sreeja, "Moment Centralization based Gradient Descent Optimizers for Convolutional Neural Networks", arXiv 2022. https://arxiv.org/abs/2207.09066

---
[Back to the Canon](../README.md)
