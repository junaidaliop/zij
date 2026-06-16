# Scheduled Weight Decay (SWD)

Implements Scheduled Weight Decay (SWD), an Adam variant that scales the weight-decay strength by the running gradient norm at each step.

SWD addresses a pitfall of fixed weight decay in adaptive optimizers: when gradient norms shrink, a constant decay coefficient over-penalizes the parameters. SWD instead schedules the decay by the inverse square root of $\bar{v}_t$, the mean of the bias-corrected second moment, so the effective regularization tracks the gradient magnitude. The momentum step is the ordinary Adam update; only the decoupled decay term is rescaled.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\bar{v}_t &= \mathrm{mean}(\hat{v}_t) \\
\theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t}+\epsilon}\, \hat{m}_t - \frac{\eta}{\sqrt{\bar{v}_t}+\epsilon}\, \lambda\, \theta_{t-1}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t,v_t$ the first/second moments with decays $\beta_1,\beta_2$, $\hat{m}_t,\hat{v}_t$ their bias-corrected forms, $\bar{v}_t$ the scalar mean of $\hat{v}_t$ acting as a gradient-norm-aware scheduler, $\lambda$ the weight decay, and $\epsilon$ a stability constant.

Reference: Zeke Xie, Zhiqiang Xu, Jingzhao Zhang, Issei Sato, Masashi Sugiyama, "On the Overlooked Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm Perspective", NeurIPS 2023. https://arxiv.org/abs/2011.11152

---
[Back to the Canon](../index.md)
