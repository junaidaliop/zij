# CAdam

Implements CAdam, a confidence-based variant of Adam that pauses momentum-driven updates when momentum and the current gradient disagree in sign.

CAdam keeps Adam's first- and second-moment estimates and bias correction unchanged. Before each step it checks, per coordinate, whether the bias-corrected momentum $\hat{m}_t$ and the current gradient $g_t$ point the same way. Where they agree it updates as usual; where they disagree it zeroes that coordinate of $\hat{m}_t$, temporarily withholding the update so it can tell a genuine distribution shift (persistent disagreement) from transient noise (fleeting disagreement). An optional AMSGrad-style running maximum can be applied to the second moment.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\hat{v}_{\max,t} &= \max(\hat{v}_{\max,t-1},\, \hat{v}_t) \\
\hat{m}_t &\leftarrow \hat{m}_t \odot \mathbb{1}(m_t \odot g_t > 0) \\
\theta_t &= \theta_{t-1} - \gamma \, \frac{\hat{m}_t}{\sqrt{\hat{v}_{\max,t}} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\gamma$ the learning rate, $g_t$ the gradient, $m_t,v_t$ the first and second moments with decays $\beta_1,\beta_2$, $\hat{m}_t,\hat{v}_t$ their bias-corrected forms, $\hat{v}_{\max,t}$ the optional AMSGrad running maximum (set $\hat{v}_{\max,t}=\hat{v}_t$ when disabled), $\mathbb{1}(\cdot)$ the element-wise indicator that zeroes coordinates where momentum and gradient disagree in sign, $\odot$ element-wise product, and $\epsilon$ a small constant for numerical stability.

Reference: Shaowen Wang, Anan Liu, Jian Xiao, Huan Liu, Yuekui Yang, Cong Xu, Qianqian Pu, Suncong Zheng, Wei Zhang, Di Wang, Jie Jiang, Jian Li, "CAdam: Confidence-Based Optimization for Online Learning", arXiv 2024. https://arxiv.org/abs/2411.19647

---
[Back to the Canon](../README.md)
