# DEAM

Implements DEAM, an Adam-style optimizer whose momentum decay is set adaptively from the angle between the running momentum and the current gradient.

Instead of a fixed $\beta_1$, DEAM measures the angle $\theta$ between the normalized previous momentum and the current gradient. When the two roughly agree (small angle) the momentum is trusted and the new gradient is blended in lightly; when they disagree (angle past $\pi/2$) a constant weight is used. DEAM keeps the AMSGrad max-tracked second moment and adds a backtrack term that partially reverses the previous step when momentum and gradient point in opposite directions.

$$
\begin{aligned}
\theta &= \angle\!\left(\frac{m_{t-1}}{\sqrt{\hat{v}_{t-1}}},\; g_t\right) \\
\beta_{1,t} &= \begin{cases} \dfrac{\sin\theta}{K} + \epsilon, & \theta \in [0, \tfrac{\pi}{2}) \\ \dfrac{1}{K}, & \theta \in [\tfrac{\pi}{2}, \pi] \end{cases} \\
m_t &= (1-\beta_{1,t})\, m_{t-1} + \beta_{1,t}\, g_t \\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, g_t \odot g_t \\
\hat{v}_t &= \max(\hat{v}_{t-1},\, v_t) \\
d_t &= \min(0.5\cos\theta,\, 0) \\
\Delta_t &= d_t\, \Delta_{t-1} - \eta\, \frac{m_t}{\sqrt{\hat{v}_t}} \\
\theta_t &= \theta_{t-1} + \Delta_t
\end{aligned}
$$

where $\theta_t$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$/$v_t$ the first and second moments, $\hat{v}_t$ the running max of the second moment, $\beta_{1,t}$ the angle-adaptive first-moment weight, $\beta_2$ the fixed second-moment decay, $\Delta_t$ the step (with backtrack coefficient $d_t$), $K = 10(2+\pi)/(2\pi)$, and $\epsilon \approx 0.001$.

Reference: Jiyang Bai, Yuxiang Ren, Jiawei Zhang, "DEAM: Adaptive Momentum with Discriminative Weight for Stochastic Optimization", arXiv 2019. https://arxiv.org/abs/1907.11307

---
[Back to the Canon](../index.md)
