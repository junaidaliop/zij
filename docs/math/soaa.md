# SOAA

Implements SOAA (Second-Order Adaptive Adam), an Adam variant that scales the step by a diagonal Fisher approximation inside an adaptive trust region.

SOAA keeps Adam's bias-corrected first and second moments but augments the denominator with a diagonal Fisher information estimate $F_t$ built from the moments. The effective step size is clamped by a trust-region scale that takes the elementwise maximum of $d_t F_t$ and $\sqrt{\hat{v}_t}$, and the trust-region radius $d_t$ is rescaled each step by the ratio of actual to predicted loss reduction, so the optimizer expands the step when predictions are accurate and contracts it otherwise.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
F_t &= \left(1 + \frac{\sum_i \hat{m}_{t,i}^2}{\sum_i (\hat{v}_{t,i} + \epsilon)}\right) \hat{v}_t \\
r_t &= \max\!\left(d_t F_t,\ \sqrt{\hat{v}_t}\right) \\
\theta_t &= \theta_{t-1} - \eta \lambda \theta_{t-1} - \eta\,\frac{\hat{m}_t\, d_t}{r_t + \epsilon} \\
d_{t} &= \min\!\left(\max\!\left(\tfrac{\hat{\ell}-\ell_t}{\max(p_t,\epsilon)}\, d_{t-1},\ (1-\gamma)^{(t-1)/T}\right),\ 1+\gamma^{(t-1)/T}\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t,v_t$ the first and second moment estimates with bias-corrected forms $\hat{m}_t,\hat{v}_t$, $\beta_1,\beta_2$ their decay rates, $\lambda$ the weight decay, $\epsilon$ a stability constant, $F_t$ the diagonal Fisher approximation, $r_t$ the trust-region scale, $d_t$ the trust-region radius, $\hat{\ell}-\ell_t$ the actual loss reduction, $p_t$ the predicted reduction, $\gamma$ the radius bound factor, and $T$ the total number of steps.

Reference: James Vo and Anh-Dung Vo, "Efficient Second-Order Neural Network Optimization via Adaptive Trust Region Methods", arXiv preprint 2024. https://arxiv.org/abs/2410.02293

---
[Back to the Canon](../index.md)
