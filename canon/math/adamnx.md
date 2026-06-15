# AdamNX

Implements AdamNX, an Adam variant with a novel time-dependent exponential decay rate for the second-moment estimate.

AdamNX folds bias correction directly into the moment-update coefficients rather than applying a separate $\hat{m}_t, \hat{v}_t$ step. For the second moment it replaces Adam's fixed $\beta_2$ with an effective decay $\hat{\beta}_{2,t}$ that increases toward $1$ as training progresses, gradually weakening the per-coordinate step-size correction so the optimizer behaves more like momentum SGD in late training. Weight decay is decoupled and applied directly to the parameters.

$$
\begin{aligned}
m_t &= \frac{\beta_1 - \beta_1^t}{1 - \beta_1^t}\, m_{t-1} + \left(1 - \frac{\beta_1 - \beta_1^t}{1 - \beta_1^t}\right) g_t \\
\hat{\beta}_{2,t} &= \frac{1 - \beta_2^{(1-\beta_2)(t-1)}}{1 - \beta_2^{(1-\beta_2)\,t}} \\
v_t &= \hat{\beta}_{2,t}\, v_{t-1} + \left(1 - \hat{\beta}_{2,t}\right) g_t^2 \\
u_t &= \frac{m_t}{\sqrt{v_t} + \epsilon} \\
\theta_t &= \theta_{t-1} - \eta\left(u_t + \lambda\, \theta_{t-1}\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t$ and $v_t$ the first- and second-moment estimates, $\beta_1, \beta_2$ the base decay rates, $\hat{\beta}_{2,t}$ the effective time-varying second-moment decay, $\lambda$ the decoupled weight decay, and $\epsilon$ a stability constant (defaults $\beta_1 = 0.9$, $\beta_2 = 0.99$, $\epsilon = 10^{-8}$).

Reference: Meng Zhu, Quan Xiao, Weidong Min, "AdamNX: An Adam improvement algorithm based on a novel exponential decay mechanism for the second-order moment estimate", arXiv 2025. https://arxiv.org/abs/2511.13465

---
[Back to the Canon](../README.md)
