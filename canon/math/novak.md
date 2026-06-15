# NOVAK

Implements NOVAK, a unified adaptive optimizer combining Adam moments, RAdam rectification, decoupled weight decay, LAMB-style trust scaling, and lookahead synchronization.

NOVAK maintains the standard first and second moments with bias correction, then forms an effective learning rate by multiplying the base rate by the RAdam variance-rectification factor $r_t$, a LAMB trust ratio $\lambda_{\mathrm{trust}}$, and a gradient-norm autoscaling factor $\alpha_{\mathrm{auto}}$. The parameter step subtracts a decoupled weight-decay term (using the base rate) plus the rectified, adaptively scaled momentum direction. An outer lookahead loop periodically pulls fast weights toward slow weights every $k$ steps.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t, & v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, & \hat{v}_t &= \frac{v_t}{1-\beta_2^t}, \\
\rho_\infty &= \frac{2}{1-\beta_2} - 1, & \rho_t &= \rho_\infty - \frac{2 t \beta_2^t}{1-\beta_2^t}, \\
r_t &= \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_\infty}{(\rho_\infty-4)(\rho_\infty-2)\rho_t}} \;\; (\rho_t \ge 5), & \alpha_{\mathrm{eff}} &= \alpha \, r_t \, \lambda_{\mathrm{trust}} \, \alpha_{\mathrm{auto}}, \\
\theta_{t+1} &= \theta_t - \alpha \lambda \, \theta_t - \alpha_{\mathrm{eff}} \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
\end{aligned}
$$

where $\theta$ are the parameters, $\alpha$ the base learning rate, $g_t$ the gradient, $m_t,v_t$ the first and second moments with bias-corrected counterparts $\hat{m}_t,\hat{v}_t$, $\beta_1,\beta_2$ the decay rates, $\lambda$ the decoupled weight-decay coefficient, $\epsilon$ a stability constant, $r_t$ the RAdam rectification factor (set to $1$ when $\rho_t < 5$), $\lambda_{\mathrm{trust}} = \lVert \theta_t \rVert_2 / \lVert u_t \rVert_2$ the LAMB-style trust ratio, and $\alpha_{\mathrm{auto}} = 1/(1+\log \mathrm{EMA}(\lVert g_t \rVert_2))$ the gradient-norm autoscaling factor.

Reference: Sergii Kavun, "NOVAK: Unified adaptive optimizer for deep neural networks", arXiv 2026. https://arxiv.org/abs/2601.07876

---
[Back to the Canon](../README.md)
