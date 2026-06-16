# C-Adam

Implements C-Adam, an AMSGrad variant that blends the running and maximal second-moment estimates instead of taking a hard maximum.

AMSGrad enforces convergence by carrying the running maximum of the second moment, which can make the effective step size overly conservative. C-Adam replaces that hard maximum with a "line of sight" convex combination between the previous adaptive estimate $\tilde{v}_{t-1}$ and $\max(\tilde{v}_{t-1}, v_t)$, with a data-dependent mixing weight $\lambda$. This retains the non-increasing behavior needed for the convergence proof while relaxing the AMSGrad bound when the running estimate has not actually grown.

$$
\begin{aligned}
m_t &= \beta_{1,t}\, m_{t-1} + (1-\beta_{1,t})\, g_t \\
v_t &= \beta_2\, \tilde{v}_{t-1} + (1-\beta_2)\, g_t^2 \\
\lambda &= \frac{\tilde{v}_{t-1}}{\max(\tilde{v}_{t-1},\, v_t)} \\
\tilde{v}_t &= (1-\lambda)\max(\tilde{v}_{t-1},\, v_t) + \lambda\, \tilde{v}_{t-1} \\
\theta_{t+1} &= \Pi_{\mathcal{F},\sqrt{\tilde{v}_t}}\!\left(\theta_t - \alpha_t\, \frac{m_t}{\sqrt{\tilde{v}_t + \epsilon}}\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\alpha_t$ the step size, $g_t$ the gradient, $m_t$ the first moment with decay $\beta_{1,t}$, $v_t$ the raw second moment with decay $\beta_2$, $\tilde{v}_t$ the adaptive (line-of-sight) second moment, $\lambda \in [0,1]$ the convex-combination weight, $\epsilon$ a stability constant, and $\Pi_{\mathcal{F},\sqrt{\tilde{v}_t}}$ the projection onto the feasible set $\mathcal{F}$ under the $\sqrt{\tilde{v}_t}$-weighted norm.

Reference: Sakshi Kumari, Shyam Kumar M, Sushmitha P, "A Theoretical and Experimental Study of a Novel Adaptive Learning Algorithm", arXiv 2026. https://arxiv.org/abs/2605.29273

---
[Back to the Canon](../index.md)
