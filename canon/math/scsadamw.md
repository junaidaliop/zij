# SCSAdamW

Implements SCSAdamW, an AdamW variant that replaces the first moment with a stochastic conjugate subgradient direction.

Instead of an exponentially weighted first moment, SCSAdamW builds the search direction by optimally blending the previous direction $d_{t-1}$ with the current gradient $g_t$. The blend weight $\lambda_t^\ast$ comes from a one-dimensional projected line search that minimizes the norm of the combined direction over the segment between $d_{t-1}$ and $g_t$, clamped to $[0,1]$. This conjugate direction is then bias-corrected, divided by the AdamW-style RMS of the gradient, and applied with decoupled weight decay.

$$
\begin{aligned}
\lambda_t^\ast &= \Pi_{[0,1]}\!\left(\frac{-\langle d_{t-1}, g_t\rangle + \lVert g_t\rVert^2}{\lVert d_{t-1}\rVert^2 - 2\langle d_{t-1}, g_t\rangle + \lVert g_t\rVert^2}\right) \\
d_t &= (1-\lambda_t^\ast)\, d_{t-1} + \lambda_t^\ast\, g_t \quad (t>1), \qquad d_1 = g_1 \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \\
\hat{d}_t &= \frac{d_t}{1-(\lambda_t^\ast)^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \\
\theta_t &= \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t}+\zeta}\, \hat{d}_t \\
\theta_t &\leftarrow \theta_t - \eta\lambda\,\theta_{t-1}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $d_t$ the conjugate subgradient direction with line-search weight $\lambda_t^\ast \in [0,1]$, $v_t$ the second moment with decay $\beta_2$, $\Pi_{[0,1]}$ projection onto the unit interval, $\lambda$ the decoupled weight decay, and $\zeta$ a small stability constant.

Reference: Di Zhang, Yihang Zhang, "Beyond First-Order: Training LLMs with Stochastic Conjugate Subgradients and AdamW", arXiv preprint 2025. https://arxiv.org/abs/2507.01241

---
[Back to the Canon](../README.md)
