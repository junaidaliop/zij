# MGUP

Implements MGUP, a momentum-gradient alignment update policy that scales each coordinate's step size by whether its momentum and gradient agree.

MGUP is a plug-and-play wrapper for momentum-based optimizers (shown here on AdamW). At each step it forms an alignment score from the element-wise product of the update direction and the gradient, sorts the coordinates by that score, and selects the top $K = \lfloor \tau d \rfloor$ entries. Coordinates in this top-$K$ set get their step size amplified by $\alpha > 1$, while the rest get a smaller but nonzero factor $\gamma < 1$; in practice $\alpha = 1/\tau$ and $\gamma = \tau$. Unlike cautious updates that zero out misaligned coordinates, MGUP keeps every step nonzero, which the paper shows is needed to preserve stochastic convergence, and the greedy sorting raises the average update magnitude by a factor $1 + \tau - \tau^2$.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)(g_t \odot g_t) \\
u_t &= \frac{m_t}{\sqrt{v_t} + \epsilon}, \qquad \eta_t = \eta\,\frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t} \\
\phi_{t,i} &= \begin{cases} 1/\tau & i \in I_{\mathrm{topK}} \\ \tau & i \notin I_{\mathrm{topK}} \end{cases}, \qquad I_{\mathrm{topK}} = \text{top-}K\text{ indices of } (u_t \odot g_t),\; K = \lfloor \tau d \rfloor \\
\theta_t &\leftarrow (1 - \eta_t \lambda)\,\theta_t \\
\theta_{t+1} &= \theta_t - \eta_t\, \phi_t \odot u_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the base learning rate and $\eta_t$ its bias-corrected value, $g_t$ the stochastic gradient, $m_t$/$v_t$ the first and second moments, $u_t$ the AdamW update direction, $\beta_1,\beta_2$ the decay rates, $\lambda$ the weight decay, $\epsilon$ the stability constant, $\tau \in (0,1)$ the selection ratio (default $\tfrac12$), $\phi_t$ the per-coordinate step-size factor, and $I_{\mathrm{topK}}$ the index set of the $K$ coordinates with largest alignment score $u_t \odot g_t$.

Reference: Da Chang, Ganzhao Yuan, "MGUP: A Momentum-Gradient Alignment Update Policy for Stochastic Optimization", NeurIPS 2025. https://openreview.net/forum?id=TDFSKAspoQ

---
[Back to the Canon](../README.md)
