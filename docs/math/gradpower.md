# GradPower

Implements GradPower, a sign-power gradient transformation applied before a base optimizer's update.

GradPower is a lightweight, plug-in modification that reshapes the raw gradient through an elementwise signed power before it enters the usual momentum and adaptive-scaling machinery. With exponent $p>1$ it sharpens large coordinates and suppresses small ones, while $p<1$ does the reverse; the sign is preserved so the descent direction per coordinate is unchanged. The example below instantiates it inside AdamW (AdamPower), where the transformation costs a single extra line over vanilla Adam.

$$
\begin{aligned}
g_t &\leftarrow |g_t|^{p}\,\mathrm{sign}(g_t) \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad \hat{m}_t = \frac{m_t}{1-\beta_1^{t}} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,g_t^{2}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{t}} \\
\theta_t &= \theta_{t-1} - \eta_t\!\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\,\theta_{t-1}\right)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t$ the gradient, $p>0$ the power exponent (e.g. $p=1.2$), $m_t,v_t$ the first and second moments with decay rates $\beta_1,\beta_2$, $\lambda$ the decoupled weight decay, and $\epsilon$ a stability constant; all powers and the sign act elementwise.

Reference: Mingze Wang, Jinbo Wang, Jiaqi Zhang, Wei Wang, Peng Pei, Xunliang Cai, Weinan E, Lei Wu, "GradPower: Powering Gradients for Faster Language Model Pre-Training", arXiv 2025. https://arxiv.org/abs/2505.24275

---
[Back to the Canon](../index.md)
