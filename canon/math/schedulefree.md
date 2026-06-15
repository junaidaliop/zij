# ScheduleFree+

Implements ScheduleFree+, a learning-rate-free and schedule-free variant of AdamW that adds a Polyak step size.

ScheduleFree+ keeps the three-sequence structure of Schedule-Free learning: an averaged iterate $x_t$, a raw optimizer iterate $z_t$, and an evaluation point $y_t$ that interpolates between them and is where the gradient is taken. On top of this it removes the need to tune a learning rate by setting the effective step from a Polyak rule, scaling the AdamW step by $\max(0, F_t + I_t)$ divided by a bias-corrected exponential average of the $\ell_1$ gradient norm (the $\sqrt{\pi/2}$ factor converts the $\ell_1$ norm to an $\ell_2$ estimate). The interpolation weight $\tilde\beta_t$ is annealed from $\beta_{\mathrm{sf}}$ toward $\beta_{\mathrm{sf}}^{\max}$ over $T_{\mathrm{anneal}}$ steps, and weight decay follows the decoupled AdamC form (scaled by $\alpha_t^2$). The returned parameters are the averaged iterate $x_t$.

$$
\begin{aligned}
g_t &= \nabla f(y_{t-1}), \quad F_t = f(y_{t-1}), \quad I_t = \tilde\beta_t \langle g_t,\, z_{t-1} - x_{t-1}\rangle \\
e_t &= \beta_p e_{t-1} + (1 - \beta_p)\,\lVert g_t\rVert_1 \sqrt{\tfrac{\pi}{2}}, \qquad \hat{e}_t = \frac{e_t}{1 - \beta_p^{\,t}} \\
\alpha_t &= \gamma_t \cdot \frac{\max(0,\, F_t + I_t)}{\hat{e}_t} \\
z_t &= z_{t-1} - \alpha_t^2 \lambda\, y_{t-1} \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^{\,t}} \\
z_t &= z_t - \alpha_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \\
x_t &= (1 - c_t)\, x_{t-1} + c_t\, z_t \\
y_t &= \tilde\beta_t\, x_t + (1 - \tilde\beta_t)\, z_t
\end{aligned}
$$

where $\theta$ are the parameters (returned as $x_t$), $\gamma_t$ the warmup factor, $\alpha_t$ the Polyak effective step, $g_t$ the gradient at $y_{t-1}$, $m_t,v_t$ the Adam moments with decays $\beta_1,\beta_2$ and bias-corrected forms $\hat{m}_t,\hat{v}_t$, $\lambda$ the decoupled weight decay, $\epsilon$ a stability constant, $\beta_p$ the EMA coefficient for the $\ell_1$ gradient-norm estimate $\hat{e}_t$, $\tilde\beta_t$ the annealed interpolation weight, and $c_t = w_t / W_t$ the averaging weight with $w_t = t^r\, \gamma_{\max}^{\,p}$ (and $c_t = 1$ during the first $C_{\mathrm{warmup}}$ steps).

Reference: Aaron Defazio, "ScheduleFree+: Scaling Learning-Rate-Free & Schedule-Free Learning to Large Language Models", arXiv 2026. https://arxiv.org/abs/2605.19095

---
[Back to the Canon](../README.md)
