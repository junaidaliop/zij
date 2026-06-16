# Adalite

Implements Adalite, a memory-efficient adaptive optimizer combining Adafactor-style factored second moments with LAMB-style trust scaling.

Adalite targets full-parameter 16-bit fine-tuning on a single consumer GPU. It fuses the optimizer step into the backward pass (one parameter is updated as soon as its gradient is ready, then the gradient is freed), so no full gradient buffer is held. Memory for the second moment is reduced by factoring: for a 2D weight only a per-column row vector $c_t$ is stored, recovered Adafactor-style; for other shapes the full per-element estimate $v_t$ is kept.

Each step centralizes the gradient, forms the (possibly factored) running squared-gradient estimate $u_t$ with an increasing decay $\beta_t$, preconditions the gradient by $u_t^{-1/2}$, optionally clips by its root-mean-square, then applies a LAMB-style trust ratio $\lVert\theta\rVert/\lVert g_t\rVert$ together with a decoupled weight-decay pull toward the unit-norm direction.

$$
\begin{aligned}
g_t &\leftarrow g_t - \mathrm{mean}(g_t), \qquad \beta_t = 1 - t^{-\beta_d}, \\
u_t &= (1-\beta_t)\, g_t^2 + \beta_t\, \hat{v}_{t-1} + \epsilon, \\
c_t &= \mathrm{mean}_{\mathrm{rows}}(u_t) \;\;\text{(2D)}, \qquad v_t = u_t \;\;\text{(otherwise)}, \\
m_t &= u_t^{-1/2} \odot g_t, \qquad m_t \leftarrow m_t \,/\, \max\!\big(1,\ \mathrm{RMS}(m_t)\big), \\
m_t &\leftarrow \frac{\lVert \theta_t \rVert}{\lVert g_t \rVert}\, m_t + \lambda\Big(\theta_t - \tfrac{\theta_t}{\lVert \theta_t \rVert}\Big), \\
\theta_{t+1} &= \theta_t - \eta\, m_t.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the (column-centralized) gradient, $u_t$ the current squared-gradient estimate and $\hat{v}_{t-1}$ its previous value (the broadcast factor $c_{t-1}$ for 2D parameters, else $v_{t-1}$), $\beta_d$ the decay exponent (default $0.8$), $\lambda$ the weight-decay strength, $\mathrm{RMS}(m_t)=\sqrt{\mathrm{mean}(m_t^2)}$, and $\epsilon$ the stability constant.

Reference: euclaise, "Adalite (SlimTrainer)", code-only, 2023. https://github.com/euclaise/SlimTrainer

---
[Back to the Canon](../index.md)
