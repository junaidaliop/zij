# Refined SF-AdamW

Implements Refined SF-AdamW, a Schedule-Free AdamW variant that decouples the averaging window from the momentum coefficient.

Schedule-Free methods replace a learning-rate schedule with an online weighted average of the iterates: gradients are taken at an interpolated point $y_t$ between the running iterate $z_t$ and the average $x_t$, and the average is updated each step. In vanilla SF-AdamW the averaging weight is tied to $\beta_1$, which couples the effective momentum to the width of the averaging window. The refined variant introduces a separate decoupling constant $C$ that scales the averaging weight $c_{t+1}$, so $\beta_1$ controls momentum while $C$ independently sets how quickly the average concentrates on recent iterates. Setting $C = 1/(1-\beta_1)$ recovers the original SF-AdamW.

$$
\begin{aligned}
y_t &= (1-\beta_1)\, z_t + \beta_1\, x_t \\
g_t &\in \partial f(y_t, \zeta_t) \\
v_t &= \beta_2\, v_{t-1} + (1-\beta_2)\, g_t^2, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}} \\
z_{t+1} &= z_t - \frac{\gamma_t\, g_t}{\sqrt{\hat{v}_t}+\epsilon} - \gamma_t\, \lambda\, y_t \\
c_{t+1} &= \min\!\left\{ \frac{\gamma_t^2}{\sum_{i=1}^{t} \gamma_i^2}\,(1-\beta_1)\,C,\; 1 \right\} \\
x_{t+1} &= (1-c_{t+1})\, x_t + c_{t+1}\, z_{t+1}
\end{aligned}
$$

where $z_t$ is the running iterate, $x_t$ the averaged (returned) iterate, $y_t$ the interpolated point at which the gradient $g_t$ is evaluated, $v_t$ the second-moment estimate with bias correction $\hat{v}_t$, $\gamma_t$ the (warmup-scaled) learning rate, $\beta_1,\beta_2$ the interpolation and second-moment decay rates, $\lambda$ the decoupled weight decay, $\epsilon$ a stability constant, and $C$ the decoupling constant governing the averaging window.

Reference: Minhak Song, Beomhan Baek, Kwangjun Ahn, Chulhee Yun, "Through the River: Understanding the Benefit of Schedule-Free Methods for Language Model Training", arXiv 2025. https://arxiv.org/abs/2507.09846

---
[Back to the Canon](../README.md)
