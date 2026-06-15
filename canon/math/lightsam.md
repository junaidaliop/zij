# LightSAM

Implements LightSAM, a parameter-agnostic Sharpness-Aware Minimization that makes both the ascent and descent steps adaptive.

Standard SAM perturbs to the worst-case point $w_t = \theta_t + \rho\, s_t/\lVert s_t\rVert$ and then descends with $\theta_{t+1} = \theta_t - \eta\, g_t$, so its behavior is highly sensitive to the chosen radius $\rho$ and learning rate $\eta$. LightSAM removes this tuning burden by replacing the fixed-scale rules with AdaGrad-style accumulation in both phases: a running sum of squared gradient norms rescales the perturbation, and a second running sum rescales the update. The result is convergence guaranteed for any initial $\rho, \eta > 0$. Coordinate-wise (AdaGrad) and Adam variants follow the same template, replacing the scalar accumulators with per-coordinate second moments and, for Adam, exponential moving averages with $\beta_1, \beta_2$.

$$
\begin{aligned}
s_t &= \nabla f(\theta_t, \xi_t), \\
u_t &= u_{t-1} + \lVert s_t \rVert^2, \\
w_t &= \theta_t + \rho \, \frac{s_t}{\sqrt{u_t}}, \\
g_t &= \nabla f(w_t, \xi_t), \\
v_t &= v_{t-1} + \lVert g_t \rVert^2, \\
\theta_{t+1} &= \theta_t - \eta \, \frac{g_t}{\sqrt{v_t}}.
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\rho$ the perturbation radius, $s_t$ the gradient at $\theta_t$ and $g_t$ the gradient at the perturbed point $w_t$, $\xi_t$ the sampled minibatch, $u_t, v_t$ the accumulated squared-norm denominators (initialized $u_0 = v_0 = \epsilon^2$), and $\epsilon$ a small stability constant.

Reference: Yifei Cheng, Li Shen, Hao Sun, Nan Yin, Xiaochun Cao, Enhong Chen, "LightSAM: Parameter-Agnostic Sharpness-Aware Minimization", 2025. https://arxiv.org/abs/2505.24399

---
[Back to the Canon](../README.md)
