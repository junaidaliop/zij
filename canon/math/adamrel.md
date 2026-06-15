# Adam-Rel

Implements Adam-Rel, Adam with a relative timestep that resets at each target change.

In reinforcement learning the optimization target shifts between epochs, which makes the global Adam timestep a poor proxy for how stationary the current objective is. Adam-Rel keeps Adam's moment estimates and update form unchanged but measures the bias-correction timestep locally: $t$ is reset to $0$ at the start of each epoch (after each target change), so the bias correction always treats the new objective as if optimization had just begun. This prevents Adam from taking oversized steps when the gradient distribution abruptly changes, yielding approximately unit-scale updates right after a target shift.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^{\,t}}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^{\,t}} \\
\theta_t &= \theta_{t-1} - \eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $m_t,v_t$ the first and second moment estimates, $\beta_1,\beta_2$ their decay rates, $\epsilon$ a stability constant, and $t$ the relative timestep counted from the most recent epoch boundary (reset to $0$ on each target change) rather than the global step.

Reference: Benjamin Ellis, Matthew T. Jackson, Andrei Lupu, Alexander D. Goldie, Mattie Fellows, Shimon Whiteson, Jakob N. Foerster, "Adam on Local Time: Addressing Nonstationarity in RL with Relative Adam Timesteps", arXiv 2024. https://arxiv.org/abs/2412.17113

---
[Back to the Canon](../README.md)
