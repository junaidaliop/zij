# PadamP

Implements PadamP, partially adaptive Adam with projection-based decoupling for scale-invariant weights.

PadamP combines two ideas. From Padam it borrows partial adaptivity: the second-moment normalizer is raised to a power $p \in (0, 1/2]$ rather than the usual $1/2$, interpolating between SGD-with-momentum ($p \to 0$) and Adam ($p = 1/2$) to curb the generalization gap of fully adaptive methods. From AdamP it borrows a projection step that, when the parameter and gradient are nearly orthogonal, removes the component of the update lying along the parameter direction, preventing the effective step size from collapsing on scale-invariant weights.

$$
\begin{aligned}
m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\
v_t &\leftarrow \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\
\hat{m}_t &\leftarrow m_t / (1 - \beta_1^t), \qquad \hat{v}_t \leftarrow v_t / (1 - \beta_2^t) \\
p_t &\leftarrow \hat{m}_t / (\hat{v}_t + \epsilon)^p \\
q_t &\leftarrow
\begin{cases}
\Pi_{\theta_t}(p_t) & \text{if } \cos(\theta_t, g_t) < \delta\, \eta_t / \sqrt{\dim(\theta)} \\
p_t & \text{otherwise}
\end{cases} \\
\theta_{t+1} &\leftarrow \theta_t - \eta_t\, q_t
\end{aligned}
$$

where $g_t = \nabla_\theta f_t(\theta_t)$, the projection $\Pi_{\theta_t}(p_t) = p_t - \langle \hat{\theta}_t, p_t \rangle\, \hat{\theta}_t$ removes the radial component with $\hat{\theta}_t = \theta_t / \lVert \theta_t \rVert_2$, $\eta_t$ is the learning rate, $\beta_1, \beta_2$ are the moment decay rates, $p \in (0, 1/2]$ is the partial-adaptivity exponent, $\delta$ is the projection threshold (e.g. $0.1$), and $\epsilon$ is a stability constant.

Reference: Yongqi Li, Xiaowei Zhang, "Adaptive Moment Estimation Optimization Algorithm Using Projection Gradient for Deep Learning", arXiv 2025. https://arxiv.org/abs/2503.10005

---
[Back to the Canon](../README.md)
