# FAGH

Implements FAGH (Federated learning with Approximated Global Hessian), a second-order federated optimizer that approximates the global Hessian from only its first row.

Each round, clients send back their local gradient $g_i$ and the first row of their local Hessian $v_i = \partial g_i[0]/\partial \theta$; the server aggregates both with client weights $p_i$ into a global gradient $g_t$ and a global first-row vector $v_t$. Exploiting Hessian symmetry, the full $d\times d$ Hessian is approximated by the rank-one matrix $H_a = z_t v_t^{\top}$, where $z_t$ normalizes the first row by its leading entry. A regularized Newton step $(H_a+\rho I)^{-1} g_t$ is then computed in $O(d)$ via the Sherman–Morrison formula, never forming the matrix.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\, g_t, \qquad \hat m_t = \frac{m_t}{1-\beta_1^{\,t}} \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\, h_t, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
z_t &= \frac{\hat v_t}{\hat v_t[0]}, \qquad H_a = z_t\, \hat v_t^{\top} \\
(H_a+\rho I)^{-1}\hat m_t &= \frac{\hat m_t}{\rho} - \frac{z_t\,(\hat v_t^{\top}\hat m_t)/\rho^2}{1 + (\hat v_t^{\top} z_t)/\rho} \\
\theta_t &= \theta_{t-1} - \eta\,(H_a+\rho I)^{-1}\hat m_t
\end{aligned}
$$

where $g_t=\sum_i p_i g_i$ is the aggregated global gradient, $h_t=\sum_i p_i v_i$ the aggregated first row of the global Hessian, $\hat m_t/\hat v_t$ their bias-corrected first moments, $z_t$ the first row normalized by its diagonal entry $\hat v_t[0]=\partial^2 F/\partial x_1^2$, $\rho>0$ the regularizer keeping $H_a+\rho I$ positive definite, $\eta$ the learning rate, and $\beta_1,\beta_2$ the moment decay rates.

Reference: Mrinmay Sen, A. K. Qin, Krishna Mohan C, "FAGH: Accelerating Federated Learning with Approximated Global Hessian", arXiv 2024. https://arxiv.org/abs/2403.11041

---
[Back to the Canon](../README.md)
