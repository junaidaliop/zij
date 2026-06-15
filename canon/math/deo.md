# DEO

Implements DEO (Dimer-Enhanced Optimization), a first-order method that escapes saddle points by removing low-curvature gradient components.

DEO adapts the Dimer method from molecular simulation to estimate, with only gradient evaluations, the local lowest-curvature direction $\hat{N}_t$. A second point $\theta_2 = \theta_t + \Delta R\,\hat{N}_t$ is used to compute a rotational force that aligns $\hat{N}_t$ with the smallest-eigenvalue eigenvector of the Hessian, all without forming the Hessian explicitly. The raw gradient is then projected to subtract its component along $\hat{N}_t$, biasing the step away from flat or negatively curved directions, and the corrected gradient is fed into a standard Adam update.

$$
\begin{aligned}
F_R &= (g_2 - g_t) - \big((g_2 - g_t)\cdot \hat{N}_t\big)\hat{N}_t, \\
\hat{N}_{t+1} &= \frac{\hat{N}_t + \eta_{\mathrm{rot}} F_R}{\lVert \hat{N}_t + \eta_{\mathrm{rot}} F_R \rVert}, \\
g_{\mathrm{mod}} &= g_t - \alpha\,(g_t\cdot \hat{N}_t)\,\hat{N}_t, \\
m_t &= \beta_1 m_{t-1} + (1-\beta_1)\,g_{\mathrm{mod}}, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)\,g_{\mathrm{mod}}^2, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}, \\
\theta_{t+1} &= \theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon},
\end{aligned}
$$

where $g_t$ is the gradient at $\theta_t$, $g_2$ is the gradient at the dimer point $\theta_2 = \theta_t + \Delta R\,\hat{N}_t$, $\hat{N}_t$ is the unit dimer direction estimating the lowest-curvature eigenvector, $\Delta R$ is the dimer separation, $\eta_{\mathrm{rot}}$ is the rotation step size, $\alpha$ is the correction coefficient scaling the projection removal, $\eta$ is the learning rate, $\beta_1,\beta_2$ are the moment decay rates, and $\epsilon$ is a stability constant.

Reference: Yue Hu, Zanxia Cao, Yingchao Liu, "Dimer-Enhanced Optimization: A First-Order Approach to Escaping Saddle Points in Neural Network Training", arXiv 2025. https://arxiv.org/abs/2507.19968

---
[Back to the Canon](../README.md)
