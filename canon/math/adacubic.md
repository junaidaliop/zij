# AdaCubic

Implements AdaCubic, an adaptive cubic-regularization optimizer with a trust-region acceptance test.

At each step AdaCubic builds a cubic-regularized local model around the current parameters using the gradient and a diagonal Hessian approximation, and minimizes it to obtain the step $s_t$. The cubic term keeps the step bounded without an explicit trust radius; instead a scalar $\xi_t$ caps $\lVert s_t\rVert_2^3$, and the step is found by solving the regularized linear system $(H_t + \tfrac{\nu r}{2} I)\,s = -g_t$ for the dual variable $\nu$ that makes the step satisfy the constraint, with $r=\lVert s\rVert_2$.

The step is accepted only if the ratio of actual to predicted decrease exceeds $\eta_1$, and the cap $\xi_t$ is expanded on highly successful steps and contracted on rejected ones.

$$
\begin{aligned}
s_t &= \arg\min_{s}\ \Big[ g_t^\top s + \tfrac{1}{2} s^\top H_t s + \tfrac{M}{6}\lVert s\rVert_2^3 \Big],
\quad\text{equivalently}\quad (H_t + \tfrac{\nu r}{2} I)\,s_t = -g_t,\ \ r=\lVert s_t\rVert_2 \\
\rho_t &= \frac{F(\theta_t) - F(\theta_t + s_t)}{F(\theta_t) - m_{\nu}(s_t)} \\
\theta_{t+1} &= \begin{cases} \theta_t + s_t & \rho_t \ge \eta_1 \\ \theta_t & \rho_t < \eta_1 \end{cases} \\
\xi_{t+1} &= \begin{cases} \max\{\alpha_1 \lVert s_t\rVert_2^3,\ \xi_t\} & \rho_t \ge \eta_2 \\ \xi_t & \eta_1 \le \rho_t < \eta_2 \\ \max\{\alpha_2 \lVert s_t\rVert_2^3,\ \epsilon\} & \rho_t < \eta_1 \end{cases}
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ the (batch) gradient, $H_t$ a diagonal Hessian approximation, $F$ the objective, $M$ the cubic-regularization weight, $m_{\nu}(s) = F(\theta_t) + g_t^\top s + \tfrac{1}{2}s^\top H_t s + \tfrac{M}{6}\lVert s\rVert_2^3$ the predicted value, $\nu$ the dual variable found by root-finding on $\phi(\nu,r)=\lVert s(\nu,r)\rVert_2^{-1} - \xi^{-1/3}$, $\xi$ the cap on $\lVert s\rVert_2^3$, $\rho_t$ the actual-to-predicted decrease ratio, $\eta_1,\eta_2$ the acceptance thresholds, $\alpha_1\ge 1$ and $\alpha_2\in(0,1)$ the expansion/contraction factors, and $\epsilon$ a floor on $\xi$.

Reference: Ioannis Tsingalis, Constantine Kotropoulos, Corentin Briat, "AdaCubic: An Adaptive Cubic Regularization Optimizer for Deep Learning", arXiv 2026. https://arxiv.org/abs/2604.09437

---
[Back to the Canon](../README.md)
