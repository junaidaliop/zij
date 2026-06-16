# SASSHA

Implements SASSHA, a sharpness-aware adaptive second-order optimizer with a stabilized diagonal-Hessian preconditioner.

SASSHA pairs a SAM-style sharpness-aware perturbation with an adaptive second-order preconditioner built from the diagonal of the Hessian. Each step first ascends to the worst-case neighbor of the current parameters, then evaluates both the gradient and a Hessian-diagonal estimate at that perturbed point. Because sharpness minimization drives curvature toward zero, the diagonal Hessian is made positive by an element-wise absolute value, exponentially averaged, and square-rooted rather than damped or clipped, which smoothly lifts near-zero entries and keeps the preconditioner stable. The Hessian diagonal is estimated by Hutchinson's method and refreshed lazily every $k$ steps to amortize its cost.

$$
\begin{aligned}
\epsilon_t &= \rho \, \frac{g_t}{\lVert g_t \rVert_2}, \qquad \tilde g_t = \nabla f_{\mathcal{B}}(\theta_t + \epsilon_t) \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \tilde g_t, \qquad \bar m_t = \frac{m_t}{1 - \beta_1^{\,t}} \\
\tilde H_t &= \hat H(\theta_t + \epsilon_t), \qquad D_t = \beta_2 D_{t-1} + (1 - \beta_2)\, \lvert \tilde H_t \rvert \\
\bar D_t &= \sqrt{\frac{D_t}{1 - \beta_2^{\,t}}} \\
\theta_{t+1} &= \theta_t - \eta_t\, \bar D_t^{-1} \bar m_t - \eta_t \lambda \theta_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t = \nabla f_{\mathcal{B}}(\theta_t)$ the minibatch gradient, $\rho$ the perturbation radius, $\tilde g_t$ the gradient at the perturbed point, $\hat H(\cdot)$ the Hutchinson diagonal-Hessian estimate, $\lvert \cdot \rvert$ the element-wise absolute value, $m_t$ and $D_t$ the first-moment and absolute-Hessian moving averages with bias-corrected forms $\bar m_t, \bar D_t$ and decays $\beta_1, \beta_2$, and $\lambda$ the decoupled weight decay; $\tilde H_t$ (hence $D_t$) is recomputed every $k$ steps and reused otherwise.

Reference: Dahun Shin, Dongyeop Lee, Jinseok Chung, Namhoon Lee, "Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation", ICML 2025. https://arxiv.org/abs/2502.18153

---
[Back to the Canon](../index.md)
