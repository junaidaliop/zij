# SASSHA

Implements SASSHA, a sharpness-aware adaptive second-order optimizer with a stabilized Hessian-diagonal preconditioner.

SASSHA combines a SAM-style sharpness-aware perturbation with an adaptive diagonal-Hessian preconditioner. At each step it perturbs the parameters toward the worst-case direction, then evaluates the gradient and a Hessian-diagonal estimate at the perturbed point. The Hessian diagonal is approximated by Hutchinson's method, made positive by taking absolute values, exponentially averaged, and square-rooted to stabilize the preconditioner against small or sign-indefinite curvature. The Hessian estimate is refreshed lazily every $k$ steps to amortize its cost.

$$
\begin{aligned}
\epsilon_t &= \rho \, \frac{g_t}{\lVert g_t \rVert_2}, \qquad \tilde g_t = \nabla f_{\mathcal{B}}(\theta_t + \epsilon_t) \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, \tilde g_t, \qquad \bar m_t = \frac{m_t}{1 - \beta_1^{\,t}} \\
D_t &= \beta_2 D_{t-1} + (1 - \beta_2)\, \lvert \tilde H_t \rvert, \qquad \tilde H_t = \hat H(\theta_t + \epsilon_t) \\
\bar D_t &= \sqrt{\frac{D_t}{1 - \beta_2^{\,t}}} \\
\theta_{t+1} &= \theta_t - \eta_t\, \bar D_t^{-1} \bar m_t - \eta_t \lambda \theta_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_t$ the learning rate, $g_t = \nabla f_{\mathcal{B}}(\theta_t)$ the minibatch gradient, $\rho$ the perturbation radius, $\tilde g_t$ the gradient at the perturbed point, $\hat H(\cdot)$ the Hutchinson diagonal-Hessian estimate, $\lvert \cdot \rvert$ the element-wise absolute value, $m_t$ and $D_t$ the bias-corrected first moment and absolute-Hessian moving average with decays $\beta_1, \beta_2$, and $\lambda$ the weight decay; $D_t$ (hence $\tilde H_t$) is recomputed every $k$ steps and reused otherwise.

Reference: Dahun Shin, Dongyeop Lee, Jinseok Chung, Namhoon Lee, "Sassha: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation", ICML 2025. https://arxiv.org/abs/2502.18153

---
[Back to the Canon](../index.md)
