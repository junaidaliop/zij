# LE-SAM

Implements LE-SAM (Loss-Equated SAM), a sharpness-aware method that fixes a loss-space budget instead of a fixed perturbation radius.

Standard SAM perturbs the weights by a fixed-radius ascent step $\rho\,g_t/\|g_t\|$, so the resulting rise in loss scales with the gradient norm and the first-order term dominates the learning signal. LE-SAM inverts this: it chooses the radius $\rho_t$ so that the linearized loss increase equals a fixed budget $\sigma$. The first-order ascent contribution then becomes a parameter-independent constant, and the descent gradient is driven by the curvature (second-order) term, which is what flat minima actually concern. The budget $\sigma$ is cosine-annealed to zero over the final training epochs, and $\rho_t$ is clipped to a ceiling for stability.

$$
\begin{aligned}
\rho_t &= \min\!\left(\frac{\sigma_t}{\|g_t\| + \varrho},\; \rho_{\max}\right), \\
\epsilon_t &= \rho_t \, \frac{g_t}{\|g_t\|}, \\
\hat{g}_t &= \nabla_\theta L\big(\theta_t + \epsilon_t\big), \\
\theta_{t+1} &= \mathrm{Opt}\big(\theta_t,\, \hat{g}_t\big).
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the gradient at $\theta_t$, $\sigma_t$ is the cosine-annealed loss budget, $\rho_t$ is the loss-equated perturbation radius, $\varrho$ is a small stability constant, $\rho_{\max}$ caps the radius, $\epsilon_t$ is the adversarial perturbation, $\hat{g}_t$ is the gradient evaluated at the perturbed point, and $\mathrm{Opt}$ is the base optimizer (e.g. SGD or Adam) applied to $\hat{g}_t$.

Reference: Jinping Wang, Qinhan Liu, Zhiwu Xie, Zhiqiang Gao, "Fix the Loss, Not the Radius: Rethinking the Adversarial Perturbation of Sharpness-Aware Minimization", ICML 2026. https://arxiv.org/abs/2605.10183

---
[Back to the Canon](../README.md)
