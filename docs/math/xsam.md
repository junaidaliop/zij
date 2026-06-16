# X-SAM

Implements X-SAM, sharpness-aware minimization with dominant-eigenvector gradient correction.

Sharpness-Aware Minimization (SAM) perturbs the weights toward the locally worst-case direction and steps with the gradient evaluated at that adversarial point. X-SAM augments this by correcting the perturbed gradient along the principal Hessian eigenvector $v$: it subtracts the component of the gradient parallel to $v$ (the most curved direction), with the sign chosen so the step is pushed away from the sharpest direction. The eigenvector $v$ is estimated intermittently by power iteration, so the correction adds little overhead.

$$
\begin{aligned}
\epsilon_t &= \rho \frac{g_t}{\lVert g_t \rVert}, \qquad \theta_t^{\mathrm{adv}} = \theta_t + \epsilon_t, \\
g_t^{\mathrm{adv}} &= \nabla f(\theta_t^{\mathrm{adv}}), \qquad g_{t,\parallel}^{\mathrm{adv}} = \langle g_t^{\mathrm{adv}}, v \rangle\, v, \\
\tilde{g}_t &= \frac{g_t^{\mathrm{adv}}}{\lVert g_t^{\mathrm{adv}} \rVert} - \alpha\, \mathrm{sign}\!\left(\langle g_t^{\mathrm{adv}}, v \rangle\right) g_{t,\parallel}^{\mathrm{adv}}, \\
\theta_{t+1} &= \theta_t - \gamma\, \tilde{g}_t.
\end{aligned}
$$

where $g_t = \nabla f(\theta_t)$ is the gradient, $\rho > 0$ the perturbation radius, $\theta_t^{\mathrm{adv}}$ the adversarial (worst-case) point, $v$ the principal Hessian eigenvector, $g_{t,\parallel}^{\mathrm{adv}}$ the projection of the perturbed gradient onto $v$, $\alpha \in [0,2]$ the correction strength, and $\gamma$ the learning rate.

Reference: Hongru Duan, Yongle Chen, Lei Guan, "X-SAM: Boosting Sharpness-Aware Minimization with Dominant-Eigenvector Gradient Correction", arXiv 2026. https://arxiv.org/abs/2601.10251

---
[Back to the Canon](../index.md)
