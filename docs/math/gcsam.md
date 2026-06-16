# GCSAM

Implements GCSAM, sharpness-aware minimization with gradient centralization applied to the ascent and descent gradients.

GCSAM extends SAM by centralizing the gradient — subtracting its mean over the weight-vector elements — before forming the perturbation. The centered gradient $g^{\mathrm{GC}}_t = g_t - \frac{1}{n}\sum_{j=1}^{n} (g_t)_j$ defines the ascent direction toward the local maximum of the loss inside a radius-$\rho$ ball, which stabilizes the perturbation and the subsequent descent step.

$$
\begin{aligned}
g^{\mathrm{GC}}_t &= g_t - \frac{1}{n}\sum_{j=1}^{n} (g_t)_j, \\
\hat{\epsilon}_t &= \rho \, \frac{g^{\mathrm{GC}}_t}{\lVert g^{\mathrm{GC}}_t \rVert_p}, \\
\theta_{t+1} &= \theta_t - \eta \, g^{\mathrm{GC}}\!\big(\theta_t + \hat{\epsilon}_t\big),
\end{aligned}
$$

where $g_t = \nabla_\theta L_S(\theta_t)$ is the training-loss gradient, $g^{\mathrm{GC}}$ is the gradient centralization operator (mean removal across the $n$ weight elements), $\rho$ is the neighborhood radius, $\lVert \cdot \rVert_p$ is the $p$-norm (default Euclidean), $\hat{\epsilon}_t$ is the weight perturbation, and $\eta$ is the learning rate; the final step uses the centralized gradient evaluated at the perturbed point $\theta_t + \hat{\epsilon}_t$.

Reference: Mohamed Hassan, Aleksandar Vakanski, Boyu Zhang, Min Xian, "GCSAM: Gradient Centralized Sharpness Aware Minimization", arXiv 2025. https://arxiv.org/abs/2501.11584

---
[Back to the Canon](../index.md)
