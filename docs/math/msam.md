# MSAM

Implements MSAM (Momentum-SAM), sharpness-aware minimization that perturbs parameters along the accumulated momentum direction instead of the current gradient.

SAM ascends along the normalized gradient before computing the descent step, doubling the per-step cost with an extra forward-backward pass. MSAM replaces this ascent direction with the SGD momentum vector, which is already available, so the perturbation is essentially free. Parameters are perturbed by $-\rho$ times the unit momentum vector, the gradient is evaluated at that perturbed point, and a standard SGD-with-momentum step follows.

$$
\begin{aligned}
\tilde{\theta}_t &= \theta_t - \rho \, \frac{v_t}{\lVert v_t \rVert} \\
g_t &= \nabla_\theta L_{B_t}(\tilde{\theta}_t) \\
v_{t+1} &= \mu \, v_t + g_t \\
\theta_{t+1} &= \theta_t - \eta \, v_{t+1}
\end{aligned}
$$

where $\theta_t$ are the (unperturbed) parameters, $\tilde{\theta}_t$ the perturbed parameters at which the gradient is taken, $v_t$ the momentum buffer, $\rho$ the perturbation radius, $\mu$ the momentum coefficient, $\eta$ the learning rate, and $g_t$ the minibatch gradient evaluated at $\tilde{\theta}_t$.

Reference: Marlon Becker, Frederick Altrock, Benjamin Risse, "Momentum-SAM: Sharpness Aware Minimization without Computational Overhead", arXiv 2024. https://arxiv.org/abs/2401.12033

---
[Back to the Canon](../index.md)
