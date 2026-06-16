# SAMPa

Implements SAMPa, a parallelized sharpness-aware minimizer that removes the sequential gradient dependency of SAM.

Standard SAM needs two sequential gradient evaluations per step: one to form the ascent perturbation and one at the perturbed point. SAMPa keeps an auxiliary iterate $y_t$ and computes the perturbation from $\nabla f(y_t)$, so the two remaining gradients $\nabla f(\tilde\theta_t)$ and $\nabla f(y_{t+1})$ depend only on already-known points and can be evaluated fully in parallel, roughly halving wall-clock time per step. The SAMPa-$\lambda$ variant interpolates the perturbed-point gradient with the auxiliary gradient, recovering plain SAMPa at $\lambda = 0$.

$$
\begin{aligned}
\tilde\theta_t &= \theta_t + \rho \, \frac{\nabla f(y_t)}{\lVert \nabla f(y_t) \rVert} \\
y_{t+1} &= \theta_t - \eta_t \, \nabla f(y_t) \\
\theta_{t+1} &= \theta_t - \eta_t \big[ (1 - \lambda)\, \nabla f(\tilde\theta_t) + \lambda \, \nabla f(y_{t+1}) \big]
\end{aligned}
$$

where $\theta_t$ are the parameters, $y_t$ the auxiliary iterate, $\tilde\theta_t$ the perturbed point, $\eta_t$ the learning rate, $\rho$ the perturbation radius, $f$ the (mini-batch) loss, and $\lambda \in [0, 1]$ the interpolation weight ($\lambda = 0$ gives plain SAMPa). The gradients $\nabla f(\tilde\theta_t)$ and $\nabla f(y_{t+1})$ are computed in parallel.

Reference: Wanyun Xie, Thomas Pethick, Volkan Cevher, "SAMPa: Sharpness-aware Minimization Parallelized", NeurIPS 2024. https://arxiv.org/abs/2410.10683

---
[Back to the Canon](../index.md)
