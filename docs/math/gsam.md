# GSAM

Implements GSAM, surrogate gap guided sharpness-aware minimization
wrapping a base optimizer.


$$
\begin{aligned}
&\theta_t^{adv} = \theta_t
    + \rho_t \frac{g_t}{\lVert g_t \rVert + \epsilon}, \qquad
 g_t^p = \nabla f(\theta) \big\rvert_{\theta = \theta_t^{adv}}      \\
&g_t = g_{t,\parallel} + g_{t,\perp}, \qquad
 g_{t,\parallel} = \frac{\langle g_t, g_t^p \rangle}
    {\lVert g_t^p \rVert^2} \, g_t^p                                \\
&\theta_{t+1} = \theta_t - \eta_t \left( g_t^p
    - \alpha \, g_{t,\perp} \right)
\end{aligned}
$$

where $g_t$ is the gradient of the loss $f$,
$g_t^p$ is the gradient of the perturbed loss, and the
perturbation radius follows the schedule
$\rho_t = \rho_{min} + (\rho_{max} - \rho_{min})
(\eta_t - \eta_{min}) / (\eta_{max} - \eta_{min})$.

Reference: Juntang Zhuang et al., "Surrogate Gap Minimization Improves
Sharpness-Aware Training", ICLR 2022.
https://arxiv.org/abs/2203.08065


**Note:** Each step needs two forward-backward passes: call `set_closure` and then `step`, or pass `step` a closure that zeroes gradients, computes the loss, and calls `backward()`. Pass `rho_scheduler` (for example `ProportionScheduler`) and call `update_rho_t` once per step to anneal $\rho_t$ with the learning rate as in the paper; without one, $\rho_t$ stays at the constant `rho`. The state dict covers only the GSAM perturbation state; save `base_optimizer.state_dict()` separately.


---
[Back to the Canon](../index.md)
