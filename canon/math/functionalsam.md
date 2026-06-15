# Functional SAM

Implements Functional-SAM, a sharpness-aware update that perturbs only the network Jacobian and keeps the loss derivative at the unperturbed point.

For a loss $L$ composed with a network function $F$ (the concatenated logits), the SAM gradient at the perturbed point $\theta+\rho\,\epsilon^*$ expands by the chain rule into a "logit path", which moves the loss derivative $\nabla_F L$ along the perturbation, and a "functional path", which moves the Jacobian $\nabla_\theta F$. The paper argues the logit path drives spurious sharpness minimization that fails to improve generalization on harder problems. Functional-SAM discards the logit-path contribution: it evaluates the network Jacobian at the perturbed parameters but multiplies it by the loss derivative taken at the original parameters, so only the functional path contributes to the descent direction.

$$
\begin{aligned}
\epsilon^* &= \frac{\nabla_\theta L(\theta_t)}{\lVert \nabla_\theta L(\theta_t) \rVert} \\
g_t &= \nabla_\theta F(\theta_t + \rho\,\epsilon^*) \cdot \nabla_F L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta\, g_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $\rho$ the perturbation radius, $\epsilon^*$ the normalized ascent direction, $F(\theta)$ the network function (logits), $\nabla_\theta F$ its Jacobian with respect to the parameters, and $\nabla_F L$ the gradient of the loss with respect to the function outputs; standard SAM instead uses $g_t = \nabla_\theta L(\theta_t + \rho\,\epsilon^*)$, which the chain rule splits into this functional path plus the discarded logit path.

Reference: Sidak Pal Singh, Hossein Mobahi, Atish Agarwala, Yann Dauphin, "Avoiding spurious sharpness minimization broadens applicability of SAM", arXiv 2025. https://arxiv.org/abs/2502.02407

---
[Back to the Canon](../README.md)
