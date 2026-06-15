# CFGD (Conformable)

Implements CFGD (Conformable Fractional Gradient Descent), a local fractional optimizer that rescales the gradient by a conformable factor.

Classical fractional gradient methods built on the Caputo or Riemann–Liouville operators are nonlocal: they accumulate a weighted history of past gradients, which is costly to maintain during training. CFGD instead uses the conformable fractional derivative of Khalil et al., $T_\alpha f(\theta) = \theta^{1-\alpha}\,f'(\theta)$, which is local and reduces to a simple power-law rescaling of the ordinary derivative. The update therefore costs essentially the same as plain gradient descent while the fractional order $\alpha \in (0,1]$ modulates the effective step per coordinate.

Applied coordinate-wise to the loss, the conformable gradient rescales each component $g_t$ by $|\theta_t|^{1-\alpha}$, recovering standard gradient descent at $\alpha = 1$. A variable-order variant (vCFGD) lets the order grow toward $1$ during training, $\alpha_t \uparrow 1$, giving stronger fractional modulation early and gradient-descent–like behavior asymptotically.

$$
\begin{aligned}
\theta_{t+1} &= \theta_t - \eta\,|\theta_t|^{\,1-\alpha_t}\, g_t, \\
g_t &= \nabla_\theta \mathcal{L}(\theta_t), \\
\alpha_t &\uparrow 1 \quad (\text{vCFGD}).
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $g_t$ the gradient, $\alpha_t \in (0,1]$ the (possibly time-varying) conformable order, and $|\theta_t|^{1-\alpha_t}$ the conformable rescaling factor applied per coordinate; setting $\alpha_t = 1$ recovers ordinary gradient descent.

Reference: Hayman Thabet, "Conformable fractional gradient descent: A local optimizer for neural network training", Journal of Computational and Applied Mathematics 488 (2026). https://doi.org/10.1016/j.cam.2026.117842

---
[Back to the Canon](../README.md)
