# CFDNN

Implements CFDNN (Conformable Fractional Deep Neural Network), a deep network trained by conformable fractional gradient descent in place of standard backpropagation.

The conformable fractional derivative of order $\alpha$ replaces the limit definition of the ordinary derivative with $D^\alpha f(t) = t^{1-\alpha} f'(t)$, which contains no Gamma function and no memory integral — it is purely local. Applied to a weight, the conformable fractional gradient is the ordinary gradient rescaled by $\theta^{1-\alpha}$. CFDNN trains in the super-integer regime $\alpha \in [1.2, 1.8]$, where this rescaling smooths the loss landscape and accelerates convergence. An optional weight-decay term enters with the same conformable scaling.

$$
\begin{aligned}
g_t &= \nabla_\theta E(\theta_t), \\
\theta_{t+1} &= \theta_t - \eta \left( \theta_t^{\,1-\alpha}\, g_t - \lambda\, \theta_t^{\,2-\alpha} \right).
\end{aligned}
$$

where $\theta$ are the weights, $\eta$ the learning rate, $g_t$ the gradient of the loss $E$, $\alpha \in [1.2, 1.8]$ the conformable fractional order, $\lambda$ the weight-decay coefficient, and $\theta^{1-\alpha}$ the conformable rescaling factor (setting $\lambda = 0$ and $\alpha = 1$ recovers ordinary gradient descent).

Reference: B. Ajarmah and H. Iwidat, "Conformable Fractional Deep Neural Networks (CFDNN) for high-speed cyber-attack detection", Scientific Reports 2026. https://doi.org/10.1038/s41598-026-45213-w

---
[Back to the Canon](../README.md)
