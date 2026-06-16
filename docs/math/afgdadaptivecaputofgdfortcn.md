# AFGD (adaptive Caputo FGD for TCN)

Implements AFGD, an adaptive Caputo fractional gradient descent for training temporal convolutional networks.

AFGD replaces the integer-order gradient with a Caputo fractional-order gradient, so each step carries a memory term that weights the recent parameter displacement, scaled by $1/\Gamma(2-\alpha)$. The non-local, memory-dependent nature of the Caputo derivative lets the update absorb historical information rather than only the instantaneous slope. The fractional order $\alpha$ is not fixed: it is adapted during training, and the authors prove that under suitable conditions on the activations and the learning rate the loss decreases monotonically. As $\alpha \to 1$ the rule recovers ordinary gradient descent.

Per parameter, the practical one-term Caputo fractional gradient and the resulting update are

$$
\begin{aligned}
g_t^{C} &= \frac{\left|\theta_t - c\right|^{\,1-\alpha_t}}{\Gamma(2-\alpha_t)}\, g_t, \\
\theta_{t+1} &= \theta_t - \eta\, g_t^{C},
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t = \nabla f(\theta_t)$ is the ordinary gradient, $g_t^{C}$ is the Caputo fractional gradient, $\alpha_t \in (0,1]$ is the adaptive fractional order, $c$ is the fixed lower terminal of the fractional derivative (the memory anchor, often the previous iterate $\theta_{t-1}$), and $\Gamma(\cdot)$ is the gamma function.

Reference: Xiao et al., "Monotonic convergence of adaptive Caputo fractional gradient descent for temporal convolutional networks", Neurocomputing 656 (2025). https://doi.org/10.1016/j.neucom.2025.131491

---
[Back to the Canon](../index.md)
