# FNGD

Implements FNGD, natural gradient descent in which the ordinary gradient is replaced by a fractional-order gradient.

FNGD is an information-geometric method: rather than following the raw gradient, it preconditions the descent direction by the inverse Fisher information matrix of a chosen output distribution (the authors use a Dirichlet distribution), so the step accounts for the curvature of the loss surface. On top of this, the loss gradient $\nabla \mathcal{L}$ is replaced by a fractional gradient $\nabla^{\alpha} \mathcal{L}$ of order $\alpha$, evaluated through the Riemann-Liouville, Caputo, or Grünwald-Letnikov definitions. The memory of the fractional derivative reshapes the direction and lets the iterate settle in the neighborhood of the global minimum, while a Tikhonov term $\lambda I$ keeps the Fisher matrix invertible. The authors also give momentum and Nesterov-accelerated forms of this update; setting $\alpha \to 1$ recovers ordinary natural gradient descent.

$$
\begin{aligned}
\theta_{t+1} &= \theta_{t} - \eta\, (F_t + \lambda I)^{-1}\, \nabla^{\alpha} \mathcal{L}(\theta_t), \\
\left(\nabla^{\alpha} \mathcal{L}\right)_j &= {}_{a}D_{\theta_j}^{\alpha}\, \mathcal{L}(\theta_t), \qquad
F_t = \mathbb{E}\!\left[\nabla \log p(\theta_t)\, \nabla \log p(\theta_t)^{\top}\right].
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $F_t$ the Fisher information matrix of the model distribution $p$ (Dirichlet), $\lambda$ the Tikhonov damping, $I$ the identity, $\nabla^{\alpha}\mathcal{L}$ the fractional gradient of the loss with per-coordinate left fractional derivative ${}_{a}D_{\theta_j}^{\alpha}$ of order $\alpha$ (Riemann-Liouville, Caputo, or Grünwald-Letnikov, with lower terminal $a$), and $\nabla \log p$ the score function. Setting $\alpha \to 1$ reduces $\nabla^{\alpha}\mathcal{L}$ to the ordinary gradient and recovers natural gradient descent.

Reference: Ruslan Ibrahimovich Abdulkadirov, Pavel Alekseevich Lyakhov, Valentina Alekseevna Baboshina, Nikolay Nikolaevich Nagornov, "Improving the Accuracy of Neural Network Pattern Recognition by Fractional Gradient Descent", IEEE Access 2024. https://doi.org/10.1109/ACCESS.2024.3491614

---
[Back to the Canon](../README.md)
