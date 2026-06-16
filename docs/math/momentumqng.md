# Momentum-QNG

Implements Momentum-QNG, quantum natural gradient with a momentum term derived from Langevin dynamics.

Standard quantum natural gradient preconditions the gradient by the inverse Fubini-Study metric tensor of the variational quantum state, yielding steepest descent in the natural (information) geometry of the parameter manifold. Momentum-QNG adds an inertial term: by discretizing the underdamped Langevin equation on this manifold, the parameter increment carries momentum from the previous step, which helps the optimizer traverse flat regions and escape local minima of the cost landscape.

$$
\begin{aligned}
\Delta\theta_{t+1} &= \rho\, \Delta\theta_t - \eta\, g^{-1}(\theta_t)\, \nabla \mathcal{L}(\theta_t) \\
\theta_{t+1} &= \theta_t + \Delta\theta_{t+1}
\end{aligned}
$$

where $\theta$ are the variational parameters, $\eta$ the learning rate, $\rho \in [0,1)$ the momentum coefficient, $g^{-1}(\theta_t)$ the inverse Fubini-Study metric tensor at $\theta_t$, and $\nabla\mathcal{L}(\theta_t)$ the gradient of the cost function. Setting $\rho = 0$ recovers plain quantum natural gradient.

Reference: Oleksandr Borysenko, Mykhailo Bratchenko, Ilya Lukin, Mykola Luhanko, Ihor Omelchenko, Andrii Sotnikov, Alessandro Lomi, "Application of Langevin Dynamics to Advance the Quantum Natural Gradient Optimization Algorithm", arXiv 2024. https://arxiv.org/abs/2409.01978

---
[Back to the Canon](../index.md)
