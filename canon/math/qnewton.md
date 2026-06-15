# Q-Newton

Implements Q-Newton, Newton's gradient descent with hybrid quantum-classical scheduling of the Hessian inversion.

Q-Newton trains neural networks with the full Newton step, preconditioning the gradient by the inverse Hessian. Its contribution is a scheduler that, at each step, estimates the cost of inverting the Hessian on a classical LU solver versus a quantum linear-systems solver and dispatches to whichever is cheaper; the parameter-update rule itself is the standard regularized Newton iteration. The Hessian is optionally regularized as $H + \epsilon I$ to improve conditioning before inversion.

$$
\begin{aligned}
\Delta\theta_t &= -\,(H_t + \epsilon I)^{-1}\, g_t, \\
\theta_{t+1} &= \theta_t + \beta\, \Delta\theta_t.
\end{aligned}
$$

where $g_t = \nabla_\theta \mathcal{J}(\theta_t)$ is the gradient of the loss, $H_t = \nabla_\theta^2 \mathcal{J}(\theta_t)$ is its Hessian, $\beta$ is the (fixed) learning rate, $\epsilon$ is the regularization term, and $I$ is the identity matrix.

Reference: Pingzhi Li, Junyu Liu, Hanrui Wang, Tianlong Chen, "Hybrid Quantum-Classical Scheduling for Accelerating Neural Network Training with Newton's Gradient Descent", arXiv 2024. https://arxiv.org/abs/2405.00252

---
[Back to the Canon](../README.md)
