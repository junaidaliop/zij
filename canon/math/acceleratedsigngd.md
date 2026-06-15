# Accelerated SignGD

Implements Accelerated SignGD, a momentum-accelerated variant of sign gradient descent.

SignGD takes a step in the direction of the sign of the gradient, decoupling the step magnitude from the gradient norm. Accelerated SignGD prepends a Nesterov-style extrapolation: the gradient is evaluated at a point pushed ahead along the previous update direction by a momentum factor $\beta$, and the sign step is taken from there. A safeguard resets the extrapolation ($\beta \to 0$ for that step) whenever the look-ahead point increases the objective, keeping the descent monotone.

$$
\begin{aligned}
v_t &= \theta_t + \beta\,(\theta_t - \theta_{t-1}) \\
\theta_{t+1} &= v_t - \eta\,\mathrm{sign}\!\big(\nabla f(v_t)\big)
\end{aligned}
$$

where $\theta_t$ are the parameters, $v_t$ is the extrapolated look-ahead point, $\eta$ is the step size, $\beta \in [0,1)$ is the momentum coefficient, $\nabla f(v_t)$ is the gradient at the look-ahead point, and $\mathrm{sign}(\cdot)$ is applied elementwise; if $f(v_t) > f(\theta_t)$ the step restarts with $v_t \leftarrow \theta_t$.

Reference: Valentin Leplat, Sergio Mayorga, Roland Hildebrand, Alexander Gasnikov, "Norm-Constrained Flows and Sign-Based Optimization: Theory and Algorithms", arXiv 2025. https://arxiv.org/abs/2508.18510

---
[Back to the Canon](../README.md)
