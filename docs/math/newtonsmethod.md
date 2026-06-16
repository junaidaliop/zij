# Newton's Method

Implements Newton's Method, a second-order step that minimizes the local quadratic model of the objective.

Newton's method models the objective around the current iterate with a second-order Taylor expansion and steps to the minimizer of that quadratic. Setting the gradient of the model to zero yields a step given by the inverse Hessian applied to the gradient. A step size $\eta$ (chosen by line search, or $\eta = 1$ for the pure method) damps the step to ensure progress when the quadratic model is inaccurate. The method exhibits local quadratic convergence near a minimizer where the Hessian is positive definite.

$$
\begin{aligned}
g_t &= \nabla f(\theta_t) \\
H_t &= \nabla^2 f(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta\, H_t^{-1} g_t
\end{aligned}
$$

where $\theta$ are the parameters, $g_t$ is the gradient, $H_t$ is the Hessian of the objective $f$, $H_t^{-1} g_t$ is the Newton direction, and $\eta$ is the step size (equal to $1$ for the undamped method).

Reference: J. J. Moré and D. C. Sorensen, "Newton's Method", Argonne National Laboratory technical report ANL-82-8, 1982. https://www.osti.gov/biblio/5326201

---
[Back to the Canon](../index.md)
