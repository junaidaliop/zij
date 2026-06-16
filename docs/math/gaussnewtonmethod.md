# Gauss-Newton Method

Implements the Gauss-Newton Method, a second-order method for nonlinear least-squares problems.

The Gauss-Newton method minimizes a sum of squared residuals $\frac{1}{2}\sum_i r_i(\theta)^2$, with residual vector $r(\theta)$ and Jacobian $J_t$. Rather than forming the true Hessian, it approximates it by $J_t^\top J_t$, dropping the terms involving second derivatives of the residuals. This makes each step a linear least-squares solve and avoids computing individual residual Hessians. Wedderburn showed the same iteration extends naturally to maximum quasi-likelihood estimation in generalized linear models, where it coincides with iteratively reweighted least squares.

$$
\begin{aligned}
J_t &= \frac{\partial r(\theta_t)}{\partial \theta}, \\
\theta_{t+1} &= \theta_t - \gamma \, \bigl(J_t^\top J_t\bigr)^{-1} J_t^\top \, r(\theta_t).
\end{aligned}
$$

where $\theta$ are the parameters, $r(\theta)$ is the residual vector, $J_t$ is its Jacobian at $\theta_t$, $J_t^\top J_t$ is the Gauss-Newton approximation to the Hessian, and $\gamma$ is the step size (with $\gamma = 1$ for the classical full step).

Reference: R. W. M. Wedderburn, "Quasi-likelihood functions, generalized linear models, and the Gauss—Newton method", Biometrika 61(3), 1974, 439–447. https://doi.org/10.1093/biomet/61.3.439

---
[Back to the Canon](../index.md)
