# Hessian-aware Scaling

Implements Hessian-aware Scaling, a curvature-aware rescaling of gradient descent that picks the step magnitude from the curvature along the gradient.

Rather than computing a full Newton step, the method scales the negative gradient by a scalar $s_t$ derived from the Hessian-vector product $H_t g_t$ along the current gradient direction. When the curvature $\langle g_t, H_t g_t\rangle$ is strongly positive, the scaling matches a one-dimensional second-order step; an Armijo line search then sets the step size $\alpha_t$, which is provably $1$ near a minimizer. The CG, MR, and GM variants below differ only in how $s_t$ is formed from the same curvature quantities.

$$
\begin{aligned}
s_t^{\mathrm{CG}} &= \frac{\lVert g_t\rVert^2}{\langle g_t, H_t g_t\rangle}, \quad
s_t^{\mathrm{MR}} = \frac{\langle g_t, H_t g_t\rangle}{\lVert H_t g_t\rVert^2}, \quad
s_t^{\mathrm{GM}} = \sqrt{s_t^{\mathrm{MR}}\, s_t^{\mathrm{CG}}} = \frac{\lVert g_t\rVert}{\lVert H_t g_t\rVert}, \\
\theta_{t+1} &= \theta_t - \alpha_t\, s_t\, g_t,
\end{aligned}
$$

where $g_t = \nabla f(\theta_t)$, $H_t$ is the Hessian at $\theta_t$, $s_t > 0$ is the Hessian-aware scaling (chosen as above when $\langle g_t, H_t g_t\rangle > \sigma\lVert g_t\rVert^2$ for a tolerance $\sigma \ll 1$, and from prescribed ranges under limited or negative curvature), and $\alpha_t > 0$ is the Armijo backtracking step size accepting $f(\theta_t - \alpha s_t g_t) \le f(\theta_t) - \rho\,\alpha\, s_t \lVert g_t\rVert^2$ with $\rho \in (0, \tfrac{1}{2})$, initialized at $\alpha = 1$.

Reference: Oscar Smee, Fred Roosta, Stephen J. Wright, "First-ish Order Methods: Hessian-aware Scalings of Gradient Descent", arXiv 2025. https://arxiv.org/abs/2502.03701

---
[Back to the Canon](../README.md)
