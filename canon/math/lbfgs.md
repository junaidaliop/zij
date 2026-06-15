# LBFGS

Implements L-BFGS, a limited-memory quasi-Newton method that approximates
the inverse Hessian from the most recent curvature pairs.

L-BFGS takes a Newton-like step $\theta_t = \theta_{t-1} - \eta\, H_t g_t$,
where $H_t$ approximates the inverse Hessian. Instead of storing the full
$H_t$, it keeps only the last $m$ pairs of parameter and gradient
differences $(s_i, y_i)$ and reconstructs the action $H_t g_t$ on the fly
through the two-loop recursion below, giving linear memory and cost per
step.

$$
\begin{aligned}
s_{t-1} &= \theta_{t-1} - \theta_{t-2}, \qquad
y_{t-1} = g_{t-1} - g_{t-2}, \qquad
\rho_i = \frac{1}{y_i^{\top} s_i} \\
q &= g_t \\
&\text{for } i = t-1, \dots, t-m: \quad
\alpha_i = \rho_i\, s_i^{\top} q, \quad
q \leftarrow q - \alpha_i\, y_i \\
r &= \frac{y_{t-1}^{\top} s_{t-1}}{y_{t-1}^{\top} y_{t-1}}\, q \\
&\text{for } i = t-m, \dots, t-1: \quad
\beta = \rho_i\, y_i^{\top} r, \quad
r \leftarrow r + (\alpha_i - \beta)\, s_i \\
\theta_t &= \theta_{t-1} - \eta\, r
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the step length (chosen by line
search), $g_t$ is the gradient, $s_i$ and $y_i$ are the parameter and
gradient differences of the stored pairs, $m$ is the history size, and
$r = H_t g_t$ is the resulting search direction.

Reference: D. C. Liu and J. Nocedal, "On the limited memory BFGS method for
large scale optimization", Mathematical Programming 1989.
https://doi.org/10.1007/BF01589116

---
[Back to the Canon](../README.md)
