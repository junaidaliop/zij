# AEGDM

Implements AEGDM, an adaptive gradient method that couples an energy variable with momentum.

AEGDM rescales each gradient by twice the square root of the shifted loss, then accumulates these transformed gradients with classical momentum. A per-coordinate energy variable $r_t$ relaxes toward zero through an unconditionally dissipative update, and the step is the product of energy and momentum. The energy mechanism guarantees stability for any base learning rate while preserving the effective adaptivity of the step size.

$$
\begin{aligned}
v_t &= \frac{g_t}{2\sqrt{f_t(\theta_t) + c}} \\
m_{t+1} &= \mu\, m_t + v_t \\
r_{t+1} &= \frac{r_t}{1 + 2\eta\, v_t \odot v_t} \\
\theta_{t+1} &= \theta_t - 2\eta\, r_{t+1} \odot m_{t+1}
\end{aligned}
$$

where $g_t = \nabla f_t(\theta_t)$ is the gradient, $f_t(\theta_t)$ is the loss, $c$ is a constant with $f_t(\theta) + c > 0$, $\eta$ is the base learning rate, $\mu$ is the momentum parameter, $r_t$ is the per-coordinate energy initialized to $r_0 = \sqrt{f_0(\theta_0) + c}$, $m_0 = 0$, and $\odot$ denotes elementwise multiplication.

Reference: Hailiang Liu, Xuping Tian, "An Adaptive Gradient Method with Energy and Momentum", arXiv 2022. https://arxiv.org/abs/2203.12191

---
[Back to the Canon](../index.md)
