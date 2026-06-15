# ScheduleFreeWrapper

Implements ScheduleFree, a wrapper that makes any base optimizer
schedule-free by replacing a learning-rate schedule with an interpolation
and a running average of the iterates.

The base optimizer maintains the iterate $z_t$. Gradients are evaluated at
the interpolated point $y_t$, a convex combination of $z_t$ and the running
average $x_t$ controlled by the momentum $\beta$. After the base step
produces $z_{t+1}$, the average $x_{t+1}$ is updated as a weighted mean of
all past iterates. The averaging plays the role a decaying schedule would,
so no schedule is required; evaluation and checkpointing use $x_t$, while
training computes gradients at $y_t$.

$$
\begin{aligned}
y_t &= (1 - \beta) z_t + \beta x_t \\
z_{t+1} &= z_t + \mathrm{base}\bigl(\nabla f(y_t)\bigr) \\
c_{t+1} &= \frac{w_{t+1}}{\sum_{i=1}^{t+1} w_i}, \qquad
w_t = t^{\gamma} \left(\max_{i \le t} \eta_i\right)^{p} \\
x_{t+1} &= (1 - c_{t+1}) x_t + c_{t+1} z_{t+1}
\end{aligned}
$$

where $z_t$ is the base iterate, $x_t$ is the running average, $y_t$ is the
interpolated point where the gradient $\nabla f$ is taken, $\beta$ is the
momentum, $\eta$ is the learning rate, $\gamma$ is the polynomial weighting
power, $p$ is the learning-rate weighting power, and $c_t$ is the average
checkpoint weight built from the per-step weights $w_t$.

Reference: Aaron Defazio, Xingyu Alice Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky, "The Road Less Scheduled", NeurIPS 2024.
https://arxiv.org/abs/2405.15682

---
[Back to the Canon](../README.md)
