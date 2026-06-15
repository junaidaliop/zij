# SparseAdam

Implements SparseAdam, a variant of Adam for gradients that are sparse.

SparseAdam runs the Adam update lazily: at each step it touches only the
coordinates where the current gradient is nonzero. For those coordinates the
moments are advanced and the parameters are updated exactly as in Adam, while
the remaining coordinates and their moment estimates are left untouched. The
bias correction uses the per-coordinate count of nonzero updates rather than a
global step, so each active coordinate is corrected as if it had been the only
one updated. Let $\mathcal{I}_t = \{ i : (g_t)_i \neq 0 \}$ be the support of
the gradient at step $t$.

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\theta_t &= \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\quad \text{restricted to } i \in \mathcal{I}_t
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ is the learning rate, $g_t$ is the
sparse gradient, $\mathcal{I}_t$ is the set of its nonzero coordinates, $m_t$
and $v_t$ are the first and second moment estimates, $\beta_1, \beta_2$ are the
decay rates, and $\epsilon$ is the numerical-stability term. Coordinates outside
$\mathcal{I}_t$ keep their previous values of $\theta$, $m$, and $v$.

Reference: Diederik P. Kingma, Jimmy Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015.
https://arxiv.org/abs/1412.6980

---
[Back to the Canon](../README.md)
