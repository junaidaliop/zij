# GDA-AM

Implements GDA-AM, gradient descent ascent accelerated by Anderson mixing for minimax problems.

GDA-AM treats the simultaneous gradient descent ascent step as a fixed-point iteration $w_{t+1} = g(w_t)$ on the stacked variables $w = [x; y]$, and accelerates it with Anderson mixing. Instead of taking a single GDA step, it forms a window of the last $p$ residuals, solves a constrained least-squares problem for mixing coefficients, and extrapolates the next iterate as a weighted combination of recent GDA steps. This nonlinear extrapolation damps the rotational dynamics that cause vanilla GDA to diverge on saddle-point objectives.

$$
\begin{aligned}
g(w) &= w - \gamma\, V(w), \quad V(w) = \begin{bmatrix} \nabla_x f(x,y) \\ -\nabla_y f(x,y) \end{bmatrix} \\
f_i &= g(w_i) - w_i, \quad F_t = [\, f_{t-p_t},\ \dots,\ f_t \,], \quad p_t = \min(t, p) \\
\beta &= \arg\min_{\beta}\ \lVert F_t\, \beta \rVert_2 \quad \text{s.t.}\quad \sum_{i=0}^{p_t} \beta_i = 1 \\
w_{t+1} &= \sum_{i=0}^{p_t} \beta_i\, g(w_{t-p_t+i})
\end{aligned}
$$

where $w = [x; y]$ stacks the min and max variables, $V$ is the GDA vector field, $\gamma$ is the learning rate, $f_i$ are the fixed-point residuals collected in the matrix $F_t$, $p$ is the window (table) size with $p_t$ the current fill, and $\beta$ are the mixing coefficients constrained to sum to one. The table is restarted (cleared) every $p$ iterations.

Reference: Huan He, Shifan Zhao, Yuanzhe Xi, Joyce C. Ho, Yousef Saad, "GDA-AM: On the effectiveness of solving minimax optimization via Anderson Acceleration", ICLR 2022. https://arxiv.org/abs/2110.02457

---
[Back to the Canon](../index.md)
