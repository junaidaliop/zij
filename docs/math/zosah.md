# ZO-SAH

Implements ZO-SAH, a zeroth-order method that builds an approximate Hessian inside randomly chosen two-dimensional subspaces to take Newton-style steps from function evaluations alone.

ZO-SAH targets problems where gradients are unavailable and a full Hessian estimate would be prohibitively expensive. Instead of working in the ambient space, it draws a set of $m/2$ orthogonal two-dimensional subspaces. Within each subspace $j$ it forms a zeroth-order gradient $g_j$ by coordinate-wise forward finite differences and an approximate Hessian $H_j$ by fitting a quadratic polynomial to a small grid of function values via regularized least squares. The local Newton direction $H_j^{-1} g_j$ is projected back to the full space by the inverse of the orthogonal projection $P_j$, and the contributions of all subspaces are summed into one descent direction. A periodic subspace-switching schedule reuses function evaluations across iterations to keep the query cost low.

$$
\begin{aligned}
g_j &= \frac{1}{\epsilon}\big(f(\theta + \epsilon\, e_1) - f(\theta),\; f(\theta + \epsilon\, e_2) - f(\theta)\big)^{\top}, \qquad \theta = P_j x \\
H_j &= \arg\min_{H}\; \sum_i \big(q(\bar{\theta}_i) - \phi(\bar{\theta}_i)^{\top} H\big)^2 \\
v &= \sum_{j=1}^{m/2} P_j^{-1}\big(H_j^{-1} g_j\big) \\
x_{t+1} &= x_t - \eta\, v
\end{aligned}
$$

where $x$ are the parameters, $\eta$ the step size, $\epsilon > 0$ the finite-difference smoothing size, $e_1, e_2$ the basis vectors of the two-dimensional subspace, $P_j$ the orthogonal projection onto subspace $j$ (with $P_j^{-1}$ embedding the subspace direction back into the full space), $g_j$ the zeroth-order subspace gradient, $H_j$ the approximate subspace Hessian obtained by fitting the quadratic $\phi(\cdot)^{\top} H$ to sampled function values $q(\bar{\theta}_i)$, $m$ the intermediate subspace dimension, and $v$ the aggregated descent direction.

Reference: Dongyoon Kim, Sungjae Lee, Wonjin Lee, Kwang In Kim, "Subspace-based Approximate Hessian Method for Zeroth-Order Optimization", arXiv 2025. https://arxiv.org/abs/2507.06125

---
[Back to the Canon](../index.md)
