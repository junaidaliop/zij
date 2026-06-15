# AFOGD / AFOAGD

Implements AFOGD / AFOAGD, adaptive fractional-order gradient descent with an optional Nesterov-style acceleration term.

The method approximates the Caputo fractional derivative by truncating its Taylor expansion to the leading term, which introduces a factor of the form $(x_k - x_{k-1})^{1-\mu}$ with fractional order $\mu$. To avoid complex-valued powers of a signed vector, the displacement is replaced by its Euclidean norm plus a small constant, giving a scalar fractional weight $(\|x_k - x_{k-1}\|_2 + \delta)^{1-\mu}$ on the gradient step. A per-iteration adaptive coefficient $\beta_k$ keeps the effective step size bounded, which is what yields the robust-control convergence guarantees.

AFOGD applies this fractional weight directly to the gradient descent step. AFOAGD first forms an extrapolated point $y_k$ by momentum and evaluates the fractionally weighted gradient there.

$$
\begin{aligned}
\text{AFOGD:}\quad & x_{k+1} = x_k - \alpha\,\beta_k\,(\|x_k - x_{k-1}\|_2 + \delta)^{1-\mu}\,\nabla f(x_k) \\
\text{AFOAGD:}\quad & y_k = x_k + \eta\,(x_k - x_{k-1}) \\
& x_{k+1} = y_k - \alpha\,\beta_k\,(\|y_k - y_{k-1}\|_2 + \delta)^{1-\mu}\,\nabla f(y_k) \\
\text{subject to}\quad & 0 < c_1 \le \beta_k\,(\|x_k - x_{k-1}\|_2 + \delta)^{1-\mu} \le c_2 < \infty
\end{aligned}
$$

where $x$ are the parameters, $\nabla f$ the gradient, $\alpha > 0$ the learning rate, $\mu \in (0,2)$ the fractional order, $\delta > 0$ a small regularization constant, $\beta_k$ the per-iteration adaptive coefficient bounded by $c_1, c_2$, and $\eta \ge 0$ the momentum coefficient.

Reference: Jiaxu Liu, Song Chen, Shengze Cai, Chao Xu, "The Novel Adaptive Fractional Order Gradient Decent Algorithms Design Via Robust Control", arXiv 2023. https://arxiv.org/abs/2303.04328

---
[Back to the Canon](../README.md)
