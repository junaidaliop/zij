# VR-SZD

Implements VR-SZD, a proximal stochastic variance-reduced zeroth-order method that builds gradient estimates from structured orthogonal directions.

VR-SZD minimizes a composite objective $\frac{1}{n}\sum_{i=1}^{n} f_i + h$ using only function evaluations. It follows a SVRG-style two-loop schedule: each outer round $\tau$ computes a full finite-difference gradient approximation $g_\tau$ at the snapshot point along the canonical basis, then runs $m$ inner steps that correct cheap stochastic zeroth-order estimates with this anchor.

The inner estimator probes each sampled component $f_i$ along $\ell$ directions drawn from an orthogonal matrix $G \in O(d)$, forming a structured forward-difference estimate $\hat{g}_i$. Subtracting the snapshot estimate from the current-point estimate and adding the full anchor yields the variance-reduced direction $v_k^\tau$, which drives a proximal step on the nonsmooth term $h$.

$$
\begin{aligned}
g_\tau &= \sum_{j=1}^{d} \frac{f(x_0^\tau + \beta_\tau e_j) - f(x_0^\tau)}{\beta_\tau}\, e_j, \\
\hat{g}_i(x, G, \beta) &= \frac{d}{\ell} \sum_{j=1}^{\ell} \frac{f_i(x + \beta G e_j) - f_i(x)}{\beta}\, G e_j, \\
v_k^\tau &= \frac{1}{b} \sum_{j=1}^{b} \left[ \hat{g}_{i_{j,k}^\tau}(x_k^\tau, G_{j,k}^\tau, \beta_\tau) - \hat{g}_{i_{j,k}^\tau}(x_0^\tau, G_{j,k}^\tau, \beta_\tau) \right] + g_\tau, \\
x_{k+1}^\tau &= \mathrm{prox}_{\gamma h}\!\left( x_k^\tau - \gamma\, v_k^\tau \right)
\end{aligned}
$$

where $x$ are the parameters, $\gamma$ the stepsize, $\beta_\tau$ the finite-difference discretization, $e_j$ the canonical basis vectors, $G \in O(d)$ an orthogonal matrix sampled uniformly at each inner step, $\ell$ the number of structured directions ($1 \le \ell \le d$), $b$ the batch size of components $i_{j,k}^\tau$ drawn uniformly from $[n]$, $\mathrm{prox}_{\gamma h}$ the proximal operator of the nonsmooth term $h$, and snapshots are refreshed by $x_0^{\tau+1} = x_m^\tau$.

Reference: Marco Rando, Cheik Traoré, Cesare Molinari, Lorenzo Rosasco, Silvia Villa, "A Structured Proximal Stochastic Variance Reduced Zeroth-order Algorithm", arXiv 2025. https://arxiv.org/abs/2506.23758

---
[Back to the Canon](../index.md)
