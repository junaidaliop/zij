# MOAOCFGD

Implements MOAOCFGD, an adaptive-order Caputo fractional gradient descent method for multi-objective optimization.

For a vector objective $f=(f_1,\dots,f_m)$ the method replaces each integer gradient with a regularized Caputo fractional gradient and then picks a single common descent direction by solving a quadratic subproblem that improves every objective at once. The fractional order $\alpha$ and the smoothing parameter $\beta$ are not fixed: the run is split into stages, and each stage $s$ uses its own pair $(\alpha_s,\beta_s)$, so the memory weighting sharpens as the iterates approach a Pareto-critical point. The direction $d^k$ is a convex combination of the fractional gradients (the multipliers come from the subproblem's KKT conditions), and the step size $\eta_k$ is found by an Armijo-type backtracking line search that must decrease all objectives simultaneously.

$$
\begin{aligned}
{}^{C}_{c}\nabla^{\alpha}_{x} f_{j,\alpha,\beta}(x) &= \mathrm{diag}\!\bigl({}^{C}_{c}\nabla^{\alpha}_{x} I(x)\bigr)^{-1}\Bigl[{}^{C}_{c}\nabla^{\alpha}_{x} f_{j}(x) + \beta\,\mathrm{diag}(|x-c|)\,{}^{C}_{c}\nabla^{1+\alpha}_{x} f_{j}(x)\Bigr] \\
(t^k, d^k) &= \arg\min_{(t,d)\,\in\,\mathbb{R}\times\mathbb{R}^n}\; t + \tfrac{1}{2}\,\|d\|^2 \quad \text{s.t. } {}^{C}_{c}\nabla^{\alpha}_{x} f_{j,\alpha,\beta}(x^k)^{\top} d - t \le 0,\;\; j=1,\dots,m \\
d^k &= -\sum_{j=1}^{m} \lambda_j^k\, {}^{C}_{c}\nabla^{\alpha}_{x} f_{j,\alpha,\beta}(x^k), \qquad \sum_{j=1}^{m}\lambda_j^k = 1,\;\; \lambda_j^k \ge 0 \\
\eta_k &= \max\bigl\{ r^{\,i} : i\in\mathbb{N}_0,\; f_j(x^k + r^{\,i} d^k) \le f_j(x^k) + \sigma\, r^{\,i}\, t^k \;\; \forall j \bigr\} \\
x^{k+1} &= x^k + \eta_k\, d^k
\end{aligned}
$$

where ${}^{C}_{c}\nabla^{\alpha}_{x}$ is the order-$\alpha$ Caputo fractional gradient with lower terminal $c$, $f_{j,\alpha,\beta}$ is its $\beta$-regularized form ($I(x)$ the integrand normalizer, $|x-c|$ applied componentwise), $\alpha\in(0,1]$ and $\beta\in\bigl[\tfrac{1-\alpha}{2-\alpha},\infty\bigr)$ are the stage-wise order and smoothing parameters, $d^k$ is the common descent direction with KKT multipliers $\lambda_j^k$, $t^k<0$ is the subproblem value at non-critical points, $\sigma\in(0,1)$ is the Armijo constant, $r\in(0,1)$ the backtracking factor, $\eta_k$ the step size, and the loop stops when $\|d^k\|<\epsilon$.

Reference: Barsha Shaw, Md Abu Talhamainuddin Ansary, "An Adaptive Order Caputo Fractional Gradient Descent Method for Multi-objective Optimization Problems", arXiv 2025. https://arxiv.org/abs/2507.07674

---
[Back to the Canon](../index.md)
