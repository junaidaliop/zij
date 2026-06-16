# CFGD (Caputo)

Implements CFGD, gradient descent driven by the Caputo fractional derivative instead of the ordinary gradient.

CFGD replaces each coordinate's gradient by a Caputo fractional derivative of order $\alpha \in (0,1)$, taken with respect to a per-coordinate lower terminal $c_j$. Because the fractional derivative is a weighted integral of the slope between $c_j$ and the current point, the resulting direction carries nonlocal history of the objective along each axis, which acts as an implicit Tikhonov-style regularizer and smooths the landscape. A second term scaled by $\beta$ adds the order-$(1+\alpha)$ derivative, and the bracket is normalized by the fractional derivative of the identity so the step reduces to ordinary gradient descent as $\alpha \to 1$.

The terminal $c_k$ may be fixed (non-adaptive, NA-CFGD) or lagged, $c_k = \theta^{(k-L)}$ (adaptive-terminal, AT-CFGD). In practice the per-coordinate fractional derivatives are evaluated by Gauss-Jacobi quadrature on the slope, giving the second line below.

$$
\begin{aligned}
\theta^{(k+1)} &= \theta^{(k)} - \eta_k\, d_k, \\
d_k &= \mathrm{diag}\!\left({}_{c_j}^{C}D_x^{\alpha_k} I(\theta_j^{(k)})\right)^{-1}\!\left[\nabla_{c_k}^{\alpha_k} f(\theta^{(k)}) + \beta_k\, \mathrm{diag}\!\left(|\theta_j^{(k)} - c_j^{(k)}|\right) \nabla_{c_k}^{1+\alpha_k} f(\theta^{(k)})\right], \\
\left(\nabla_{c_k}^{\alpha_k} f\right)_j &= \frac{1}{\Gamma(1-\alpha_k)} \int_{c_j^{(k)}}^{\theta_j^{(k)}} \frac{f_{j}'(t)}{(\theta_j^{(k)} - t)^{\alpha_k}}\, dt .
\end{aligned}
$$

where $\theta$ are the parameters, $\eta_k$ the learning rate, $\alpha_k \in (0,1)$ the fractional order, $c_j$ the per-coordinate integral terminal, $\beta_k$ the smoothing parameter, $f_j'$ the partial derivative of $f$ along coordinate $j$, $\Gamma$ the gamma function, ${}_{c}^{C}D_x^{\alpha}$ the left Caputo fractional derivative, $\nabla_c^{\alpha}$ and $\nabla_c^{1+\alpha}$ the per-coordinate Caputo fractional gradients of order $\alpha$ and $1+\alpha$, $I$ the identity map, and $\mathrm{diag}(\cdot)$ the diagonal matrix of a per-coordinate vector. Setting $\alpha_k \to 1$ recovers standard gradient descent.

Reference: Yeonjong Shin, Jérôme Darbon, George Em Karniadakis, "A Caputo fractional derivative-based algorithm for optimization", arXiv 2021. https://arxiv.org/abs/2104.02259

---
[Back to the Canon](../index.md)
