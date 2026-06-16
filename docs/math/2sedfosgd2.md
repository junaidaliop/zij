# 2SEDFOSGD

Implements 2SEDFOSGD, fractional-order SGD whose fractional exponent is adapted per layer from the Two-Scale Effective Dimension.

Fractional-order SGD replaces the integer-order gradient step with a Caputo-style fractional difference, weighting the gradient by a power of the recent parameter change and a Gamma-function factor. 2SEDFOSGD makes the fractional order $\alpha$ dynamic: it estimates a Two-Scale Effective Dimension $d_\zeta$ for each layer from the Fisher information curvature and lowers $\alpha$ where the effective dimension is large, so flatter, higher-dimensional layers take more memory-weighted steps while sharp directions stay closer to plain SGD.

$$
\begin{aligned}
\alpha_t^{(\ell)} &= \alpha_0 - \beta\,\frac{d_\zeta^{(\ell)}(\varepsilon)\big|_t}{d_{\max}} \\
\mu_t &= \frac{\mu_0}{t^{\rho}}, \qquad 0.5 < \rho < 1 \\
\theta_{t+1}^{(\ell)} &= \theta_t^{(\ell)} - \frac{\mu_t}{\Gamma\!\left(2 - \alpha_t^{(\ell)}\right)}\,\left(\left|\theta_t^{(\ell)} - \theta_{t-1}^{(\ell)}\right| + \delta\right)^{1 - \alpha_t^{(\ell)}} g_t^{(\ell)}
\end{aligned}
$$

where $\theta^{(\ell)}$ are the parameters of layer $\ell$, $g_t^{(\ell)}$ its stochastic gradient, $\mu_t$ the decaying step size, $\Gamma$ the Gamma function, $\alpha_t^{(\ell)} \in (0,1]$ the adaptive fractional order with base $\alpha_0$ and tuning gain $\beta$, $\delta > 0$ a small offset preventing stalls, $\rho$ the step-size decay exponent, and $d_\zeta^{(\ell)}$ the Two-Scale Effective Dimension of layer $\ell$ normalized by its maximum $d_{\max} = \max_{\ell,k} d_\zeta^{(\ell)}(\varepsilon)\big|_k$.

Reference: Mohammad Partohaghighi, Roummel Marcia, YangQuan Chen, "More Optimal Fractional-Order Stochastic Gradient Descent for Non-Convex Optimization Problems", arXiv 2025. https://arxiv.org/abs/2505.02985

---
[Back to the Canon](../index.md)
