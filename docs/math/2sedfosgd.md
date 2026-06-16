# 2SEDFOSGD

Implements 2SEDFOSGD, a fractional-order SGD whose per-layer fractional exponent is steered by the Two-Scale Effective Dimension.

Fractional-order SGD replaces the integer first derivative with a Caputo-style fractional derivative, so each step carries a memory term built from the most recent parameter displacement, scaled by $1/\Gamma(2-\alpha)$. The novelty here is that the fractional order is not fixed: for each layer $\ell$ the order $\alpha_t^{(\ell)}$ is lowered in proportion to that layer's Two-Scale Effective Dimension (2SED), a curvature-aware measure blending nominal parameter count with local Fisher-information geometry. Layers with higher effective dimension get a smaller $\alpha$, sharpening the memory weighting where the loss landscape is more sensitive.

$$
\begin{aligned}
\alpha_t^{(\ell)} &= \alpha_0 - \beta\,\frac{d_\zeta^{(\ell)}(\varepsilon)\big|_t}{d_{\max}}, \\
\theta_{t+1}^{(\ell)} &= \theta_t^{(\ell)} - \frac{\eta_t}{\Gamma\!\left(2-\alpha_t^{(\ell)}\right)}\,\left(\left|\theta_t^{(\ell)}-\theta_{t-1}^{(\ell)}\right|+\delta\right)^{1-\alpha_t^{(\ell)}} g_t^{(\ell)}
\end{aligned}
$$

where $\theta^{(\ell)}$ are the parameters of layer $\ell$, $\eta_t$ is the diminishing step size (e.g. $\eta_t=\eta_0/\sqrt{t}$), $g_t^{(\ell)}=\nabla f(\theta_t)^{(\ell)}$ is the gradient, $\Gamma(\cdot)$ is the gamma function, $\alpha_t^{(\ell)}\in(0,1)$ is the adaptive fractional order with base $\alpha_0$ and sensitivity $\beta>0$, $d_\zeta^{(\ell)}(\varepsilon)$ is the layer's 2SED with $d_{\max}$ the maximum across layers, and $\delta>0$ guards the displacement term against stalling.

Reference: Mohammad Partohaghighi, Roummel Marcia, YangQuan Chen, "Effective Dimension Aware Fractional-Order Stochastic Gradient Descent for Convex Optimization Problems", arXiv 2025. https://arxiv.org/abs/2503.13764

---
[Back to the Canon](../index.md)
