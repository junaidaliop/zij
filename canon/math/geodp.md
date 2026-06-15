# GeoDP

Implements GeoDP, a geometric perturbation for DP-SGD that adds noise to a gradient's direction and magnitude separately.

Standard DP-SGD adds Gaussian noise to the clipped gradient as a whole, which injects biased, accumulating noise into the gradient direction. GeoDP instead converts the clipped gradient into spherical coordinates, a magnitude $r_t = \|g_t\|$ and a vector of direction angles $\theta_t$, then perturbs the magnitude and the direction independently with separately calibrated noise before converting back to rectangular coordinates. This keeps the perturbed direction unbiased while spending the same privacy budget.

$$
\begin{aligned}
g_t &\;\longleftrightarrow\; \big(\,r_t,\; \theta_t\,\big), \qquad r_t = \|g_t\| = \sqrt{\textstyle\sum_{z=1}^{d} g_{t,z}^2} \\
r_t^{\star} &= r_t + \frac{C}{B}\, n_\sigma \\
\theta_t^{\star} &= \theta_t + \frac{\sqrt{d+2}\,\beta\pi}{B}\, n_\sigma \\
g_t^{\star} &\;\longleftarrow\; \big(\,r_t^{\star},\; \theta_t^{\star}\,\big)
\end{aligned}
$$

where $g_t$ is the clipped mini-batch gradient, $C$ the clipping bound, $B$ the batch size, $d$ the dimension, $\beta\in(0,1]$ a bound on the angular sensitivity region, $n_\sigma\sim\mathcal{N}(0,\sigma^2 I)$ the Gaussian noise with multiplier $\sigma$, $\sqrt{d+2}\,\beta\pi$ the $\ell_2$ sensitivity of the direction component, and $g_t^{\star}$ the perturbed gradient used to update $\theta$.

Reference: Jiawei Duan, Haibo Hu, Qingqing Ye, Xinyue Sun, "Analyzing and Optimizing Perturbation of DP-SGD Geometrically", arXiv 2025. https://arxiv.org/abs/2504.05618

---
[Back to the Canon](../README.md)
