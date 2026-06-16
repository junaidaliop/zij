# AGS-GD

Implements AGS-GD, gradient descent on an anisotropic Gaussian-smoothed objective.

Instead of descending on the raw loss, AGS-GD descends on a Gaussian convolution of it, replacing the gradient with the gradient of the smoothed function $f_\Sigma$. The smoothing covariance $\Sigma_t$ is anisotropic, so each direction can be smoothed at a different scale, letting the iterate average over local roughness while preserving sharp directions. In practice the smoothed gradient has no closed form and is estimated by Monte Carlo sampling of Gaussian directions.

$$
\begin{aligned}
\theta_t &= \theta_{t-1} - \eta\, \nabla f_{\Sigma_t}(\theta_{t-1}), \\
\nabla f_{\Sigma}(\theta) &= \frac{2}{\pi^{d/2}}\, \Sigma^{-1} \int_{\mathbb{R}^d} u\, f(\theta + \Sigma u)\, e^{-\lVert u \rVert^2}\, du, \\
\nabla f_{\Sigma}(\theta) &\approx \frac{1}{N} \sum_{n=1}^{N} \delta_{\Sigma}(\theta, u_n)\, \Sigma^{-1} u_n, \qquad \delta_{\Sigma}(\theta, u) = f(\theta + \Sigma u) - f(\theta - \Sigma u)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the learning rate, $f$ the loss, $\Sigma_t$ the symmetric invertible smoothing matrix at step $t$, $f_\Sigma$ the anisotropic Gaussian-smoothed objective, $d$ the dimension, and $u_n$ standard Gaussian samples used in the $N$-point Monte Carlo estimate of the smoothed gradient.

Reference: Andrew Starnes, Guannan Zhang, Viktor Reshniak, Clayton Webster, "Anisotropic Gaussian Smoothing for Gradient-based Optimization", arXiv 2024. https://arxiv.org/abs/2411.11747

---
[Back to the Canon](../index.md)
