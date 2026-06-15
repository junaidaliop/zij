# Lap2

Implements Lap2, a differentially private SGD that replaces the Gaussian mechanism with calibrated Laplace noise.

Lap2 follows the DP-SGD template: per-example gradients are clipped to bound their $\ell_2$ sensitivity to $C$, noise is added, and a plain SGD step is taken. Unlike standard DP-SGD, the perturbation is drawn from a multivariate Laplace distribution rather than a Gaussian, with scale $b$ calibrated jointly with the clipping norm through a majorization-theoretic moments accountant. Privacy loss depends on the sensitivity-to-noise ratio $\rho = C/b$, and the analysis yields a noise scale that grows with the clipping norm and the iteration count $T$.

$$
\begin{aligned}
\hat g_t(x) &= \frac{g_t(x)}{\max\!\left(1,\ \lVert g_t(x)\rVert_2 / C\right)} \\
\tilde g_t &= \frac{1}{L}\sum_{x} \left( \hat g_t(x) + w \right),\qquad w \sim \mathrm{Lap}(b)^{n} \\
b^{*} &\approx \frac{2\zeta}{\epsilon_{\mathrm{tar}}}\,\sqrt{T\,\log(1/\delta)}\; C \\
\theta_{t+1} &= \theta_t - \eta\, \tilde g_t
\end{aligned}
$$

where $g_t(x)$ is the per-example gradient, $C$ the clipping norm, $L$ the lot size, $w$ a length-$n$ vector of i.i.d. Laplace samples with scale $b$ and density $\propto \exp(-\lVert w\rVert_1 / b)$, $\eta$ the learning rate, $T$ the number of steps, $(\epsilon_{\mathrm{tar}},\delta)$ the target privacy budget, $\zeta$ the sampling rate, and $\rho = C/b$ the sensitivity-to-noise ratio optimized by the accountant.

Reference: Meisam Mohammady, Qin Yang, Nicholas Stout, Ayesha Samreen, Han Wang, Christopher J. Quinn, Yuan Hong, "Lap2: Revisiting Laplace DP-SGD for High Dimensions via Majorization Theory", arXiv 2026. https://arxiv.org/abs/2602.23516

---
[Back to the Canon](../README.md)
