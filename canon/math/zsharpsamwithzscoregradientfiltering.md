# ZSharp (SAM with Z-Score Gradient Filtering)

Implements ZSharp, a sharpness-aware optimizer that applies layer-wise Z-score filtering to the gradient before computing the SAM ascent step.

ZSharp builds on Sharpness-Aware Minimization (SAM), which perturbs the parameters toward the worst-case direction within a radius $\rho$ and then descends using the gradient evaluated at that perturbed point. ZSharp refines the ascent direction: within each layer it normalizes the gradient to a Z-score, keeps only the components whose magnitude exceeds a high percentile threshold, and forms the perturbation from this filtered gradient. By retaining only the most statistically salient coordinates, the ascent probes the directions most responsible for sharpness while suppressing noisy, low-magnitude components.

$$
\begin{aligned}
\mu^{(\ell)} &= \frac{1}{d_\ell} \sum_{j=1}^{d_\ell} g_{t,j}^{(\ell)}, \qquad
\sigma^{(\ell)} = \Big( \frac{1}{d_\ell} \sum_{j=1}^{d_\ell} \big(g_{t,j}^{(\ell)} - \mu^{(\ell)}\big)^2 \Big)^{1/2} \\
z_j^{(\ell)} &= \frac{g_{t,j}^{(\ell)} - \mu^{(\ell)}}{\sigma^{(\ell)} + \epsilon} \\
m_j &= \begin{cases} 1 & \text{if } |z_j| > q_{Q_p} \\ 0 & \text{otherwise} \end{cases}, \qquad
\tilde g_t = g_t \odot m \\
\hat\epsilon_t &= \rho \, \frac{\tilde g_t}{\lVert \tilde g_t \rVert_2 + \epsilon} \\
\theta_{t+1} &= \theta_t - \gamma \, \nabla \mathcal{L}(\theta_t + \hat\epsilon_t)
\end{aligned}
$$

where $g_t = \nabla \mathcal{L}(\theta_t)$ is the gradient, $\mu^{(\ell)}$ and $\sigma^{(\ell)}$ are the mean and standard deviation of the gradient over the $d_\ell$ coordinates of layer $\ell$, $z_j$ is the per-coordinate Z-score, $m$ is the binary mask that keeps coordinates above the Z-score threshold $q_{Q_p}$ at percentile $Q_p$ (default $0.95$, retaining the top 5%), $\tilde g_t$ is the filtered gradient, $\hat\epsilon_t$ is the SAM perturbation of radius $\rho$ (default $0.05$), $\gamma$ is the learning rate, and $\epsilon = 10^{-8}$ guards against division by zero. If $\lVert \tilde g_t \rVert_2 = 0$, the unfiltered gradient $g_t$ is used for the perturbation.

Reference: Vincent-Daniel Yun, "Sharpness-Aware Minimization with Z-Score Gradient Filtering for Neural Networks", arXiv 2025. https://arxiv.org/abs/2505.02369

---
[Back to the Canon](../README.md)
