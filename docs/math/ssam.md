# SSAM

Implements Stable SAM (SSAM), a renormalized variant of sharpness-aware minimization.

Sharpness-aware minimization (SAM) takes an ascent step to a nearby worst-case point, then descends using the gradient measured there. SSAM observes that this surrogate gradient can shrink in norm near saddle points and flat regions, stalling progress, and that SAM is therefore only well-behaved within a narrow learning-rate range. The fix is a single renormalization: rescale the perturbed gradient so its norm equals that of the clean gradient before the update, which widens the stable learning-rate regime without adding hyperparameters.

$$
\begin{aligned}
\theta_t^{\mathrm{adv}} &= \theta_t + \rho\,\frac{g_t}{\lVert g_t \rVert_2}, \\
\tilde{g}_t &= \nabla_\theta F_{B_t}(\theta)\big|_{\theta=\theta_t^{\mathrm{adv}}}, \\
\hat{g}_t &= \frac{\lVert g_t \rVert_2}{\lVert \tilde{g}_t \rVert_2}\,\tilde{g}_t, \\
\theta_{t+1} &= \theta_t - \eta\,\hat{g}_t.
\end{aligned}
$$

where $g_t = \nabla_\theta F_{B_t}(\theta_t)$ is the clean mini-batch gradient, $\rho > 0$ the perturbation radius, $\tilde{g}_t$ the gradient at the perturbed point $\theta_t^{\mathrm{adv}}$, $\hat{g}_t$ its renormalized version, and $\eta$ the learning rate.

Reference: Chengli Tan, Jiangshe Zhang, Junmin Liu, Yicheng Wang, Yunda Hao, "Stabilizing Sharpness-aware Minimization Through A Simple Renormalization Strategy", arXiv 2024. https://arxiv.org/abs/2401.07250

---
[Back to the Canon](../index.md)
