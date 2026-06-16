# Step-Tuned SGD

Implements Step-Tuned SGD, an SGD variant that tunes its step size online from a second-order curvature estimate built out of gradient differences.

The method takes two consecutive half-steps per iteration and uses the change in parameters and gradients across the first half-step to approximate local curvature. An exponential moving average of these gradient differences yields a curvature vector, and the ratio of the squared parameter change to its inner product with that curvature gives a Barzilai-Borwein-style multiplier $\gamma$. The multiplier is clipped to a safe range and reused on a slowly decaying base schedule, so only the base learning rate needs careful tuning.

$$
\begin{aligned}
\theta_{k+1/2} &= \theta_k - \frac{\eta}{(k+1)^{1/2+\delta}}\,\gamma_k\, g_t(\theta_k) \\
\theta_{k+1} &= \theta_{k+1/2} - \frac{\eta}{(k+1)^{1/2+\delta}}\,\gamma_k\, g_t(\theta_{k+1/2}) \\
\Delta\theta &= \theta_{k+1/2} - \theta_k, \quad \Delta g = g_t(\theta_{k+1/2}) - g_t(\theta_k) \\
G_k &= \beta\, G_{k-1} + (1-\beta)\,\Delta g, \quad \hat{G}_k = \frac{G_k}{1-\beta^{\,k+1}} \\
\gamma_{k+1} &= \begin{cases} \dfrac{\lVert \Delta\theta \rVert^2}{\langle \hat{G}_k, \Delta\theta \rangle} & \text{if } \langle \hat{G}_k, \Delta\theta \rangle > 0 \\ \nu & \text{otherwise} \end{cases} \\
\gamma_{k+1} &= \min\!\big(\max(\gamma_{k+1}, \tilde{m}),\, \tilde{M}\big)
\end{aligned}
$$

where $\theta$ are the parameters, $\eta$ the base learning rate, $g_t(\cdot)$ the minibatch gradient, $\delta \in (0, 1/2)$ a decay exponent, $\gamma_k$ the tuned step multiplier, $\hat{G}_k$ the bias-corrected EMA of gradient differences with decay $\beta$, $\nu$ a fallback value for negative curvature, and $\tilde{m}, \tilde{M}$ the clipping bounds.

Reference: Camille Castera, Jérôme Bolte, Cédric Févotte, Edouard Pauwels, "Second-order step-size tuning of SGD for non-convex optimization", arXiv 2021. https://arxiv.org/abs/2103.03570

---
[Back to the Canon](../index.md)
