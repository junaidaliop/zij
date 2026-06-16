# Spectral Clipping (matrix-valued)

Implements Spectral Clipping, a matrix-valued gradient clipping rule that clamps the singular values of layer-wise gradient matrices.

Classical gradient clipping rescales a gradient by its vector norm, treating all parameters as a flat vector. The authors observe that data outliers tend to inflate only a few leading singular values of a layer's gradient matrix while the rest of the spectrum is unaffected. Spectral clipping addresses this directly: it takes the SVD of the gradient, caps any singular value above a threshold $\tau_t$ to that threshold, and reconstructs the gradient with the singular directions left untouched. The clipped gradient then drives an ordinary descent step.

$$
\begin{aligned}
G_t &= U_t\, \mathrm{diag}(\sigma_t)\, V_t^{\top} \\
\mathrm{clamp}_{\tau_t}(\sigma_t)_i &= \min(\sigma_{t,i},\, \tau_t) \\
\mathcal{C}_{\tau_t}(G_t) &= U_t\, \mathrm{diag}\!\big(\mathrm{clamp}_{\tau_t}(\sigma_t)\big)\, V_t^{\top} \\
\theta_{t+1} &= \theta_t - \eta\, \mathcal{C}_{\tau_t}(G_t)
\end{aligned}
$$

where $\theta$ is the matrix-valued parameter, $G_t$ its gradient with SVD factors $U_t, V_t$ and singular values $\sigma_t$, $\tau_t > 0$ is the spectral clipping threshold (optionally adapted from a moving average or quantile of the spectrum), $\eta$ is the learning rate, and $\mathcal{C}_{\tau_t}$ is the spectral clipping operator.

Reference: Alexander Yukhimchuk, Mladen Kolar, Martin Takáč, Sayantan Choudhury, "Gradient Clipping Beyond Vector Norms: A Spectral Approach for Matrix-Valued Parameters", arXiv 2026. https://arxiv.org/abs/2605.11838

---
[Back to the Canon](../index.md)
